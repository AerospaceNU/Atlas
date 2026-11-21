from __future__ import annotations

import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import Tool, ToolResult
from atlas.agent.tools import _jobs
from atlas.agent.tools._process import OUTPUT_LIMIT_NOTE, run_captured

MAX_TIMEOUT = 900

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")
_TRIMMED_MARKER = "... [truncated "
_TRIMMED_NOTE = (
    "note: output was trimmed to its head and tail; redirect to a file (command > out.txt) "
    "and use read_file or grep for the full text"
)
_START_WAIT = 0.5


class BashInputModel(BaseModel):
    command: str = Field(description="The shell command to execute")
    timeout: int = Field(
        default=30,
        ge=1,
        description=(
            "Seconds (1-900) to allow the command to run before it's killed; "
            "larger values are clamped to 900; ignored for background jobs"
        ),
    )
    working_directory: str = Field(
        default=".",
        description=(
            "Directory to run in. A relative path resolves against the workspace; "
            "an absolute path is used as given."
        ),
    )
    run_in_background: bool = Field(
        default=False,
        description=(
            "Start a managed background job for servers, watchers or long builds; read it "
            "with the jobs tool. The command must stay in the foreground: no trailing & or nohup."
        ),
    )
    lifetime_seconds: int = Field(
        default=3600,
        ge=1,
        description="Background only: seconds (60-86400) before the job is stopped.",
    )


class BashTool(Tool):
    """Run a shell command in the workspace with no path confinement.

    The working directory defaults to the project workspace when one is
    mounted, and the artifact root otherwise. Absolute paths, ``~``, ``..``,
    command substitution, globs, and interpreter eval are all allowed. The
    child inherits the process environment, so credentials and toolchain
    variables the host holds are visible to the command.

    Each call still uses a bounded timeout and kills the whole process group
    when it returns. A child that calls ``setsid()`` leaves that group.
    """

    name = "bash"
    description = (
        "Run a shell command with bash. Each call is a new shell that starts in the workspace "
        "root (or working_directory); cd and exported variables do not carry over to the next "
        "call. stdout and stderr are captured and a non-zero exit is a result, not an error. "
        "timeout is in seconds (default 30, max 900). Prefer run_tests for test suites and "
        "grep, glob and read_file for searching and reading files. Foreground commands are "
        "killed, with their children, when the call returns; use run_in_background for "
        "servers and watchers, then the jobs tool."
    )
    trust: ClassVar[Literal["default", "opt_in"]] = "default"
    input_model = BashInputModel

    max_output_chars: ClassVar[int] = 200_000

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        request = BashInputModel.model_validate(arguments)

        cwd = self._resolve_cwd(request.working_directory, store)
        if not cwd.is_dir():
            return ToolResult(text=f"Working directory not found: {request.working_directory}")

        if request.run_in_background:
            return _start_background(request, cwd, store)

        timeout = min(request.timeout, MAX_TIMEOUT)
        captured = run_captured(
            _shell_argv(request.command),
            cwd=cwd,
            env=_command_env(store),
            timeout=timeout,
            max_chars=self.max_output_chars,
        )
        stdout = self._sanitize(captured.stdout)
        stderr = self._sanitize(captured.stderr)
        note = f"note: timeout clamped to {MAX_TIMEOUT}s\n" if request.timeout > MAX_TIMEOUT else ""
        if captured.output_limited:
            note += f"{OUTPUT_LIMIT_NOTE}\n"
        if _TRIMMED_MARKER in stdout or _TRIMMED_MARKER in stderr:
            note += _TRIMMED_NOTE + "\n"
        if captured.cancelled:
            return ToolResult(text=f"{note}Command cancelled\nstdout:\n{stdout}\nstderr:\n{stderr}")
        if captured.timed_out:
            return ToolResult(
                text=(
                    f"{note}Command timed out after {timeout}s: {request.command}\n"
                    f"stdout:\n{stdout}\nstderr:\n{stderr}\n"
                    f"hint: rerun with a larger timeout (max {MAX_TIMEOUT}), "
                    "narrow the command, or use run_in_background for long-running processes"
                )
            )
        return ToolResult(
            text=f"{note}exit_code: {captured.exit_code}\nstdout:\n{stdout}\nstderr:\n{stderr}"
        )

    def _resolve_cwd(self, working_directory: str, store: LocalArtifactStore) -> Path:
        """Return the directory for the command.

        The default (``.``) is the project workspace when one is mounted, else
        the artifact root. Any other value is resolved against the working
        tree, and an absolute path is used as given.
        """
        if working_directory in ("", "."):
            return store.read_root or store.root
        return store.resolve_read(working_directory)

    def _sanitize(self, text: str) -> str:
        return _ANSI_ESCAPE_RE.sub("", text)


def _shell_argv(command: str) -> list[str]:
    """Return argv that runs ``command`` under bash, or sh when bash is absent."""
    bash = shutil.which("bash")
    if bash is None:
        return ["/bin/sh", "-c", command]
    return [bash, "-c", command]


def _analysis_dir(store: LocalArtifactStore) -> Path:
    from atlas.geo.paths import analysis_dir

    return analysis_dir(store)


def _command_env(store: LocalArtifactStore) -> dict[str, str]:
    """Environment shared by foreground and background commands."""
    return {
        **os.environ,
        "ATLAS_PYTHON": sys.executable,
        "ATLAS_WORKSPACE": str(store.read_root or store.root),
        "ATLAS_ARTIFACT_ROOT": str(store.artifact_root),
        "ATLAS_ANALYSIS_DIR": str(_analysis_dir(store)),
    }


def _start_background(request: BashInputModel, cwd: Path, store: LocalArtifactStore) -> ToolResult:
    """Start a managed job and report its id, log path and first output."""
    try:
        job = _jobs.start_job(
            _shell_argv(request.command),
            command=request.command,
            cwd=cwd,
            env=_command_env(store),
            artifact_root=store.artifact_root,
            lifetime=request.lifetime_seconds,
        )
    except _jobs.JobError as exc:
        return ToolResult(text=str(exc))
    deadline = time.monotonic() + _START_WAIT
    while time.monotonic() < deadline and job.proc.poll() is None:
        time.sleep(0.1)
    first = _jobs.read_output(job, from_byte=0, max_chars=2000)
    text = f"started job {job.id} (pid {job.pid})\nlog: {job.log_path}\n"
    text += f"lifetime: {job.lifetime}s; stopped when the session ends\n"
    if job.lifetime != request.lifetime_seconds:
        text += f"note: lifetime clamped to {job.lifetime}s\n"
    if not first.running:
        text += f"exited with code {first.exit_code} within {_START_WAIT}s\n"
    return ToolResult(text=text + f"first output:\n{first.text}")
