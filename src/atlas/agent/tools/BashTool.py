from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import Tool, ToolResult
from atlas.agent.tools._process import OUTPUT_LIMIT_NOTE, run_captured

MAX_TIMEOUT = 900

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


class BashInputModel(BaseModel):
    command: str = Field(description="The shell command to execute")
    timeout: int = Field(
        default=30,
        ge=1,
        description=(
            "Seconds (1-900) to allow the command to run before it's killed; "
            "larger values are clamped to 900"
        ),
    )
    working_directory: str = Field(
        default=".",
        description=(
            "Directory to run in. A relative path resolves against the workspace; "
            "an absolute path is used as given."
        ),
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
        "grep, glob and read_file for searching and reading files."
    )
    trust: ClassVar[Literal["default", "opt_in"]] = "default"
    input_model = BashInputModel

    max_output_chars: ClassVar[int] = 200_000

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        request = BashInputModel.model_validate(arguments)

        cwd = self._resolve_cwd(request.working_directory, store)
        if not cwd.is_dir():
            return ToolResult(text=f"Working directory not found: {request.working_directory}")

        timeout = min(request.timeout, MAX_TIMEOUT)
        env = {
            **os.environ,
            "ATLAS_PYTHON": sys.executable,
            "ATLAS_WORKSPACE": str(store.read_root or store.root),
            "ATLAS_ARTIFACT_ROOT": str(store.artifact_root),
        }
        captured = run_captured(
            _shell_argv(request.command),
            cwd=cwd,
            env=env,
            timeout=timeout,
            max_chars=self.max_output_chars,
        )
        stdout = self._sanitize(captured.stdout)
        stderr = self._sanitize(captured.stderr)
        note = f"note: timeout clamped to {MAX_TIMEOUT}s\n" if request.timeout > MAX_TIMEOUT else ""
        if captured.output_limited:
            note += f"{OUTPUT_LIMIT_NOTE}\n"
        if captured.timed_out:
            return ToolResult(
                text=(
                    f"{note}Command timed out after {timeout}s: {request.command}\n"
                    f"stdout:\n{stdout}\nstderr:\n{stderr}\n"
                    f"hint: rerun with a larger timeout (max {MAX_TIMEOUT}) "
                    "or narrow the command"
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
