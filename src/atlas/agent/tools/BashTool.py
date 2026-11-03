from __future__ import annotations

import contextlib
import os
import re
import shutil
import signal
import subprocess
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import Tool, ToolResult

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


class BashInputModel(BaseModel):
    command: str = Field(description="The shell command to execute")
    timeout: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Seconds (1-300) to allow the command to run before it's killed",
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

    name = "bash_tool"
    description = (
        "Run a shell command. The working directory defaults to the workspace; "
        "relative paths resolve there and absolute paths are allowed. "
        "stdout and stderr are captured, and a non-zero exit is a result rather than an exception."
    )
    trust: ClassVar[Literal["default", "opt_in"]] = "default"
    input_model = BashInputModel

    max_output_chars: ClassVar[int] = 200_000

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        request = BashInputModel.model_validate(arguments)

        cwd = self._resolve_cwd(request.working_directory, store)
        if not cwd.is_dir():
            return ToolResult(text=f"Working directory not found: {request.working_directory}")

        process = subprocess.Popen(
            _shell_argv(request.command),
            cwd=cwd,
            # The session's own stdin carries the user's protocol messages.
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            start_new_session=True,
        )
        # start_new_session=True makes the shell the leader of a new process
        # group whose id is its pid. Captured now because communicate() reaps
        # the shell, and on macOS os.getpgid() fails once the leader has exited.
        pgid = process.pid
        try:
            stdout, stderr = process.communicate(timeout=request.timeout)
        except subprocess.TimeoutExpired:
            return ToolResult(text=f"Command timed out after {request.timeout}s: {request.command}")
        finally:
            # A background child that redirects its stdio lets communicate()
            # return normally, so the group is killed on every exit path.
            self._kill_process_group(process, pgid)

        output = (
            f"exit_code: {process.returncode}\n"
            f"stdout:\n{self._sanitize(stdout)}\n"
            f"stderr:\n{self._sanitize(stderr)}"
        )
        return ToolResult(text=output)

    def _resolve_cwd(self, working_directory: str, store: LocalArtifactStore) -> Path:
        """Return the directory for the command.

        The default (``.``) is the project workspace when one is mounted, else
        the artifact root. Any other value is resolved against the working
        tree, and an absolute path is used as given.
        """
        if working_directory in ("", "."):
            return store.read_root or store.root
        return store.resolve_read(working_directory)

    def _kill_process_group(self, process: subprocess.Popen[str], pgid: int) -> None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pgid, signal.SIGKILL)
        process.wait()

    def _sanitize(self, text: str) -> str:
        cleaned = _ANSI_ESCAPE_RE.sub("", text)
        if len(cleaned) > self.max_output_chars:
            omitted = len(cleaned) - self.max_output_chars
            cleaned = cleaned[: self.max_output_chars] + f"\n... [truncated {omitted} characters]"
        return cleaned


def _shell_argv(command: str) -> list[str]:
    """Return argv that runs ``command`` under bash, or sh when bash is absent."""
    bash = shutil.which("bash")
    if bash is None:
        return ["/bin/sh", "-c", command]
    return [bash, "-c", command]
