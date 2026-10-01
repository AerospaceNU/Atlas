from __future__ import annotations

import contextlib
import os
import re
import signal
import subprocess
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import Tool, ToolResult

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

# Only pass through the environment a shell needs to find and run ordinary
# programs. Everything else the host process holds (API keys, tokens, ...)
# stays out of the child's reach.
_INHERITED_ENV_KEYS = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "USER", "SHELL")


class BashInputModel(BaseModel):
    command: str = Field(description="The shell command to execute")
    timeout: int = Field(
        default=30, description="Maximum seconds to allow the command to run before it's killed"
    )
    working_directory: str = Field(
        default=".",
        description="Directory to run the command in, relative to the artifact workspace root",
    )


class BashTool(Tool):
    """Run a shell command confined to one local artifact workspace.

    Three invariants hold: the working directory is resolved only through
    :class:`~atlas.agent.artifacts.LocalArtifactStore`, so it cannot leave the
    workspace root; the child runs with a minimal, explicit environment
    rather than inheriting the host process's secrets; and a timeout kills
    the whole process group, so a backgrounded or disowned child cannot
    outlive it.
    """

    name = "bash_tool"
    description = (
        "Run a command with a working directory inside the sandbox, a timeout, "
        "captured stdout/stderr, and a non-zero exit reported as a result rather "
        "than an exception"
    )
    trust = "default"
    input_model = BashInputModel

    max_output_chars: ClassVar[int] = 200_000

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        request = BashInputModel.model_validate(arguments)

        cwd = store.resolve(request.working_directory)
        if not cwd.is_dir():
            return ToolResult(text=f"Working directory not found: {request.working_directory}")

        env = {key: os.environ[key] for key in _INHERITED_ENV_KEYS if key in os.environ}

        process = subprocess.Popen(
            request.command,
            shell=True,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=request.timeout)
        except subprocess.TimeoutExpired:
            self._kill_process_group(process)
            return ToolResult(text=f"Command timed out after {request.timeout}s: {request.command}")

        output = (
            f"exit_code: {process.returncode}\n"
            f"stdout:\n{self._sanitize(stdout)}\n"
            f"stderr:\n{self._sanitize(stderr)}"
        )
        return ToolResult(text=output)

    def _kill_process_group(self, process: subprocess.Popen[str]) -> None:
        # start_new_session=True makes this process its own session and
        # process-group leader, so its pid *is* the group id for as long as
        # any member of the group is alive. We rely on that identity rather
        # than re-querying os.getpgid() at kill time: on macOS, getpgid()
        # raises ProcessLookupError once this particular pid has zombied
        # (which happens almost immediately for a shell that only
        # backgrounds a job and exits), even though other processes in the
        # same group are still very much alive.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()

    def _sanitize(self, text: str) -> str:
        cleaned = _ANSI_ESCAPE_RE.sub("", text)
        if len(cleaned) > self.max_output_chars:
            omitted = len(cleaned) - self.max_output_chars
            cleaned = cleaned[: self.max_output_chars] + f"\n... [truncated {omitted} characters]"
        return cleaned
