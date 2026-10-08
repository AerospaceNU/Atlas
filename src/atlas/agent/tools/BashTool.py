from __future__ import annotations

import contextlib
import os
import re
import signal
import subprocess
from pathlib import Path
from typing import Any, ClassVar, Literal

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
        default=30,
        ge=1,
        le=300,
        description="Seconds (1-300) to allow the command to run before it's killed",
    )
    working_directory: str = Field(
        default=".",
        description="Directory to run in, relative to the workspace (default: the workspace root)",
    )


class BashTool(Tool):
    """Run a shell command confined to one local artifact workspace.

    Three invariants hold: the working directory is resolved only through
    :class:`~atlas.agent.artifacts.LocalArtifactStore`, so the command starts
    inside the sandbox; the child runs with a minimal, explicit environment
    rather than inheriting the host process's secrets; and every call ends by
    killing the whole process group, so a backgrounded or disowned child
    cannot outlive it. Only the starting directory is confined: the command
    itself runs with the user's privileges, and a child that calls
    ``setsid()`` leaves the group. See docs/adr/0001-sandbox-model.md.
    """

    name = "bash_tool"
    description = (
        "Run a shell command with the working directory confined to the artifact/workspace "
        "sandbox, a timeout, captured stdout/stderr, and a non-zero exit reported as a result "
        "rather than an exception."
    )
    trust: ClassVar[Literal["default", "opt_in"]] = "default"
    input_model = BashInputModel

    max_output_chars: ClassVar[int] = 200_000

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        request = BashInputModel.model_validate(arguments)

        cwd = self._resolve_cwd(request.working_directory, store)
        if not cwd.is_dir():
            return ToolResult(text=f"Working directory not found: {request.working_directory}")

        env = {key: os.environ[key] for key in _INHERITED_ENV_KEYS if key in os.environ}

        process = subprocess.Popen(
            request.command,
            shell=True,
            cwd=cwd,
            env=env,
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
        """Return a sandbox-confined directory for the command.

        The default (``.``) is the project workspace when one is mounted, else
        the artifact root. Any other path goes through the store; an absolute
        path or a ``..`` escape raises ``SandboxDenied``.
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
