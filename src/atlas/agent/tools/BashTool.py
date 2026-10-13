from __future__ import annotations

import contextlib
import os
import re
import signal
import subprocess
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from atlas.agent.artifacts import LocalArtifactStore, SandboxDenied
from atlas.agent.contracts import Tool, ToolResult
from atlas.agent.shell_guard import (
    guard_shell_command,
    sandbox_unavailable,
    sandboxed_shell_argv,
)

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

    The working directory is the project workspace when one is mounted, and
    the artifact root otherwise. Every path the command names is resolved
    before launch: absolute paths and ``..`` that leave the workspace fail,
    a symlink is judged by its target, ``~/.atlas/keys`` is refused even when
    a path only lands there after resolution, and the session ``proposals/``
    directory is not a runnable path. The child gets a minimal environment,
    and on Linux Landlock keeps it from opening anything else on the host.
    Every call still kills the whole process group. A child that calls
    ``setsid()`` leaves that group. See docs/adr/0001-sandbox-model.md.
    """

    name = "bash_tool"
    description = (
        "Run a shell command inside the workspace. The working directory is the workspace. "
        "Absolute paths, '..' escapes, the user key directory, and drafted tool paths are denied. "
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

        workspace = store.read_root or store.root
        env = {key: os.environ[key] for key in _INHERITED_ENV_KEYS if key in os.environ}
        home = Path(env["HOME"]) if "HOME" in env else None
        guard_shell_command(
            request.command,
            cwd=cwd,
            workspace=workspace,
            proposals=store.root / "proposals",
            home=home,
        )

        process = subprocess.Popen(
            sandboxed_shell_argv(workspace, request.command),
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

        if sandbox_unavailable(process.returncode, stderr):
            raise SandboxDenied("Shell sandbox is unavailable")

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
        path or a ``..`` escape raises ``SandboxDenied``. A directory that
        resolves into the user key directory or the session proposal directory
        is refused by :func:`atlas.agent.shell_guard.guard_shell_command`.
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
