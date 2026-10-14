from __future__ import annotations

import contextlib
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from atlas.agent.artifacts import LocalArtifactStore, SandboxDenied
from atlas.agent.contracts import Tool, ToolResult
from atlas.agent.layout import user_keys_dir
from atlas.agent.shell_guard import (
    MACOS_SANDBOX_DISABLED,
    SANDBOX_EXEC_FAILED,
    SANDBOX_EXEC_READY,
    WINDOWS_BASH_WARNING,
    bash_tool_trust,
    cleanup_mount_points,
    guard_shell_command,
    prepare_mount_points,
    refuse_shell_start,
    require_landlock_kernel,
    sandbox_unavailable,
    sandboxed_shell_argv,
    shell_excluded_paths,
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
    the artifact root otherwise. A best-effort text scan rejects obvious
    paths before launch. The boundary is the OS sandbox: Landlock on Linux
    5.13 or newer, and ``sandbox-exec`` on macOS. The shell does not start
    when the user key directory or ``ATLAS_HOME`` resolves inside the
    workspace, or when that sandbox is unavailable. Hiding an artifact root
    or ``proposals/`` that already sits inside the workspace requires a user
    namespace. Without one the shell does not start. A ``sandbox-exec``
    launch or profile failure is a sandbox denial. On Windows the tool is
    opt-in and
    warns that there is no OS confinement. The network is not restricted.
    Every call still kills the whole process group. A child that calls
    ``setsid()`` leaves that group. See docs/adr/0001-sandbox-model.md.
    """

    name = "bash_tool"
    description = (
        "Run a shell command inside the workspace. The working directory is the workspace. "
        "A best-effort check rejects obvious escapes; Landlock or sandbox-exec is the boundary. "
        "The network is not restricted. "
        "stdout and stderr are captured, and a non-zero exit is a result rather than an exception."
    )
    trust: ClassVar[Literal["default", "opt_in"]] = bash_tool_trust()
    input_model = BashInputModel

    max_output_chars: ClassVar[int] = 200_000

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        request = BashInputModel.model_validate(arguments)

        cwd = self._resolve_cwd(request.working_directory, store)
        if not cwd.is_dir():
            return ToolResult(text=f"Working directory not found: {request.working_directory}")

        workspace = store.read_root or store.root
        refuse_shell_start(workspace)
        if sys.platform == "darwin" and shutil.which("sandbox-exec") is None:
            raise SandboxDenied(MACOS_SANDBOX_DISABLED)
        if sys.platform == "linux":
            require_landlock_kernel()
        env = {key: os.environ[key] for key in _INHERITED_ENV_KEYS if key in os.environ}
        # Tests force the no-namespace failure. The shell command cannot set
        # this; the launcher reads it before bash starts.
        if os.environ.get("ATLAS_SHELL_NO_USERNS") == "1":
            env["ATLAS_SHELL_NO_USERNS"] = "1"
        home = Path(env["HOME"]) if "HOME" in env else None
        proposals = store.root / "proposals"
        guard_shell_command(
            request.command,
            cwd=cwd,
            workspace=workspace,
            proposals=proposals,
            home=home,
        )
        excluded = shell_excluded_paths(workspace, store.root, proposals)
        created: list[Path] = []
        tmpdir: Path | None = None
        mask_dir: Path | None = None
        if sys.platform == "linux":
            created = prepare_mount_points(excluded)
            mask_dir = Path(tempfile.mkdtemp(prefix="atlas-shell-mask-"))
        try:
            tmpdir = Path(tempfile.mkdtemp(prefix=".shell-tmp-", dir=workspace))
            os.chmod(tmpdir, 0o700)
        except OSError as exc:
            if mask_dir is not None:
                shutil.rmtree(mask_dir, ignore_errors=True)
            cleanup_mount_points(created)
            raise SandboxDenied("Shell sandbox is unavailable") from exc
        env["TMPDIR"] = str(tmpdir)
        env["TMP"] = str(tmpdir)
        env["TEMP"] = str(tmpdir)
        keys: list[Path] = []
        with contextlib.suppress(RuntimeError):
            keys.append(user_keys_dir())
        if "HOME" in os.environ:
            keys.append(Path(os.environ["HOME"]) / ".atlas" / "keys")

        try:
            argv = sandboxed_shell_argv(
                workspace,
                request.command,
                tmpdir=tmpdir,
                mask_dir=mask_dir,
                excluded=excluded,
                key_dirs=keys,
            )
            try:
                process = subprocess.Popen(
                    argv,
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
            except OSError as exc:
                if sys.platform == "darwin":
                    raise SandboxDenied(SANDBOX_EXEC_FAILED) from exc
                raise
            # start_new_session=True makes the shell the leader of a new process
            # group whose id is its pid. Captured now because communicate() reaps
            # the shell, and on macOS os.getpgid() fails once the leader has exited.
            pgid = process.pid
            try:
                stdout, stderr = process.communicate(timeout=request.timeout)
            except subprocess.TimeoutExpired:
                return ToolResult(
                    text=f"Command timed out after {request.timeout}s: {request.command}"
                )
            finally:
                # A background child that redirects its stdio lets communicate()
                # return normally, so the group is killed on every exit path.
                self._kill_process_group(process, pgid)
        finally:
            self._cleanup_shell_dirs(tmpdir, mask_dir, created)

        if sys.platform == "darwin":
            ready = f"{SANDBOX_EXEC_READY}\n"
            if stdout.startswith(ready):
                stdout = stdout[len(ready) :]
            elif process.returncode not in (0, None):
                raise SandboxDenied(SANDBOX_EXEC_FAILED)

        if sandbox_unavailable(process.returncode, stderr):
            detail = (stderr or "").strip() or "Shell sandbox is unavailable"
            raise SandboxDenied(detail)

        warning = f"{WINDOWS_BASH_WARNING}\n" if sys.platform == "win32" else ""
        output = (
            f"{warning}"
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

    def _cleanup_shell_dirs(
        self,
        tmpdir: Path | None,
        mask_dir: Path | None,
        created: list[Path],
    ) -> None:
        if tmpdir is not None:
            shutil.rmtree(tmpdir, ignore_errors=True)
        if mask_dir is not None:
            shutil.rmtree(mask_dir, ignore_errors=True)
        cleanup_mount_points(created)

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
