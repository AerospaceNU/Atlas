from __future__ import annotations

import subprocess
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import Tool, ToolResult


class BashInputModel(BaseModel):
    command: str = Field(description="The shell command to execute")
    timeout: int = Field(
        default=30, description="Maximum seconds to allow the command to run before it's killed"
    )
    working_directory: str = Field(
        default=".",
        description="Directory to run in, relative to the workspace (default: the workspace root)",
    )


class BashTool(Tool):
    name = "bash_tool"
    description = (
        "Run a shell command with the working directory confined to the artifact/workspace "
        "sandbox, a timeout, captured stdout/stderr, and a non-zero exit reported as a result "
        "rather than an exception."
    )
    trust: ClassVar[Literal["default", "opt_in"]] = "default"
    input_model = BashInputModel

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        request = BashInputModel.model_validate(arguments)
        try:
            cwd = self._resolve_cwd(request.working_directory, store)
        except ValueError as exc:
            return ToolResult(
                text=f"Invalid working directory {request.working_directory!r}: {exc}"
            )
        try:
            result = subprocess.run(
                request.command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=request.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(text=f"Command timed out after {request.timeout}s: {request.command}")
        except FileNotFoundError:
            return ToolResult(text=f"Working directory not found: {request.working_directory}")

        output = (
            f"exit_code: {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        return ToolResult(text=output)

    def _resolve_cwd(self, working_directory: str, store: LocalArtifactStore) -> str:
        """Return a sandbox-confined directory path for the command.

        The default (``.``) is the project workspace when one is mounted, else
        the artifact root. Any other path must stay inside the sandbox; an
        absolute path or a ``..`` escape raises ``ValueError``.
        """
        if working_directory in ("", "."):
            return str(store.read_root or store.root)
        return str(store.resolve_read(working_directory))
