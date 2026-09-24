from __future__ import annotations

import subprocess
from typing import Any

from pydantic import BaseModel, Field

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import Tool, ToolResult


class BashInputModel(BaseModel):
    command: str = Field(description="The shell command to execute")
    timeout: int = Field(
        default=30, description="Maximum seconds to allow the command to run before it's killed"
    )

    working_directory: str = Field(
        default=".", description="Directory to run the command in, relative or absolute path"
    )


class BashTool(Tool):
    name = "bash_tool"
    description = "Run a command with a working directory inside the sandbox, a timeout, captured stdout/stderr, and a non-zero exit reported as a result rather than an exception"
    trust = "default"
    input_model = BashInputModel

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        request = BashInputModel.model_validate(arguments)
        try:
            result = subprocess.run(
                request.command,
                shell=True,
                cwd=request.working_directory,
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
