import subprocess

from atlas.agent.contracts import Tool, ToolResult
from pydantic import BaseModel, Field


class BashInputModel(BaseModel):
    command: str = Field(
        description="The shell command to execute"
    )
    timeout: int = Field(
        default=30,
        description="Maximum seconds to allow the command to run before it's killed"
    )

    working_directory: str = Field(
        default=".",
        description="Directory to run the command in, relative or absolute path"
    )


class BashTool(Tool):
    name = "bash_tool"
    description = "Run a command with a working directory inside the sandbox, a timeout, captured stdout/stderr, and a non-zero exit reported as a result rather than an exception"
    trust = "default"
    input_model = BashInputModel

    def run(self, arguments, store):
        #
        try:
            result = subprocess.run(
                arguments.command,
                shell=True,
                cwd=arguments.working_directory,
                capture_output=True,
                text=True,
                timeout=arguments.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                text=f"Command timed out after {arguments.timeout}s: {arguments.command}"
            )
        except FileNotFoundError:
            return ToolResult(
                text=f"Working directory not found: {arguments.working_directory}"
            )

        output = (
            f"exit_code: {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
        return ToolResult(text=output)
