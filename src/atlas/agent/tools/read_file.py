from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import Tool, ToolResult


class ReadFileInput(BaseModel):
    path: str = Field(description="Path relative to the local artifact workspace.")


class ReadFileTool(Tool):
    """Read one UTF-8 text file from the artifact workspace.

    The path is resolved through
    :class:`~atlas.agent.artifacts.LocalArtifactStore`. Relative paths resolve
    against the working tree and absolute paths are used as given. The file is
    read as text, never executed or imported. Reads prefer the session artifact
    directory and fall back to the project workspace when one is mounted.
    """

    name = "read_file"
    description = (
        "Read a UTF-8 text file from the local artifact workspace, then the project workspace."
    )
    trust: ClassVar[Literal["default", "opt_in"]] = "default"
    max_bytes: ClassVar[int] = 100_000

    @property
    def input_model(self) -> type[BaseModel]:
        return ReadFileInput

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        request = ReadFileInput.model_validate(arguments)
        try:
            data = store.read_bytes(request.path, max_bytes=self.max_bytes)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"File does not exist: {request.path}") from exc
        try:
            text = data.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
        except UnicodeDecodeError as exc:
            raise ValueError(f"File {request.path} is not valid UTF-8") from exc
        return ToolResult(text=text)
