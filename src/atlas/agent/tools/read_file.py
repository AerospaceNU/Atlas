from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import Tool, ToolResult


class ReadFileInput(BaseModel):
    path: str = Field(description="Path relative to the local artifact workspace.")


class ReadFileTool(Tool):
    """Read one UTF-8 text file from the artifact workspace.

    Two invariants hold: the path is resolved only through
    :class:`~atlas.agent.artifacts.LocalArtifactStore`, so it cannot leave the
    workspace root, and the file is read as text, never executed or imported.
    """

    name = "read_file"
    description = "Read a UTF-8 text file from the local artifact workspace."
    trust: ClassVar[Literal["default", "opt_in"]] = "default"
    max_bytes: ClassVar[int] = 100_000

    @property
    def input_model(self) -> type[BaseModel]:
        return ReadFileInput

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        request = ReadFileInput.model_validate(arguments)
        path = store.resolve(request.path)
        if not path.is_file():
            raise FileNotFoundError(f"File does not exist: {request.path}")
        if path.stat().st_size > self.max_bytes:
            raise ValueError(f"File {request.path} exceeds the {self.max_bytes} byte read limit")
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"File {request.path} is not valid UTF-8") from exc
        return ToolResult(text=text)
