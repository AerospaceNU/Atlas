from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import Tool, ToolResult


class EditFileInput(BaseModel):
    """The docstring should describe the three fields (workspace-relative path, exact text to find, replacement), not "Inputs the relative path..."."""

    path: str = Field(description="Path relative to the local artifact workspace.")
    old_string: str = Field(description="Exact old string to replace.")
    new_string: str = Field(description="Exact new string to replace.")


class EditFileTool(Tool):
    """Edits a string from a UTF-8 text file from the artifact workspace.

    Two invariants hold: the path is resolved only through
    :class:`~atlas.agent.artifacts.LocalArtifactStore`, so it cannot leave the
    workspace root, and the file is read as text, never executed or imported.
    """

    name = "edit_file"
    description = "Edits a UTF-8 text file by replacing an existing string with a new input."
    trust: ClassVar[Literal["default", "opt_in"]] = "default"
    max_bytes: ClassVar[int] = 100_000

    @property
    def input_model(self) -> type[BaseModel]:
        return EditFileInput

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        request = EditFileInput.model_validate(arguments)
        path = store.resolve(request.path)
        old_string = request.old_string
        new_string = request.new_string

        if not path.is_file():
            raise FileNotFoundError(f"File does not exist: {request.path}")
        if path.stat().st_size > self.max_bytes:
            raise ValueError(f"File {request.path} exceeds the {self.max_bytes} byte read limit")
        if not old_string:
            raise ValueError("old_string must not be empty")

        try:
            text = path.read_text(encoding="utf-8")

        except UnicodeDecodeError as exc:
            raise ValueError(f"File {request.path} is not valid UTF-8") from exc

        count = text.count(old_string)

        if count > 1:
            raise ValueError("The input appears multiple times.")

        elif count == 0:
            raise ValueError("The input cannot be found.")

        else:
            new_text = text.replace(old_string, new_string, 1)
            byte_count = len(new_text.encode())

            if byte_count <= self.max_bytes:
                path.write_text(new_text, encoding="utf-8")

        return ToolResult(
            text=f"File {request.path} edited successfully (bytes: {byte_count}).",
            artifacts=[store.relative(path)],
        )
