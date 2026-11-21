"""Tool for writing content to a file in the local artifact workspace."""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from atlas.agent.artifacts import ArtifactError, LocalArtifactStore
from atlas.agent.contracts import Tool, ToolResult
from atlas.agent.tools._edits import normalize_newlines, python_syntax_warning


class WriteFileInput(BaseModel):
    path: str = Field(description="Path relative to the workspace, or absolute.")
    content: str = Field(
        description=(
            "The content that should be written to the file path. Input should be a raw "
            "string, and it will be written to the file as UTF-8 text."
        )
    )


class WriteTool(Tool):
    """Create or overwrite one UTF-8 text file.

    The path resolves through :class:`~atlas.agent.artifacts.LocalArtifactStore`
    (relative to the workspace; absolute paths and ``..`` are accepted) and missing
    parent directories are created. Content is written as text, never executed.
    """

    name = "write_file"
    description = (
        "Create or overwrite a UTF-8 text file (parent directories are created). "
        "To change part of an existing file use edit_file or patch_file."
    )
    trust: ClassVar[Literal["default", "opt_in"]] = "default"
    max_bytes: ClassVar[int] = 1_000_000

    @property
    def input_model(self) -> type[BaseModel]:
        return WriteFileInput

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        request = WriteFileInput.model_validate(arguments)
        path = store.resolve(request.path)

        content_size = len(request.content.encode("utf-8"))
        if content_size > self.max_bytes:
            raise ArtifactError(f"Content exceeds the {self.max_bytes} byte write limit")

        previous_size = path.stat().st_size if path.is_file() else None
        previous = self._previous_text(path) if previous_size is not None else None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(request.content, encoding="utf-8")

        text = f"File {request.path} created and written successfully (bytes: {content_size})."
        if previous_size is not None:
            text += f" Replaced an existing file of {previous_size} bytes."
        warning = python_syntax_warning(request.path, previous, normalize_newlines(request.content))
        if warning:
            text += "\n" + warning
        return ToolResult(text=text, artifacts=[store.relative(path)])

    def _previous_text(self, path: Any) -> str | None:
        """Return the file's current text for the syntax comparison, or None if unreadable."""
        try:
            if path.stat().st_size > self.max_bytes:
                return None
            return normalize_newlines(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError):
            return None
