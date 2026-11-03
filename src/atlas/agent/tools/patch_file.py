from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import Tool, ToolResult


class PatchEdit(BaseModel):
    """One exact-string replacement inside a patch.

    ``old_string`` is matched against the text left by the previous edit, so
    edits are applied in order. ``replace_all`` allows a non-unique match.
    """

    old_string: str = Field(description="Exact old string to replace.")
    new_string: str = Field(description="Exact new string to replace it with.")
    replace_all: bool = Field(
        default=False,
        description="Replace every occurrence instead of requiring exactly one.",
    )


class PatchFileInput(BaseModel):
    """Arguments for ``patch_file``.

    ``path`` is relative to the artifact workspace. ``edits`` holds 1..50
    replacements, applied in the given order.
    """

    path: str = Field(description="Path relative to the local artifact workspace.")
    edits: list[PatchEdit] = Field(
        min_length=1,
        max_length=50,
        description="Edits to apply in order, against the progressively updated text.",
    )


class PatchFileTool(Tool):
    """Apply several exact-string edits to one file atomically.

    Two invariants hold: the path is resolved only through
    :class:`~atlas.agent.artifacts.LocalArtifactStore`, so it cannot leave the
    workspace root, and every edit is validated against the running text before
    the single final write. A failing edit aborts the whole patch and leaves the
    file byte-identical.
    """

    name = "patch_file"
    description = "Apply multiple exact-string edits to a UTF-8 text file in one atomic patch."
    trust: ClassVar[Literal["default", "opt_in"]] = "default"
    max_bytes: ClassVar[int] = 1_000_000

    @property
    def input_model(self) -> type[BaseModel]:
        return PatchFileInput

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        request = PatchFileInput.model_validate(arguments)
        path = store.resolve(request.path)

        if not path.is_file():
            raise FileNotFoundError(f"File does not exist: {request.path}")
        if path.stat().st_size > self.max_bytes:
            raise ValueError(f"File {request.path} exceeds the {self.max_bytes} byte read limit")

        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"File {request.path} is not valid UTF-8") from exc

        for index, edit in enumerate(request.edits):
            text = self._apply_edit(text, index, edit)

        byte_count = len(text.encode("utf-8"))
        if byte_count > self.max_bytes:
            raise ValueError(f"Patched content exceeds the {self.max_bytes} byte limit")
        path.write_text(text, encoding="utf-8")

        return ToolResult(
            text=(f"Applied {len(request.edits)} edits to {request.path} (bytes: {byte_count})."),
            artifacts=[store.relative(path)],
        )

    def _apply_edit(self, text: str, index: int, edit: PatchEdit) -> str:
        """Return ``text`` with one edit applied, or raise before any write."""
        old_string = edit.old_string

        if not old_string:
            raise ValueError(f"Edit {index}: old_string must not be empty")

        count = text.count(old_string)

        if count == 0:
            raise ValueError(f"Edit {index}: old_string was not found in the file")
        if count > 1 and not edit.replace_all:
            raise ValueError(
                f"Edit {index}: old_string occurs {count} times; set replace_all or make it unique"
            )

        return text.replace(old_string, edit.new_string, -1 if edit.replace_all else 1)
