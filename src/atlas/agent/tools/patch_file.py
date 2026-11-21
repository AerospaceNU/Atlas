from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import Tool, ToolResult
from atlas.agent.tools._edits import (
    changed_region,
    changed_span,
    encode_text,
    normalize_newlines,
    not_found_hint,
    occurrence_lines,
    python_syntax_warning,
    read_text_keeping_newlines,
)

_REGION_LINES = 40

Span = tuple[int, int]


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

    ``path`` is relative to the workspace, or absolute. ``edits`` holds 1..50
    replacements, applied in the given order.
    """

    path: str = Field(description="Path relative to the workspace, or absolute.")
    edits: list[PatchEdit] = Field(
        min_length=1,
        max_length=50,
        description="Edits to apply in order, against the progressively updated text.",
    )


def _shift(span: Span, index: int, old_end: int, new_end: int) -> Span:
    """Move an earlier edit's span across a later replacement of [index, old_end)."""
    delta = new_end - old_end
    start, end = span
    if start >= old_end:
        start += delta
    elif start > index:
        start = index
    if end >= old_end:
        end += delta
    elif end > index:
        end = new_end
    return start, end


class PatchFileTool(Tool):
    """Apply several exact-string edits to one file atomically.

    The path is resolved through :class:`~atlas.agent.artifacts.LocalArtifactStore`
    (relative to the workspace; absolute paths and ``..`` are accepted). Every edit
    is validated against the running text before the single final write, and the
    file's newline style is kept. A failing edit aborts the whole patch and leaves
    the file byte-identical.
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

        original, newline = read_text_keeping_newlines(path, request.path, self.max_bytes)
        text = original
        spans: list[Span] = []
        for index, edit in enumerate(request.edits):
            text = self._apply_edit(text, index, edit, spans)

        data = encode_text(text, newline)
        if len(data) > self.max_bytes:
            raise ValueError(f"Patched content exceeds the {self.max_bytes} byte limit")
        path.write_bytes(data)

        lines = [f"Applied {len(request.edits)} edits to {request.path} (bytes: {len(data)})."]
        lines.extend(self._regions(text, spans))
        warning = python_syntax_warning(request.path, original, text)
        if warning:
            lines.append(warning)
        return ToolResult(text="\n".join(lines), artifacts=[store.relative(path)])

    def _apply_edit(self, text: str, index: int, edit: PatchEdit, spans: list[Span]) -> str:
        """Return ``text`` with one edit applied, or raise before any write."""
        old_string = normalize_newlines(edit.old_string)
        new_string = normalize_newlines(edit.new_string)
        if not old_string:
            raise ValueError(f"Edit {index}: old_string must not be empty")
        count = text.count(old_string)
        if count == 0:
            hint = not_found_hint(text, old_string)
            raise ValueError(
                f"Edit {index}: old_string was not found in the file"
                + (f"\n{hint}" if hint else "")
            )
        if count > 1 and not edit.replace_all:
            lines = ", ".join(str(n) for n in occurrence_lines(text, old_string))
            raise ValueError(
                f"Edit {index}: old_string occurs {count} times; set replace_all or make it "
                f"unique (at lines {lines})"
            )
        first = text.index(old_string)
        pieces: list[str] = []
        cursor = 0
        shift = 0
        for position in self._positions(text, old_string, count if edit.replace_all else 1):
            pieces.append(text[cursor:position])
            pieces.append(new_string)
            cursor = position + len(old_string)
            here = position + shift
            for slot, span in enumerate(spans):
                spans[slot] = _shift(span, here, here + len(old_string), here + len(new_string))
            shift += len(new_string) - len(old_string)
        pieces.append(text[cursor:])
        spans.append((first, first + len(new_string)))
        return "".join(pieces)

    @staticmethod
    def _positions(text: str, needle: str, count: int) -> list[int]:
        """Return the first ``count`` non-overlapping start offsets of ``needle``."""
        found: list[int] = []
        position = text.find(needle)
        while position != -1 and len(found) < count:
            found.append(position)
            position = text.find(needle, position + len(needle))
        return found

    def _regions(self, text: str, spans: list[Span]) -> list[str]:
        """Return the changed regions, merged when they touch, within the line budget."""
        merged: list[Span] = []
        for start, end in sorted(spans):
            if merged:
                prev_start, prev_end = merged[-1]
                if text.count("\n", prev_end, start) <= 7:
                    merged[-1] = (prev_start, max(prev_end, end))
                    continue
            merged.append((start, end))
        out: list[str] = []
        left = _REGION_LINES
        for number, (start, end) in enumerate(merged):
            if left <= 2:
                out.append(f"... [{len(merged) - number} more changed regions not shown]")
                break
            low, high = changed_span(text, start, end, max_lines=left - 1)
            out.append(f"Lines {low}-{high} now read:")
            out.append(changed_region(text, start, end, max_lines=left - 1))
            left -= high - low + 2
        return out
