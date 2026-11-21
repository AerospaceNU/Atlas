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


class EditFileInput(BaseModel):
    """Arguments for ``edit_file``.

    ``path`` is relative to the workspace, or absolute. ``old_string`` is the exact
    text to find; it must occur exactly once unless ``replace_all`` is set.
    ``new_string`` is the text that replaces it.
    """

    path: str = Field(description="Path relative to the workspace, or absolute.")
    old_string: str = Field(description="Exact old string to replace.")
    new_string: str = Field(description="Exact new string to replace it with.")
    replace_all: bool = Field(
        default=False,
        description="Replace every occurrence instead of requiring exactly one.",
    )


class EditFileTool(Tool):
    """Edit a UTF-8 text file by replacing one string.

    The path is resolved through :class:`~atlas.agent.artifacts.LocalArtifactStore`
    (relative to the workspace; absolute paths and ``..`` are accepted). The file is
    read as text, never executed or imported. Matching is exact on LF-normalised
    text and the file's own newline style is kept. A failed match changes nothing;
    the error carries advice but an inexact match is never applied.
    """

    name = "edit_file"
    description = (
        "Edit a UTF-8 text file by replacing an existing string with a new one. "
        "old_string must match exactly once, or set replace_all."
    )
    trust: ClassVar[Literal["default", "opt_in"]] = "default"
    max_bytes: ClassVar[int] = 1_000_000

    @property
    def input_model(self) -> type[BaseModel]:
        return EditFileInput

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        request = EditFileInput.model_validate(arguments)
        path = store.resolve(request.path)
        if not path.is_file():
            raise FileNotFoundError(f"File does not exist: {request.path}")
        old_string = normalize_newlines(request.old_string)
        new_string = normalize_newlines(request.new_string)
        if not old_string:
            raise ValueError("old_string must not be empty")
        if old_string == new_string:
            raise ValueError("old_string and new_string are identical")

        text, newline = read_text_keeping_newlines(path, request.path, self.max_bytes)
        count = text.count(old_string)
        if count == 0:
            hint = not_found_hint(text, old_string)
            raise ValueError("The input cannot be found." + (f"\n{hint}" if hint else ""))
        if count > 1 and not request.replace_all:
            lines = ", ".join(str(n) for n in occurrence_lines(text, old_string))
            more = ", ..." if count > 8 else ""
            raise ValueError(
                f"The input appears multiple times. It occurs {count} times, at lines "
                f"{lines}{more}. Add surrounding lines to make old_string unique, "
                "or set replace_all."
            )
        start = text.index(old_string)
        replaced = count if request.replace_all else 1
        new_text = text.replace(old_string, new_string, -1 if request.replace_all else 1)
        data = encode_text(new_text, newline)
        if len(data) > self.max_bytes:
            raise ValueError(f"Edited content exceeds the {self.max_bytes} byte limit")
        path.write_bytes(data)
        return ToolResult(
            text=self._report(request, text, new_text, start, len(new_string), len(data), replaced),
            artifacts=[store.relative(path)],
        )

    def _report(
        self,
        request: EditFileInput,
        before: str,
        after: str,
        start: int,
        length: int,
        byte_count: int,
        replaced: int,
    ) -> str:
        """Return the result text: success line, the changed lines, any syntax warning."""
        lines = [f"File {request.path} edited successfully (bytes: {byte_count})."]
        if request.replace_all:
            lines.append(f"Replaced {replaced} occurrences; showing the first.")
        low, high = changed_span(after, start, start + length)
        lines.append(f"Lines {low}-{high} now read:")
        lines.append(changed_region(after, start, start + length))
        warning = python_syntax_warning(request.path, before, after)
        if warning:
            lines.append(warning)
        return "\n".join(lines)
