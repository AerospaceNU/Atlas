from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from atlas.agent.artifacts import ArtifactError, LocalArtifactStore
from atlas.agent.contracts import Tool, ToolResult
from atlas.agent.tools._edits import split_lines

_FOOTER_RESERVE = 70
_RESPONSE_RESERVE = 100
_MIN_FILE_BUDGET = 300
MAX_LIMIT = 5000


class ReadTarget(BaseModel):
    """One file in a ``read_files`` batch.

    ``offset`` is a 1-based line number, so the default reads from the top.
    ``limit`` caps how many lines are returned for this file.
    """

    path: str = Field(description="Path relative to the workspace, or absolute.")
    offset: int = Field(default=1, ge=1, description="First line to return (1-based).")
    limit: int = Field(
        default=2000,
        ge=1,
        description=(
            "Maximum number of lines to return for this file (values above 5000 are clamped to 5000)."
        ),
    )


class ReadFilesInput(BaseModel):
    """Arguments for ``read_files``.

    ``files`` holds between 1 and 20 targets, read in the given order.
    """

    files: list[ReadTarget] = Field(
        min_length=1,
        max_length=20,
        description="Files to read, in order (1-20).",
    )


class ReadFilesTool(Tool):
    """Read several UTF-8 text files in one call.

    Every path is resolved through
    :class:`~atlas.agent.artifacts.LocalArtifactStore` (relative to the workspace;
    absolute paths and ``..`` are accepted). A missing, binary, or oversized file
    becomes an error section instead of failing the batch. The whole response stays
    under ``max_output_chars``: each file gets an equal share of what is left, is cut
    at a line boundary, and its header states the lines actually returned.
    """

    name = "read_files"
    description = (
        "Read several UTF-8 text files in one call, with line numbers and optional "
        "line ranges. A file that cannot be read is reported without failing the batch."
    )
    trust: ClassVar[Literal["default", "opt_in"]] = "default"
    max_file_bytes: ClassVar[int] = 1_000_000
    max_output_chars: ClassVar[int] = 18_000
    max_line_chars: ClassVar[int] = 2000

    @property
    def input_model(self) -> type[BaseModel]:
        return ReadFilesInput

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        request = ReadFilesInput.model_validate(arguments)
        sections: list[str] = []
        left = self.max_output_chars - _RESPONSE_RESERVE
        total = len(request.files)
        skipped = 0
        for index, target in enumerate(request.files):
            share = left // (total - index)
            if share < _MIN_FILE_BUDGET:
                skipped = total - index
                break
            section = self._read_one(target, store, share)
            sections.append(section)
            left -= len(section) + 2
        if skipped:
            sections.append(f"... [{skipped} more files not read: output limit]")
        return ToolResult(text="\n\n".join(sections))

    def _read_one(self, target: ReadTarget, store: LocalArtifactStore, budget: int) -> str:
        """Return one labelled file section of at most about ``budget`` characters."""
        header = f"== {target.path}"
        try:
            data = store.read_bytes(target.path, max_bytes=self.max_file_bytes)
        except FileNotFoundError:
            return f"{header}\n[missing: file does not exist]"
        except ArtifactError as exc:
            return f"{header}\n[refused: {exc}]"
        except OSError as exc:
            return f"{header}\n[error: {type(exc).__name__}: {exc}]"
        if b"\x00" in data:
            return f"{header}\n[skipped: binary file]"
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            return f"{header}\n[skipped: not valid UTF-8]"
        lines = split_lines(text.replace("\r\n", "\n").replace("\r", "\n"))
        if not lines:
            return f"{header}\n(empty file)"
        start = target.offset - 1
        if start >= len(lines):
            return f"{header}\n[offset {target.offset} is past the end ({len(lines)} lines)]"
        selected = lines[start : start + min(target.limit, MAX_LIMIT)]
        fitted = self._fit(selected, start + 1, budget - len(header) - _FOOTER_RESERVE - 40)
        end = start + len(fitted)
        section = f"{header} (lines {start + 1}-{end} of {len(lines)})\n" + "\n".join(fitted)
        if end < len(lines):
            more = len(lines) - end
            if len(fitted) < len(selected):
                section += f"\n... [{more} more lines; continue with offset={end + 1}]"
            else:
                section += f"\n... [{more} more lines]"
        return section

    def _fit(self, selected: list[str], first: int, room: int) -> list[str]:
        """Return gutter-formatted lines that fit ``room`` characters, at least one."""
        fitted: list[str] = []
        used = 0
        for number, line in enumerate(selected, first):
            rendered = f"{number:>6}| {line[: self.max_line_chars]}"
            if fitted and used + len(rendered) + 1 > room:
                break
            if not fitted and len(rendered) > room:
                rendered = rendered[: max(room, 80)]
            fitted.append(rendered)
            used += len(rendered) + 1
        return fitted
