from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from atlas.agent.artifacts import ArtifactError, LocalArtifactStore
from atlas.agent.contracts import Tool, ToolResult


class ReadTarget(BaseModel):
    """One file in a ``read_files`` batch.

    ``offset`` is a 1-based line number, so the default reads from the top.
    ``limit`` caps how many lines are returned for this file.
    """

    path: str = Field(description="Path relative to the local artifact workspace.")
    offset: int = Field(default=1, ge=1, description="First line to return (1-based).")
    limit: int = Field(
        default=2000,
        ge=1,
        le=5000,
        description="Maximum number of lines to return for this file (1-5000).",
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
    :class:`~atlas.agent.artifacts.LocalArtifactStore`. A missing, binary, or
    oversized file becomes an error section instead of failing the batch. Each
    section is capped and the whole response stops at a fixed character budget.
    """

    name = "read_files"
    description = (
        "Read several UTF-8 text files in one call, with line numbers and optional "
        "line ranges. A file that cannot be read is reported without failing the batch."
    )
    trust: ClassVar[Literal["default", "opt_in"]] = "default"
    max_file_bytes: ClassVar[int] = 1_000_000
    max_output_chars: ClassVar[int] = 400_000
    max_line_chars: ClassVar[int] = 2000

    @property
    def input_model(self) -> type[BaseModel]:
        return ReadFilesInput

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        request = ReadFilesInput.model_validate(arguments)
        sections: list[str] = []
        used = 0
        skipped = 0
        for index, target in enumerate(request.files):
            section = self._read_one(target, store)
            if used + len(section) > self.max_output_chars:
                room = self.max_output_chars - used
                if room > 0:
                    sections.append(section[:room] + "\n... [output limit reached]")
                skipped = len(request.files) - index
                break
            sections.append(section)
            used += len(section)
        if skipped:
            sections.append(f"... [{skipped} more files not read: output limit]")
        return ToolResult(text="\n\n".join(sections))

    def _read_one(self, target: ReadTarget, store: LocalArtifactStore) -> str:
        """Return one labelled file section, or an error section."""
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
        lines = text.replace("\r\n", "\n").replace("\r", "\n").splitlines()
        if not lines:
            return f"{header}\n(empty file)"
        start = target.offset - 1
        if start >= len(lines):
            return f"{header}\n[offset {target.offset} is past the end ({len(lines)} lines)]"
        selected = lines[start : start + target.limit]
        end = start + len(selected)
        body = "\n".join(
            f"{number:>6}| {line[: self.max_line_chars]}"
            for number, line in enumerate(selected, start + 1)
        )
        section = f"{header} (lines {start + 1}-{end} of {len(lines)})\n{body}"
        if end < len(lines):
            section += f"\n... [{len(lines) - end} more lines]"
        return section
