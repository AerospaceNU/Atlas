from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from atlas.agent.artifacts import ArtifactError, LocalArtifactStore
from atlas.agent.contracts import Tool, ToolResult
from atlas.agent.tools._edits import split_lines

MAX_LIMIT = 2000


class ReadFileInput(BaseModel):
    path: str = Field(description="Path relative to the workspace, or absolute.")
    offset: int = Field(default=1, ge=1, description="First line to return (1-based).")
    limit: int | None = Field(
        default=None,
        ge=1,
        description="Maximum number of lines to return (default 500; values above 2000 are clamped to 2000).",
    )


class ReadFileTool(Tool):
    """Read one UTF-8 text file, whole when small and as a window of lines otherwise.

    Relative paths resolve against the workspace; absolute paths and ``..`` are
    accepted. The file is read as text, never executed or imported. A file that fits
    the character budget is returned verbatim (newlines normalised to LF); anything
    larger comes back as whole lines plus a final line saying where to continue.
    """

    name = "read_file"
    description = (
        "Read a UTF-8 text file. A small file is returned whole; a larger one as a window "
        "of lines. Use offset (1-based line number) and limit to read a specific part or to "
        "continue: the last line of a partial result says where to continue."
    )
    trust: ClassVar[Literal["default", "opt_in"]] = "default"
    max_bytes: ClassVar[int] = 5_000_000
    max_chars: ClassVar[int] = 16_000
    default_limit: ClassVar[int] = 500

    @property
    def input_model(self) -> type[BaseModel]:
        return ReadFileInput

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        request = ReadFileInput.model_validate(arguments)
        try:
            data = store.read_bytes(request.path, max_bytes=self.max_bytes)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"File does not exist: {request.path}") from exc
        except ArtifactError as exc:
            raise ArtifactError(f"{exc}; search it with grep or read a slice through bash") from exc
        try:
            text = data.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
        except UnicodeDecodeError as exc:
            raise ValueError(f"File {request.path} is not valid UTF-8") from exc
        return ToolResult(text=self._render(text, request.offset, request.limit))

    def _render(self, text: str, offset: int, limit: int | None) -> str:
        """Return the whole text when it fits, else the window that starts at ``offset``."""
        lines = split_lines(text)
        total = len(lines)
        wanted = min(limit or self.default_limit, MAX_LIMIT)
        if offset == 1 and total <= wanted and len(text) <= self.max_chars:
            return text
        if offset > total:
            return f"[offset {offset} is past the end of the file ({total} lines)]"
        window = self._window(lines, offset, wanted)
        last = offset + len(window) - 1
        body = "\n".join(window)
        if last < total:
            return (
                f"{body}\n... [lines {offset}-{last} of {total}; continue with offset={last + 1}]"
            )
        if offset > 1:
            return f"{body}\n... [lines {offset}-{total} of {total}]"
        return body

    def _window(self, lines: list[str], offset: int, wanted: int) -> list[str]:
        """Return whole lines from ``offset``, within ``wanted`` lines and the char budget."""
        picked: list[str] = []
        used = 0
        for number, line in enumerate(lines[offset - 1 : offset - 1 + wanted], offset):
            cost = len(line) + (1 if picked else 0)
            if picked and used + cost > self.max_chars:
                break
            if not picked and len(line) > self.max_chars:
                extra = len(line) - self.max_chars
                line = (
                    f"{line[: self.max_chars]} ... [line {number} truncated: {extra} more "
                    "characters; lines this long are better searched with grep or sliced "
                    "through bash]"
                )
                cost = len(line)
            picked.append(line)
            used += cost
        return picked
