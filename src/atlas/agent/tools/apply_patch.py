from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import Tool, ToolResult

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_SEARCH_LIMIT = 200


@dataclass
class _Hunk:
    """One parsed ``@@`` hunk, kept in patch order."""

    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[tuple[str, str]] = field(default_factory=list)
    old_no_newline: bool = False
    new_no_newline: bool = False


@dataclass
class _FilePatch:
    """One file section of a unified diff."""

    old_path: str | None
    new_path: str | None
    hunks: list[_Hunk] = field(default_factory=list)


@dataclass
class _PendingWrite:
    """Validated new content for one file, written after every file checks out."""

    display: str
    path: Path
    text: str
    hunks: int
    created: bool


def _header_path(value: str) -> str | None:
    """Return the path from one ``---``/``+++`` header, or None for ``/dev/null``."""
    path = value.split("\t", 1)[0].strip()
    if path == "/dev/null":
        return None
    if len(path) >= 2 and path.startswith('"') and path.endswith('"'):
        path = path[1:-1]
    if path.startswith(("a/", "b/")):
        path = path[2:]
    if not path:
        raise ValueError("empty path in patch header")
    return path


def _mark_no_newline(hunk: _Hunk) -> None:
    r"""Record a ``\ No newline at end of file`` marker on the preceding line."""
    if not hunk.lines:
        raise ValueError("no-newline marker must follow a content line")
    operator = hunk.lines[-1][0]
    if operator in ("-", " "):
        hunk.old_no_newline = True
    if operator in ("+", " "):
        hunk.new_no_newline = True


def _parse_hunk(raw: list[str], index: int) -> tuple[_Hunk, int]:
    """Parse one hunk starting at ``raw[index]``, returning it and the next index."""
    match = _HUNK_RE.match(raw[index])
    if match is None:
        raise ValueError(f"line {index + 1}: malformed hunk header")
    hunk = _Hunk(
        old_start=int(match.group(1)),
        old_count=int(match.group(2) or 1),
        new_start=int(match.group(3)),
        new_count=int(match.group(4) or 1),
    )
    index += 1
    old_seen = new_seen = 0
    while index < len(raw) and (old_seen < hunk.old_count or new_seen < hunk.new_count):
        line = raw[index]
        if not line:
            raise ValueError(
                f"line {index + 1}: empty line in hunk; context lines start with a space"
            )
        operator, text = line[0], line[1:]
        if operator == "\\":
            _mark_no_newline(hunk)
            index += 1
            continue
        if operator not in (" ", "-", "+"):
            raise ValueError(f"line {index + 1}: unexpected hunk line prefix {operator!r}")
        hunk.lines.append((operator, text))
        if operator in (" ", "-"):
            old_seen += 1
        if operator in (" ", "+"):
            new_seen += 1
        index += 1
    while index < len(raw) and raw[index].startswith("\\"):
        _mark_no_newline(hunk)
        index += 1
    if old_seen != hunk.old_count or new_seen != hunk.new_count:
        raise ValueError(
            f"hunk at line {hunk.old_start}: header says {hunk.old_count} old / "
            f"{hunk.new_count} new lines, body has {old_seen} / {new_seen}"
        )
    if not hunk.lines:
        raise ValueError(f"hunk at line {hunk.old_start} is empty")
    return hunk, index


def _parse_patch(patch: str) -> list[_FilePatch]:
    """Parse a unified diff into file sections without touching the workspace."""
    if "\x00" in patch:
        raise ValueError("patch must not contain NUL bytes")
    raw = patch.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    files: list[_FilePatch] = []
    index = 0
    while index < len(raw):
        if not raw[index].startswith("--- "):
            index += 1
            continue
        old_path = _header_path(raw[index][4:])
        if index + 1 >= len(raw) or not raw[index + 1].startswith("+++ "):
            raise ValueError(f"line {index + 1}: '---' header must be followed by a '+++' header")
        new_path = _header_path(raw[index + 1][4:])
        index += 2
        current = _FilePatch(old_path=old_path, new_path=new_path)
        while index < len(raw) and raw[index].startswith("@@"):
            hunk, index = _parse_hunk(raw, index)
            current.hunks.append(hunk)
        if not current.hunks:
            name = new_path or old_path or "unknown"
            raise ValueError(f"file section for {name} has no hunks")
        files.append(current)
    if not files:
        raise ValueError("patch contains no file sections; expected '---' and '+++' headers")
    return files


def _split_lines(text: str) -> tuple[list[str], bool]:
    """Return the text's lines and whether it ended with a newline."""
    if not text:
        return [], False
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return normalized.splitlines(), normalized.endswith("\n")


def _locate(lines: list[str], block: list[str], expected: int, floor: int) -> int | None:
    """Find ``block`` near ``expected``, searching forward then backward."""
    start = max(expected, floor)
    if not block:
        return start
    last = len(lines) - len(block)
    lower = min(last, start + _SEARCH_LIMIT)
    for candidate in range(start, lower + 1):
        if lines[candidate : candidate + len(block)] == block:
            return candidate
    upper = max(floor, start - _SEARCH_LIMIT)
    for candidate in range(start - 1, upper - 1, -1):
        if lines[candidate : candidate + len(block)] == block:
            return candidate
    return None


class ApplyPatchInput(BaseModel):
    """Arguments for ``apply_patch``.

    ``patch`` is a unified diff, possibly covering several files. ``dry_run``
    validates every hunk and reports what would change without writing.
    """

    patch: str = Field(
        min_length=1,
        max_length=2_000_000,
        description="Unified diff text with one or more '---'/'+++' file sections.",
    )
    dry_run: bool = Field(
        default=False,
        description="Validate and report only; leave every file unchanged.",
    )


class ApplyPatchTool(Tool):
    """Apply a multi-file unified diff atomically.

    Every hunk is matched against the current file before any write happens, so
    one bad hunk aborts the whole patch and leaves every file byte-identical.
    Paths resolve through :class:`~atlas.agent.artifacts.LocalArtifactStore`;
    absolute paths and ``..`` are allowed. Files are read as text and never
    executed or imported. Deleting a file is not expressible through this tool.
    """

    name = "apply_patch"
    description = (
        "Apply a unified diff across one or more UTF-8 text files atomically. "
        "Every hunk is validated before any write; a mismatch aborts the whole patch. "
        "Supports creating a file from /dev/null and a dry_run preview."
    )
    trust: ClassVar[Literal["default", "opt_in"]] = "default"
    max_bytes: ClassVar[int] = 1_000_000

    @property
    def input_model(self) -> type[BaseModel]:
        return ApplyPatchInput

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        request = ApplyPatchInput.model_validate(arguments)
        sections = _parse_patch(request.patch)
        writes = [self._build(section, store) for section in sections]
        total_hunks = sum(write.hunks for write in writes)
        if request.dry_run:
            return ToolResult(text=self._report(writes, total_hunks, dry_run=True))
        for write in writes:
            write.path.parent.mkdir(parents=True, exist_ok=True)
            write.path.write_text(write.text, encoding="utf-8")
        return ToolResult(
            text=self._report(writes, total_hunks, dry_run=False),
            artifacts=[store.relative(write.path) for write in writes],
        )

    def _build(self, section: _FilePatch, store: LocalArtifactStore) -> _PendingWrite:
        """Validate one file section and return the content to write."""
        if section.new_path is None:
            raise ValueError("deleting a file is not supported by apply_patch")
        path = store.resolve(section.new_path)
        created = section.old_path is None
        if created:
            if path.exists():
                raise ValueError(f"new file already exists: {section.new_path}")
            original = ""
            ends_with_newline = True
        else:
            if not path.is_file():
                raise FileNotFoundError(f"File does not exist: {section.new_path}")
            if path.stat().st_size > self.max_bytes:
                raise ValueError(f"File {section.new_path} exceeds the {self.max_bytes} byte limit")
            try:
                original = path.read_text(encoding="utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(f"File {section.new_path} is not valid UTF-8") from exc
            _, ends_with_newline = _split_lines(original)

        lines, _ = _split_lines(original)
        result: list[str] = []
        index = 0
        touched_end = False
        for number, hunk in enumerate(section.hunks, 1):
            old_block = [text for operator, text in hunk.lines if operator in (" ", "-")]
            new_block = [text for operator, text in hunk.lines if operator in (" ", "+")]
            if len(old_block) != hunk.old_count or len(new_block) != hunk.new_count:
                raise ValueError(f"hunk {number} header and body disagree")
            expected = hunk.old_start - 1 if hunk.old_count else hunk.old_start
            if created and expected != 0:
                raise ValueError(f"hunk {number}: a new file must start at line 0")
            found = _locate(lines, old_block, expected, index)
            if found is None:
                preview = old_block[0] if old_block else ""
                raise ValueError(
                    f"hunk {number}: context does not match {section.new_path} at line "
                    f"{expected + 1}" + (f" (expected {preview!r})" if preview else "")
                )
            result.extend(lines[index:found])
            result.extend(new_block)
            index = found + len(old_block)
            if index == len(lines):
                touched_end = True
            if hunk.new_no_newline:
                ends_with_newline = False
        result.extend(lines[index:])
        if touched_end and not any(hunk.new_no_newline for hunk in section.hunks):
            ends_with_newline = True

        text = "\n".join(result)
        if result and ends_with_newline:
            text += "\n"
        if len(text.encode("utf-8")) > self.max_bytes:
            raise ValueError(f"Patched {section.new_path} exceeds the {self.max_bytes} byte limit")
        return _PendingWrite(
            display=section.new_path,
            path=path,
            text=text,
            hunks=len(section.hunks),
            created=created,
        )

    def _report(self, writes: list[_PendingWrite], hunks: int, *, dry_run: bool) -> str:
        """Return the human-readable summary for a run or a dry run."""
        prefix = "dry run: would apply" if dry_run else "applied"
        lines = [f"{prefix} {hunks} hunks to {len(writes)} files"]
        for write in writes:
            note = " (created)" if write.created else ""
            lines.append(f"  {write.display}: {write.hunks} hunks{note}")
        return "\n".join(lines)
