from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import Tool, ToolResult
from atlas.agent.tools._edits import (
    encode_text,
    python_syntax_warning,
    read_text_keeping_newlines,
    split_lines,
)

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_SEARCH_LIMIT = 200
_LENIENT_NOTE = (
    "note: read leniently (hunk line counts in the patch were wrong or the format was non-standard)"
)
_WHITESPACE_NOTE = "note: matched ignoring trailing whitespace"
_UPDATE_PREFIX = "*** Update File:"
_ADD_PREFIX = "*** Add File:"
_DELETE_PREFIX = "*** Delete File:"
_IGNORED_PREFIXES = ("*** Begin Patch", "*** End Patch", "diff --git ", "index ")
_DELETE_ERROR = "deleting a file is not supported by apply_patch"


@dataclass
class _Hunk:
    """One parsed hunk, kept in patch order.

    ``has_hint`` is False for a bare ``@@`` (lenient parsing only); the hunk is then
    searched for forward from the end of the previous hunk.
    """

    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[tuple[str, str]] = field(default_factory=list)
    old_no_newline: bool = False
    new_no_newline: bool = False
    has_hint: bool = True


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
    newline: str
    hunks: int
    created: bool
    first_line: int
    last_line: int
    loose: bool
    warning: str | None
    original: str | None = None


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


def _looks_like_hunk_line(raw: list[str], index: int) -> bool:
    """Return whether ``raw[index]`` is a hunk body line rather than a section header."""
    line = raw[index]
    if line.startswith("--- ") and index + 1 < len(raw) and raw[index + 1].startswith("+++ "):
        return False
    return line[:1] in (" ", "+", "-")


def _parse_hunk(raw: list[str], index: int, *, reject_surplus: bool = True) -> tuple[_Hunk, int]:
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
    if reject_surplus and index < len(raw) and _looks_like_hunk_line(raw, index):
        raise ValueError(
            f"line {index + 1}: hunk has more lines than its header counts "
            f"({hunk.old_count} old / {hunk.new_count} new)"
        )
    return hunk, index


def _patch_lines(patch: str) -> list[str]:
    """Return the patch split into lines, refusing NUL bytes."""
    if "\x00" in patch:
        raise ValueError("patch must not contain NUL bytes")
    return split_lines(patch.replace("\r\n", "\n").replace("\r", "\n"))


def _parse_patch(patch: str, *, reject_surplus: bool = True) -> list[_FilePatch]:
    """Parse a unified diff strictly into file sections without touching the workspace.

    With ``reject_surplus`` False, lines after a hunk's counted body (such as a git
    signature trailer) are skipped, as the original strict parser did.
    """
    raw = _patch_lines(patch)
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
            hunk, index = _parse_hunk(raw, index, reject_surplus=reject_surplus)
            current.hunks.append(hunk)
        if not current.hunks:
            name = new_path or old_path or "unknown"
            raise ValueError(f"file section for {name} has no hunks")
        files.append(current)
    if not files:
        raise ValueError("patch contains no file sections; expected '---' and '+++' headers")
    return files


def _is_ignored(line: str) -> bool:
    """Return whether a lenient parse skips this envelope or git metadata line."""
    return line.startswith(_IGNORED_PREFIXES)


def _is_section_start(raw: list[str], index: int) -> bool:
    """Return whether ``raw[index]`` begins a new file section."""
    line = raw[index]
    if line.startswith((_UPDATE_PREFIX, _ADD_PREFIX, _DELETE_PREFIX)):
        return True
    return line.startswith("--- ") and index + 1 < len(raw) and raw[index + 1].startswith("+++ ")


def _trim_patch(raw: list[str]) -> list[str]:
    """Drop trailing blank lines and one surrounding markdown fence."""
    lines = list(raw)
    while lines and not lines[-1].strip():
        lines.pop()
    if lines and lines[0].startswith("```"):
        lines.pop(0)
        if lines and lines[-1].strip() == "```":
            lines.pop()
    elif lines and lines[-1].strip() == "```":
        lines.pop()
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def _lenient_hunk(raw: list[str], index: int) -> tuple[_Hunk, int]:
    """Parse one hunk from its ``@@`` line, computing counts from the body."""
    match = _HUNK_RE.match(raw[index])
    start = int(match.group(1)) if match else 0
    hunk = _Hunk(0, 0, 0, 0, has_hint=match is not None)
    hunk.old_start = start
    number = index + 1
    index += 1
    while (
        index < len(raw) and not raw[index].startswith("@@") and not _is_section_start(raw, index)
    ):
        line = raw[index]
        index += 1
        if _is_ignored(line):
            continue
        if line.startswith("\\"):
            _mark_no_newline(hunk)
            continue
        operator = line[:1] or " "
        if operator not in (" ", "-", "+"):
            raise ValueError(
                f"line {index}: unexpected hunk line prefix {operator!r}; "
                "expected a line starting with ' ', '-' or '+'"
            )
        hunk.lines.append((operator, line[1:]))
    hunk.old_count = sum(1 for op, _ in hunk.lines if op in (" ", "-"))
    hunk.new_count = sum(1 for op, _ in hunk.lines if op in (" ", "+"))
    if not hunk.lines:
        raise ValueError(
            f"line {number}: hunk is empty; expected lines starting with ' ', '-' or '+'"
        )
    return hunk, index


def _lenient_added_file(raw: list[str], index: int) -> tuple[_Hunk, int]:
    """Parse the ``+`` lines that follow ``*** Add File:`` into one hunk."""
    hunk = _Hunk(0, 0, 1, 0)
    while index < len(raw) and not _is_section_start(raw, index):
        line = raw[index]
        index += 1
        if _is_ignored(line):
            continue
        if not line.startswith("+"):
            raise ValueError(
                f"line {index}: expected a line starting with '+' in an added file, got {line!r}"
            )
        hunk.lines.append(("+", line[1:]))
    hunk.new_count = len(hunk.lines)
    if not hunk.lines:
        raise ValueError("an added file section has no '+' lines")
    return hunk, index


def _lenient_section(raw: list[str], index: int) -> tuple[_FilePatch, int]:
    """Parse one file section starting at its header line."""
    line = raw[index]
    if line.startswith(_DELETE_PREFIX):
        raise ValueError(_DELETE_ERROR)
    if line.startswith(_ADD_PREFIX):
        section = _FilePatch(None, line[len(_ADD_PREFIX) :].strip() or None)
        index += 1
    elif line.startswith(_UPDATE_PREFIX):
        path = line[len(_UPDATE_PREFIX) :].strip() or None
        section = _FilePatch(path, path)
        index += 1
    else:
        section = _FilePatch(_header_path(line[4:]), _header_path(raw[index + 1][4:]))
        index += 2
    if section.new_path is None and section.old_path is not None:
        raise ValueError(_DELETE_ERROR)
    if section.new_path is None:
        raise ValueError(f"line {index}: file section has no path")
    if line.startswith(_ADD_PREFIX):
        hunk, index = _lenient_added_file(raw, index)
        section.hunks.append(hunk)
        return section, index
    while index < len(raw) and not _is_section_start(raw, index):
        if _is_ignored(raw[index]) or not raw[index].strip():
            index += 1
        elif raw[index].startswith("@@"):
            hunk, index = _lenient_hunk(raw, index)
            section.hunks.append(hunk)
        else:
            raise ValueError(f"line {index + 1}: expected a hunk header starting with '@@'")
    if not section.hunks:
        raise ValueError(f"file section for {section.new_path} has no hunks")
    return section, index


def _parse_patch_lenient(patch: str) -> list[_FilePatch]:
    """Parse a patch tolerantly: envelopes, fences, bare ``@@``, and wrong hunk counts."""
    raw = _trim_patch(_patch_lines(patch))
    files: list[_FilePatch] = []
    index = 0
    while index < len(raw):
        if _is_section_start(raw, index):
            section, index = _lenient_section(raw, index)
            files.append(section)
        elif raw[index].startswith("--- "):
            raise ValueError(f"line {index + 1}: '---' header must be followed by a '+++' header")
        else:
            index += 1
    if not files:
        raise ValueError(
            "patch contains no file sections; expected '--- path' and '+++ path' headers "
            "or '*** Update File: path'"
        )
    return files


def _split_lines(text: str) -> tuple[list[str], bool]:
    """Return the text's lines and whether it ended with a newline."""
    if not text:
        return [], False
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return split_lines(normalized), normalized.endswith("\n")


def _same(file_line: str, patch_line: str, loose: bool) -> bool:
    if loose:
        return file_line.rstrip() == patch_line.rstrip()
    return file_line == patch_line


def _fits(lines: list[str], block: list[str], at: int, loose: bool) -> bool:
    if at < 0 or at + len(block) > len(lines):
        return False
    return all(_same(lines[at + k], line, loose) for k, line in enumerate(block))


def _candidates(lines: list[str], block: list[str], expected: int, floor: int, span: int) -> range:
    """Return the start positions a hunk may match at, nearest-forward first."""
    start = max(expected, floor)
    last = len(lines) - len(block)
    return range(start, min(last, start + span) + 1)


def _locate(
    lines: list[str], block: list[str], expected: int, floor: int, *, loose: bool, span: int
) -> int | None:
    """Find ``block`` near ``expected``, searching forward then backward."""
    start = max(expected, floor)
    if not block:
        return start
    for candidate in _candidates(lines, block, expected, floor, span):
        if _fits(lines, block, candidate, loose):
            return candidate
    for candidate in range(start - 1, max(floor, start - span) - 1, -1):
        if _fits(lines, block, candidate, loose):
            return candidate
    return None


def _all_fits(lines: list[str], block: list[str], floor: int, loose: bool) -> list[int]:
    """Return every start at or after ``floor`` where ``block`` fits, in order."""
    last = len(lines) - len(block)
    return [at for at in range(max(floor, 0), last + 1) if _fits(lines, block, at, loose)]


def _nearest_miss(lines: list[str], block: list[str], expected: int, floor: int, span: int) -> str:
    """Describe the first disagreement at the position where most lines agree."""
    start = max(expected, floor)
    window = list(range(start, min(len(lines) - len(block), start + span) + 1))
    window += list(range(start - 1, max(floor, start - span) - 1, -1))
    best_at, best_score = -1, -1
    for at in window:
        score = sum(1 for k, line in enumerate(block) if lines[at + k].rstrip() == line.rstrip())
        if score > best_score:
            best_at, best_score = at, score
    if best_at < 0:
        return ""
    for k, line in enumerate(block):
        if lines[best_at + k] != line:
            return f"; near line {best_at + k + 1}: expected {line!r}, found {lines[best_at + k]!r}"
    return ""


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
    relative paths resolve against the workspace, absolute paths and ``..`` are
    allowed. Files are read as text, never executed or imported, and keep their
    newline style. A patch that parses strictly is applied as written; only a
    patch that fails strict parsing is re-read leniently. Deleting a file is not
    expressible through this tool.
    """

    name = "apply_patch"
    description = (
        "Apply a unified diff to one or more text files atomically: every hunk is checked "
        "before anything is written. Accepts standard ---/+++/@@ diffs and *** Begin Patch "
        "envelopes; wrong hunk line counts are recomputed from the hunk body. Creating a "
        "file from /dev/null is supported, deleting is not. dry_run validates only."
    )
    trust: ClassVar[Literal["default", "opt_in"]] = "default"
    max_bytes: ClassVar[int] = 1_000_000

    @property
    def input_model(self) -> type[BaseModel]:
        return ApplyPatchInput

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        request = ApplyPatchInput.model_validate(arguments)
        lenient = False
        try:
            sections = _parse_patch(request.patch)
        except ValueError as strict_error:
            if "NUL" in str(strict_error):
                raise
            try:
                sections = _parse_patch_lenient(request.patch)
                lenient = True
            except ValueError as lenient_error:
                # A patch the original strict parser accepted keeps working even when
                # its trailing lines (e.g. a git signature) defeat lenient parsing.
                try:
                    sections = _parse_patch(request.patch, reject_surplus=False)
                except ValueError:
                    raise lenient_error from None
        pending: dict[Path, _PendingWrite] = {}
        for section in sections:
            built = self._build(section, store, pending)
            pending[built.path.resolve()] = built
        writes = list(pending.values())
        total_hunks = sum(write.hunks for write in writes)
        if not request.dry_run:
            for write in writes:
                write.path.parent.mkdir(parents=True, exist_ok=True)
                write.path.write_bytes(encode_text(write.text, write.newline))
        text = self._report(writes, total_hunks, dry_run=request.dry_run, lenient=lenient)
        if request.dry_run:
            return ToolResult(text=text)
        return ToolResult(text=text, artifacts=[store.relative(write.path) for write in writes])

    def _build(
        self,
        section: _FilePatch,
        store: LocalArtifactStore,
        pending: dict[Path, _PendingWrite],
    ) -> _PendingWrite:
        """Validate one file section and return the content to write.

        A path already in ``pending`` is patched from that earlier result, never from
        disk, so several sections for one file compose; the returned write replaces it.
        """
        if section.new_path is None:
            raise ValueError(_DELETE_ERROR)
        path = store.resolve(section.new_path)
        earlier = pending.get(path.resolve())
        created = section.old_path is None
        newline = "\n"
        original: str | None = None
        if created:
            if earlier is not None or path.exists():
                raise ValueError(f"new file already exists: {section.new_path}")
            source = ""
            ends_with_newline = True
        elif earlier is not None:
            source, newline, original = earlier.text, earlier.newline, earlier.original
            _, ends_with_newline = _split_lines(source)
        else:
            if not path.is_file():
                raise FileNotFoundError(f"File does not exist: {section.new_path}")
            source, newline = read_text_keeping_newlines(path, section.new_path, self.max_bytes)
            original = source
            _, ends_with_newline = _split_lines(source)

        lines, _ = _split_lines(source)
        result: list[str] = []
        index = 0
        touched_end = False
        loose_used = False
        first_line = last_line = 1
        for number, hunk in enumerate(section.hunks, 1):
            old_block = [text for operator, text in hunk.lines if operator in (" ", "-")]
            new_block = [text for operator, text in hunk.lines if operator in (" ", "+")]
            if len(old_block) != hunk.old_count or len(new_block) != hunk.new_count:
                raise ValueError(f"hunk {number} header and body disagree")
            if hunk.has_hint:
                expected = hunk.old_start - 1 if hunk.old_count else hunk.old_start
                span = _SEARCH_LIMIT
            else:
                expected = index
                span = len(lines)
            if created and (expected != 0 or (not hunk.has_hint and index)):
                raise ValueError(f"hunk {number}: a new file must start at line 0")
            found, loose = self._find(
                hunk, number, lines, old_block, expected, index, span, section.new_path, created
            )
            loose_used = loose_used or loose
            result.extend(lines[index:found])
            if number == 1:
                first_line = len(result) + 1
            self._emit(hunk, lines, found, result)
            last_line = max(len(result), first_line)
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
        if len(encode_text(text, newline)) > self.max_bytes:
            raise ValueError(f"Patched {section.new_path} exceeds the {self.max_bytes} byte limit")
        if earlier is not None:
            hunks_total = earlier.hunks + len(section.hunks)
            first_line = min(first_line, earlier.first_line)
            last_line = max(last_line, earlier.last_line)
            loose_used = loose_used or earlier.loose
            created = earlier.created
        else:
            hunks_total = len(section.hunks)
        return _PendingWrite(
            display=earlier.display if earlier is not None else section.new_path,
            path=path,
            text=text,
            newline=newline,
            hunks=hunks_total,
            created=created,
            first_line=first_line,
            last_line=last_line,
            loose=loose_used,
            warning=python_syntax_warning(section.new_path, original, text),
            original=original,
        )

    @staticmethod
    def _find(
        hunk: _Hunk,
        number: int,
        lines: list[str],
        old_block: list[str],
        expected: int,
        floor: int,
        span: int,
        display: str,
        created: bool,
    ) -> tuple[int, bool]:
        """Return where the hunk applies and whether whitespace-loose matching was needed.

        A hunk with a line hint takes the nearest match. A hint-less hunk never guesses:
        it must fit in exactly one place at or after ``floor``.
        """
        if hunk.has_hint:
            found = _locate(lines, old_block, expected, floor, loose=False, span=span)
            if found is not None:
                return found, False
            found = _locate(lines, old_block, expected, floor, loose=True, span=span)
            if found is not None:
                return found, True
        else:
            if not old_block:
                if created:
                    return 0, False
                raise ValueError(
                    f"hunk {number}: an insertion needs context lines or a line number"
                )
            for loose in (False, True):
                places = _all_fits(lines, old_block, floor, loose)
                if len(places) == 1:
                    return places[0], loose
                if places:
                    shown = ", ".join(str(at + 1) for at in places[:8])
                    raise ValueError(
                        f"hunk {number}: its lines match {len(places)} places in {display} "
                        f"(lines {shown}); add more context lines or a "
                        "@@ -line,count +line,count @@ header"
                    )
        miss = _nearest_miss(lines, old_block, expected, floor, span)
        raise ValueError(
            f"hunk {number}: context does not match {display} at line "
            f"{max(expected, floor) + 1}{miss}"
        )

    @staticmethod
    def _emit(hunk: _Hunk, lines: list[str], found: int, result: list[str]) -> None:
        """Append the hunk's new text; context keeps the file's own line text."""
        position = found
        for operator, text in hunk.lines:
            if operator == " ":
                result.append(lines[position])
                position += 1
            elif operator == "-":
                position += 1
            else:
                result.append(text)

    def _report(
        self, writes: list[_PendingWrite], hunks: int, *, dry_run: bool, lenient: bool
    ) -> str:
        """Return the human-readable summary for a run or a dry run."""
        prefix = "dry run: would apply" if dry_run else "applied"
        lines = [f"{prefix} {hunks} hunks to {len(writes)} files"]
        for write in writes:
            note = " (created)" if write.created else ""
            span = f" (now lines {write.first_line}-{write.last_line})"
            lines.append(f"  {write.display}: {write.hunks} hunks{note}{span}")
        if lenient:
            lines.append(_LENIENT_NOTE)
        if any(write.loose for write in writes):
            lines.append(_WHITESPACE_NOTE)
        lines.extend(write.warning for write in writes if write.warning)
        return "\n".join(lines)
