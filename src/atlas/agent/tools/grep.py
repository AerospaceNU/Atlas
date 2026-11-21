from __future__ import annotations

import fnmatch
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import Tool, ToolResult
from atlas.agent.tools._walk import iter_files

_GNU_ESCAPES = ("\\|", "\\(", "\\)")
_GNU_NOTE = (
    "note: no match for the pattern as written; retried reading \\| \\( \\) as regex operators "
    "(this tool uses Python regex: write | ( ) unescaped)"
)
_LIMIT_PREFIX = "no matches (scan limit reached"
_MAX_RESULT_CHARS = 16_000
MAX_CONTEXT = 30
MAX_MATCHES = 1000


class GrepInput(BaseModel):
    """Arguments for ``grep``.

    ``pattern`` is a Python regular expression unless ``fixed_strings`` is set.
    ``path`` and ``glob`` are workspace-relative; ``glob`` filters file names
    such as ``*.py``.
    """

    pattern: str = Field(description="Python regular expression (use | for alternation).")
    path: str = Field(default=".", description="File or directory to search.")
    glob: str | None = Field(default=None, description="File-name filter, e.g. '*.py'.")
    max_matches: int = Field(
        default=200,
        ge=1,
        description="Maximum matching lines to return (at least 1; values above 1000 are clamped to 1000).",
    )
    case_sensitive: bool = Field(default=True, description="Match case sensitively.")
    fixed_strings: bool = Field(default=False, description="Treat the pattern as a literal.")
    include_ignored: bool = Field(
        default=False, description="Also search .git, virtualenvs and caches."
    )
    context: int = Field(
        default=0,
        ge=0,
        description="Lines shown before and after a match (values above 30 are clamped to 30).",
    )
    output_mode: Literal["content", "files_with_matches", "count"] = Field(
        default="content", description="content, files_with_matches (paths only) or count."
    )


def _clamped(request: GrepInput) -> tuple[GrepInput, list[str]]:
    """Clamp over-large numeric arguments to their maximum and describe each clamp."""
    update: dict[str, int] = {}
    notes: list[str] = []
    if request.context > MAX_CONTEXT:
        update["context"] = MAX_CONTEXT
        notes.append(f"note: context clamped to {MAX_CONTEXT}")
    if request.max_matches > MAX_MATCHES:
        update["max_matches"] = MAX_MATCHES
        notes.append(f"note: max_matches clamped to {MAX_MATCHES}")
    return (request.model_copy(update=update) if update else request), notes


def _compile(pattern: str, *, fixed_strings: bool, case_sensitive: bool) -> re.Pattern[str]:
    """Compile the search pattern, escaping it when it is a fixed string."""
    expression = re.escape(pattern) if fixed_strings else pattern
    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        return re.compile(expression, flags)
    except re.error as exc:
        raise ValueError(f"Invalid regular expression: {exc}") from exc


def _has_gnu_escapes(request: GrepInput) -> bool:
    """Return True when the pattern uses GNU-style escaped operators."""
    return not request.fixed_strings and any(seq in request.pattern for seq in _GNU_ESCAPES)


def _rewrite_gnu(pattern: str) -> str:
    """Rewrite the GNU-escaped alternation and group operators into plain regex operators."""
    return pattern.replace("\\|", "|").replace("\\(", "(").replace("\\)", ")")


def _split_lines(text: str) -> list[str]:
    """Split on the newline character only, so numbers match editors and read_file.

    ``str.splitlines`` also breaks on form feed, U+2028 and similar; those must
    stay inside their line. One trailing carriage return per line is dropped.
    """
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return [line[:-1] if line.endswith("\r") else line for line in lines]


@dataclass
class _Scan:
    """Accumulated result of one pass over the files."""

    rows: list[tuple[str, bool]] = field(default_factory=list)
    counts: list[tuple[str, int]] = field(default_factory=list)
    overflow: int = 0
    total: int = 0
    files: int = 0
    stopped: bool = False


class GrepTool(Tool):
    """Search text files for a regular expression.

    Every path is resolved through :class:`~atlas.agent.artifacts.LocalArtifactStore`,
    and absolute paths and ``..`` are allowed. Binary files (containing a NUL
    byte) and files above the per-file byte cap are skipped and do not count toward
    the scan budget; the whole scan stops at a wall-clock and input-byte budget, and
    a stopped scan is always reported, even with zero matches.
    """

    name = "grep"
    description = (
        "Search file contents with a Python regular expression (use | for alternation). "
        "Returns path:line:text. context=N adds surrounding lines, output_mode lists only "
        "file names or per-file counts, glob filters file names. Skips .git, virtualenvs and "
        "caches unless include_ignored is true. Binary files and files over 1 MB are skipped."
    )
    trust: ClassVar[Literal["default", "opt_in"]] = "default"
    max_file_bytes: ClassVar[int] = 1 << 20
    max_scan_bytes: ClassVar[int] = 100_000_000
    scan_seconds: ClassVar[float] = 10.0
    max_line_chars: ClassVar[int] = 300
    max_result_chars: ClassVar[int] = _MAX_RESULT_CHARS

    @property
    def input_model(self) -> type[BaseModel]:
        return GrepInput

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        request, notes = _clamped(GrepInput.model_validate(arguments))
        matcher = _compile(
            request.pattern,
            fixed_strings=request.fixed_strings,
            case_sensitive=request.case_sensitive,
        )
        base = store.resolve_read(request.path)
        if not base.exists():
            raise FileNotFoundError(f"Path does not exist: {request.path}")

        text, found = self._search(request, matcher, base, store)
        prefix = "".join(f"{note}\n" for note in notes)
        if found or not _has_gnu_escapes(request):
            return ToolResult(text=f"{prefix}{text}")
        if text.startswith(_LIMIT_PREFIX):
            return ToolResult(text=f"{prefix}{text}\n{_GNU_NOTE}")
        try:
            retry = _compile(
                _rewrite_gnu(request.pattern),
                fixed_strings=False,
                case_sensitive=request.case_sensitive,
            )
        except ValueError:
            return ToolResult(text=f"{prefix}{text}\n{_GNU_NOTE}")
        retried, retry_found = self._search(request, retry, base, store)
        if retry_found:
            return ToolResult(text=f"{prefix}{_GNU_NOTE}\n{retried}")
        return ToolResult(text=f"{prefix}{text}\n{_GNU_NOTE}")

    def _search(
        self, request: GrepInput, matcher: re.Pattern[str], base: Path, store: LocalArtifactStore
    ) -> tuple[str, bool]:
        """Scan once and render the result; the flag says whether anything matched."""
        scan = self._scan(request, matcher, base, store)
        if scan.total == 0:
            if scan.stopped:
                return (
                    f"{_LIMIT_PREFIX} after {scan.files} files; narrow path or glob)",
                    False,
                )
            return "no matches", False
        rows = self._mode_rows(request, scan)
        lines, hidden = self._cap(rows, reserve=min(450, self.max_result_chars // 4))
        if hidden:
            lines.append(
                f"... [{hidden} more matching lines; narrow the pattern or use "
                'output_mode="files_with_matches"]'
            )
        if request.output_mode == "count":
            lines.append(f"total: {scan.total}")
        if scan.overflow:
            lines.append(f"... [truncated {scan.overflow} more matches]")
        if scan.stopped:
            lines.append("... [scan limit reached]")
        return "\n".join(lines), True

    def _mode_rows(self, request: GrepInput, scan: _Scan) -> list[tuple[str, bool]]:
        """Return the body rows for the requested output mode."""
        if request.output_mode == "files_with_matches":
            return [(path, True) for path, _ in scan.counts]
        if request.output_mode == "count":
            return [(f"{path}:{count}", True) for path, count in scan.counts]
        return scan.rows

    def _cap(self, rows: list[tuple[str, bool]], reserve: int) -> tuple[list[str], int]:
        """Keep rows within the cap less ``reserve`` for trailer lines; return them and hidden count."""
        kept: list[str] = []
        used = 0
        for index, (row, _) in enumerate(rows):
            if used + len(row) + 1 > self.max_result_chars - reserve:
                return kept, sum(1 for _, is_match in rows[index:] if is_match)
            kept.append(row)
            used += len(row) + 1
        return kept, 0

    def _scan(
        self, request: GrepInput, matcher: re.Pattern[str], base: Path, store: LocalArtifactStore
    ) -> _Scan:
        """Walk the files and collect matches within the byte and time budget."""
        scan = _Scan()
        scanned = 0
        shown = 0
        started = time.monotonic()
        for path in iter_files(base, root=store.root, include_ignored=request.include_ignored):
            if scanned >= self.max_scan_bytes or time.monotonic() - started > self.scan_seconds:
                scan.stopped = True
                break
            if request.glob is not None and not fnmatch.fnmatch(path.name, request.glob):
                continue
            try:
                if path.stat().st_size > self.max_file_bytes:
                    continue
                data = path.read_bytes()
            except OSError:
                continue
            if b"\x00" in data:
                continue
            scanned += len(data)
            scan.files += 1
            lines = _split_lines(data.decode("utf-8", errors="replace"))
            hits = [number for number, line in enumerate(lines, 1) if matcher.search(line)]
            if not hits:
                continue
            scan.total += len(hits)
            relative = store.relative(path)
            scan.counts.append((relative, len(hits)))
            if request.output_mode != "content":
                continue
            keep = hits[: max(request.max_matches - shown, 0)]
            shown += len(keep)
            scan.overflow += len(hits) - len(keep)
            self._add_rows(scan, relative, lines, keep, request.context)
        return scan

    def _add_rows(
        self, scan: _Scan, relative: str, lines: list[str], hits: list[int], context: int
    ) -> None:
        """Append grep -C style rows for one file; windows that touch are merged."""
        if not hits:
            return
        matched = set(hits)
        previous_end = -1
        for number in sorted(self._window_lines(hits, context, len(lines))):
            if context and scan.rows and previous_end != number - 1:
                scan.rows.append(("--", False))
            previous_end = number
            sep = ":" if number in matched else "-"
            text = lines[number - 1][: self.max_line_chars]
            scan.rows.append((f"{relative}{sep}{number}{sep}{text}", number in matched))

    @staticmethod
    def _window_lines(hits: list[int], context: int, total: int) -> set[int]:
        """Return every 1-based line number inside a match window."""
        shown: set[int] = set()
        for number in hits:
            shown.update(range(max(1, number - context), min(total, number + context) + 1))
        return shown
