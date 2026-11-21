from __future__ import annotations

import re
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import Tool, ToolDefinition, ToolResult
from atlas.agent.scope import cancel_requested
from atlas.agent.tools._grep_scan import (
    WorkerConfig,
    run_worker,
    worker_available,
)
from atlas.agent.tools._schema import slim_definition
from atlas.agent.tools._walk import iter_files, load_ignore_set, match_glob

_GNU_ESCAPES = ("\\|", "\\(", "\\)")
_GNU_NOTE = (
    "note: no match for the pattern as written; retried reading \\| \\( \\) as regex operators "
    "(this tool uses Python regex: write | ( ) unescaped)"
)
_LIMIT_PREFIX = "no matches (scan limit reached"
_FINAL_PREFIXES = (
    _LIMIT_PREFIX,
    "no matches (search cancelled",
    "pattern too slow",
)
_NAMED_FILE_BYTES = 32 << 20
_MAX_WORKER_PATHS = 200_000
_PER_FILE_SECONDS = 2.0
_SLOW_ABORT = 3
_MAX_RESULT_CHARS = 16_000
MAX_CONTEXT = 30
MAX_MATCHES = 1000


class GrepInput(BaseModel):
    """Arguments for ``grep``."""

    pattern: str = Field(description="Python regular expression (use | for alternation).")
    path: str = Field(default=".", description="File or directory to search.")
    glob: str | None = Field(default=None, description="File-name filter, e.g. '*.py'.")
    max_matches: int = Field(
        default=200,
        ge=1,
        description="Max matching lines (clamped to 1000).",
    )
    case_sensitive: bool = Field(default=True, description="Match case sensitively.")
    fixed_strings: bool = Field(default=False, description="Treat the pattern as a literal.")
    include_ignored: bool = Field(
        default=False, description="Also search .git, virtualenvs and caches."
    )
    context: int = Field(
        default=0,
        ge=0,
        description="Lines before and after a match (clamped to 30).",
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


def _glob_matches(path: Path, base: Path, glob: str) -> bool:
    """Match `glob` against the path relative to `base` (a file base matches by its name)."""
    relative = path.name if base.is_file() else path.relative_to(base).as_posix()
    return bool(match_glob(glob, relative))


def _names(names: list[str]) -> str:
    """Join up to three names, noting how many more were left out."""
    shown = ", ".join(names[:3])
    return f"{shown}, +{len(names) - 3} more" if len(names) > 3 else shown


@dataclass
class _Scan:
    """Accumulated result of one pass over the files."""

    rows: list[tuple[str, bool]] = field(default_factory=list)
    counts: list[tuple[str, int]] = field(default_factory=list)
    overflow: int = 0
    total: int = 0
    files: int = 0
    stopped: bool = False
    oversize: int = 0
    binary: int = 0
    slow: list[str] = field(default_factory=list)
    aborted: bool = False
    cancelled: bool = False
    ignored: str = ""
    error: str | None = None


class GrepTool(Tool):
    """Search text files for a regular expression.

    Every path is resolved through :class:`~atlas.agent.artifacts.LocalArtifactStore`,
    and absolute paths and ``..`` are allowed. Binary files (containing a NUL
    byte) and files above the per-file byte cap are skipped and do not count toward
    the scan budget; the whole scan stops at a wall-clock and input-byte budget, and
    a stopped scan is always reported, even with zero matches. A file named by
    ``path`` is searched up to 32 MB. Regex scans run in a killable child process.
    """

    name = "grep"
    description = (
        "Search file contents with a Python regular expression (use | for alternation). "
        "Returns path:line:text. context=N adds surrounding lines, output_mode lists only "
        "file names or per-file counts, glob filters file names. Skips .git, virtualenvs and "
        "caches unless include_ignored is true. Binary files and files over 1 MB are skipped "
        "(a file you name is searched up to 32 MB); respects .gitignore."
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

    def definition(self) -> ToolDefinition:
        """Return the advertised schema without null branches or the model docstring."""
        return slim_definition(super().definition())

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
        if text.startswith(_FINAL_PREFIXES):
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
        if scan.error:
            return scan.error, False
        if scan.total == 0:
            return self._zero_text(scan, base), False
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
        if scan.cancelled:
            lines.append(f"... [search cancelled after {scan.files} files]")
        if scan.slow:
            lines.append(
                f"... [skipped {len(scan.slow)} file(s) where the pattern ran over "
                f"{_PER_FILE_SECONDS:.0f} s: {_names(scan.slow)}]"
            )
        return "\n".join(lines), True

    def _zero_text(self, scan: _Scan, base: Path) -> str:
        """Explain a zero-match scan: cancelled, too slow, limit reached, or what was skipped."""
        if scan.cancelled:
            return f"no matches (search cancelled after {scan.files} files)"
        if scan.aborted:
            return (
                f"pattern too slow: {len(scan.slow)} files each exceeded "
                f"{_PER_FILE_SECONDS:.0f} s ({_names(scan.slow)}); "
                "simplify the pattern or narrow path/glob"
            )
        if scan.stopped:
            return f"{_LIMIT_PREFIX} after {scan.files} files; narrow path or glob)"
        cap = _NAMED_FILE_BYTES if base.is_file() else self.max_file_bytes
        parts = [
            f"{scan.oversize} file(s) over {cap >> 20} MB" if scan.oversize else "",
            f"{scan.binary} binary file(s)" if scan.binary else "",
            scan.ignored,
        ]
        skipped = ", ".join(part for part in parts if part)
        text = f"no matches (not searched: {skipped})" if skipped else "no matches"
        if scan.slow:
            text += (
                f" [skipped {len(scan.slow)} file(s) where the pattern ran over "
                f"{_PER_FILE_SECONDS:.0f} s: {_names(scan.slow)}]"
            )
        return text

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

    def _candidates(
        self, request: GrepInput, base: Path, store: LocalArtifactStore, scan: _Scan
    ) -> Iterator[Path]:
        """Yield files passing the glob and size cap; count oversize files on `scan`."""
        ignore = None if request.include_ignored else load_ignore_set(base, root=store.root)
        cap = _NAMED_FILE_BYTES if base.is_file() else self.max_file_bytes
        for path in iter_files(
            base, root=store.root, include_ignored=request.include_ignored, ignore=ignore
        ):
            if request.glob is not None and not _glob_matches(path, base, request.glob):
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size > cap:
                scan.oversize += 1
                continue
            yield path
        scan.ignored = ignore.summary() if ignore is not None else ""

    def _scan(
        self, request: GrepInput, matcher: re.Pattern[str], base: Path, store: LocalArtifactStore
    ) -> _Scan:
        """Collect matches within the byte and time budget, in a killable child for regexes."""
        scan = _Scan()
        started = time.monotonic()
        if request.fixed_strings or sys.platform == "win32" or not worker_available():
            return self._scan_in_process(request, matcher, base, store, scan, started)
        paths: list[Path] = []
        for path in self._candidates(request, base, store, scan):
            paths.append(path)
            if time.monotonic() - started > self.scan_seconds or len(paths) >= _MAX_WORKER_PATHS:
                scan.stopped = True
                break
        config = WorkerConfig(
            pattern=matcher.pattern,
            flags=matcher.flags,
            max_scan_bytes=self.max_scan_bytes,
            scan_seconds=max(self.scan_seconds - (time.monotonic() - started), 0.0),
            max_file_bytes=_NAMED_FILE_BYTES if base.is_file() else self.max_file_bytes,
            max_hits=request.max_matches,
            per_file_seconds=_PER_FILE_SECONDS,
            slow_abort=_SLOW_ABORT,
        )
        try:
            result = run_worker(config, [str(p) for p in paths], deadline=self.scan_seconds + 4)
        except OSError:
            return self._scan_in_process(request, matcher, base, store, _Scan(), started)
        scan.files = result.files
        scan.stopped = scan.stopped or result.stopped
        scan.binary = result.binary
        scan.aborted = result.aborted
        scan.cancelled = result.cancelled
        scan.error = result.error and (
            f"{result.error}; retry with fixed_strings=true if the pattern is literal"
        )
        scan.slow = [store.relative(paths[index]) for index in result.slow]
        shown = 0
        for index, line_numbers, count in result.hits:
            if index in result.slow:
                continue
            shown = self._record_hit(
                request,
                scan,
                store.relative(paths[index]),
                paths[index],
                line_numbers,
                count,
                shown,
            )
        return scan

    def _record_hit(
        self,
        request: GrepInput,
        scan: _Scan,
        relative: str,
        path: Path,
        hits: list[int],
        count: int,
        shown: int,
    ) -> int:
        """Re-read a worker hit file and account its rows; return the updated shown count."""
        try:
            lines = _split_lines(path.read_bytes().decode("utf-8", errors="replace"))
        except OSError:
            return shown
        hits = [number for number in hits if number <= len(lines)]
        return self._account(request, scan, relative, lines, hits, count, shown)

    def _account(
        self,
        request: GrepInput,
        scan: _Scan,
        relative: str,
        lines: list[str],
        hits: list[int],
        count: int,
        shown: int,
    ) -> int:
        """Add one file's matches to the scan totals and rows; return the new shown count."""
        scan.total += count
        scan.counts.append((relative, count))
        if request.output_mode != "content":
            return shown
        keep = hits[: max(request.max_matches - shown, 0)]
        scan.overflow += count - len(keep)
        self._add_rows(scan, relative, lines, keep, request.context)
        return shown + len(keep)

    def _scan_in_process(
        self,
        request: GrepInput,
        matcher: re.Pattern[str],
        base: Path,
        store: LocalArtifactStore,
        scan: _Scan,
        started: float,
    ) -> _Scan:
        """Search in this process; used for literals, Windows and a missing worker."""
        scanned = 0
        shown = 0
        for path in self._candidates(request, base, store, scan):
            if cancel_requested():
                scan.cancelled = True
                break
            if scanned >= self.max_scan_bytes or time.monotonic() - started > self.scan_seconds:
                scan.stopped = True
                break
            try:
                data = path.read_bytes()
            except OSError:
                continue
            if b"\x00" in data:
                scan.binary += 1
                continue
            scanned += len(data)
            scan.files += 1
            lines = _split_lines(data.decode("utf-8", errors="replace"))
            hits = [number for number, line in enumerate(lines, 1) if matcher.search(line)]
            if hits:
                shown = self._account(
                    request, scan, store.relative(path), lines, hits, len(hits), shown
                )
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
