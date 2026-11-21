"""Shared helpers for the file-editing tools.

Invariants: matching always happens on LF-normalised text, a file's own newline
style is restored on write, and nothing here executes or imports file content.
"""

from __future__ import annotations

import ast
import difflib
import warnings
from collections.abc import Callable
from pathlib import Path

_HINT_CHARS = 1_200
_LINE_CHARS = 200
_MAX_FIRST_LINE_OCCURRENCES = 8
_SIMILARITY_CANDIDATES = 5_000


def normalize_newlines(text: str) -> str:
    """Return ``text`` with CRLF and lone CR converted to LF."""
    if "\r" not in text:
        return text
    return text.replace("\r\n", "\n").replace("\r", "\n")


def read_text_keeping_newlines(path: Path, display: str, max_bytes: int) -> tuple[str, str]:
    """Read a UTF-8 file and return its LF-normalised text and its newline style.

    The style is CRLF when CRLF lines outnumber bare LF lines, else LF.
    The read is bounded, so a file over ``max_bytes`` is refused, not truncated.
    """
    with path.open("rb") as stream:
        data = stream.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError(f"File {display} exceeds the {max_bytes} byte read limit")
    try:
        raw = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"File {display} is not valid UTF-8") from exc
    crlf = raw.count("\r\n")
    bare_lf = raw.count("\n") - crlf
    return normalize_newlines(raw), "\r\n" if crlf > bare_lf else "\n"


def encode_text(text: str, newline: str) -> bytes:
    """Return ``text`` (LF-normalised) as UTF-8 bytes in the given newline style."""
    if newline != "\n":
        text = text.replace("\n", newline)
    return text.encode("utf-8")


def write_text_keeping_newlines(path: Path, text: str, newline: str) -> None:
    """Write LF-normalised ``text`` to ``path`` using the file's newline style."""
    path.write_bytes(encode_text(text, newline))


def split_lines(text: str) -> list[str]:
    """Return the lines of LF-normalised text, split on the newline character only.

    Unlike ``str.splitlines``, form feed, vertical tab, NEL and the Unicode line and
    paragraph separators stay inside their line. A trailing newline adds no empty line.
    """
    if not text:
        return []
    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()
    return lines


def _display_lines(text: str) -> list[str]:
    return split_lines(text) or [""]


def numbered(lines: list[str], first: int) -> str:
    """Render lines in the numbered gutter used by the read tools."""
    return "\n".join(f"{n:>6}| {line[:_LINE_CHARS]}" for n, line in enumerate(lines, first))


def _cap_hint(hint: str) -> str:
    if len(hint) <= _HINT_CHARS:
        return hint
    return hint[: _HINT_CHARS - 3] + "..."


def _loose_match(
    text_lines: list[str], needle_lines: list[str], norm: Callable[[str], str]
) -> list[int]:
    """Return start indexes where every line matches once ``norm`` is applied."""
    wanted = [norm(line) for line in needle_lines]
    hits: list[int] = []
    for start in range(len(text_lines) - len(wanted) + 1):
        if norm(text_lines[start]) != wanted[0]:
            continue
        if all(norm(text_lines[start + k]) == wanted[k] for k in range(1, len(wanted))):
            hits.append(start)
    return hits


def _whitespace_hint(text_lines: list[str], needle_lines: list[str]) -> str:
    for norm in (str.rstrip, str.lstrip, str.strip):
        hits = _loose_match(text_lines, needle_lines, norm)
        if len(hits) == 1:
            first = hits[0]
            last = first + len(needle_lines)
            return (
                f"A match that differs only in whitespace is at lines {first + 1}-{last}:\n"
                + numbered(text_lines[first:last], first + 1)
            )
    return ""


def not_found_hint(text: str, needle: str) -> str:
    """Return advice for an ``old_string`` that is not in ``text``, or an empty string.

    The advice is informational only: callers never apply an inexact match.
    """
    text_lines = text.split("\n")
    needle_lines = needle.rstrip("\n").split("\n")
    if not any(line.strip() for line in needle_lines):
        return ""
    hint = _whitespace_hint(text_lines, needle_lines)
    if hint:
        return _cap_hint(hint)
    target = next(line.strip() for line in needle_lines if line.strip())
    where = [i for i, line in enumerate(text_lines) if target in line]
    if where:
        shown = ", ".join(str(i + 1) for i in where[:_MAX_FIRST_LINE_OCCURRENCES])
        return _cap_hint(
            f"The first line of old_string is at line(s) {shown}; the text after it differs:\n"
            + numbered(text_lines[where[0] : where[0] + 6], where[0] + 1)
        )
    candidates: dict[str, int] = {}
    for index, line in enumerate(text_lines[:_SIMILARITY_CANDIDATES]):
        stripped = line.strip()
        if stripped:
            candidates.setdefault(stripped, index)
    close = difflib.get_close_matches(target, list(candidates), n=1, cutoff=0.6)
    if close:
        return _cap_hint(f"Most similar line {candidates[close[0]] + 1}: {close[0][:_LINE_CHARS]}")
    return ""


def occurrence_lines(text: str, needle: str, limit: int = 8) -> list[int]:
    """Return the 1-based line of each non-overlapping occurrence, at most ``limit``."""
    lines: list[int] = []
    if not needle:
        return lines
    position = text.find(needle)
    counted_to = 0
    line = 1
    while position != -1 and len(lines) < limit:
        line += text.count("\n", counted_to, position)
        counted_to = position
        lines.append(line)
        position = text.find(needle, position + len(needle))
    return lines


def changed_span(
    text: str, start: int, end: int, *, context: int = 3, max_lines: int = 40
) -> tuple[int, int]:
    """Return the 1-based first and last line that ``changed_region`` shows."""
    total = len(_display_lines(text))
    first = text.count("\n", 0, start) + 1
    last = text.count("\n", 0, max(start, end - 1)) + 1
    low = max(1, first - context)
    high = min(total, last + context)
    return low, max(low, min(high, low + max_lines - 1))


def changed_region(
    text: str, start: int, end: int, *, context: int = 3, max_lines: int = 40
) -> str:
    """Return the lines covering [``start``, ``end``) of ``text`` plus context, numbered.

    A ``... [K more lines]`` line follows when the region was capped at ``max_lines``.
    """
    lines = _display_lines(text)
    last = text.count("\n", 0, max(start, end - 1)) + 1
    wanted_high = min(len(lines), last + context)
    low, high = changed_span(text, start, end, context=context, max_lines=max_lines)
    body = numbered(lines[low - 1 : high], low)
    if wanted_high > high:
        body += f"\n... [{wanted_high - high} more lines]"
    return body


def python_syntax_warning(display: str, before: str | None, after: str) -> str | None:
    """Warn when an edit leaves a Python file unparseable that parsed before.

    Only ``.py`` and ``.pyi`` paths are checked. ``ast.parse`` never executes the code.
    """
    if not display.endswith((".py", ".pyi")):
        return None
    error = _syntax_error(after)
    if error is None:
        return None
    if before is not None and _syntax_error(before) is not None:
        return None
    return (
        f"warning: {display} no longer parses as Python ({error}). Fix it before running the code."
    )


def _syntax_error(source: str) -> str | None:
    """Return ``line N: message`` when ``source`` does not parse, else None."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ast.parse(source)
    except SyntaxError as exc:
        return f"line {exc.lineno}: {exc.msg}"
    except (ValueError, RecursionError):
        return None
    return None
