from __future__ import annotations

import fnmatch
import os
import re
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import Tool, ToolResult


class GrepInput(BaseModel):
    """Arguments for ``grep``.

    ``pattern`` is a Python regular expression unless ``fixed_strings`` is set.
    ``path`` and ``glob`` are workspace-relative; ``glob`` filters file names
    such as ``*.py``.
    """

    pattern: str = Field(description="Python regular expression to search for.")
    path: str = Field(default=".", description="Path relative to the local artifact workspace.")
    glob: str | None = Field(
        default=None,
        description="Only search files whose name matches this shell-style pattern, e.g. '*.py'.",
    )
    max_matches: int = Field(
        default=200,
        ge=1,
        le=1000,
        description="Maximum number of matching lines to return (1-1000).",
    )
    case_sensitive: bool = Field(default=True, description="Match case sensitively.")
    fixed_strings: bool = Field(
        default=False, description="Treat the pattern as a literal string instead of a regex."
    )


def _compile(pattern: str, *, fixed_strings: bool, case_sensitive: bool) -> re.Pattern[str]:
    """Compile the search pattern, escaping it when it is a fixed string."""
    expression = re.escape(pattern) if fixed_strings else pattern
    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        return re.compile(expression, flags)
    except re.error as exc:
        raise ValueError(f"Invalid regular expression: {exc}") from exc


def _iter_files(base: Path) -> Iterator[Path]:
    """Yield regular files under ``base`` in sorted order."""
    if base.is_file():
        yield base
        return
    for root, dir_names, file_names in os.walk(base, followlinks=False):
        current = Path(root)
        dir_names[:] = sorted(dir_names)
        for name in sorted(file_names):
            candidate = current / name
            if candidate.is_file():
                yield candidate


def _display_path(store: LocalArtifactStore, path: Path) -> str:
    """Return a relative path when it sits inside the workspace, else absolute."""
    return store.relative(path)


class GrepTool(Tool):
    """Search text files for a regular expression.

    Every path is resolved through :class:`~atlas.agent.artifacts.LocalArtifactStore`,
    and absolute paths and ``..`` are allowed. Binary files (containing a NUL
    byte) and files above the per-file byte cap are skipped, and the whole scan
    stops at a wall-clock and input-byte budget.
    """

    name = "grep"
    description = (
        "Search text files in the local artifact workspace for a Python regular expression "
        "and return matching lines as <relative_path>:<line_number>:<line_text>."
    )
    trust: ClassVar[Literal["default", "opt_in"]] = "default"
    max_file_bytes: ClassVar[int] = 1 << 20
    max_scan_bytes: ClassVar[int] = 20_000_000
    scan_seconds: ClassVar[float] = 5.0
    max_line_chars: ClassVar[int] = 300

    @property
    def input_model(self) -> type[BaseModel]:
        return GrepInput

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        request = GrepInput.model_validate(arguments)
        matcher = _compile(
            request.pattern,
            fixed_strings=request.fixed_strings,
            case_sensitive=request.case_sensitive,
        )
        base = store.resolve_read(request.path)
        if not base.exists():
            raise FileNotFoundError(f"Path does not exist: {request.path}")

        lines: list[str] = []
        overflow = 0
        scanned = 0
        started = time.monotonic()
        stopped = False
        for path in _iter_files(base):
            if scanned >= self.max_scan_bytes or time.monotonic() - started > self.scan_seconds:
                stopped = True
                break
            if request.glob is not None and not fnmatch.fnmatch(path.name, request.glob):
                continue
            try:
                if path.stat().st_size > self.max_file_bytes:
                    continue
                data = path.read_bytes()
            except OSError:
                continue
            scanned += len(data)
            if b"\x00" in data:
                continue
            relative = _display_path(store, path)
            for number, line in enumerate(data.decode("utf-8", errors="replace").splitlines(), 1):
                if not matcher.search(line):
                    continue
                if len(lines) < request.max_matches:
                    lines.append(f"{relative}:{number}:{line[: self.max_line_chars]}")
                else:
                    overflow += 1
        if not lines and not overflow:
            return ToolResult(text="no matches")
        if overflow:
            lines.append(f"... [truncated {overflow} more matches]")
        if stopped:
            lines.append("... [scan limit reached]")
        return ToolResult(text="\n".join(lines))
