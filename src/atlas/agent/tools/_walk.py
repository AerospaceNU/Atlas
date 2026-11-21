"""Shared directory walking for the search and navigation tools.

Invariants: no symlink following, sorted traversal, and the base path handed in by the
caller is never pruned, so an explicit path inside a skipped directory is still searched.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

SKIP_DIR_NAMES = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        ".venv",
        "venv",
        "__pycache__",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        ".tox",
    }
)
# Relative to the store root: file copies, conversation logs and saved tool output.
# `.atlas/data` and `.atlas/artifacts` stay searchable, except each session's
# `.atlas/artifacts/<session>/tool-output` (see `_is_session_tool_output`).
SKIP_RELATIVE = frozenset({".atlas/checkpoints", ".atlas/sessions", ".atlas/tool-output"})


def _relative_posix(path: Path, root: Path) -> str | None:
    """Return ``path`` relative to ``root`` as posix text, or None when outside it."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return None


def is_skipped_dir(
    current: Path, name: str, *, root: Path, extra_skip: frozenset[str] = frozenset()
) -> bool:
    """Return True when the child directory ``name`` of ``current`` is pruned by default."""
    if name in SKIP_DIR_NAMES or name in extra_skip:
        return True
    relative = _relative_posix(current / name, root)
    if relative is None:
        return False
    return relative in SKIP_RELATIVE or _is_session_tool_output(relative)


def _is_session_tool_output(relative: str) -> bool:
    """True for ``.atlas/artifacts/<session>/tool-output``, where saved tool output lives."""
    parts = relative.split("/")
    return len(parts) == 4 and parts[:2] == [".atlas", "artifacts"] and parts[3] == "tool-output"


def prune_dirs(
    current: Path,
    dir_names: list[str],
    *,
    root: Path,
    include_ignored: bool = False,
    extra_skip: frozenset[str] = frozenset(),
) -> None:
    """Sort ``dir_names`` in place and drop skipped directories, for use inside os.walk."""
    if include_ignored:
        dir_names[:] = sorted(dir_names)
        return
    dir_names[:] = sorted(
        name
        for name in dir_names
        if not is_skipped_dir(current, name, root=root, extra_skip=extra_skip)
    )


def is_ignored_path(path: Path, base: Path, *, root: Path) -> bool:
    """Return True when ``path`` sits under (or is) a skipped directory below ``base``."""
    try:
        parts = path.relative_to(base).parts
    except ValueError:
        return False
    current = base
    for index, name in enumerate(parts):
        is_last = index == len(parts) - 1
        if (not is_last or (current / name).is_dir()) and is_skipped_dir(current, name, root=root):
            return True
        current = current / name
    return False


def iter_files(base: Path, *, root: Path, include_ignored: bool = False) -> Iterator[Path]:
    """Yield regular files under ``base`` in sorted order, pruning skipped directories."""
    if base.is_file():
        yield base
        return
    for current_name, dir_names, file_names in os.walk(base, followlinks=False):
        current = Path(current_name)
        prune_dirs(current, dir_names, root=root, include_ignored=include_ignored)
        for name in sorted(file_names):
            candidate = current / name
            if candidate.is_file():
                yield candidate
