"""Shared directory walking for the search and navigation tools.

Invariants: no symlink following, sorted traversal, and the base path handed in by the
caller is never pruned, so an explicit path inside a skipped directory is still searched.
"""

from __future__ import annotations

import fnmatch
import os
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
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


_GIT_TIMEOUT_S = 5
_GIT_MAX_BYTES = 32_000_000
_MAX_BRACE_RESULTS = 64


@dataclass(frozen=True)
class IgnoreSet:
    """Git-ignored paths below ``base``, as reported by git (untracked files only).

    Paths are posix, relative to ``base``. Fully ignored directories are collapsed by git
    into one entry, so pruning that directory hides its children.
    """

    base: Path
    root: Path
    dirs: frozenset[str]
    files: frozenset[str]

    def _relative(self, current: Path, name: str) -> str | None:
        return _relative_posix(current / name, self.base)

    def hides_dir(self, current: Path, name: str) -> bool:
        """Return True when the child directory ``name`` of ``current`` is git-ignored."""
        return self._relative(current, name) in self.dirs

    def hides_file(self, current: Path, name: str) -> bool:
        """Return True when the child file ``name`` of ``current`` is git-ignored."""
        return self._relative(current, name) in self.files

    def summary(self, limit: int = 4) -> str:
        """Return a short list of ignored directories plus a file count; "" when empty."""
        parts = [f"{name}/" for name in sorted(self.dirs)[:limit]]
        if len(self.dirs) > limit:
            parts.append(f"{len(self.dirs) - limit} more dirs")
        if self.files:
            parts.append(f"{len(self.files)} files")
        return ", ".join(parts)


def _run_git(base: Path, *args: str) -> subprocess.CompletedProcess[bytes] | None:
    """Run git in ``base`` with a time cap; None when git is missing or slow."""
    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
    try:
        return subprocess.run(
            ["git", "-C", str(base), *args],
            capture_output=True,
            timeout=_GIT_TIMEOUT_S,
            env=env,
            check=False,
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def load_ignore_set(base: Path, *, root: Path) -> IgnoreSet | None:
    """Return the git-ignored set below ``base``, or None when git cannot say.

    None covers: ``base`` is a file, git is missing or slow, ``base`` is not in a work tree,
    or ``base`` itself is ignored (an explicit search inside an ignored directory is never
    filtered). ``.atlas`` is never hidden. Never raises.
    """
    if not base.is_dir():
        return None
    probe = _run_git(base, "check-ignore", "-q", "--", ".")
    if probe is None or probe.returncode != 1:
        return None
    listing = _run_git(
        base, "ls-files", "--others", "--ignored", "--exclude-standard", "--directory", "-z"
    )
    if listing is None or listing.returncode != 0 or len(listing.stdout) > _GIT_MAX_BYTES:
        return None
    dirs: set[str] = set()
    files: set[str] = set()
    for raw in listing.stdout.split(b"\0"):
        entry = raw.decode("utf-8", errors="replace")
        if not entry or entry.split("/", 1)[0] == ".atlas":
            continue
        if entry.endswith("/"):
            dirs.add(entry.rstrip("/"))
        else:
            files.add(entry)
    return IgnoreSet(base=base, root=root, dirs=frozenset(dirs), files=frozenset(files))


def prune_dirs(
    current: Path,
    dir_names: list[str],
    *,
    root: Path,
    include_ignored: bool = False,
    extra_skip: frozenset[str] = frozenset(),
    ignore: IgnoreSet | None = None,
) -> None:
    """Sort ``dir_names`` in place and drop skipped directories, for use inside os.walk."""
    if include_ignored:
        dir_names[:] = sorted(dir_names)
        return
    dir_names[:] = sorted(
        name
        for name in dir_names
        if not is_skipped_dir(current, name, root=root, extra_skip=extra_skip)
        and not (ignore is not None and ignore.hides_dir(current, name))
    )


def expand_braces(pattern: str) -> list[str]:
    """Expand the first top-level ``{a,b}`` group recursively; unbalanced braces stay literal."""
    results = _expand(pattern)
    return results[:_MAX_BRACE_RESULTS]


def _expand(pattern: str) -> list[str]:
    start = pattern.find("{")
    if start < 0:
        return [pattern]
    depth = 0
    pieces: list[str] = []
    piece_start = start + 1
    for index in range(start, len(pattern)):
        char = pattern[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                pieces.append(pattern[piece_start:index])
                head, tail = pattern[:start], pattern[index + 1 :]
                out: list[str] = []
                for piece in pieces:
                    out.extend(_expand(head + piece + tail))
                    if len(out) >= _MAX_BRACE_RESULTS:
                        return out[:_MAX_BRACE_RESULTS]
                return out
        elif char == "," and depth == 1:
            pieces.append(pattern[piece_start:index])
            piece_start = index + 1
    return [pattern]


def _match_parts(pattern: list[str], parts: list[str]) -> bool:
    """Match path segments against pattern segments; ``**`` spans zero or more segments."""
    if not pattern:
        return not parts
    head, rest = pattern[0], pattern[1:]
    if head == "**":
        return any(_match_parts(rest, parts[skip:]) for skip in range(len(parts) + 1))
    if not parts:
        return False
    return fnmatch.fnmatchcase(parts[0], head) and _match_parts(rest, parts[1:])


def match_glob(pattern: str, rel: str) -> bool:
    """Return True when posix path ``rel`` (relative to the search base) matches ``pattern``.

    A pattern without ``/`` or ``**`` matches the basename at any depth; any other pattern
    matches at any depth below the base (``src/x.py`` also matches ``a/src/x.py``).
    """
    parts = rel.split("/")
    for alternative in expand_braces(pattern):
        segments = [part for part in alternative.split("/") if part]
        bare = alternative[3:] if alternative.startswith("**/") else alternative
        if "/" not in bare and "**" not in bare:
            if fnmatch.fnmatchcase(parts[-1], bare):
                return True
        elif _match_parts(["**", *segments], parts):
            return True
    return False


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


def iter_files(
    base: Path,
    *,
    root: Path,
    include_ignored: bool = False,
    ignore: IgnoreSet | None = None,
) -> Iterator[Path]:
    """Yield regular files under ``base`` in sorted order, pruning skipped directories.

    Git-ignored paths are pruned too unless ``include_ignored`` is set.
    """
    if base.is_file():
        yield base
        return
    if ignore is None and not include_ignored:
        ignore = load_ignore_set(base, root=root)
    for current_name, dir_names, file_names in os.walk(base, followlinks=False):
        current = Path(current_name)
        prune_dirs(current, dir_names, root=root, include_ignored=include_ignored, ignore=ignore)
        for name in sorted(file_names):
            if ignore is not None and ignore.hides_file(current, name):
                continue
            candidate = current / name
            if candidate.is_file():
                yield candidate
