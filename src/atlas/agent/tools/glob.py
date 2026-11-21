from __future__ import annotations

import fnmatch
import glob as glob_module
import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import Tool, ToolResult
from atlas.agent.tools._walk import SKIP_DIR_NAMES, prune_dirs


class GlobInput(BaseModel):
    """Arguments for ``glob``.

    ``pattern`` is a shell-style pattern such as ``**/*.py``, relative to
    ``path``. ``pattern`` may also be absolute. ``path`` narrows the search to
    one directory.
    """

    pattern: str = Field(
        description="Shell-style pattern, e.g. '**/*.py'. Relative patterns resolve against path."
    )
    path: str = Field(default=".", description="Directory to search in, relative or absolute.")
    max_results: int = Field(
        default=500, ge=1, le=10_000, description="Maximum number of paths to return."
    )
    include_ignored: bool = Field(
        default=False, description="Also match inside .git, virtualenvs and caches."
    )


def _names_skipped_dir(pattern: str) -> bool:
    """Return True when the first segment of a multi-segment pattern is a skipped directory.

    Such a pattern (``.git/*``, ``node_modules/pkg/*.js``) is an explicit request.
    """
    parts = [part for part in pattern.split("/") if part]
    return len(parts) > 1 and parts[0] in SKIP_DIR_NAMES


def _single_segment(pattern: str) -> str | None:
    """Return the file-name pattern when it matches at any depth, else None."""
    bare = pattern[3:] if pattern.startswith("**/") else pattern
    if "/" in bare or "**" in bare:
        return None
    return bare


def _walk_matches(
    base: Path, name_pattern: str, *, root: Path, include_ignored: bool
) -> Iterator[str]:
    """Yield entries at any depth whose name matches, pruning skipped directories."""
    for current_name, dir_names, file_names in os.walk(base, followlinks=False):
        current = Path(current_name)
        prune_dirs(current, dir_names, root=root, include_ignored=include_ignored)
        for name in (*dir_names, *file_names):
            if fnmatch.fnmatchcase(name, name_pattern):
                yield str(current / name)


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


def _walk_path_matches(
    base: Path, pattern: str, *, root: Path, include_ignored: bool
) -> Iterator[str]:
    """Yield entries whose path below ``base`` matches at any depth, pruning during the walk.

    Mirrors ``Path.rglob`` for multi-segment patterns without entering skipped directories.
    """
    pattern_parts = ["**", *[part for part in pattern.split("/") if part]]
    for current_name, dir_names, file_names in os.walk(base, followlinks=False):
        current = Path(current_name)
        prune_dirs(current, dir_names, root=root, include_ignored=include_ignored)
        for name in (*dir_names, *file_names):
            entry = current / name
            if _match_parts(pattern_parts, list(entry.relative_to(base).parts)):
                yield str(entry)


class GlobTool(Tool):
    """List entries matching a shell-style glob pattern.

    The pattern is matched under a directory resolved by
    :class:`~atlas.agent.artifacts.LocalArtifactStore`. Absolute patterns and
    ``..`` are allowed. Results that sit outside the working tree are returned
    as absolute paths. Relative patterns skip .git, virtualenvs and caches unless
    ``include_ignored`` is set; absolute and ``..`` patterns are explicit and unfiltered.
    """

    name = "glob"
    description = (
        "List files and directories matching a shell-style pattern such as '**/*.py'. "
        "Relative patterns resolve against the workspace; absolute patterns are allowed. "
        ".git, virtualenvs and caches are skipped unless include_ignored is true."
    )
    trust: ClassVar[Literal["default", "opt_in"]] = "default"

    @property
    def input_model(self) -> type[BaseModel]:
        return GlobInput

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        request = GlobInput.model_validate(arguments)
        if not request.pattern:
            raise ValueError("Glob pattern must not be empty")
        base = store.resolve_read(request.path)
        if not base.exists():
            raise FileNotFoundError(f"Path does not exist: {request.path}")

        candidate_paths = self._candidates(request, base, store)
        matches = [store.relative(Path(name)) for name in candidate_paths if Path(name).exists()]
        matches.sort()

        overflow = len(matches) - request.max_results
        if overflow > 0:
            matches = matches[: request.max_results]
            matches.append(f"... [truncated {overflow} more results]")
        return ToolResult(text="\n".join(matches))

    def _candidates(
        self, request: GlobInput, base: Path, store: LocalArtifactStore
    ) -> Iterable[str]:
        """Return raw candidate paths for the pattern."""
        pattern = os.path.expanduser(request.pattern)
        path_pattern = Path(pattern)
        if path_pattern.is_absolute():
            return glob_module.glob(pattern, recursive=True)
        if ".." in path_pattern.parts:
            return glob_module.glob(str(base / pattern), recursive=True)
        if not base.is_dir():
            return (str(path) for path in base.rglob(pattern))
        include_ignored = request.include_ignored or _names_skipped_dir(pattern)
        name_pattern = _single_segment(pattern)
        if name_pattern is not None:
            return _walk_matches(
                base, name_pattern, root=store.root, include_ignored=include_ignored
            )
        return _walk_path_matches(base, pattern, root=store.root, include_ignored=include_ignored)
