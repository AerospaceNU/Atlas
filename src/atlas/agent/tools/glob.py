from __future__ import annotations

import glob as glob_module
import itertools
import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import Tool, ToolResult
from atlas.agent.tools._walk import (
    SKIP_DIR_NAMES,
    IgnoreSet,
    expand_braces,
    load_ignore_set,
    match_glob,
    prune_dirs,
)


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
        default=False,
        description="Also match inside .git, virtualenvs, caches and git-ignored paths.",
    )


def _names_skipped_dir(pattern: str) -> bool:
    """Return True when the first segment of a multi-segment pattern is a skipped directory.

    Such a pattern (``.git/*``, ``node_modules/pkg/*.js``) is an explicit request.
    """
    parts = [part for part in pattern.split("/") if part]
    return len(parts) > 1 and parts[0] in SKIP_DIR_NAMES


def _walk_matches(
    base: Path,
    pattern: str,
    *,
    root: Path,
    include_ignored: bool,
    ignore: IgnoreSet | None,
) -> Iterator[str]:
    """Yield entries below ``base`` matching ``pattern``, pruning during the walk."""
    for current_name, dir_names, file_names in os.walk(base, followlinks=False):
        current = Path(current_name)
        prune_dirs(current, dir_names, root=root, include_ignored=include_ignored, ignore=ignore)
        for name in (*dir_names, *file_names):
            if ignore is not None and name in file_names and ignore.hides_file(current, name):
                continue
            entry = current / name
            if match_glob(pattern, entry.relative_to(base).as_posix()):
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
        "Respects .gitignore in git repos plus .git, virtualenvs and caches; set include_ignored "
        "to search them. Supports {a,b} braces."
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

        ignore = self._ignore_set(request, base, store)
        candidate_paths = self._candidates(request, base, store, ignore)
        matches = [store.relative(Path(name)) for name in candidate_paths if Path(name).exists()]
        matches.sort()
        if not matches:
            return ToolResult(text=self._no_matches(request, base, store, ignore))

        overflow = len(matches) - request.max_results
        if overflow > 0:
            matches = matches[: request.max_results]
            matches.append(f"... [truncated {overflow} more results]")
        return ToolResult(text="\n".join(matches))

    def _ignore_set(
        self, request: GlobInput, base: Path, store: LocalArtifactStore
    ) -> IgnoreSet | None:
        """Return the git-ignore set for a relative pattern over a directory base."""
        pattern = os.path.expanduser(request.pattern)
        path_pattern = Path(pattern)
        if path_pattern.is_absolute() or ".." in path_pattern.parts or not base.is_dir():
            return None
        if request.include_ignored or _names_skipped_dir(pattern):
            return None
        return load_ignore_set(base, root=store.root)

    def _no_matches(
        self,
        request: GlobInput,
        base: Path,
        store: LocalArtifactStore,
        ignore: IgnoreSet | None,
    ) -> str:
        """Return ``no matches``, noting git-ignored hits only when some really exist."""
        if ignore is None:
            return "no matches"
        hidden = self._candidates(request, base, store, None)
        hits = sorted(
            store.relative(Path(name))
            for name in itertools.islice(hidden, 3)
            if Path(name).exists()
        )
        if not hits:
            return "no matches"
        return (
            f"no matches outside git-ignored paths; {len(hits)}+ match(es) are git-ignored "
            f"(e.g. {', '.join(hits)}); set include_ignored=true to search them"
        )

    def _candidates(
        self,
        request: GlobInput,
        base: Path,
        store: LocalArtifactStore,
        ignore: IgnoreSet | None,
    ) -> Iterable[str]:
        """Return raw candidate paths for the pattern."""
        pattern = os.path.expanduser(request.pattern)
        path_pattern = Path(pattern)
        if path_pattern.is_absolute():
            return itertools.chain.from_iterable(
                glob_module.glob(alt, recursive=True) for alt in expand_braces(pattern)
            )
        if ".." in path_pattern.parts:
            return itertools.chain.from_iterable(
                glob_module.glob(str(base / alt), recursive=True) for alt in expand_braces(pattern)
            )
        if not base.is_dir():
            return (str(path) for path in base.rglob(pattern))
        include_ignored = request.include_ignored or _names_skipped_dir(pattern)
        return _walk_matches(
            base, pattern, root=store.root, include_ignored=include_ignored, ignore=ignore
        )
