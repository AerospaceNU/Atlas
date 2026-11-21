from __future__ import annotations

import os
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import Tool, ToolResult
from atlas.agent.tools._walk import load_ignore_set, prune_dirs


class LsInput(BaseModel):
    """Arguments for ``ls``.

    ``path`` is workspace-relative. ``recursive`` walks subdirectories and
    ``max_entries`` bounds how many entries are returned.
    """

    path: str = Field(default=".", description="Path relative to the local artifact workspace.")
    recursive: bool = Field(default=False, description="List entries in subdirectories too.")
    max_entries: int = Field(
        default=500, ge=1, le=10_000, description="Maximum number of entries to return."
    )
    include_ignored: bool = Field(
        default=False,
        description="Also list inside .git, virtualenvs, caches and git-ignored paths.",
    )


def _display_path(store: LocalArtifactStore, path: Path) -> str:
    """Return a relative path when it sits inside the workspace, else absolute."""
    return store.relative(path)


def _entries(
    store: LocalArtifactStore,
    base: Path,
    *,
    recursive: bool,
    include_ignored: bool = False,
    notes: list[str] | None = None,
) -> list[str]:
    """Collect entries under ``base`` with directories suffixed by ``/``.

    A recursive listing appends a line to ``notes`` when git-ignored paths were left out.
    """
    entries: list[str] = []
    if not recursive:
        for child in base.iterdir():
            name = _display_path(store, child)
            entries.append(f"{name}/" if child.is_dir() else name)
        return sorted(entries)
    ignore = None if include_ignored else load_ignore_set(base, root=store.root)
    hidden = 0
    for root, dir_names, file_names in os.walk(base, followlinks=False):
        current = Path(root)
        if ignore is not None:
            hidden += sum(ignore.hides_dir(current, name) for name in dir_names)
            hidden += sum(ignore.hides_file(current, name) for name in file_names)
        prune_dirs(
            current, dir_names, root=store.root, include_ignored=include_ignored, ignore=ignore
        )
        for name in dir_names:
            entries.append(f"{_display_path(store, current / name)}/")
        for name in sorted(file_names):
            if ignore is not None and ignore.hides_file(current, name):
                continue
            entries.append(_display_path(store, current / name))
    if ignore is not None and hidden and notes is not None:
        notes.append(f"(git-ignored paths not listed: {ignore.summary()})")
    return sorted(entries)


class LsTool(Tool):
    """List one directory.

    The path is resolved through :class:`~atlas.agent.artifacts.LocalArtifactStore`,
    and absolute paths and ``..`` are allowed. Results are workspace-relative
    names when they sit inside the working tree, and absolute names otherwise,
    with directories suffixed by ``/``.
    """

    name = "ls"
    description = (
        "List files and directories in a local artifact workspace directory, one "
        "workspace-relative entry per line with directories suffixed by '/'. Recursive "
        "listings skip git-ignored paths, .git, virtualenvs and caches unless include_ignored "
        "is true."
    )
    trust: ClassVar[Literal["default", "opt_in"]] = "default"

    @property
    def input_model(self) -> type[BaseModel]:
        return LsInput

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        request = LsInput.model_validate(arguments)
        base = store.resolve_read(request.path)
        if not base.exists():
            raise FileNotFoundError(f"Path does not exist: {request.path}")
        if not base.is_dir():
            raise ValueError(f"Path is not a directory: {request.path}")

        notes: list[str] = []
        entries = _entries(
            store,
            base,
            recursive=request.recursive,
            include_ignored=request.include_ignored,
            notes=notes,
        )
        overflow = len(entries) - request.max_entries
        if overflow > 0:
            entries = entries[: request.max_entries]
            entries.append(f"... [truncated {overflow} more entries]")
        return ToolResult(text="\n".join([*entries, *notes]))
