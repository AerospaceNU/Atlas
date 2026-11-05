from __future__ import annotations

import os
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import Tool, ToolResult


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


def _display_path(store: LocalArtifactStore, path: Path) -> str:
    """Return a relative path when it sits inside the workspace, else absolute."""
    return store.relative(path)


def _entries(store: LocalArtifactStore, base: Path, *, recursive: bool) -> list[str]:
    """Collect entries under ``base`` with directories suffixed by ``/``."""
    entries: list[str] = []
    if not recursive:
        for child in base.iterdir():
            name = _display_path(store, child)
            entries.append(f"{name}/" if child.is_dir() else name)
        return sorted(entries)
    for root, dir_names, file_names in os.walk(base, followlinks=False):
        current = Path(root)
        dir_names[:] = sorted(dir_names)
        for name in dir_names:
            entries.append(f"{_display_path(store, current / name)}/")
        for name in sorted(file_names):
            entries.append(_display_path(store, current / name))
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
        "workspace-relative entry per line with directories suffixed by '/'."
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

        entries = _entries(store, base, recursive=request.recursive)
        overflow = len(entries) - request.max_entries
        if overflow > 0:
            entries = entries[: request.max_entries]
            entries.append(f"... [truncated {overflow} more entries]")
        return ToolResult(text="\n".join(entries))
