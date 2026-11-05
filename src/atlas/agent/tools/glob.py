from __future__ import annotations

import glob as glob_module
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import Tool, ToolResult


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


class GlobTool(Tool):
    """List entries matching a shell-style glob pattern.

    The pattern is matched under a directory resolved by
    :class:`~atlas.agent.artifacts.LocalArtifactStore`. Absolute patterns and
    ``..`` are allowed. Results that sit outside the working tree are returned
    as absolute paths.
    """

    name = "glob"
    description = (
        "List files and directories matching a shell-style pattern such as '**/*.py'. "
        "Relative patterns resolve against the workspace; absolute patterns are allowed."
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

        pattern = os.path.expanduser(request.pattern)
        path_pattern = Path(pattern)
        candidate_paths: Iterable[str]
        if path_pattern.is_absolute():
            candidate_paths = glob_module.glob(pattern, recursive=True)
        elif ".." in path_pattern.parts:
            candidate_paths = glob_module.glob(str(base / pattern), recursive=True)
        else:
            candidate_paths = (str(path) for path in base.rglob(pattern))
        matches = [store.relative(Path(name)) for name in candidate_paths if Path(name).exists()]
        matches.sort()

        overflow = len(matches) - request.max_results
        if overflow > 0:
            matches = matches[: request.max_results]
            matches.append(f"... [truncated {overflow} more results]")
        return ToolResult(text="\n".join(matches))
