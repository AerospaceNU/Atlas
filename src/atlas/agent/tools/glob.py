from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from atlas.agent.artifacts import LocalArtifactStore, SandboxDenied
from atlas.agent.contracts import Tool, ToolResult


class GlobInput(BaseModel):
    """Arguments for ``glob``.

    ``pattern`` is a shell-style pattern such as ``**/*.py``, relative to the
    workspace. ``path`` narrows the search to one workspace-relative directory.
    """

    pattern: str = Field(
        description="Shell-style pattern relative to the workspace, e.g. '**/*.py'."
    )
    path: str = Field(default=".", description="Path relative to the local artifact workspace.")
    max_results: int = Field(
        default=500, ge=1, le=10_000, description="Maximum number of paths to return."
    )


def _reject_unsafe_pattern(pattern: str) -> None:
    """Refuse empty, absolute, and parent-traversing glob patterns."""
    if not pattern:
        raise ValueError("Glob pattern must not be empty")
    if Path(pattern).is_absolute():
        raise SandboxDenied("Glob patterns must be relative to the artifact root")
    if ".." in pattern:
        raise SandboxDenied("Glob patterns must not contain '..'")


def _display_path(store: LocalArtifactStore, base: Path, path: Path) -> str:
    """Return a workspace-relative path without exposing a host path."""
    try:
        return store.relative(path)
    except SandboxDenied:
        return path.relative_to(base).as_posix()


class GlobTool(Tool):
    """List workspace entries matching a shell-style glob pattern.

    The pattern is matched under a path resolved by
    :class:`~atlas.agent.artifacts.LocalArtifactStore`, so no search can leave
    the sandbox. Absolute and ``..`` patterns are refused, and symlinks are
    skipped so a result never points outside the workspace.
    """

    name = "glob"
    description = (
        "List files and directories in the local artifact workspace matching a shell-style "
        "pattern such as '**/*.py', one workspace-relative path per line."
    )
    trust: ClassVar[Literal["default", "opt_in"]] = "default"

    @property
    def input_model(self) -> type[BaseModel]:
        return GlobInput

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        request = GlobInput.model_validate(arguments)
        _reject_unsafe_pattern(request.pattern)
        base = store.resolve_read(request.path)
        if not base.exists():
            raise FileNotFoundError(f"Path does not exist: {request.path}")

        matches: list[str] = []
        for candidate in base.rglob(request.pattern):
            if candidate.is_symlink():
                continue
            try:
                candidate.resolve().relative_to(base)
            except ValueError:
                continue
            matches.append(_display_path(store, base, candidate))
        matches.sort()

        overflow = len(matches) - request.max_results
        if overflow > 0:
            matches = matches[: request.max_results]
            matches.append(f"... [truncated {overflow} more results]")
        return ToolResult(text="\n".join(matches))
