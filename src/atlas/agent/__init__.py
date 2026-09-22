from __future__ import annotations

from typing import Any

from atlas.agent.artifacts import ImageInfo, LocalArtifactStore
from atlas.agent.contracts import (
    Tool,
    ToolRegistry,
    default_registry,
    discover_tool_classes,
)
from atlas.agent.runtime import Agent, AgentRun, AgentStep

__all__ = [
    "Agent",
    "AgentRun",
    "AgentSession",
    "AgentStep",
    "ImageInfo",
    "LocalArtifactStore",
    "Tool",
    "ToolRegistry",
    "default_registry",
    "discover_tool_classes",
]


def __getattr__(name: str) -> Any:
    """Load the session type on demand.

    Importing it here would place ``atlas.agent.session`` in ``sys.modules``
    before ``python -m atlas.agent.session`` runs, which makes runpy warn and
    reuse the already-imported module.
    """
    if name == "AgentSession":
        from atlas.agent.session import AgentSession

        return AgentSession
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
