"""Delegate one bounded task to a child agent.

A subagent gets a fresh message history, its own tool-call budget, and its own
artifact directory. It runs the same registry as its parent, minus this tool,
so delegation cannot recurse: there is no handler for a nested
``spawn_agent`` call.

The child reports its final answer as the tool result. Its steps are not added
to the parent transcript, which is the point: the parent keeps one summary
instead of every intermediate tool call.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import Tool, ToolResult


class SubagentInput(BaseModel):
    """Arguments for ``spawn_agent``."""

    task: str = Field(
        min_length=1,
        max_length=8000,
        description="Self-contained instruction. The child cannot see this conversation.",
    )
    allow_writes: bool = Field(
        default=False,
        description="Let the child use write_file, edit_file, patch_file, and bash.",
    )
    max_tool_calls: int = Field(
        default=40,
        ge=1,
        le=200,
        description="Tool-call budget for the child turn.",
    )


class SubagentTool(Tool):
    """Run one task in a child agent and return its final answer."""

    name = "spawn_agent"
    description = (
        "Delegate one self-contained task to a child agent with a fresh context. "
        "The child sees only the task text and the workspace, not this conversation. "
        "It returns a short report. Use it for independent exploration or a scoped edit; "
        "writes are refused unless allow_writes is true."
    )
    trust: ClassVar[Literal["default", "opt_in"]] = "default"
    max_report_chars: ClassVar[int] = 8000

    def __init__(self, make_agent: Callable[[LocalArtifactStore, bool, int], Any]) -> None:
        """Build the tool.

        Args:
            make_agent: Factory taking the child store, whether writes are
                allowed, and the tool-call budget. It returns an object with a
                ``run(task)`` method, normally :class:`atlas.agent.runtime.Agent`.
        """
        self._make_agent = make_agent

    @property
    def input_model(self) -> type[BaseModel]:
        return SubagentInput

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        request = SubagentInput.model_validate(arguments)
        child_id = uuid.uuid4().hex[:8]
        artifact_root = store.artifact_root / "subagents" / child_id
        child_store = LocalArtifactStore(store.root, artifact_root=artifact_root)
        agent = self._make_agent(child_store, request.allow_writes, request.max_tool_calls)
        try:
            run = agent.run(request.task)
        except Exception as exc:
            return ToolResult(text=f"subagent failed: {type(exc).__name__}: {exc}")
        steps = len(run.steps)
        report = run.response.strip() or "(no answer)"
        if len(report) > self.max_report_chars:
            report = report[: self.max_report_chars] + "\n... [truncated]"
        return ToolResult(
            text=f"subagent {child_id} ({steps} tool calls, writes="
            f"{'allowed' if request.allow_writes else 'refused'}):\n{report}",
        )
