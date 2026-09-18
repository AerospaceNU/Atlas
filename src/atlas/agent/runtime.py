from __future__ import annotations

import json

from pydantic import BaseModel, Field

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import ChatMessage, ToolCall, ToolCapableModel
from atlas.agent.tools import ToolRegistry, ToolResult

_SYSTEM_PROMPT = """You are Atlas, a local image-workspace assistant.
Use tools when they help answer the request. All tool paths are relative to the
artifact workspace. Do not claim to have inspected an image unless a tool has
returned information about it. Tool-created artifacts are local workspace files.
Scripts staged for review are never executable tools; explain that distinction."""


class AgentStep(BaseModel):
    """One tool action recorded in an autonomous run."""

    call: ToolCall
    result: ToolResult | None = None
    error: str | None = None


class AgentRun(BaseModel):
    """Final response plus a transparent record of every tool action."""

    response: str
    steps: list[AgentStep] = Field(default_factory=list)
    stopped_for_limit: bool = False


class Agent:
    """Run a tools-capable model against a fixed local artifact workspace.

    The agent has a strict maximum number of tool rounds. Only tools registered
    on the supplied registry can be invoked; a model cannot invent a capability
    by naming it in a response.
    """

    def __init__(
        self,
        model: ToolCapableModel,
        tools: ToolRegistry,
        artifacts: LocalArtifactStore,
        *,
        max_tool_rounds: int = 8,
    ) -> None:
        if max_tool_rounds < 1:
            raise ValueError("max_tool_rounds must be at least 1")
        self.model = model
        self.tools = tools
        self.artifacts = artifacts
        self.max_tool_rounds = max_tool_rounds

    def run(self, request: str) -> AgentRun:
        """Answer ``request``, making bounded, registry-approved tool calls."""
        messages = [
            ChatMessage(role="system", content=_SYSTEM_PROMPT),
            ChatMessage(role="user", content=request),
        ]
        steps: list[AgentStep] = []
        for _ in range(self.max_tool_rounds):
            response = self.model.complete(messages, self.tools.definitions)
            messages.append(
                ChatMessage(
                    role="assistant", content=response.content, tool_calls=response.tool_calls
                )
            )
            if not response.tool_calls:
                return AgentRun(response=response.content or "", steps=steps)
            for call in response.tool_calls:
                step = self._execute(call)
                steps.append(step)
                messages.append(
                    ChatMessage(
                        role="tool",
                        tool_call_id=call.id,
                        content=_tool_message(step),
                    )
                )
        return AgentRun(
            response="Stopped after reaching the configured tool-use limit.",
            steps=steps,
            stopped_for_limit=True,
        )

    def _execute(self, call: ToolCall) -> AgentStep:
        try:
            result = self.tools.execute(call.name, call.arguments, self.artifacts)
        except Exception as exc:
            return AgentStep(call=call, error=f"{type(exc).__name__}: {exc}")
        return AgentStep(call=call, result=result)


def _tool_message(step: AgentStep) -> str:
    if step.result is not None:
        return step.result.model_dump_json()
    return json.dumps({"error": step.error})
