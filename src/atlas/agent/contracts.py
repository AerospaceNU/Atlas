"""Shared, model-neutral contracts for tool-using agents."""

from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field


class ToolDefinition(BaseModel):
    """A function schema advertised to a tool-capable model."""

    name: str
    description: str
    parameters: dict[str, Any]


class ToolCall(BaseModel):
    """One structured request returned by the model."""

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ChatMessage(BaseModel):
    """A portable chat-completions message."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None


class ModelResponse(BaseModel):
    """The subset of a model response needed by the runtime."""

    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)


class ToolCapableModel(Protocol):
    """Minimal model contract needed by :class:`atlas.agent.runtime.Agent`."""

    def complete(
        self, messages: list[ChatMessage], tools: list[ToolDefinition]
    ) -> ModelResponse: ...
