from __future__ import annotations

import json
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import (
    ChatMessage,
    Role,
    ToolCall,
    ToolCapableModel,
    ToolRegistry,
    ToolResult,
)

DEFAULT_MAX_TOOL_CALLS = 256
DEFAULT_MODEL = "google/gemini-3.8-flash"

_SYSTEM_PROMPT = """You are Atlas, a local workspace assistant.
Use tools when they help answer the request. All tool paths are relative to the
artifact workspace. Do not claim to have read a file unless a tool has returned
its contents."""

_STOPPED_RESPONSE = "Stopped after reaching the configured tool-use limit."

ErrorKind = Literal["validation", "unknown_tool", "tool_failure", "limit"]


class AgentStep(BaseModel):
    """One tool action recorded in an autonomous run."""

    call: ToolCall
    result: ToolResult | None = None
    error: str | None = None
    error_kind: ErrorKind | None = None


class AgentRun(BaseModel):
    """Final response plus a transparent record of every tool action."""

    response: str
    steps: list[AgentStep] = Field(default_factory=list)
    stopped_for_limit: bool = False


def _load_agent_config(root: Path) -> dict[str, object]:
    """Return ``.atlas/agent.toml`` under ``root``, or an empty mapping if absent.

    Malformed TOML raises ``ValueError`` (``tomllib.TOMLDecodeError``).
    """
    config_path = root / ".atlas" / "agent.toml"
    if not config_path.is_file():
        return {}
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("agent.toml must be a table")
    return data


def load_max_tool_calls(root: Path) -> int:
    """Read ``.atlas/agent.toml`` under ``root``; missing key means the default.

    A present key must be an int (not a bool) greater than or equal to 1.
    Anything else, including malformed TOML, raises ``ValueError``.
    """
    data = _load_agent_config(root)
    if "max_tool_calls" not in data:
        return DEFAULT_MAX_TOOL_CALLS
    value = data["max_tool_calls"]
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("max_tool_calls must be an integer >= 1")
    return value


def ensure_agent_config(root: Path) -> Path:
    """Create ``.atlas/agent.toml`` with defaults when the file is missing.

    An existing file is left unchanged, including a file that omits a key.
    """
    config_dir = root / ".atlas"
    config_path = config_dir / "agent.toml"
    if config_path.is_file():
        return config_path
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                "# OpenRouter model id. Change this to switch the agent TUI.",
                f'model = "{DEFAULT_MODEL}"',
                "",
                "# Tool-call budget for one turn. Must be an integer >= 1.",
                f"max_tool_calls = {DEFAULT_MAX_TOOL_CALLS}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def load_model(root: Path) -> str:
    """Read ``model`` from ``.atlas/agent.toml``; missing key means the default.

    A present key must be a non-empty string. The value is an OpenRouter model id.
    """
    data = _load_agent_config(root)
    if "model" not in data:
        return DEFAULT_MODEL
    value = data["model"]
    if not isinstance(value, str) or not value.strip():
        raise ValueError("model must be a non-empty string")
    return value.strip()


class Agent:
    """Run a tools-capable model against a fixed local artifact workspace.

    The agent spends a bounded budget of tool calls. Only tools registered on
    the supplied registry can be invoked; a model cannot invent a capability by
    naming it in a response.
    """

    def __init__(
        self,
        model: ToolCapableModel,
        tools: ToolRegistry,
        artifacts: LocalArtifactStore,
        *,
        max_tool_calls: int | None = None,
    ) -> None:
        if max_tool_calls is None:
            max_tool_calls = load_max_tool_calls(artifacts.root)
        elif isinstance(max_tool_calls, bool) or max_tool_calls < 1:
            raise ValueError("max_tool_calls must be an integer >= 1")
        self.model = model
        self.tools = tools
        self.artifacts = artifacts
        self.max_tool_calls = max_tool_calls

    def run(self, request: str) -> AgentRun:
        """Answer ``request``, making bounded, registry-approved tool calls."""
        messages = [
            ChatMessage(role=Role.SYSTEM, content=_SYSTEM_PROMPT),
            ChatMessage(role=Role.USER, content=request),
        ]
        return self.advance(messages)

    def advance(
        self,
        messages: list[ChatMessage],
        *,
        on_step: Callable[[AgentStep], None] | None = None,
    ) -> AgentRun:
        """Continue ``messages`` in place while the tool-call budget lasts.

        The caller owns the history: no user or system message is appended here.
        Every tool call that fits in the budget runs in order, even after one
        fails. A call past the budget is recorded as a ``limit`` step and still
        gets a tool message so every ``tool_call_id`` is answered.
        """
        steps: list[AgentStep] = []
        executed = 0
        while executed < self.max_tool_calls:
            response = self.model.complete(messages, self.tools.definitions)
            messages.append(
                ChatMessage(
                    role=Role.ASSISTANT,
                    content=response.content,
                    tool_calls=response.tool_calls,
                )
            )
            if not response.tool_calls:
                return AgentRun(response=response.content or "", steps=steps)
            hit_limit = False
            for call in response.tool_calls:
                if executed < self.max_tool_calls:
                    step = self._execute(call)
                    executed += 1
                else:
                    hit_limit = True
                    step = AgentStep(call=call, error="tool call limit reached", error_kind="limit")
                steps.append(step)
                messages.append(
                    ChatMessage(
                        role=Role.TOOL,
                        tool_call_id=call.id,
                        content=_tool_message(step),
                    )
                )
                if on_step is not None:
                    on_step(step)
            if executed >= self.max_tool_calls or hit_limit:
                return AgentRun(
                    response=_STOPPED_RESPONSE,
                    steps=steps,
                    stopped_for_limit=True,
                )
        return AgentRun(
            response=_STOPPED_RESPONSE,
            steps=steps,
            stopped_for_limit=True,
        )

    def _execute(self, call: ToolCall) -> AgentStep:
        try:
            result = self.tools.execute(call.name, call.arguments, self.artifacts)
        except Exception as exc:
            return AgentStep(call=call, error=f"{type(exc).__name__}: {exc}", error_kind=_kind(exc))
        return AgentStep(call=call, result=result)


def _kind(exc: Exception) -> ErrorKind:
    if isinstance(exc, KeyError):
        return "unknown_tool"
    if isinstance(exc, ValidationError):
        return "validation"
    return "tool_failure"


def _tool_message(step: AgentStep) -> str:
    if step.result is not None:
        return step.result.model_dump_json()
    return json.dumps({"error": step.error, "error_kind": step.error_kind})
