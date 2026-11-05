from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import (
    ChatMessage,
    ModelResponse,
    TokenUsage,
    Tool,
    ToolCall,
    ToolDefinition,
    ToolRegistry,
    ToolResult,
)
from atlas.agent.model import ModelConfig, OpenRouterHTTPError, OpenRouterModel
from atlas.agent.runtime import Agent


class _EchoInput(BaseModel):
    value: int = 0


class _EchoTool(Tool):
    name = "echo"
    description = "Echo one value."
    trust = "default"

    @property
    def input_model(self) -> type[BaseModel]:
        return _EchoInput

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        request = _EchoInput.model_validate(arguments)
        return ToolResult(text=f"echo {request.value}")


class _SlowTool(Tool):
    name = "slow"
    description = "Sleep briefly, then answer."

    trust = "default"
    delay = 0.2
    active = 0
    peak = 0

    @property
    def input_model(self) -> type[BaseModel]:
        return _EchoInput

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        type(self).active += 1
        type(self).active_max = max(getattr(type(self), "active_max", 0), type(self).active)
        time.sleep(self.delay)
        type(self).active -= 1
        return ToolResult(text="slept")


class _ScriptedModel:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = responses
        self.requests: list[list[ChatMessage]] = []

    def complete(self, messages: list[ChatMessage], tools: list[ToolDefinition]) -> ModelResponse:
        self.requests.append(list(messages))
        return self.responses.pop(0)


def _calls(*names: str) -> ModelResponse:
    return ModelResponse(
        tool_calls=[
            ToolCall(id=f"call-{index}", name=name, arguments={"value": index})
            for index, name in enumerate(names)
        ]
    )


def _registry(*tools: Tool) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


def test_parallel_tool_calls_run_concurrently(tmp_path: Path) -> None:
    _SlowTool.active_max = 0
    model = _ScriptedModel([_calls("slow", "slow", "slow"), ModelResponse(content="done")])
    agent = Agent(model, _registry(_SlowTool()), LocalArtifactStore(tmp_path))

    run = agent.run("go")

    assert len(run.steps) == 3
    assert all(step.result is not None and step.result.text == "slept" for step in run.steps)
    assert [step.call.id for step in run.steps] == ["call-0", "call-1", "call-2"]
    assert _SlowTool.active_max >= 2


def test_token_budget_stops_the_turn(tmp_path: Path) -> None:
    model = _ScriptedModel(
        [
            ModelResponse(
                content="",
                tool_calls=[ToolCall(id="a", name="echo", arguments={"value": 1})],
                usage=TokenUsage(prompt_tokens=80, completion_tokens=40, total_tokens=120),
            ),
            ModelResponse(content="should not be requested"),
        ]
    )
    agent = Agent(
        model,
        _registry(_EchoTool()),
        LocalArtifactStore(tmp_path),
        max_tokens=100,
    )

    run = agent.run("go")

    assert run.stopped_for_budget is True
    assert "budget" in run.response
    assert len(model.requests) == 1


def test_spend_budget_stops_the_turn(tmp_path: Path) -> None:
    model = _ScriptedModel(
        [
            ModelResponse(
                tool_calls=[ToolCall(id="a", name="echo", arguments={"value": 1})],
                usage=TokenUsage(prompt_tokens=1, completion_tokens=1, cost=0.5),
            ),
            ModelResponse(content="should not be requested"),
        ]
    )
    agent = Agent(
        model,
        _registry(_EchoTool()),
        LocalArtifactStore(tmp_path),
        max_spend_usd=0.25,
    )

    run = agent.run("go")

    assert run.stopped_for_budget is True
    assert len(model.requests) == 1


def test_cancel_before_a_tool_batch_answers_every_call(tmp_path: Path) -> None:
    model = _ScriptedModel([_calls("echo", "echo"), ModelResponse(content="never")])
    agent = Agent(model, _registry(_EchoTool()), LocalArtifactStore(tmp_path))
    messages = [ChatMessage(role="user", content="go")]

    run = agent.advance(messages, should_cancel=lambda: len(model.requests) >= 1)

    assert run.cancelled is True
    tool_messages = [message for message in messages if message.role == "tool"]
    assert sorted(message.tool_call_id or "" for message in tool_messages) == ["call-0", "call-1"]
    assert all(message.content and "cancel" in message.content for message in tool_messages)


class _FailThenSucceedClient:
    """One 429 response, then a valid completion."""

    def __init__(self) -> None:
        self.calls = 0

    def post(self, _url: str, **_kwargs: Any) -> Any:
        self.calls += 1
        if self.calls == 1:
            return httpx.Response(429, text="slow down", request=httpx.Request("POST", _url))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "recovered", "tool_calls": []}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
            request=httpx.Request("POST", _url),
        )


def test_rate_limit_is_retried_and_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("atlas.agent.model.time.sleep", sleeps.append)
    client = _FailThenSucceedClient()
    model = OpenRouterModel(
        ModelConfig(model="vendor/model", api_key="test-key"),
        client=client,  # type: ignore[arg-type]
    )

    response = model.complete([ChatMessage(role="user", content="hi")], [])

    assert response.content == "recovered"
    assert client.calls == 2
    assert sleeps == [0.5]


class _AlwaysFailsClient:
    def __init__(self, status: int) -> None:
        self.status = status
        self.calls = 0

    def post(self, _url: str, **_kwargs: Any) -> Any:
        self.calls += 1
        return httpx.Response(self.status, text="nope", request=httpx.Request("POST", _url))


def test_a_bad_request_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("atlas.agent.model.time.sleep", sleeps.append)
    client = _AlwaysFailsClient(400)
    model = OpenRouterModel(
        ModelConfig(model="vendor/model", api_key="test-key"),
        client=client,  # type: ignore[arg-type]
    )

    with pytest.raises(OpenRouterHTTPError):
        model.complete([ChatMessage(role="user", content="hi")], [])

    assert client.calls == 1
    assert sleeps == []


def test_rate_limit_gives_up_after_the_attempt_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("atlas.agent.model.time.sleep", sleeps.append)
    client = _AlwaysFailsClient(503)
    model = OpenRouterModel(
        ModelConfig(model="vendor/model", api_key="test-key"),
        client=client,  # type: ignore[arg-type]
    )

    with pytest.raises(OpenRouterHTTPError):
        model.complete([ChatMessage(role="user", content="hi")], [])

    assert client.calls == 3
    assert sleeps == [0.5, 1.0]
