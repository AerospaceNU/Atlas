from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from atlas.agent import runtime
from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import (
    ChatMessage,
    ModelResponse,
    Role,
    TokenUsage,
    Tool,
    ToolCall,
    ToolDefinition,
    ToolRegistry,
    ToolResult,
)
from atlas.agent.runtime import Agent, AgentStep, _compact, _estimate_tokens, _tool_message


class ScriptedModel:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = responses
        self.requests: list[list[ChatMessage]] = []

    def complete(self, messages: list[ChatMessage], tools: list[ToolDefinition]) -> ModelResponse:
        self.requests.append(list(messages))
        return self.responses.pop(0)


def _call(name: str) -> ToolCall:
    return ToolCall(id=name, name="large_result", arguments={})


def _history(count: int = 42) -> list[ChatMessage]:
    messages = [
        ChatMessage(role=Role.SYSTEM, content="system instructions"),
        ChatMessage(role=Role.USER, content="project guidance"),
        ChatMessage(role=Role.USER, content="original request"),
    ]
    for index in range(count):
        call = _call(str(index))
        messages.extend(
            [
                ChatMessage(role=Role.ASSISTANT, tool_calls=[call]),
                ChatMessage(
                    role=Role.TOOL,
                    tool_call_id=call.id,
                    content=ToolResult(
                        text="x" * runtime.MAX_TOOL_RESULT_CHARS, artifacts=[f"{index}.txt"]
                    ).model_dump_json(),
                ),
            ]
        )
    return messages


def _assert_paired(messages: list[ChatMessage]) -> None:
    calls = [call.id for message in messages for call in message.tool_calls]
    results = [message.tool_call_id for message in messages if message.role is Role.TOOL]
    assert sorted(calls) == sorted(results)
    assert len(calls) == len(set(calls))


def test_ingest_truncates_only_model_text_and_preserves_original_result() -> None:
    original = ToolResult(text="x" * (runtime.MAX_TOOL_RESULT_CHARS + 123), artifacts=["map.tif"])
    step = AgentStep(call=_call("one"), result=original)

    visible = ToolResult.model_validate_json(_tool_message(step))

    assert visible.text == "x" * runtime.MAX_TOOL_RESULT_CHARS + "...[truncated 123 characters]"
    assert visible.artifacts == ["map.tif"]
    assert step.result == original
    assert len(original.text) == runtime.MAX_TOOL_RESULT_CHARS + 123


def test_clearing_preserves_escaped_json_and_error_metadata() -> None:
    content = ToolResult(text='"line\n' * 1000, artifacts=["a.tif", "b.json"]).model_dump_json()
    visible = ToolResult.model_validate_json(runtime._compact_tool_content(content))
    assert visible.text.startswith(('"line\n' * 1000)[: runtime.COMPACT_PREVIEW_CHARS])
    assert visible.artifacts == ["a.tif", "b.json"]
    assert runtime._compact_tool_content(visible.model_dump_json()) == visible.model_dump_json()

    error = json.loads(
        runtime._compact_tool_content(
            json.dumps({"error": "bad" * 1000, "error_kind": "validation"})
        )
    )
    assert error["error_kind"] == "validation"
    assert error["error"].endswith("characters]")
    assert runtime._compact_tool_content("legacy error " * 1000).endswith("characters]")


def test_parallel_tool_batch_crossing_recent_boundary_is_preserved_whole(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime, "MAX_CONTEXT_TOKENS", 8000)
    calls = [_call(name) for name in ("a", "b", "c")]
    batch = [ChatMessage(role=Role.ASSISTANT, tool_calls=calls)] + [
        ChatMessage(
            role=Role.TOOL,
            tool_call_id=call.id,
            content=ToolResult(text="x" * 2000).model_dump_json(),
        )
        for call in reversed(calls)
    ]
    messages = [
        ChatMessage(role=Role.SYSTEM, content="instructions"),
        ChatMessage(role=Role.USER, content="request"),
        ChatMessage(role=Role.ASSISTANT, content="old" * 20_000),
        *batch,
        *[ChatMessage(role=Role.ASSISTANT, content="recent") for _ in range(4)],
        ChatMessage(role=Role.USER, content="continue"),
    ]

    result = _compact(messages)

    assert result.messages_dropped == 1
    assert all(message in messages for message in batch)
    _assert_paired(messages)
    assert _estimate_tokens(messages) <= 8000


def test_current_request_is_kept_after_many_tool_rounds() -> None:
    messages = _history(50)
    current = ChatMessage(role=Role.USER, content="Keep this current request.")
    messages.insert(20, current)

    _compact(messages, protected_user=current)

    assert any(message is current for message in messages)
    _assert_paired(messages)


class LargeResultInput(BaseModel):
    pass


class LargeResultTool(Tool):
    name = "large_result"
    description = "Return a large offline observation."
    input_model = LargeResultInput

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        LargeResultInput.model_validate(arguments)
        return ToolResult(text="x" * 30_000, artifacts=["observation.txt"])


def test_agent_does_not_send_irreducible_over_budget_context(tmp_path: Path) -> None:
    model = ScriptedModel([ModelResponse(content="must not be called")])
    agent = Agent(model, ToolRegistry(), LocalArtifactStore(tmp_path))
    messages = [
        ChatMessage(role=Role.SYSTEM, content="Required instructions"),
        ChatMessage(role=Role.USER, content="x" * (runtime.MAX_CONTEXT_TOKENS * 4 + 1000)),
    ]

    run = agent.advance(messages)

    assert model.requests == []
    assert run.stopped_for_context
    assert "Shorten the request" in run.response
    assert not run.compaction.occurred


def test_budget_stop_answers_pending_calls_before_the_next_turn(tmp_path: Path) -> None:
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=[_call("a"), _call("b")],
                usage=TokenUsage(prompt_tokens=100, completion_tokens=1),
            ),
            ModelResponse(content="next turn"),
        ]
    )
    agent = Agent(model, ToolRegistry(), LocalArtifactStore(tmp_path), max_tokens=50)
    messages = [ChatMessage(role=Role.USER, content="go")]

    stopped = agent.advance(messages)
    _assert_paired(messages)
    assert stopped.stopped_for_budget
    assert [step.error_kind for step in stopped.steps] == ["budget", "budget"]
    assert all(step.result is None for step in stopped.steps)

    messages.append(ChatMessage(role=Role.USER, content="Continue with a new request."))
    assert agent.advance(messages).response == "next turn"
    _assert_paired(model.requests[-1])


def test_tool_images_are_marked_as_observations_for_resume(tmp_path: Path) -> None:
    from PIL import Image

    Image.new("RGB", (2, 2)).save(tmp_path / "image.png")

    class ImageTool(LargeResultTool):
        def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
            return ToolResult(text="image", artifacts=["image.png"])

    registry = ToolRegistry()
    registry.register(ImageTool())
    model = ScriptedModel([ModelResponse(tool_calls=[_call("image")])])
    agent = Agent(model, registry, LocalArtifactStore(tmp_path), max_tool_calls=1)
    messages = [ChatMessage(role=Role.USER, content="Inspect the image.")]

    run = agent.advance(messages)

    assert run.stopped_for_limit
    assert messages[-1].tool_observation
    assert messages[-1].images
    assert runtime._current_user_message(messages) is messages[0]
    assert runtime._context_groups(messages)[-1] == range(1, 4)
