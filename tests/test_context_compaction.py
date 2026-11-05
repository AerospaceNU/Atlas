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
from atlas.agent.session import AgentSession


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


def test_tool_clearing_keeps_valid_json_artifacts_ids_and_recent_context() -> None:
    messages = _history()
    originals = list(messages)
    before = _estimate_tokens(messages)

    result = _compact(messages)

    assert before > runtime.MAX_CONTEXT_TOKENS
    assert _estimate_tokens(messages) <= int(runtime.MAX_CONTEXT_TOKENS * 0.8)
    assert result.occurred
    assert result.tool_results_shortened > 0
    assert result.messages_dropped == 0
    assert result.tokens_reclaimed == before - _estimate_tokens(messages)
    assert messages[:3] == originals[:3]
    assert messages[-runtime.KEEP_RECENT_MESSAGES :] == originals[-runtime.KEEP_RECENT_MESSAGES :]
    for message in messages:
        if message.role is Role.TOOL:
            visible = ToolResult.model_validate_json(message.content or "")
            assert visible.artifacts == [f"{message.tool_call_id}.txt"]
    assert all(
        len(ToolResult.model_validate_json(message.content or "").text)
        == runtime.MAX_TOOL_RESULT_CHARS
        for message in originals
        if message.role is Role.TOOL
    )
    _assert_paired(messages)

    # A normal result fits in the reclaimed headroom without another compaction.
    messages.extend(_history(1)[3:])
    assert _estimate_tokens(messages) < runtime.MAX_CONTEXT_TOKENS


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


def test_second_layer_drops_non_tool_history_but_keeps_instructions_and_current_request() -> None:
    messages = _history(0)
    opening = list(messages)
    for _ in range(35):
        messages.extend(
            [
                ChatMessage(role=Role.USER, content="u" * 15_000),
                ChatMessage(role=Role.ASSISTANT, content="a" * 15_000),
            ]
        )
    current = ChatMessage(role=Role.USER, content="Use the new dates.")
    messages.append(current)
    recent = messages[-runtime.KEEP_RECENT_MESSAGES :]

    result = _compact(messages)

    assert result.messages_dropped > 0
    assert result.tool_results_shortened == 0
    assert _estimate_tokens(messages) <= int(runtime.MAX_CONTEXT_TOKENS * 0.8)
    assert messages[:3] == opening
    assert messages[-runtime.KEEP_RECENT_MESSAGES :] == recent
    assert any(message is current for message in messages)


def test_second_layer_drops_tool_batches_whole() -> None:
    messages = _history(60)
    # Large artifact lists cannot be shortened with the observation text.
    for message in messages:
        if message.role is Role.TOOL:
            message.content = ToolResult(
                text="short", artifacts=["data/" + "p" * 1000] * 20
            ).model_dump_json()

    result = _compact(messages)

    assert result.messages_dropped > 0
    assert result.messages_dropped % 2 == 0
    assert _estimate_tokens(messages) <= int(runtime.MAX_CONTEXT_TOKENS * 0.8)
    _assert_paired(messages)


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


def test_scripted_agent_compacts_before_model_call_and_retains_full_run_results(
    tmp_path: Path,
) -> None:
    model = ScriptedModel(
        [ModelResponse(tool_calls=[_call(str(index))]) for index in range(42)]
        + [ModelResponse(content="done")]
    )
    registry = ToolRegistry()
    registry.register(LargeResultTool())
    agent = Agent(model, registry, LocalArtifactStore(tmp_path))

    run = agent.run("Inspect the observations.")

    assert run.response == "done"
    assert run.compaction.occurred
    assert run.compaction.tokens_reclaimed > 0
    assert run.compaction.tool_results_shortened > 0
    assert all(
        _estimate_tokens(request) <= runtime.MAX_CONTEXT_TOKENS for request in model.requests
    )
    assert all(step.result and len(step.result.text) == 30_000 for step in run.steps)
    visible = [message for message in model.requests[-1] if message.role is Role.TOOL]
    assert any("[compacted" in (message.content or "") for message in visible)
    assert all(
        ToolResult.model_validate_json(message.content or "").artifacts == ["observation.txt"]
        for message in visible
    )
    _assert_paired(model.requests[-1])
    assert run.model_dump()["compaction"]["occurred"] is True


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


def test_resumed_compaction_keeps_request_before_synthetic_image_observations(
    tmp_path: Path,
) -> None:
    messages = _history(60)
    current = ChatMessage(role=Role.USER, content="Keep the user's real request.")
    messages.insert(19, current)
    for message in messages:
        if message.role is Role.TOOL:
            message.content = ToolResult(
                text="short", artifacts=["a" * 1000] * 20
            ).model_dump_json()
    observation = ChatMessage(role=Role.USER, content=None, tool_observation=True)
    messages.append(observation)
    # Session persistence strips image bytes but retains the observation marker.
    messages = [ChatMessage.model_validate(message.model_dump()) for message in messages]
    model = ScriptedModel([ModelResponse(content="done")])
    agent = Agent(model, ToolRegistry(), LocalArtifactStore(tmp_path))
    session = AgentSession(agent)
    session.messages = messages
    session.stopped_for_limit = True

    run = session.resume()

    assert run.compaction.messages_dropped > 0
    assert any(message.content == current.content for message in model.requests[0])
    assert messages[-2].tool_observation
    _assert_paired(model.requests[0])


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
