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
    original = '"line\n' * 1000
    content = ToolResult(text=original, artifacts=["a.tif", "b.json"]).model_dump_json()
    visible = ToolResult.model_validate_json(runtime._compact_tool_content(content))
    head = runtime.COMPACT_PREVIEW_CHARS // 2
    tail = runtime.COMPACT_PREVIEW_CHARS - head
    assert visible.text.startswith(original[:head])
    assert visible.text.endswith(original[-tail:])
    assert "characters compacted" in visible.text
    assert visible.artifacts == ["a.tif", "b.json"]
    assert runtime._compact_tool_content(visible.model_dump_json()) == visible.model_dump_json()

    error = json.loads(
        runtime._compact_tool_content(
            json.dumps({"error": "bad" * 1000, "error_kind": "validation"})
        )
    )
    assert error["error_kind"] == "validation"
    assert "characters compacted" in error["error"]
    assert "characters compacted" in runtime._compact_tool_content("legacy error " * 1000)


def test_compact_text_is_idempotent_and_keeps_both_ends() -> None:
    original = "HEAD" + "x" * 5000 + "TAIL"

    once = runtime._compact_text(original)

    assert once.startswith("HEAD")
    assert once.endswith("TAIL")
    assert "characters compacted" in once
    assert runtime._compact_text(once) == once


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


def _exchange(index: int) -> list[ChatMessage]:
    """One assistant tool call plus a large observation, sized to force drops."""
    call = ToolCall(
        id=f"c{index}",
        name="run_tests",
        arguments={"command": ["pytest", "-q"]},
    )
    result = ToolResult(
        text=f"command: pytest -q\nexit_code: 1\n{'x' * runtime.MAX_TOOL_RESULT_CHARS}",
        artifacts=[f"out{index}.txt"],
    )
    return [
        ChatMessage(role=Role.ASSISTANT, tool_calls=[call]),
        ChatMessage(role=Role.TOOL, tool_call_id=call.id, content=result.model_dump_json()),
    ]


def _digest_text(messages: list[ChatMessage]) -> str:
    digests = [message for message in messages if runtime._is_digest(message)]
    assert len(digests) == 1
    return digests[0].content or ""


def test_compaction_replaces_dropped_exchanges_with_a_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runtime, "MAX_CONTEXT_TOKENS", 6000)
    plan_dir = tmp_path / ".atlas"
    plan_dir.mkdir()
    (plan_dir / "plan.json").write_text(
        json.dumps({"items": [{"text": "ship tools", "status": "in_progress"}]}),
        encoding="utf-8",
    )
    messages = [
        ChatMessage(role=Role.SYSTEM, content="system instructions"),
        ChatMessage(role=Role.USER, content="do the work"),
    ]
    for index in range(80):
        messages.extend(_exchange(index))
    messages.append(ChatMessage(role=Role.USER, content="now summarise"))

    result = _compact(messages, root=tmp_path)

    assert result.occurred
    assert result.digest_inserted
    assert result.messages_dropped > 0
    body = _digest_text(messages)
    assert "messages compacted" in body
    assert "pytest -q -> exit 1" in body
    assert "out0.txt" in body
    assert "Files:" in body
    assert "[~] ship tools" in body
    _assert_paired(messages)


def test_digest_rolls_forward_instead_of_stacking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runtime, "MAX_CONTEXT_TOKENS", 6000)
    messages = [
        ChatMessage(role=Role.SYSTEM, content="system instructions"),
        ChatMessage(role=Role.USER, content="do the work"),
    ]
    for index in range(60):
        messages.extend(_exchange(index))

    _compact(messages)
    first = _digest_text(messages)
    first_count = int(first.split(": ", 1)[1].split(" ", 1)[0])
    assert first_count > 0

    for index in range(60, 120):
        messages.extend(_exchange(index))
    messages.append(ChatMessage(role=Role.USER, content="continue"))

    _compact(messages)

    second = _digest_text(messages)
    second_count = int(second.split(": ", 1)[1].split(" ", 1)[0])
    assert second_count > first_count
    _assert_paired(messages)


def test_default_trigger_elides_a_history_sized_for_the_constant() -> None:
    messages = _history(300)
    assert _estimate_tokens(messages) > runtime.MAX_CONTEXT_TOKENS

    result = _compact(messages)

    assert result.occurred
    assert result.tool_results_shortened > 0
    assert _estimate_tokens(messages) <= runtime.MAX_CONTEXT_TOKENS
    _assert_paired(messages)


def test_digest_is_bounded_and_keeps_protected_messages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runtime, "MAX_CONTEXT_TOKENS", 6000)
    messages = [
        ChatMessage(role=Role.SYSTEM, content="system instructions"),
        ChatMessage(role=Role.USER, content="original request"),
    ]
    for index in range(120):
        exchange_result = _exchange(index)
        exchange_result[1] = exchange_result[1].model_copy(
            update={
                "content": ToolResult(
                    text=f"path/file_{index}.py was edited many times\n{'y' * 4000}",
                    artifacts=[f"artifact_{index}.txt"],
                ).model_dump_json()
            }
        )
        messages.extend(exchange_result)
    current = ChatMessage(role=Role.USER, content="keep this request")
    messages.append(current)

    _compact(messages, protected_user=current)

    body = _digest_text(messages)
    assert len(body) <= runtime._DIGEST_LIMIT_CHARS
    assert messages[0].content == "system instructions"
    assert any(message is current for message in messages)
    assert messages[-1] is current
    _assert_paired(messages)
