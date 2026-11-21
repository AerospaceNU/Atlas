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
from atlas.agent.runtime import Agent


class _NoArgs(BaseModel):
    pass


class PingTool(Tool):
    name = "ping"
    description = "Return pong."
    input_model = _NoArgs

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        return ToolResult(text="pong")


class ScriptedModel:
    def __init__(self, responses: list[ModelResponse | Exception]) -> None:
        self.responses = responses
        self.requests: list[list[ChatMessage]] = []

    def complete(self, messages: list[ChatMessage], tools: list[ToolDefinition]) -> ModelResponse:
        self.requests.append(list(messages))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _agent(tmp_path: Path, model: ScriptedModel, **kwargs: Any) -> Agent:
    registry = ToolRegistry()
    registry.register(PingTool())
    return Agent(model, registry, LocalArtifactStore(tmp_path), **kwargs)


def _ping(call_id: str) -> ModelResponse:
    return ModelResponse(
        tool_calls=[ToolCall(id=call_id, name="ping", arguments={})],
        usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


def test_an_empty_reply_is_nudged_and_the_run_continues(tmp_path: Path) -> None:
    model = ScriptedModel([ModelResponse(content=None), ModelResponse(content="done")])
    agent = _agent(tmp_path, model)
    messages = [ChatMessage(role=Role.USER, content="go")]

    run = agent.advance(messages)

    assert run.response == "done"
    assert run.empty_replies == 1
    nudge = messages[1]
    assert nudge.role is Role.USER and nudge.tool_observation
    assert "empty" in (nudge.content or "")
    assert [m.role for m in messages] == [Role.USER, Role.USER, Role.ASSISTANT]
    assert runtime._current_user_message(messages) is messages[0]


def test_a_whitespace_reply_counts_as_empty(tmp_path: Path) -> None:
    model = ScriptedModel([ModelResponse(content="  \n"), ModelResponse(content="ok")])
    run = _agent(tmp_path, model).advance([ChatMessage(role=Role.USER, content="go")])
    assert run.response == "ok"
    assert run.empty_replies == 1


def test_length_finish_reason_gets_its_own_wording(tmp_path: Path) -> None:
    model = ScriptedModel(
        [ModelResponse(content=None, finish_reason="length"), ModelResponse(content="ok")]
    )
    messages = [ChatMessage(role=Role.USER, content="go")]
    _agent(tmp_path, model).advance(messages)
    assert messages[1].content == (
        "Your last reply was cut off at the output limit before it contained any text or "
        "tool call. Continue the task in smaller steps: call a tool, or give your final answer."
    )


def test_third_empty_reply_ends_the_run(tmp_path: Path) -> None:
    model = ScriptedModel([ModelResponse(content=None) for _ in range(3)])
    messages = [ChatMessage(role=Role.USER, content="go")]

    run = _agent(tmp_path, model).advance(messages)

    assert run.response == ""
    assert run.empty_replies == 3
    assert len(model.requests) == 3
    assert not any(m.role is Role.ASSISTANT for m in messages)  # no empty assistant message


def test_a_nudge_does_not_spend_the_tool_budget(tmp_path: Path) -> None:
    model = ScriptedModel(
        [ModelResponse(content=None), ModelResponse(content=None), _ping("a"), _ping("b")]
    )
    run = _agent(tmp_path, model, max_tool_calls=2).advance(
        [ChatMessage(role=Role.USER, content="go")]
    )
    assert run.stopped_for_limit
    assert len(run.steps) == 2
    assert run.empty_replies == 2


def test_budget_stop_wins_over_a_nudge(tmp_path: Path) -> None:
    model = ScriptedModel(
        [
            ModelResponse(content=None, usage=TokenUsage(prompt_tokens=100, completion_tokens=1)),
            ModelResponse(content="never"),
        ]
    )
    run = _agent(tmp_path, model, max_tokens=50).advance(
        [ChatMessage(role=Role.USER, content="go")]
    )
    assert run.stopped_for_budget
    assert len(model.requests) == 1
    assert len(run.calls) == 1


def test_discarded_reply_still_records_usage_and_thought(tmp_path: Path) -> None:
    thoughts: list[str] = []
    model = ScriptedModel(
        [
            ModelResponse(
                content=None,
                reasoning="hmm",
                usage=TokenUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5),
            ),
            ModelResponse(content="ok"),
        ]
    )
    run = _agent(tmp_path, model).advance(
        [ChatMessage(role=Role.USER, content="go")],
        on_thought=lambda text, speed: thoughts.append(text),
    )
    assert thoughts == ["hmm"]
    assert len(run.calls) == 1


def test_model_exception_leaves_a_partial_run_and_propagates(tmp_path: Path) -> None:
    boom = RuntimeError("provider down")
    model = ScriptedModel([_ping("a"), boom, ModelResponse(content="fine")])
    agent = _agent(tmp_path, model)
    assert agent.partial_run is None

    with pytest.raises(RuntimeError) as caught:
        agent.advance([ChatMessage(role=Role.USER, content="go")])

    assert caught.value is boom
    partial = agent.partial_run
    assert partial is not None
    assert len(partial.steps) == 1
    assert len(partial.calls) == 1
    assert partial.error == "RuntimeError: provider down"
    assert partial.response == ""

    run = agent.advance([ChatMessage(role=Role.USER, content="again")])
    assert run.response == "fine"
    assert agent.partial_run is None


# --- history validity, empty replies, serialisation, redaction, checkpoints ---


def _tool_ids(messages: list[ChatMessage]) -> set[str | None]:
    return {m.tool_call_id for m in messages if m.role is Role.TOOL}


def _assert_every_call_answered(messages: list[ChatMessage]) -> None:
    called = {call.id for m in messages for call in m.tool_calls}
    assert called <= _tool_ids(messages)


def _two_pings() -> ModelResponse:
    return ModelResponse(
        tool_calls=[
            ToolCall(id="p1", name="ping", arguments={}),
            ToolCall(id="p2", name="ping", arguments={}),
        ],
        usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


def test_an_on_step_exception_leaves_every_tool_call_answered(tmp_path: Path) -> None:
    agent = _agent(tmp_path, ScriptedModel([_two_pings()]))
    messages = [ChatMessage(role=Role.USER, content="go")]

    def boom(step: runtime.AgentStep) -> None:
        raise RuntimeError("callback failed")

    with pytest.raises(RuntimeError, match="callback failed"):
        agent.advance(messages, on_step=boom)

    _assert_every_call_answered(messages)
    assert agent.partial_run is not None
    assert agent.partial_run.error == "RuntimeError: callback failed"
    second = next(m for m in messages if m.tool_call_id == "p2")
    assert "interrupted: RuntimeError" in (second.content or "")
    assert "cancelled" in (second.content or "")


def test_an_on_thought_exception_leaves_every_tool_call_answered(tmp_path: Path) -> None:
    response = _two_pings()
    response.reasoning = "thinking"
    agent = _agent(tmp_path, ScriptedModel([response]))
    messages = [ChatMessage(role=Role.USER, content="go")]

    def boom(text: str, speed: float | None) -> None:
        raise RuntimeError("thought callback failed")

    with pytest.raises(RuntimeError, match="thought callback failed"):
        agent.advance(messages, on_thought=boom)

    _assert_every_call_answered(messages)
    assert agent.partial_run is not None
    assert agent.partial_run.error == "RuntimeError: thought callback failed"


def test_a_keyboard_interrupt_in_a_tool_is_not_swallowed_and_history_stays_valid(
    tmp_path: Path,
) -> None:
    class Interrupting(PingTool):
        name = "ping"

        def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
            raise KeyboardInterrupt

    registry = ToolRegistry()
    registry.register(Interrupting())
    agent = Agent(ScriptedModel([_two_pings()]), registry, LocalArtifactStore(tmp_path))
    messages = [ChatMessage(role=Role.USER, content="go")]

    with pytest.raises(KeyboardInterrupt):
        agent.advance(messages)

    _assert_every_call_answered(messages)
    assert agent.partial_run is None  # only Exception subclasses are recorded by the round


def test_an_approver_that_raises_leaves_history_valid(tmp_path: Path) -> None:
    def approver(request: runtime.PermissionRequest) -> bool:
        raise ValueError("approver crashed")

    registry = ToolRegistry()
    from atlas.agent.contracts import default_registry

    registry = default_registry()
    response = ModelResponse(
        tool_calls=[
            ToolCall(id="w1", name="write_file", arguments={"path": "a.txt", "content": "x"}),
            ToolCall(id="w2", name="write_file", arguments={"path": "b.txt", "content": "y"}),
        ],
        usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )
    agent = Agent(
        ScriptedModel([response]),
        registry,
        LocalArtifactStore(tmp_path),
        approval="ask",
        on_permission=approver,
    )
    messages = [ChatMessage(role=Role.USER, content="go")]

    with pytest.raises(ValueError, match="approver crashed"):
        agent.advance(messages)

    _assert_every_call_answered(messages)
    assert agent.partial_run is not None


def test_a_base_exception_from_the_model_sets_partial_run(tmp_path: Path) -> None:
    class Dying:
        def complete(
            self, messages: list[ChatMessage], tools: list[ToolDefinition]
        ) -> ModelResponse:
            raise SystemExit(3)

    agent = Agent(Dying(), ToolRegistry(), LocalArtifactStore(tmp_path))  # type: ignore[arg-type]
    with pytest.raises(SystemExit):
        agent.advance([ChatMessage(role=Role.USER, content="go")])
    assert agent.partial_run is not None
    assert agent.partial_run.error is not None


def test_the_final_empty_reply_is_not_appended(tmp_path: Path) -> None:
    model = ScriptedModel([ModelResponse(content=None) for _ in range(3)])
    agent = _agent(tmp_path, model)
    messages = [ChatMessage(role=Role.USER, content="go")]

    run = agent.advance(messages)

    assert run.response == ""
    assert run.empty_replies == 3
    assert not any(m.role is Role.ASSISTANT for m in messages)


def _step(**kwargs: Any) -> runtime.AgentStep:
    return runtime.AgentStep(call=ToolCall(id="c1", name="ping", arguments={}), **kwargs)


def test_a_lone_surrogate_in_a_result_does_not_raise(tmp_path: Path) -> None:
    bad = "name-" + chr(0xDCFF)
    step = _step(result=ToolResult(text=f"listing {bad}", artifacts=[bad]))
    step.notes.append(bad)

    message = runtime._tool_message(step, LocalArtifactStore(tmp_path))

    visible = ToolResult.model_validate_json(message)
    assert "listing name-" in visible.text
    assert chr(0xDCFF) not in message
    assert step.result is not None and chr(0xDCFF) in step.result.text  # stored step untouched


def test_a_surrogate_in_an_error_is_scrubbed(tmp_path: Path) -> None:
    message = runtime._tool_message(_step(error="bad " + chr(0xDCFF), error_kind="tool_failure"))
    assert json.loads(message)["error"].startswith("bad ")


def test_a_message_that_cannot_be_built_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*args: Any, **kwargs: Any) -> str:
        raise RuntimeError("no")

    monkeypatch.setattr(runtime, "_render_tool_message", broken)
    message = runtime._tool_message(_step(result=ToolResult(text="x")))
    assert json.loads(message) == {
        "error": "tool result could not be serialised",
        "error_kind": "tool_failure",
    }


def test_an_oversized_error_is_cut_and_saved(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    error = "E" * (runtime.MAX_TOOL_RESULT_CHARS * 2) + "TAIL"
    step = _step(error=error, error_kind="tool_failure")

    payload = json.loads(runtime._tool_message(step, store))

    assert len(payload["error"]) < len(error)
    assert payload["error"].endswith("TAIL")
    assert "full output saved to" in payload["error"]
    assert step.error == error
    saved = list((tmp_path / ".atlas").rglob("*.txt"))
    assert len(saved) == 1 and saved[0].read_text(encoding="utf-8") == error


def test_saved_output_is_redacted_but_the_visible_cut_is_not_changed(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    secret = "sk-live-SECRET-123"  # gitleaks:allow (fake fixture)
    text = secret + " " + "x" * (runtime.MAX_TOOL_RESULT_CHARS * 2)
    step = _step(result=ToolResult(text=text))

    runtime._tool_message(step, store, lambda value: value.replace(secret, "[redacted]"))

    saved = list((tmp_path / ".atlas").rglob("*.txt"))
    assert len(saved) == 1
    body = saved[0].read_text(encoding="utf-8")
    assert secret not in body and body.startswith("[redacted]")


def test_a_failing_redactor_keeps_the_result_and_writes_nothing(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    text = "HEAD" + "x" * (runtime.MAX_TOOL_RESULT_CHARS * 2) + "TAIL"
    step = _step(result=ToolResult(text=text))

    def broken(value: str) -> str:
        raise RuntimeError("redactor failed")

    visible = ToolResult.model_validate_json(runtime._tool_message(step, store, broken))

    assert visible.text.startswith("HEAD") and visible.text.endswith("TAIL")
    assert "full output saved to" not in visible.text
    assert not list((tmp_path / ".atlas").rglob("*.txt"))


def _recorded_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, arguments: dict[str, Any]
) -> list[str]:
    from atlas.agent.contracts import default_registry

    taken: list[str] = []
    monkeypatch.setattr(runtime, "snapshot", lambda store, path: taken.append(path))
    agent = Agent(ScriptedModel([]), default_registry(), LocalArtifactStore(tmp_path))
    agent._checkpoint(ToolCall(id="a", name="apply_patch", arguments=arguments))
    return taken


_PATCH = "*** Begin Patch\n*** Update File:g.txt\n@@\n-a\n+b\n*** End Patch\n"


def test_a_string_false_dry_run_still_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    taken = _recorded_checkpoints(tmp_path, monkeypatch, {"patch": _PATCH, "dry_run": "false"})
    assert taken == ["g.txt"]


def test_a_true_dry_run_does_not_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _recorded_checkpoints(tmp_path, monkeypatch, {"patch": _PATCH, "dry_run": True}) == []
    assert _recorded_checkpoints(tmp_path, monkeypatch, {"patch": _PATCH, "dry_run": "true"}) == []


def test_envelope_headers_without_a_space_name_their_path() -> None:
    patch = "*** Update File:g.txt\n*** Add File:  new.txt\n"
    assert runtime._patch_paths({"patch": patch}) == ["g.txt", "new.txt"]


# --- consecutive empty cap and announce-and-stop nudge ---


def _empty() -> ModelResponse:
    return ModelResponse(content=None)


def _say(text: str) -> ModelResponse:
    return ModelResponse(content=text, finish_reason="stop")


def _go() -> list[ChatMessage]:
    return [ChatMessage(role=Role.USER, content="go")]


def test_non_consecutive_empty_replies_do_not_end_the_turn(tmp_path: Path) -> None:
    model = ScriptedModel(
        [_ping("a"), _empty(), _ping("b"), _empty(), _ping("c"), _empty(), _say("done")]
    )
    run = _agent(tmp_path, model).advance(_go())
    assert run.response == "done"
    assert run.empty_replies == 3
    assert len(run.steps) == 3


def test_three_consecutive_empty_replies_after_a_tool_round_still_end_it(tmp_path: Path) -> None:
    model = ScriptedModel([_ping("a"), _empty(), _empty(), _empty()])
    run = _agent(tmp_path, model).advance(_go())
    assert run.response == ""
    assert run.empty_replies == 3


def test_an_announced_next_step_gets_one_nudge_and_the_tool_call_runs(tmp_path: Path) -> None:
    model = ScriptedModel(
        [_ping("a"), _say("Let me check the existing widget tests."), _ping("b"), _say("Fixed.")]
    )
    messages = _go()
    run = _agent(tmp_path, model).advance(messages)
    assert run.response == "Fixed."
    assert run.intent_nudges == 1
    assert len(run.steps) == 2
    nudges = [m for m in messages if m.content == runtime._NUDGE_INTENT]
    assert len(nudges) == 1 and nudges[0].role is Role.USER and nudges[0].tool_observation
    assert any(m.content == "Let me check the existing widget tests." for m in messages)


def test_a_second_announcement_in_the_same_advance_is_final(tmp_path: Path) -> None:
    model = ScriptedModel([_ping("a"), _say("Let me look."), _say("Next, I will edit it.")])
    run = _agent(tmp_path, model).advance(_go())
    assert run.response == "Next, I will edit it."
    assert run.intent_nudges == 1


def test_an_announcement_without_a_prior_tool_call_is_final(tmp_path: Path) -> None:
    model = ScriptedModel([_say("I'll explain: ")])
    run = _agent(tmp_path, model).advance(_go())
    assert run.response == "I'll explain: "
    assert run.intent_nudges == 0
    assert len(model.requests) == 1


def test_final_answers_are_returned_unchanged(tmp_path: Path) -> None:
    long_reply = "Let me summarize " + "x" * 400
    for text in ("Fixed X; tests pass.", long_reply):
        model = ScriptedModel([_ping("a"), _say(text)])
        run = _agent(tmp_path, model).advance(_go())
        assert run.response == text
        assert run.intent_nudges == 0


def test_an_intent_nudge_does_not_spend_the_tool_budget(tmp_path: Path) -> None:
    model = ScriptedModel([_ping("a"), _say("Let me continue."), _ping("b"), _ping("c")])
    run = _agent(tmp_path, model, max_tool_calls=3).advance(_go())
    assert run.stopped_for_limit
    assert len(run.steps) == 3
    assert run.intent_nudges == 1


def test_announces_next_step_detector() -> None:
    assert runtime._announces_next_step("Now I will run the tests")
    assert not runtime._announces_next_step("The files are:")
    assert not runtime._announces_next_step("")
    assert not runtime._announces_next_step("All done.")
