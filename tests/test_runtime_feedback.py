"""Loop feedback: unknown tools, validation text, notes, and truncation with a saved copy."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from atlas.agent import runtime
from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import (
    ChatMessage,
    ModelResponse,
    Role,
    Tool,
    ToolCall,
    ToolDefinition,
    ToolRegistry,
    ToolResult,
)
from atlas.agent.runtime import Agent, AgentStep, _tool_message


class _Scripted:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = responses

    def complete(self, messages: list[ChatMessage], tools: list[ToolDefinition]) -> ModelResponse:
        return self.responses.pop(0)


class _In(BaseModel):
    value: int = Field(ge=0)
    label: str = "x"


class _Echo(Tool):
    name = "echo"
    description = "Echo."
    runs = 0
    text = "same"

    @property
    def input_model(self) -> type[BaseModel]:
        return _In

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        type(self).runs += 1
        request = _In.model_validate(arguments)
        return ToolResult(text=type(self).text.format(request.value))


class _Boom(Tool):
    name = "boom"
    description = "Raises KeyError."

    @property
    def input_model(self) -> type[BaseModel]:
        return _In

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        raise KeyError("inner")


class _Bash(_Echo):
    name = "bash"


def _agent(tmp_path: Path, responses: list[ModelResponse], **kwargs: Any) -> Agent:
    registry = ToolRegistry()
    for tool in (_Echo(), _Boom(), _Bash()):
        registry.register(tool)
    return Agent(_Scripted(responses), registry, LocalArtifactStore(tmp_path), **kwargs)


def _call(i: int, name: str = "echo", **arguments: Any) -> ModelResponse:
    return ModelResponse(tool_calls=[ToolCall(id=f"c{i}", name=name, arguments=arguments)])


def test_unknown_tool_hint_and_nothing_runs(tmp_path: Path) -> None:
    _Echo.runs = 0
    agent = _agent(tmp_path, [_call(1, "bash_tool"), ModelResponse(content="ok")])
    run = agent.run("go")
    step = run.steps[0]
    assert step.error_kind == "unknown_tool"
    assert step.error is not None
    assert "Did you mean 'bash'?" in step.error
    assert "Available tools: bash, boom, echo." in step.error
    assert _Echo.runs == 0


def test_unknown_tool_without_close_match_omits_hint(tmp_path: Path) -> None:
    agent = _agent(tmp_path, [_call(1, "zzzzzzzz"), ModelResponse(content="ok")])
    error = agent.run("go").steps[0].error
    assert error is not None
    assert "Did you mean" not in error


def test_key_error_inside_a_tool_is_tool_failure(tmp_path: Path) -> None:
    agent = _agent(tmp_path, [_call(1, "boom", value=1), ModelResponse(content="ok")])
    assert agent.run("go").steps[0].error_kind == "tool_failure"


def test_validation_text_lists_expected_arguments(tmp_path: Path) -> None:
    agent = _agent(tmp_path, [_call(1, value=-5), ModelResponse(content="ok")])
    step = agent.run("go").steps[0]
    assert step.error_kind == "validation"
    assert step.error is not None
    assert "errors.pydantic.dev" not in step.error
    assert "input_value" not in step.error
    assert step.error.startswith("Invalid arguments for echo:")
    assert "- value:" in step.error
    assert "Expected arguments: value (required), label (default 'x')" in step.error
    assert len(step.error) <= 1500


def test_argument_errors_become_validation_steps_without_running(tmp_path: Path) -> None:
    _Echo.runs = 0
    response = _call(1, value=1)
    response.argument_errors = {"c1": "Invalid JSON."}
    agent = _agent(tmp_path, [response, ModelResponse(content="ok")])
    run = agent.run("go")
    step = run.steps[0]
    assert step.error_kind == "validation"
    assert step.error is not None
    assert step.error.startswith("Invalid arguments for echo: Invalid JSON. Resend")
    assert _Echo.runs == 0


def test_unknown_argument_note(tmp_path: Path) -> None:
    agent = _agent(tmp_path, [_call(1, value=1, offset=3), ModelResponse(content="ok")])
    step = agent.run("go").steps[0]
    assert step.notes == ["ignored unknown argument(s): offset; valid arguments: value, label"]
    assert step.result is not None
    assert step.result.text == "same"


def test_repeated_identical_call_note_on_third_result(tmp_path: Path) -> None:
    _Echo.text = "same"
    responses = [_call(i, value=1) for i in range(4)] + [ModelResponse(content="ok")]
    steps = _agent(tmp_path, responses).run("go").steps
    assert [len(step.notes) for step in steps] == [0, 0, 1, 1]
    assert "returned the same result 3 times" in steps[2].notes[0]
    assert steps[2].result is not None
    assert steps[2].result.text == "same"


def test_changing_results_reset_the_repeat_count(tmp_path: Path) -> None:
    class _Counter(_Echo):
        name = "counter"
        n = 0

        def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
            type(self).n += 1
            return ToolResult(text=str(type(self).n))

    registry = ToolRegistry()
    registry.register(_Counter())
    responses = [_call(i, "counter", value=1) for i in range(4)] + [ModelResponse(content="ok")]
    run = Agent(_Scripted(responses), registry, LocalArtifactStore(tmp_path)).run("go")
    assert all(step.notes == [] for step in run.steps)


def test_budget_note_at_threshold(tmp_path: Path) -> None:
    responses = [_call(i, value=i) for i in range(20)]
    agent = _agent(tmp_path, responses, max_tool_calls=20)
    run = agent.run("go")
    noted = [i for i, step in enumerate(run.steps) if step.notes]
    assert noted == [17]
    assert run.steps[17].notes[0].startswith("tool budget: 18 of 20 calls used this turn, 2 left.")
    assert run.stopped_for_limit


def test_no_budget_note_for_small_budgets(tmp_path: Path) -> None:
    responses = [_call(i, value=i) for i in range(5)]
    run = _agent(tmp_path, responses, max_tool_calls=5).run("go")
    assert all(not step.notes for step in run.steps)


def test_notes_reach_the_model_but_not_the_result() -> None:
    step = AgentStep(
        call=ToolCall(id="a", name="echo", arguments={}),
        result=ToolResult(text="body"),
        notes=["hello"],
    )
    visible = ToolResult.model_validate_json(_tool_message(step))
    assert visible.text == "body\n\n[note: hello]"
    assert step.result is not None
    assert step.result.text == "body"
    failed = AgentStep(call=step.call, error="bad", error_kind="validation", notes=["n"])
    assert json.loads(_tool_message(failed))["error"] == "bad\n\n[note: n]"


def _big() -> str:
    return "FIRST\n" + "m" * 30_000 + "\nLAST"


def test_truncation_keeps_head_tail_and_saves_a_copy(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    text = _big()
    step = AgentStep(
        call=ToolCall(id="call/1:x", name="echo", arguments={}),
        result=ToolResult(text=text, artifacts=["a.tif"]),
        notes=["tail note"],
    )
    visible = ToolResult.model_validate_json(_tool_message(step, store)).text
    assert visible.startswith("FIRST")
    assert "LAST\n\n[note: tail note]" in visible
    assert (
        f"[truncated {len(text) - runtime.MAX_TOOL_RESULT_CHARS} characters; full output saved to"
        in visible
    )
    saved = (
        tmp_path
        / ".atlas"
        / "tool-output"
        / f"call1x-{hashlib.sha256(text.encode()).hexdigest()[:12]}.txt"
    )
    assert saved.read_text(encoding="utf-8") == text
    assert f".atlas/tool-output/{saved.name}" in visible
    assert step.result is not None
    assert step.result.text == text


def test_truncation_uses_artifact_root_when_separate(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "work", artifact_root=tmp_path / "arts")
    step = AgentStep(call=ToolCall(id="z", name="e", arguments={}), result=ToolResult(text=_big()))
    _tool_message(step, store)
    assert list((tmp_path / "arts" / "tool-output").glob("z-*.txt"))


def test_unwritable_directory_still_truncates(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    (tmp_path / ".atlas").write_text("a file, not a directory", encoding="utf-8")
    step = AgentStep(call=ToolCall(id="z", name="e", arguments={}), result=ToolResult(text=_big()))
    visible = ToolResult.model_validate_json(_tool_message(step, store)).text
    assert "characters]..." in visible
    assert "saved to" not in visible
    assert runtime._spill_output(store, "z", "x") is None


def test_spill_name_falls_back_to_a_hash(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    path = runtime._spill_output(store, "!!!", "body")
    assert path is not None
    assert (tmp_path / path).read_text(encoding="utf-8") == "body"


def test_truncated_text_compacts_and_is_idempotent(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    step = AgentStep(call=ToolCall(id="k", name="e", arguments={}), result=ToolResult(text=_big()))
    visible = ToolResult.model_validate_json(_tool_message(step, store)).text
    once = runtime._compact_text(visible)
    assert len(once) < 700
    assert "characters compacted; full output saved to .atlas/tool-output/k-" in once
    assert runtime._compact_text(once) == once


def test_advance_passes_the_store_to_truncation(tmp_path: Path) -> None:
    _Echo.text = "y" * 25_000
    try:
        agent = _agent(tmp_path, [_call(1, value=1), ModelResponse(content="ok")])
        messages = [*agent.opening_messages(), ChatMessage(role=Role.USER, content="go")]
        agent.advance(messages)
    finally:
        _Echo.text = "same"
    tool_message = next(m for m in messages if m.role is Role.TOOL)
    assert "full output saved to" in (tool_message.content or "")


def test_reused_call_id_does_not_overwrite_an_earlier_copy(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    texts = ["AAA" + "x" * 30_000, "BBB" + "y" * 30_000]
    visibles = []
    for text in texts:
        step = AgentStep(
            call=ToolCall(id="call_0", name="echo", arguments={}),
            result=ToolResult(text=text),
        )
        visibles.append(ToolResult.model_validate_json(_tool_message(step, store)).text)
    files = sorted((tmp_path / ".atlas" / "tool-output").glob("call_0-*.txt"))
    assert len(files) == 2
    assert {f.read_text(encoding="utf-8") for f in files} == set(texts)
    assert visibles[0] != visibles[1]
