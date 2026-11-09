"""The trace record must survive a round trip and announce its own schema."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from atlas.agent.contracts import CallTiming, ModelCall, TokenUsage, ToolCall, ToolResult
from atlas.agent.runtime import AgentRun, AgentStep, ContextCompaction
from atlas.agent.trace import (
    TRACE_SCHEMA,
    TraceSchemaError,
    dump_trace,
    load_trace,
    read_trace,
    write_trace,
)


def _populated_run() -> AgentRun:
    """A run with every field away from its default.

    A benchmark cannot trust a record whose fields silently fell back to
    defaults, so ``test_the_fixture_populates_every_field`` keeps this honest
    when ``AgentRun`` or ``AgentStep`` grows a field.
    """
    return AgentRun(
        response="Finished after two steps.\nSecond line.",
        steps=[
            AgentStep(
                call=ToolCall(id="call-1", name="read_file", arguments={"path": "a.txt"}),
                result=ToolResult(text="hello\nworld", artifacts=["a.txt", "b.json"]),
            ),
            AgentStep(
                call=ToolCall(id="call-2", name="bash_tool", arguments={"command": "ls"}),
                error="command not found",
                error_kind="tool_failure",
            ),
        ],
        stopped_for_limit=True,
        stopped_for_budget=True,
        stopped_for_context=True,
        cancelled=True,
        calls=[
            ModelCall(
                usage=TokenUsage(
                    prompt_tokens=11,
                    completion_tokens=22,
                    total_tokens=33,
                    cached_tokens=44,
                    reasoning_tokens=55,
                    cost=0.125,
                ),
                timing=CallTiming(
                    http_status=200,
                    elapsed_seconds=1.5,
                    time_to_first_token_seconds=0.25,
                ),
            ),
            ModelCall(usage=TokenUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3)),
        ],
        compaction=ContextCompaction(
            occurred=True,
            tokens_reclaimed=99,
            tool_results_shortened=3,
            messages_dropped=4,
        ),
    )


def test_a_fully_populated_run_survives_a_round_trip() -> None:
    run = _populated_run()
    restored = load_trace(dump_trace(run))
    assert restored.run == run
    assert restored.run.steps[1].error_kind == "tool_failure"
    assert restored.run.calls[0].timing == run.calls[0].timing
    assert restored.run.compaction.messages_dropped == 4


def test_the_fixture_populates_every_field() -> None:
    """Fail when a new AgentRun or AgentStep field is left out of the fixture."""
    run = _populated_run()
    # ``response`` is the only required field, so an empty one leaves every
    # other field at its default for the comparison below.
    defaults = AgentRun(response="")
    missing_run = [
        name for name in AgentRun.model_fields if getattr(run, name) == getattr(defaults, name)
    ]
    assert not missing_run, f"populate these AgentRun fields: {missing_run}"

    base = AgentStep(call=ToolCall(id="base", name="base"))
    touched = {
        name
        for step in run.steps
        for name in AgentStep.model_fields
        if getattr(step, name) != getattr(base, name)
    }
    assert touched == set(AgentStep.model_fields), (
        f"unexercised AgentStep fields: {sorted(set(AgentStep.model_fields) - touched)}"
    )


def test_a_record_is_one_line_so_a_jsonl_store_can_hold_it() -> None:
    """Newlines inside a payload stay escaped; a trace line stays a line."""
    assert "\n" not in dump_trace(_populated_run())


def test_a_record_announces_its_schema() -> None:
    assert json.loads(dump_trace(_populated_run()))["schema_version"] == TRACE_SCHEMA


def test_an_unreadable_schema_is_refused_not_guessed() -> None:
    record = json.loads(dump_trace(_populated_run()))
    record["schema_version"] = TRACE_SCHEMA + 1
    with pytest.raises(TraceSchemaError, match="not readable by this build"):
        load_trace(json.dumps(record))


def test_a_record_without_a_schema_is_refused() -> None:
    record = json.loads(dump_trace(_populated_run()))
    del record["schema_version"]
    with pytest.raises(TraceSchemaError, match="expected"):
        load_trace(json.dumps(record))


def test_malformed_input_raises_the_same_error_type() -> None:
    with pytest.raises(TraceSchemaError, match="not JSON"):
        load_trace("{not json")
    with pytest.raises(TraceSchemaError, match="JSON object"):
        load_trace("[1, 2]")


def test_write_then_read_creates_parent_directories(tmp_path: Path) -> None:
    path = write_trace(tmp_path / "runs" / "task-1" / "attempt-0.json", _populated_run())
    assert path.exists()
    assert read_trace(path).run == _populated_run()
