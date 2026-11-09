"""End-to-end headless runs: no terminal, no network, one trace record each.

``test_agent_runtime.py`` covers a single tool call. These tests run the loop
the way a benchmark would: drive it non-interactively, then keep the whole run
as a versioned record and read it back. They also run under the ``_forbid_network``
guard in ``conftest.py``, so a smoke run that dialled out would fail.
"""

from __future__ import annotations

from pathlib import Path

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import (
    ChatMessage,
    ModelResponse,
    TokenUsage,
    ToolCall,
    ToolDefinition,
    ToolRegistry,
    default_registry,
)
from atlas.agent.runtime import _STOPPED_RESPONSE, Agent
from atlas.agent.trace import read_trace, write_trace


class ScriptedModel:
    """Return canned responses in order and remember every request."""

    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = responses
        self.requests: list[tuple[list[ChatMessage], list[ToolDefinition]]] = []

    def complete(self, messages: list[ChatMessage], tools: list[ToolDefinition]) -> ModelResponse:
        self.requests.append((list(messages), list(tools)))
        return self.responses.pop(0)


def _usage(prompt: int, completion: int) -> TokenUsage:
    return TokenUsage(
        prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion
    )


def _read(path: str, call_id: str) -> ModelResponse:
    return ModelResponse(
        tool_calls=[ToolCall(id=call_id, name="read_file", arguments={"path": path})],
        usage=_usage(100, 20),
    )


def test_a_multi_step_run_completes_and_survives_a_trace_round_trip(tmp_path: Path) -> None:
    (tmp_path / "first.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "second.txt").write_text("beta", encoding="utf-8")
    model = ScriptedModel(
        [
            _read("first.txt", "call-1"),
            _read("second.txt", "call-2"),
            ModelResponse(content="Read both files.", usage=_usage(200, 10)),
        ]
    )
    agent = Agent(model, default_registry(), LocalArtifactStore(tmp_path))

    run = agent.run("Read both files.")

    assert run.response == "Read both files."
    assert [step.call.name for step in run.steps] == ["read_file", "read_file"]
    assert [step.error for step in run.steps] == [None, None]
    assert "alpha" in (run.steps[0].result.text if run.steps[0].result else "")
    assert "beta" in (run.steps[1].result.text if run.steps[1].result else "")
    assert not run.stopped_for_limit

    # The record a benchmark would keep must describe the run it just watched.
    path = write_trace(tmp_path / "runs" / "smoke" / "attempt-0.json", run)
    assert read_trace(path).run == run


def test_a_multi_step_run_reports_usage_a_benchmark_can_total(tmp_path: Path) -> None:
    (tmp_path / "first.txt").write_text("alpha", encoding="utf-8")
    model = ScriptedModel(
        [
            _read("first.txt", "call-1"),
            ModelResponse(content="Done.", usage=_usage(200, 10)),
        ]
    )
    agent = Agent(model, default_registry(), LocalArtifactStore(tmp_path))

    run = agent.run("Read it.")

    assert len(run.calls) == 2
    assert sum(call.usage.total_tokens for call in run.calls) == 330


def test_hitting_the_tool_budget_stops_without_inventing_an_answer(tmp_path: Path) -> None:
    (tmp_path / "first.txt").write_text("alpha", encoding="utf-8")
    model = ScriptedModel(
        [
            _read("first.txt", "call-1"),
            _read("first.txt", "call-2"),
            _read("first.txt", "call-3"),
        ]
    )
    agent = Agent(model, default_registry(), LocalArtifactStore(tmp_path), max_tool_calls=1)

    run = agent.run("Keep reading.")

    assert run.stopped_for_limit
    assert run.response == _STOPPED_RESPONSE
    # One call is spent, then the loop stops without consulting the model again.
    assert len(run.calls) == 1
    assert len(model.requests) == 1
    assert len(model.responses) == 2, "the loop asked again after the budget was spent"
    assert read_trace(write_trace(tmp_path / "run.json", run)).run == run


def test_an_unknown_tool_is_recorded_in_the_trace_rather_than_run(tmp_path: Path) -> None:
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=[ToolCall(id="call-1", name="no_such_tool", arguments={})],
                usage=_usage(50, 5),
            ),
            ModelResponse(content="I could not do that.", usage=_usage(60, 5)),
        ]
    )
    agent = Agent(model, default_registry(), LocalArtifactStore(tmp_path))

    run = agent.run("Try a tool that does not exist.")

    assert run.steps[0].error is not None
    assert run.steps[0].error_kind == "unknown_tool"
    assert run.steps[0].result is None
    restored = read_trace(write_trace(tmp_path / "run.json", run)).run
    assert restored.steps[0].error_kind == "unknown_tool"


def test_an_empty_registry_offers_no_tools_but_still_answers(tmp_path: Path) -> None:
    """A session with no tools is a valid headless run, not a crash."""
    model = ScriptedModel([ModelResponse(content="No tools needed.", usage=_usage(10, 5))])
    agent = Agent(model, ToolRegistry(), LocalArtifactStore(tmp_path))

    run = agent.run("Say hello.")

    assert run.response == "No tools needed."
    assert run.steps == []
    assert model.requests[0][1] == []


def test_a_missing_file_does_not_end_the_turn(tmp_path: Path) -> None:
    model = ScriptedModel(
        [
            _read("missing.txt", "call-1"),
            ModelResponse(content="The file was not there.", usage=_usage(30, 5)),
        ]
    )
    agent = Agent(model, default_registry(), LocalArtifactStore(tmp_path))

    run = agent.run("Read missing.txt")

    assert run.steps[0].error is not None
    assert run.steps[0].error_kind == "tool_failure"
    assert run.response == "The file was not there."
    assert not run.cancelled


def test_a_parent_relative_path_is_read_rather_than_refused(tmp_path: Path) -> None:
    """There is no sandbox: `..` is accepted, so this must not be a refusal.

    AGENTS.md and docs/adr/0001 record the removal of path containment. A test
    that expected an escape error here would be asserting a guarantee the
    project deliberately dropped.
    """
    root = tmp_path / "workspace"
    root.mkdir()
    (tmp_path / "outside.txt").write_text("reachable", encoding="utf-8")
    model = ScriptedModel(
        [
            _read("../outside.txt", "call-1"),
            ModelResponse(content="Read it.", usage=_usage(30, 5)),
        ]
    )
    agent = Agent(model, default_registry(), LocalArtifactStore(root))

    run = agent.run("Read ../outside.txt")

    assert run.steps[0].error is None, run.steps[0].error
    assert "reachable" in (run.steps[0].result.text if run.steps[0].result else "")
