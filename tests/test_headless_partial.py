from __future__ import annotations

from pathlib import Path

import pytest

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import ChatMessage, ModelResponse, TokenUsage, ToolCall, ToolDefinition
from atlas.agent.runtime import AgentRun, ContextCompaction
from atlas.agent.trace import read_trace
from atlas.harness_eval.headless import HeadlessRun, _build_agent, run_headless
from atlas.harness_eval.rollout import _meta
from atlas.harness_eval.substrate import InstalledHarness


class CrashingModel:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages: list[ChatMessage], tools: list[ToolDefinition]) -> ModelResponse:
        self.calls += 1
        if self.calls == 1:
            return ModelResponse(
                tool_calls=[ToolCall(id="c1", name="read_file", arguments={"path": "a.txt"})],
                usage=TokenUsage(prompt_tokens=10, completion_tokens=2, total_tokens=12),
            )
        raise RuntimeError("provider down")


def test_crash_leaves_a_readable_partial_trace_and_propagates(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    trace = tmp_path / "out" / "trace.json"

    with pytest.raises(RuntimeError, match="provider down"):
        run_headless(
            workspace=tmp_path,
            prompt="go",
            model_id="",
            api_key="",
            model=CrashingModel(),
            trace_path=trace,
        )

    record = read_trace(trace)
    assert [step.call.name for step in record.run.steps] == ["read_file"]
    assert record.run.error is not None
    assert "provider down" in record.run.error


def test_meta_reports_empty_replies_and_compaction(tmp_path: Path) -> None:
    run = AgentRun(response="x", empty_replies=2, compaction=ContextCompaction(occurred=True))
    result = HeadlessRun(
        run=run, trace_path=tmp_path / "t.json", workspace=tmp_path, harness=InstalledHarness()
    )
    meta = _meta(result)
    assert meta["empty_replies"] == 2
    assert meta["compacted"] is True
    assert meta["status"] == "ok"


def test_headless_child_of_a_read_only_parent_is_read_only(tmp_path: Path) -> None:
    agent = _build_agent(
        store=LocalArtifactStore(tmp_path),
        model=CrashingModel(),
        harness=InstalledHarness(),
        max_tool_calls=5,
        read_only=True,
        approval="allow",
    )
    tool = agent.tools.get("spawn_agent")
    assert tool is not None
    runner = tool._make_agent(LocalArtifactStore(tmp_path), True, 3)  # type: ignore[attr-defined]
    assert runner._agent.read_only is True
