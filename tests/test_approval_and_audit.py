from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import (
    ChatMessage,
    ModelResponse,
    ToolCall,
    ToolDefinition,
    ToolRegistry,
)
from atlas.agent.runtime import Agent, PermissionRequest
from atlas.agent.tools.read_file import ReadFileTool
from atlas.agent.tools.write_file import WriteTool


class _ScriptedModel:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = responses
        self.calls = 0

    def complete(self, messages: list[ChatMessage], tools: list[ToolDefinition]) -> ModelResponse:
        self.calls += 1
        return self.responses.pop(0)


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(ReadFileTool())
    registry.register(WriteTool())
    return registry


def _write_call(path: str, content: str, call_id: str = "a") -> ModelResponse:
    return ModelResponse(
        tool_calls=[
            ToolCall(id=path, name="write_file", arguments={"path": path, "content": content})
        ]
    )


def _run(
    tmp_path: Path,
    responses: list[ModelResponse],
    *,
    approval: str = "allow",
    approver=None,
) -> tuple[Agent, Any]:
    model = _ScriptedModel(responses)
    agent = Agent(
        model,
        _registry(),
        LocalArtifactStore(tmp_path),
        approval=approval,
        on_permission=approver,
    )
    return agent, agent.run("go")


def test_ask_mode_allows_after_approval(tmp_path: Path) -> None:
    seen: list[PermissionRequest] = []

    def approve(request: PermissionRequest) -> bool:
        seen.append(request)
        return True

    _run(
        tmp_path,
        [_write_call("app.py", "print(1)\n"), ModelResponse(content="done")],
        approval="ask",
        approver=approve,
    )

    assert (tmp_path / "app.py").read_text() == "print(1)\n"
    assert seen and seen[0].tool == "write_file"
    assert "+print(1)" in seen[0].preview


def test_ask_mode_denies_and_writes_nothing(tmp_path: Path) -> None:
    _, run = _run(
        tmp_path,
        [_write_call("app.py", "print(1)\n"), ModelResponse(content="done")],
        approval="ask",
        approver=lambda request: False,
    )

    assert not (tmp_path / "app.py").exists()
    assert run.steps[0].error_kind == "denied"


def test_ask_without_an_approver_fails_closed(tmp_path: Path) -> None:
    _, run = _run(
        tmp_path,
        [_write_call("app.py", "x"), ModelResponse(content="done")],
        approval="ask",
    )

    assert not (tmp_path / "app.py").exists()
    assert "no approver" in (run.steps[0].error or "")


def test_mutating_calls_in_one_turn_run_serially(tmp_path: Path) -> None:
    model = _ScriptedModel(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="a", name="write_file", arguments={"path": "a.py", "content": "1\n"}
                    ),
                    ToolCall(
                        id="b", name="write_file", arguments={"path": "a.py", "content": "2\n"}
                    ),
                ]
            ),
            ModelResponse(content="done"),
        ]
    )
    agent = Agent(model, _registry(), LocalArtifactStore(tmp_path))

    run = agent.run("go")

    assert [step.call.id for step in run.steps] == ["a", "b"]
    assert (tmp_path / "a.py").read_text() == "2\n"


def test_audit_log_records_tools_and_models(tmp_path: Path) -> None:
    _run(
        tmp_path,
        [_write_call("app.py", "x\n"), ModelResponse(content="done")],
    )

    lines = (tmp_path / ".atlas" / "audit.log").read_text().splitlines()
    records = [json.loads(line) for line in lines]
    tools = [item for item in records if item["event"] == "tool"]
    assert tools and tools[0]["tool"] == "write_file" and tools[0]["outcome"] == "ok"
    assert "duration_ms" in tools[0]
    assert "app.py" not in json.dumps(records)


def test_read_only_refuses_even_with_an_approver(tmp_path: Path) -> None:
    model = _ScriptedModel([_write_call("app.py", "x"), ModelResponse(content="done")])
    agent = Agent(
        model,
        _registry(),
        LocalArtifactStore(tmp_path),
        read_only=True,
        approval="ask",
        on_permission=lambda request: True,
    )

    run = agent.run("go")

    assert run.steps[0].error_kind == "denied"
    assert not (tmp_path / "app.py").exists()
