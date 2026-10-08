from __future__ import annotations

import os
from pathlib import Path

from atlas.agent.artifacts import LocalArtifactStore, record_denial
from atlas.agent.contracts import (
    ChatMessage,
    ModelResponse,
    ToolCall,
    ToolDefinition,
    default_registry,
)
from atlas.agent.runtime import Agent, AgentRun


class ScriptedModel:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = responses
        self.requests: list[tuple[list[ChatMessage], list[ToolDefinition]]] = []

    def complete(self, messages: list[ChatMessage], tools: list[ToolDefinition]) -> ModelResponse:
        self.requests.append((list(messages), list(tools)))
        return self.responses.pop(0)


def test_agent_uses_registry_tool_then_returns_answer(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    (tmp_path / "notes.txt").write_text("the coast is clear", encoding="utf-8")
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(id="call-1", name="read_file", arguments={"path": "notes.txt"})
                ]
            ),
            ModelResponse(content="The notes say the coast is clear."),
        ]
    )
    agent = Agent(model, default_registry(), store)

    result = agent.run("What do the notes say?")

    assert result.response == "The notes say the coast is clear."
    assert result.steps[0].result is not None
    assert "the coast is clear" in result.steps[0].result.text
    assert model.requests[1][0][-1].role == "tool"


def test_agent_emits_reasoning_before_tools_and_echoes_it(tmp_path: Path) -> None:
    details = [{"type": "reasoning.text", "text": "read the notes"}]
    model = ScriptedModel(
        [
            ModelResponse(
                reasoning="read the notes",
                reasoning_details=details,
                tool_calls=[
                    ToolCall(id="call-1", name="read_file", arguments={"path": "notes.txt"})
                ],
            ),
            ModelResponse(reasoning="enough", content="done"),
        ]
    )
    store = LocalArtifactStore(tmp_path)
    (tmp_path / "notes.txt").write_text("hi", encoding="utf-8")
    heard: list[tuple[str, float | None]] = []

    def note(text: str, speed: float | None) -> None:
        heard.append((text, speed))

    agent = Agent(model, default_registry(), store)

    result = agent.advance(
        [ChatMessage(role="user", content="go")],
        on_thought=note,
    )

    assert heard == [("read the notes", None), ("enough", None)]
    assert result.response == "done"
    assistant = model.requests[1][0][-2]
    assert assistant.role == "assistant"
    assert assistant.reasoning == "read the notes"
    assert assistant.reasoning_details == details


def _run_denied_read(store: LocalArtifactStore) -> AgentRun:
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(id="call-1", name="read_file", arguments={"path": "../outside.txt"})
                ]
            ),
            ModelResponse(content="I could not read that file."),
        ]
    )
    return Agent(model, default_registry(), store).run("read a file outside the workspace")


def test_agent_records_a_sandbox_denial_in_the_audit_log(tmp_path: Path) -> None:
    result = _run_denied_read(LocalArtifactStore(tmp_path))

    assert result.steps[0].error_kind == "tool_failure"
    audit_log = tmp_path / ".atlas" / "sandbox_audit.log"
    assert audit_log.is_file()
    contents = audit_log.read_text(encoding="utf-8")
    assert "DENIED read_file" in contents
    assert "escapes" in contents


def test_audit_log_keeps_each_denial_on_one_line(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)

    record_denial(store, "read_file", "big\n2026-01-01T00:00:00+00:00 DENIED forged: record")

    lines = (tmp_path / ".atlas" / "sandbox_audit.log").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert "big\\n2026" in lines[0]


def test_a_failed_audit_write_does_not_fail_the_turn(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    (tmp_path / ".atlas").write_text("a file where the audit directory should be", encoding="utf-8")

    result = _run_denied_read(store)

    assert result.response == "I could not read that file."
    assert result.steps[0].error_kind == "tool_failure"


def test_audit_log_does_not_follow_a_symlinked_audit_directory(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    store = LocalArtifactStore(workspace)
    (workspace / ".atlas").symlink_to(outside_dir)

    result = _run_denied_read(store)

    assert result.response == "I could not read that file."
    assert list(outside_dir.iterdir()) == []


def test_audit_log_does_not_follow_a_symlinked_log_file(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside.txt"
    outside.write_text("", encoding="utf-8")
    store = LocalArtifactStore(workspace)
    (workspace / ".atlas").mkdir()
    (workspace / ".atlas" / "sandbox_audit.log").symlink_to(outside)

    result = _run_denied_read(store)

    assert result.response == "I could not read that file."
    assert outside.read_text(encoding="utf-8") == ""


def test_audit_log_refuses_a_hardlinked_log_file(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside.txt"
    outside.write_text("", encoding="utf-8")
    store = LocalArtifactStore(workspace)
    (workspace / ".atlas").mkdir()
    os.link(outside, workspace / ".atlas" / "sandbox_audit.log")

    result = _run_denied_read(store)

    assert result.response == "I could not read that file."
    assert outside.read_text(encoding="utf-8") == ""


def test_agent_reports_unknown_tool_without_executing_it(tmp_path: Path) -> None:
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=[ToolCall(id="call-1", name="delete_everything", arguments={})]
            ),
            ModelResponse(content="I could not use that tool."),
        ]
    )
    agent = Agent(model, default_registry(), LocalArtifactStore(tmp_path))

    result = agent.run("Remove files")

    assert result.steps[0].error == "KeyError: 'Unknown tool: delete_everything'"
    assert result.response == "I could not use that tool."
