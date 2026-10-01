from __future__ import annotations

from pathlib import Path

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import (
    ChatMessage,
    ModelResponse,
    ToolCall,
    ToolDefinition,
    default_registry,
)
from atlas.agent.runtime import Agent


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


def test_agent_records_a_sandbox_denial_in_the_audit_log(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
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
    agent = Agent(model, default_registry(), store)

    result = agent.run("read a file outside the workspace")

    assert result.steps[0].error_kind == "tool_failure"
    audit_log = tmp_path / ".atlas" / "sandbox_audit.log"
    assert audit_log.is_file()
    contents = audit_log.read_text(encoding="utf-8")
    assert "DENIED read_file" in contents
    assert "escapes" in contents


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
