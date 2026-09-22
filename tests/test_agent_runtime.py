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
