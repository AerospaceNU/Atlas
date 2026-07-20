from __future__ import annotations

from pathlib import Path

from PIL import Image

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import ChatMessage, ModelResponse, ToolCall, ToolDefinition
from atlas.agent.runtime import Agent
from atlas.agent.tools import default_registry


class ScriptedModel:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = responses
        self.requests: list[tuple[list[ChatMessage], list[ToolDefinition]]] = []

    def complete(self, messages: list[ChatMessage], tools: list[ToolDefinition]) -> ModelResponse:
        self.requests.append((list(messages), list(tools)))
        return self.responses.pop(0)


def test_agent_uses_registry_tool_then_returns_answer(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    Image.new("RGB", (20, 10), "blue").save(tmp_path / "coast.png")
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(id="call-1", name="inspect_image", arguments={"path": "coast.png"})
                ]
            ),
            ModelResponse(content="The image is 20 by 10 pixels."),
        ]
    )
    agent = Agent(model, default_registry(), store)

    result = agent.run("What are the image dimensions?")

    assert result.response == "The image is 20 by 10 pixels."
    assert result.steps[0].result is not None
    assert "20x10" in result.steps[0].result.text
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
