from __future__ import annotations

from typing import Any

from atlas.agent.contracts import ChatMessage, Role, ToolDefinition
from atlas.agent.model import ModelConfig, OpenRouterModel


class FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self.payload


class FakeClient:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def post(self, _url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append(kwargs)
        return FakeResponse(self.response)

    def close(self) -> None:
        return None


def test_complete_advertises_and_parses_native_tool_calls() -> None:
    client = FakeClient(
        {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "tool-1",
                                "function": {
                                    "name": "inspect_image",
                                    "arguments": '{"path":"coast.png"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }
    )
    model = OpenRouterModel(ModelConfig(model="example/model", api_key="test"), client)
    tool = ToolDefinition(
        name="inspect_image",
        description="Inspect a local image.",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}},
    )

    response = model.complete([ChatMessage(role=Role.USER, content="Inspect coast.png")], [tool])

    assert response.content is None
    assert response.tool_calls[0].arguments == {"path": "coast.png"}
    advertised_tool = client.calls[0]["json"]["tools"][0]["function"]
    assert advertised_tool["name"] == "inspect_image"
    assert advertised_tool["parameters"] == tool.parameters
