from __future__ import annotations

from typing import Any

import pytest

from atlas.agent.contracts import ChatMessage, Role, ToolDefinition
from atlas.agent.model import (
    MISSING_KEY_MESSAGE,
    MissingOpenRouterKey,
    ModelConfig,
    OpenRouterHTTPError,
    OpenRouterModel,
    fetch_model_catalog,
    filter_model_catalog,
    parse_model_catalog,
)


class FakeResponse:
    def __init__(
        self,
        payload: dict[str, Any],
        *,
        status_code: int = 200,
        elapsed_seconds: float = 0.0,
        time_to_first_token_seconds: float | None = None,
        text: str = "",
    ) -> None:
        self.payload = payload
        self.status_code = status_code
        self.elapsed_seconds = elapsed_seconds
        self.time_to_first_token_seconds = time_to_first_token_seconds
        self.text = text

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self.payload


class FakeClient:
    def __init__(
        self,
        response: dict[str, Any],
        *,
        status_code: int = 200,
        elapsed_seconds: float = 0.0,
        time_to_first_token_seconds: float | None = None,
    ) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []
        self.status_code = status_code
        self.elapsed_seconds = elapsed_seconds
        self.time_to_first_token_seconds = time_to_first_token_seconds

    def post(self, _url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append(kwargs)
        return FakeResponse(
            self.response,
            status_code=self.status_code,
            elapsed_seconds=self.elapsed_seconds,
            time_to_first_token_seconds=self.time_to_first_token_seconds,
        )

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


def test_complete_keeps_usage_cost_status_and_elapsed_time() -> None:
    client = FakeClient(
        {
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 4,
                "total_tokens": 16,
                "cost": 0.02,
            },
            "choices": [{"message": {"content": "clear coast", "tool_calls": []}}],
        },
        status_code=200,
        elapsed_seconds=0.5,
        time_to_first_token_seconds=0.1,
    )
    model = OpenRouterModel(ModelConfig(model="example/model", api_key="test-key"), client)

    response = model.complete([ChatMessage(role=Role.USER, content="look")], [])

    assert response.content == "clear coast"
    assert response.usage.prompt_tokens == 12
    assert response.usage.completion_tokens == 4
    assert response.usage.total_tokens == 16
    assert response.usage.cost == 0.02
    assert response.timing is not None
    assert response.timing.http_status == 200
    assert response.timing.elapsed_seconds == 0.5
    assert response.timing.time_to_first_token_seconds == 0.1
    assert client.calls[0]["headers"]["Authorization"] == "Bearer test-key"


def test_complete_keeps_non_2xx_status_and_a_short_body() -> None:
    body = "x" * 400
    client = FakeClient(
        {"error": {"message": body}},
        status_code=429,
        elapsed_seconds=0.25,
    )
    model = OpenRouterModel(ModelConfig(model="example/model", api_key="test-key"), client)

    with pytest.raises(OpenRouterHTTPError) as caught:
        model.complete([ChatMessage(role=Role.USER, content="look")], [])

    err = caught.value
    assert err.status_code == 429
    assert err.elapsed_seconds == 0.25
    assert err.body == body[:180]
    assert len(err.body) == 180
    assert "Traceback" not in str(err)
    assert "test-key" not in str(err)


def test_complete_without_a_key_does_not_call_http() -> None:
    client = FakeClient({"choices": []})
    model = OpenRouterModel(ModelConfig(model="example/model", api_key="  "), client)

    with pytest.raises(MissingOpenRouterKey) as caught:
        model.complete([ChatMessage(role=Role.USER, content="look")], [])

    assert str(caught.value) == MISSING_KEY_MESSAGE
    assert client.calls == []


def test_fetch_model_catalog_reads_id_context_and_price() -> None:
    class CatalogClient:
        def __init__(self) -> None:
            self.headers: dict[str, str] = {}

        def get(self, _url: str, **kwargs: Any) -> FakeResponse:
            self.headers = kwargs["headers"]
            return FakeResponse(
                {
                    "data": [
                        {
                            "id": "google/gemini-3.8-flash",
                            "context_length": 128000,
                            "pricing": {"prompt": "0.1", "completion": "0.4"},
                        },
                        {"id": "local/no-meta"},
                        {"name": "skipped"},
                    ]
                },
                elapsed_seconds=0.2,
            )

    client = CatalogClient()
    models = fetch_model_catalog(client, "catalog-key")

    assert client.headers["Authorization"] == "Bearer catalog-key"
    assert models[0].id == "google/gemini-3.8-flash"
    assert models[0].context_length == 128000
    assert models[0].prompt_price == "0.1"
    assert models[0].completion_price == "0.4"
    assert models[1].id == "local/no-meta"
    assert models[1].context_length is None
    assert models[1].prompt_price is None
    filtered = filter_model_catalog(models, "GEMINI")
    assert [model.id for model in filtered] == ["google/gemini-3.8-flash"]
    assert parse_model_catalog({"data": []}) == []
