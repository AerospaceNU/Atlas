from __future__ import annotations

import json
from typing import Any

import httpx
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
        self.lines: list[str] | None = None

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self.payload

    def iter_lines(self) -> list[str]:
        if self.lines is not None:
            return self.lines
        return [f"data: {json.dumps(self.payload)}", "data: [DONE]"]


class _Opened:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response

    def __enter__(self) -> FakeResponse:
        return self.response

    def __exit__(self, *_exc: object) -> None:
        return None


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
        self.lines: list[str] | None = None

    def stream(self, _method: str, _url: str, **kwargs: Any) -> _Opened:
        self.calls.append(kwargs)
        response = FakeResponse(
            self.response,
            status_code=self.status_code,
            elapsed_seconds=self.elapsed_seconds,
            time_to_first_token_seconds=self.time_to_first_token_seconds,
        )
        response.lines = self.lines
        return _Opened(response)

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
    sent = client.calls[0]["json"]
    assert sent["max_tokens"] == 64 * 1024
    assert sent["stream"] is True
    assert sent["stream_options"] == {"include_usage": True}
    timeout = client.calls[0]["timeout"]
    assert isinstance(timeout, httpx.Timeout)
    assert timeout.read == 60.0
    advertised_tool = client.calls[0]["json"]["tools"][0]["function"]
    assert advertised_tool["name"] == "inspect_image"
    assert advertised_tool["parameters"] == tool.parameters


def test_complete_joins_streamed_chunks_without_a_total_deadline() -> None:
    client = FakeClient({"choices": []})
    client.lines = [
        'data: {"choices":[{"delta":{"reasoning":"look "}}]}',
        'data: {"choices":[{"delta":{"reasoning":"closer","content":"hel"}}]}',
        'data: {"choices":[{"delta":{"content":"lo","tool_calls":[{"index":0,"id":"tool-1","function":{"name":"inspect_image","arguments":"{\\"path\\":"}}]}}]}',
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"coast.png\\"}"}}]}}],"usage":{"prompt_tokens":3,"completion_tokens":5,"total_tokens":8,"cost":0.01}}',
        "data: [DONE]",
    ]
    model = OpenRouterModel(ModelConfig(model="example/model", api_key="test"), client)

    response = model.complete([ChatMessage(role=Role.USER, content="look")], [])

    assert response.content == "hello"
    assert response.reasoning == "look closer"
    assert response.tool_calls[0].name == "inspect_image"
    assert response.tool_calls[0].arguments == {"path": "coast.png"}
    assert response.usage.completion_tokens == 5
    assert response.usage.cost == 0.01
    assert client.calls[0]["timeout"].read == 60.0


def test_complete_reads_reasoning_and_sends_it_back() -> None:
    details = [{"type": "reasoning.text", "text": "look at the coast", "format": "x"}]
    client = FakeClient(
        {
            "choices": [
                {
                    "message": {
                        "content": "clear",
                        "reasoning": "look at the coast",
                        "reasoning_details": details,
                        "tool_calls": [],
                    }
                }
            ]
        }
    )
    model = OpenRouterModel(ModelConfig(model="example/model", api_key="test"), client)

    parsed = model.complete([ChatMessage(role=Role.USER, content="look")], [])

    assert parsed.reasoning == "look at the coast"
    assert parsed.reasoning_details == details
    echoed = ChatMessage(
        role=Role.ASSISTANT,
        content=parsed.content,
        reasoning=parsed.reasoning,
        reasoning_details=parsed.reasoning_details,
    )
    model.complete([ChatMessage(role=Role.USER, content="look"), echoed], [])
    sent = client.calls[1]["json"]["messages"][1]
    assert sent["reasoning"] == "look at the coast"
    assert sent["reasoning_details"] == details
    assert "reasoning" not in client.calls[0]["json"]["messages"][0]


def test_complete_falls_back_when_reasoning_is_missing() -> None:
    content_only = FakeClient(
        {
            "choices": [
                {
                    "message": {
                        "content": "a",
                        "reasoning_content": "deepseek thought",
                        "tool_calls": [],
                    }
                }
            ]
        }
    )
    model = OpenRouterModel(ModelConfig(model="example/model", api_key="test"), content_only)
    assert model.complete([ChatMessage(role=Role.USER, content="look")], []).reasoning == (
        "deepseek thought"
    )

    details_only = FakeClient(
        {
            "choices": [
                {
                    "message": {
                        "content": "a",
                        "reasoning": "  ",
                        "reasoning_details": [{"type": "reasoning.text", "text": "from details"}],
                        "tool_calls": [],
                    }
                }
            ]
        }
    )
    model = OpenRouterModel(ModelConfig(model="example/model", api_key="test"), details_only)
    parsed = model.complete([ChatMessage(role=Role.USER, content="look")], [])
    assert parsed.reasoning == "from details"
    assert parsed.reasoning_details[0]["text"] == "from details"


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
