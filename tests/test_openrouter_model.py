from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from atlas.agent.contracts import ChatMessage, Role, ToolDefinition
from atlas.agent.model import (
    MISSING_KEY_MESSAGE,
    OPENROUTER_MODELS_URL,
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
            self.url = ""

        def get(self, url: str, **kwargs: Any) -> FakeResponse:
            self.url = url
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

    assert client.url == OPENROUTER_MODELS_URL
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
    assert [model.id for model in filter_model_catalog(models, "  ")] == [
        model.id for model in models
    ]
    assert parse_model_catalog({"data": []}) == []


class _CatalogPages:
    def __init__(self, pages: dict[str, tuple[dict[str, Any], int]]) -> None:
        self.pages = pages
        self.urls: list[str] = []
        self.headers: list[dict[str, str]] = []

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.urls.append(url)
        self.headers.append(kwargs["headers"])
        payload, status = self.pages[url]
        return FakeResponse(payload, status_code=status, elapsed_seconds=0.2)


def test_list_models_follows_a_same_host_next_page() -> None:
    first = OPENROUTER_MODELS_URL
    second = "https://openrouter.ai/api/v1/models?offset=1"
    client = _CatalogPages(
        {
            first: (
                {
                    "data": [
                        {"id": "alpha/one", "context_length": 1024, "pricing": {"prompt": "0.1"}}
                    ],
                    "links": {"next": "/api/v1/models?offset=1"},
                },
                200,
            ),
            second: (
                {
                    "data": [{"id": "beta/two", "pricing": {"completion": 0.4}}],
                    "links": {"next": None},
                },
                200,
            ),
        }
    )
    model = OpenRouterModel(ModelConfig(model="keep-me", api_key="catalog-key"), client)

    listed = model.list_models()

    assert client.urls == [first, second]
    assert client.headers == [{"Authorization": "Bearer catalog-key"}] * 2
    assert [item.id for item in listed] == ["alpha/one", "beta/two"]
    assert listed[0].context_length == 1024
    assert listed[0].prompt_price == "0.1"
    assert listed[1].completion_price == "0.4"
    assert model.config.model == "keep-me"


def test_fetch_model_catalog_stops_on_a_cycle_and_ignores_an_off_host_next() -> None:
    first = OPENROUTER_MODELS_URL
    second = "https://openrouter.ai/api/v1/models?offset=1"
    cycling = _CatalogPages(
        {
            first: ({"data": [{"id": "alpha/one"}], "links": {"next": second}}, 200),
            second: ({"data": [{"id": "beta/two"}], "links": {"next": first}}, 200),
        }
    )
    cycled = fetch_model_catalog(cycling, "catalog-key")
    assert cycling.urls == [first, second]
    assert [item.id for item in cycled] == ["alpha/one", "beta/two"]

    off_host = _CatalogPages(
        {
            first: (
                {"data": [{"id": "alpha/one"}], "links": {"next": "https://evil.example/models"}},
                200,
            )
        }
    )
    kept = fetch_model_catalog(off_host, "catalog-key")
    assert off_host.urls == [first]
    assert [item.id for item in kept] == ["alpha/one"]


def test_fetch_model_catalog_stops_at_the_page_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("atlas.agent.model._CATALOG_PAGE_LIMIT", 1)
    first = OPENROUTER_MODELS_URL
    client = _CatalogPages(
        {
            first: (
                {"data": [{"id": "alpha/one"}], "links": {"next": "/api/v1/models?offset=1"}},
                200,
            )
        }
    )

    listed = fetch_model_catalog(client, "catalog-key")

    assert client.urls == [first]
    assert [item.id for item in listed] == ["alpha/one"]


def test_fetch_model_catalog_without_a_key_does_not_call_http() -> None:
    class Boom:
        def get(self, *_args: object, **_kwargs: object) -> FakeResponse:
            raise AssertionError("http")

    with pytest.raises(MissingOpenRouterKey) as caught:
        fetch_model_catalog(Boom(), "  ")

    assert str(caught.value) == MISSING_KEY_MESSAGE


def test_fetch_model_catalog_keeps_non_2xx_status_and_a_short_body() -> None:
    body = "y" * 400
    client = _CatalogPages({OPENROUTER_MODELS_URL: ({"error": {"message": body}}, 401)})

    with pytest.raises(OpenRouterHTTPError) as caught:
        fetch_model_catalog(client, "catalog-key")

    err = caught.value
    assert err.status_code == 401
    assert err.elapsed_seconds == 0.2
    assert err.body == body[:180]
    assert "catalog-key" not in str(err)
    assert client.urls == [OPENROUTER_MODELS_URL]


def test_parse_model_catalog_skips_bad_rows_and_rejects_a_bad_payload() -> None:
    models = parse_model_catalog(
        {
            "data": [
                "nope",
                {"id": "  "},
                {"id": "  vendor/model  ", "context_length": True, "pricing": {"prompt": -1}},
                {
                    "id": "kept/zero",
                    "context_length": -5,
                    "pricing": {"prompt": 0, "completion": ""},
                },
            ]
        }
    )

    assert [model.id for model in models] == ["vendor/model", "kept/zero"]
    assert models[0].context_length is None
    assert models[0].prompt_price == "-1"
    assert models[1].context_length is None
    assert models[1].prompt_price == "0"
    assert models[1].completion_price is None
    with pytest.raises(ValueError, match="model catalog must be an object"):
        parse_model_catalog([])
    with pytest.raises(ValueError, match="model catalog is missing data"):
        parse_model_catalog({"models": []})
