from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Self

import httpx

from atlas.agent.contracts import (
    CallTiming,
    CatalogModel,
    ChatMessage,
    ModelResponse,
    Role,
    TokenUsage,
    ToolCall,
    ToolDefinition,
)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
_ERROR_BODY_LIMIT = 180

MISSING_KEY_MESSAGE = (
    "OPENROUTER_API_KEY is not set. Set it in the environment, the repo .env, "
    "or with /key session or /key user."
)


class MissingOpenRouterKey(RuntimeError):
    """The process has no OpenRouter key in memory, the environment, or the user file."""


class OpenRouterHTTPError(RuntimeError):
    """A non-2xx OpenRouter response with its status and a short body."""

    def __init__(
        self,
        status_code: int,
        body: str,
        elapsed_seconds: float,
        time_to_first_token_seconds: float | None = None,
    ) -> None:
        self.status_code = status_code
        self.body = body
        self.elapsed_seconds = elapsed_seconds
        self.time_to_first_token_seconds = time_to_first_token_seconds
        super().__init__(f"OpenRouter HTTP {status_code}: {body}")


@dataclass
class ModelConfig:
    model: str
    api_key: str
    reasoning: bool = True
    max_new_tokens: int = 2048
    timeout: float = 60.0


class OpenRouterModel:
    def __init__(self, config: ModelConfig, client: httpx.Client | None = None) -> None:
        self.config = config
        self.history: list[ChatMessage] = []
        self._client = client or httpx.Client(timeout=config.timeout)
        self._owns_client = client is None

    def __call__(self, prompt: str) -> str:
        self.history.append(ChatMessage(role=Role.USER, content=prompt))
        try:
            response = self.complete(self.history, [])
            content = response.content
            if content is None:
                raise ValueError("OpenRouter returned no text content")
        except Exception:
            self.history.pop()
            raise
        self.history.append(ChatMessage(role=Role.ASSISTANT, content=content))
        return content

    def complete(self, messages: list[ChatMessage], tools: list[ToolDefinition]) -> ModelResponse:
        """Call OpenRouter using its OpenAI-compatible native tool-calling format."""
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [_serialize_message(message) for message in messages],
            "reasoning": {"enabled": self.config.reasoning},
            "max_tokens": self.config.max_new_tokens,
        }
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in tools
            ]
        response, timing = self._post(OPENROUTER_URL, payload)
        try:
            message = response["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"Unexpected OpenRouter response shape: {response!r}") from exc
        if not isinstance(message, dict):
            raise TypeError(f"Expected response message object, got {type(message)}")
        parsed = _parse_response(message)
        parsed.usage = parse_usage(response.get("usage"))
        parsed.timing = timing
        return parsed

    def list_models(self) -> list[CatalogModel]:
        """Fetch the live OpenRouter model catalog with this client's key."""
        return fetch_model_catalog(self._client, self.config.api_key)

    def reset(self) -> None:
        self.history.clear()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def _post(self, url: str, payload: dict[str, Any]) -> tuple[dict[str, Any], CallTiming]:
        if not self.config.api_key.strip():
            raise MissingOpenRouterKey(MISSING_KEY_MESSAGE)
        started = time.perf_counter()
        resp = self._client.post(
            url,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        return _interpret_response(resp, started)


def fetch_model_catalog(client: Any, api_key: str) -> list[CatalogModel]:
    """GET the OpenRouter models endpoint and parse id, context length, and price."""
    if not api_key.strip():
        raise MissingOpenRouterKey(MISSING_KEY_MESSAGE)
    started = time.perf_counter()
    resp = client.get(
        OPENROUTER_MODELS_URL,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    payload, _timing = _interpret_response(resp, started)
    return parse_model_catalog(payload)


def parse_usage(raw: object) -> TokenUsage:
    """Read prompt, completion, total, and cost from a provider usage object."""
    if not isinstance(raw, dict):
        return TokenUsage()
    prompt = _as_int(raw.get("prompt_tokens"))
    completion = _as_int(raw.get("completion_tokens"))
    total_raw = raw.get("total_tokens")
    total = _as_int(total_raw) if total_raw is not None else prompt + completion
    return TokenUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        cost=_as_cost(raw.get("cost")),
    )


def parse_model_catalog(payload: object) -> list[CatalogModel]:
    """Parse an OpenRouter ``/models`` payload.

    Rows without an id are skipped. Missing context length or pricing stays unset.
    """
    if not isinstance(payload, dict):
        raise ValueError("model catalog must be an object")
    data = payload.get("data")
    if not isinstance(data, list):
        raise ValueError("model catalog is missing data")
    models: list[CatalogModel] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        if not isinstance(model_id, str) or not model_id.strip():
            continue
        pricing = item.get("pricing")
        prompt_price = None
        completion_price = None
        if isinstance(pricing, dict):
            prompt_price = _as_price(pricing.get("prompt"))
            completion_price = _as_price(pricing.get("completion"))
        models.append(
            CatalogModel(
                id=model_id.strip(),
                context_length=_as_optional_int(item.get("context_length")),
                prompt_price=prompt_price,
                completion_price=completion_price,
            )
        )
    return models


def filter_model_catalog(models: Sequence[CatalogModel], query: str) -> list[CatalogModel]:
    """Keep catalog rows whose id contains ``query``, case-insensitively."""
    needle = query.strip().casefold()
    if not needle:
        return list(models)
    return [model for model in models if needle in model.id.casefold()]


def _interpret_response(resp: Any, started: float) -> tuple[dict[str, Any], CallTiming]:
    elapsed = _elapsed_seconds(resp, started)
    ttft = _as_optional_seconds(getattr(resp, "time_to_first_token_seconds", None))
    status = _status_code(resp)
    timing = CallTiming(
        http_status=status,
        elapsed_seconds=elapsed,
        time_to_first_token_seconds=ttft,
    )
    if status < 200 or status >= 300:
        raise OpenRouterHTTPError(status, _short_error_body(resp), elapsed, ttft)
    response = resp.json()
    if not isinstance(response, dict):
        raise TypeError(f"Expected OpenRouter response object, got {type(response)}")
    return response, timing


def _status_code(resp: Any) -> int:
    raw = getattr(resp, "status_code", 200)
    if isinstance(raw, bool) or not isinstance(raw, int):
        return 200
    return raw


def _elapsed_seconds(resp: Any, started: float) -> float:
    explicit = getattr(resp, "elapsed_seconds", None)
    if isinstance(explicit, (int, float)) and not isinstance(explicit, bool) and explicit >= 0:
        return float(explicit)
    elapsed = getattr(resp, "elapsed", None)
    if isinstance(elapsed, timedelta):
        return max(0.0, elapsed.total_seconds())
    if isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool) and elapsed >= 0:
        return float(elapsed)
    return max(0.0, time.perf_counter() - started)


def _short_error_body(resp: Any) -> str:
    payload: object
    try:
        payload = resp.json()
    except Exception:
        payload = None
    if isinstance(payload, dict):
        err = payload.get("error")
        message = err.get("message") if isinstance(err, dict) else None
        if isinstance(message, str):
            return message[:_ERROR_BODY_LIMIT]
        if isinstance(err, str):
            return err[:_ERROR_BODY_LIMIT]
    text = getattr(resp, "text", "")
    if isinstance(text, str) and text.strip():
        return text.strip()[:_ERROR_BODY_LIMIT]
    return "request failed"


def _serialize_message(message: ChatMessage) -> dict[str, Any]:
    data: dict[str, Any] = {"role": message.role.value}
    if message.content is not None:
        data["content"] = message.content
    if message.tool_call_id is not None:
        data["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        data["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
            }
            for call in message.tool_calls
        ]
    return data


def _parse_response(message: dict[str, Any]) -> ModelResponse:
    content = message.get("content")
    if content is not None and not isinstance(content, str):
        raise TypeError(f"Expected string or null content, got {type(content)}")
    calls: list[ToolCall] = []
    raw_calls = message.get("tool_calls", [])
    if not isinstance(raw_calls, list):
        raise TypeError(f"Expected list tool_calls, got {type(raw_calls)}")
    for raw_call in raw_calls:
        try:
            function = raw_call["function"]
            raw_arguments = function["arguments"]
            arguments = (
                json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
            )
            calls.append(ToolCall(id=raw_call["id"], name=function["name"], arguments=arguments))
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Unexpected OpenRouter tool call: {raw_call!r}") from exc
    return ModelResponse(content=content, tool_calls=calls)


def _as_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return int(value)


def _as_optional_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = int(value)
    if number < 0:
        return None
    return number


def _as_optional_seconds(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value < 0:
        return None
    return float(value)


def _as_cost(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _as_price(value: object) -> str | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float, str)):
        text = str(value).strip()
        return text or None
    return None
