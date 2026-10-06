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
    max_new_tokens: int = 64 * 1024
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
        # Tokens arrive as server-sent events. The read timeout is the silence
        # between those events, so a reply longer than `timeout` is not cut off
        # while chunks are still arriving.
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
        response, timing = self._complete(payload)
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

    def _complete(self, payload: dict[str, Any]) -> tuple[dict[str, Any], CallTiming]:
        if not self.config.api_key.strip():
            raise MissingOpenRouterKey(MISSING_KEY_MESSAGE)
        started = time.perf_counter()
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        timeout = _chunk_timeout(self.config.timeout)
        stream = getattr(self._client, "stream", None)
        if callable(stream):
            with stream(
                "POST", OPENROUTER_URL, headers=headers, json=payload, timeout=timeout
            ) as resp:
                return _read_completion(resp, started)
        resp = self._client.post(OPENROUTER_URL, headers=headers, json=payload)
        return _read_completion(resp, started)


def _chunk_timeout(seconds: float) -> httpx.Timeout:
    """Bound each socket wait, not the whole completion.

    While server-sent events are arriving, ``read`` is the gap allowed
    between chunks. A quiet socket still fails after ``seconds``.
    """
    return httpx.Timeout(connect=seconds, read=seconds, write=seconds, pool=seconds)


@dataclass
class _ToolAcc:
    id: str = ""
    name: str = ""
    arguments: str = ""


@dataclass
class _StreamAcc:
    content: list[str]
    reasoning: list[str]
    reasoning_content: list[str]
    details: list[dict[str, Any]]
    calls: dict[int, _ToolAcc]
    usage: dict[str, Any] | None = None


def _read_completion(resp: Any, started: float) -> tuple[dict[str, Any], CallTiming]:
    status = _status_code(resp)
    elapsed = _completion_elapsed(resp, started)
    ttft = _completion_ttft(resp, None)
    if status < 200 or status >= 300:
        _buffer_body(resp)
        raise OpenRouterHTTPError(status, _short_error_body(resp), elapsed, ttft)
    measured: list[float] = []
    message, usage = _assemble_completion(resp, started, measured)
    elapsed = _completion_elapsed(resp, started)
    ttft = _completion_ttft(resp, measured[0] if measured else None)
    timing = CallTiming(
        http_status=status,
        elapsed_seconds=elapsed,
        time_to_first_token_seconds=ttft,
    )
    return {"choices": [{"message": message}], "usage": usage}, timing


def _completion_elapsed(resp: Any, started: float) -> float:
    """Wall time for a streamed body.

    ``Response.elapsed`` stops at the headers, which arrive before the tokens.
    An injected ``elapsed_seconds`` still wins so tests can pin the duration.
    """
    explicit = getattr(resp, "elapsed_seconds", None)
    if isinstance(explicit, (int, float)) and not isinstance(explicit, bool) and explicit >= 0:
        return float(explicit)
    return max(0.0, time.perf_counter() - started)


def _completion_ttft(resp: Any, measured: float | None) -> float | None:
    if hasattr(resp, "time_to_first_token_seconds"):
        return _as_optional_seconds(resp.time_to_first_token_seconds)
    return measured


def _buffer_body(resp: Any) -> None:
    read = getattr(resp, "read", None)
    if not callable(read):
        return
    try:
        read()
    except Exception:
        return


def _assemble_completion(
    resp: Any, started: float, measured: list[float]
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    acc = _StreamAcc(content=[], reasoning=[], reasoning_content=[], details=[], calls={})
    for payload in _iter_events(resp):
        if "error" in payload and "choices" not in payload:
            raise _stream_error(payload, resp, started)
        usage = payload.get("usage")
        if isinstance(usage, dict):
            acc.usage = usage
        choices = payload.get("choices")
        if not isinstance(choices, list):
            continue
        for choice in choices:
            if isinstance(choice, dict):
                _apply_choice(acc, choice, started, measured)
    if (
        not acc.content
        and not acc.calls
        and not acc.reasoning
        and not acc.reasoning_content
        and not acc.details
    ):
        raise ValueError("Unexpected OpenRouter response shape: no streamed message")
    message: dict[str, Any] = {"content": "".join(acc.content) if acc.content else None}
    if acc.reasoning:
        message["reasoning"] = "".join(acc.reasoning)
    if acc.reasoning_content:
        message["reasoning_content"] = "".join(acc.reasoning_content)
    if acc.details:
        message["reasoning_details"] = acc.details
    if acc.calls:
        message["tool_calls"] = [
            {
                "id": acc.calls[index].id,
                "function": {
                    "name": acc.calls[index].name,
                    "arguments": acc.calls[index].arguments,
                },
            }
            for index in sorted(acc.calls)
        ]
    return message, acc.usage


def _stream_error(payload: dict[str, Any], resp: Any, started: float) -> OpenRouterHTTPError:
    err = payload.get("error")
    message = ""
    code = 502
    if isinstance(err, dict):
        text = err.get("message")
        if isinstance(text, str):
            message = text[:_ERROR_BODY_LIMIT]
        raw_code = err.get("code")
        if isinstance(raw_code, int) and not isinstance(raw_code, bool) and 400 <= raw_code <= 599:
            code = raw_code
    elif isinstance(err, str):
        message = err[:_ERROR_BODY_LIMIT]
    if not message:
        message = "request failed"
    return OpenRouterHTTPError(
        code,
        message,
        _completion_elapsed(resp, started),
        _completion_ttft(resp, None),
    )


def _iter_events(resp: Any) -> list[dict[str, Any]]:
    lines = getattr(resp, "iter_lines", None)
    raw_lines: list[object]
    if callable(lines):
        raw_lines = list(lines())
    else:
        body = resp.json()
        raw_lines = [json.dumps(body), "[DONE]"]
    events: list[dict[str, Any]] = []
    for raw in raw_lines:
        text = raw.decode() if isinstance(raw, bytes) else raw
        if not isinstance(text, str):
            continue
        line = text.strip()
        if not line or line.startswith(":"):
            continue
        if line.startswith("data:"):
            line = line[5:].strip()
        if line == "[DONE]":
            break
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Unexpected OpenRouter stream event: {line!r}") from exc
        if isinstance(payload, dict):
            events.append(payload)
    return events


def _apply_choice(
    acc: _StreamAcc, choice: dict[str, Any], started: float, measured: list[float]
) -> None:
    for key in ("delta", "message"):
        piece = choice.get(key)
        if isinstance(piece, dict):
            _apply_piece(acc, piece, started, measured)


def _apply_piece(
    acc: _StreamAcc, piece: dict[str, Any], started: float, measured: list[float]
) -> None:
    _append_text(acc.content, piece.get("content"), started, measured)
    _append_text(acc.reasoning, piece.get("reasoning"), started, measured)
    _append_text(acc.reasoning_content, piece.get("reasoning_content"), started, measured)
    details = piece.get("reasoning_details")
    if isinstance(details, list):
        for item in details:
            if isinstance(item, dict):
                _merge_detail(acc, item)
                if not measured:
                    measured.append(max(0.0, time.perf_counter() - started))
    calls = piece.get("tool_calls")
    if isinstance(calls, list):
        for index, call in enumerate(calls):
            if isinstance(call, dict):
                _merge_tool_call(acc, call, index)
                if not measured:
                    measured.append(max(0.0, time.perf_counter() - started))


def _append_text(parts: list[str], value: object, started: float, measured: list[float]) -> None:
    if not isinstance(value, str) or not value:
        return
    parts.append(value)
    if not measured:
        measured.append(max(0.0, time.perf_counter() - started))


def _merge_detail(acc: _StreamAcc, item: dict[str, Any]) -> None:
    index = item.get("index")
    slot: dict[str, Any] | None = None
    if isinstance(index, int) and not isinstance(index, bool):
        while len(acc.details) <= index:
            acc.details.append({})
        slot = acc.details[index]
    else:
        slot = {}
        acc.details.append(slot)
    for key, value in item.items():
        if key == "index":
            continue
        current = slot.get(key)
        if (
            key in ("text", "summary", "content")
            and isinstance(value, str)
            and isinstance(current, str)
        ):
            slot[key] = current + value
        else:
            slot[key] = value


def _merge_tool_call(acc: _StreamAcc, call: dict[str, Any], fallback: int) -> None:
    raw_index = call.get("index", fallback)
    index = (
        raw_index if isinstance(raw_index, int) and not isinstance(raw_index, bool) else fallback
    )
    slot = acc.calls.get(index)
    if slot is None:
        slot = _ToolAcc()
        acc.calls[index] = slot
    call_id = call.get("id")
    if isinstance(call_id, str) and call_id:
        slot.id = call_id
    function = call.get("function")
    if not isinstance(function, dict):
        return
    name = function.get("name")
    if isinstance(name, str) and name:
        slot.name = name
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        slot.arguments += arguments
    elif isinstance(arguments, dict) and not slot.arguments:
        slot.arguments = json.dumps(arguments)


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
    if message.reasoning:
        data["reasoning"] = message.reasoning
    if message.reasoning_details:
        data["reasoning_details"] = message.reasoning_details
    return data


def _optional_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _detail_text(item: object) -> str | None:
    if isinstance(item, str):
        return _optional_text(item)
    if not isinstance(item, dict):
        return None
    for key in ("text", "summary", "content"):
        found = _optional_text(item.get(key))
        if found is not None:
            return found
    return None


def _parse_reasoning(message: dict[str, Any]) -> tuple[str | None, list[dict[str, Any]]]:
    """Return display text and the provider's reasoning payload.

    OpenRouter uses ``reasoning``. DeepSeek-style replies use
    ``reasoning_content``. ``reasoning_details`` is kept whole so a later
    request can echo it, including entries that have no displayable text.
    """
    details_raw = message.get("reasoning_details")
    details: list[dict[str, Any]] = []
    detail_parts: list[str] = []
    if isinstance(details_raw, list):
        for item in details_raw:
            if isinstance(item, dict):
                details.append(item)
            text = _detail_text(item)
            if text is not None:
                detail_parts.append(text)
    reasoning = _optional_text(message.get("reasoning"))
    if reasoning is None:
        reasoning = _optional_text(message.get("reasoning_content"))
    if reasoning is None and detail_parts:
        reasoning = "\n\n".join(detail_parts)
    return reasoning, details


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
    reasoning, reasoning_details = _parse_reasoning(message)
    return ModelResponse(
        content=content,
        tool_calls=calls,
        reasoning=reasoning,
        reasoning_details=reasoning_details,
    )


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
