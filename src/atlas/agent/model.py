from __future__ import annotations

import contextlib
import hashlib
import json
import random
import re
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any, Self
from urllib.parse import urljoin, urlparse

import httpx
from httpx_sse import EventSource
from tenacity import RetryCallState, Retrying, retry_if_exception, stop_after_attempt

from atlas.agent.contracts import (
    CallTiming,
    CatalogModel,
    ChatMessage,
    ModelResponse,
    Role,
    TokenUsage,
    ToolCall,
    ToolDefinition,
    current_cancel_event,
)
from atlas.agent.images import apply_image_budget
from atlas.agent.scope import ModelCancelled, cancel_requested

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
_ERROR_BODY_LIMIT = 180
_RETRY_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
_MAX_ATTEMPTS = 4
_RETRY_BASE_DELAY = 0.5
_RETRY_BACKOFF = 2.0
_RETRY_AFTER_CAP = 30.0
_MIN_RETRY_AFTER = 0.25
_CANCEL_SLICE = 0.05
_OVERFLOW_STATUS = frozenset({400, 413, 422})
_OVERFLOW_MARKERS = (
    "context_length_exceeded",
    "context length",
    "context window",
    "maximum context",
    "too many tokens",
    "prompt is too long",
    "input is too long",
    "reduce the length",
    "context limit",
    "context size",
    "exceed context",
    "exceeds the maximum number of tokens",
    "input length and max_tokens",
    "reduce your input",
)
_OVERFLOW_COUNT_RE = re.compile(r"\b(?:tokens?|prompt|input)\b[^.\n]{0,60}exceed")
_CATALOG_PAGE_LIMIT = 20
_CATALOG_TIME_LIMIT = 15.0
_CATALOG_CACHE_SECONDS = 30.0
_OPENROUTER_HOST = "openrouter.ai"
_ANSI_ESCAPE = re.compile(
    r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[@-Z\\-_]"
)
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_BIDI_CHARS = re.compile("[\u061c\u200e\u200f\u202a-\u202e\u2066-\u2069]")
MISSING_KEY_MESSAGE = (
    "OPENROUTER_API_KEY is not set. Set it in the environment, the repo .env, "
    "or with /key session or /key user."
)
REASONING_EFFORTS = ("max", "xhigh", "high", "medium", "low", "minimal", "none")


class MissingOpenRouterKey(RuntimeError):
    """The process has no OpenRouter key in memory, the environment, or the user file."""


class StreamCancelledError(ModelCancelled):
    """A model completion or retry wait was cancelled mid-flight."""


def _interruptible_sleep(seconds: float) -> None:
    """Sleep up to ``seconds`` in short slices, returning early once cancelled."""
    ev = current_cancel_event()
    deadline = time.monotonic() + seconds
    while not cancel_requested() and not (ev is not None and ev.is_set()):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        if ev is not None:
            if ev.wait(min(_CANCEL_SLICE, remaining)):
                return
        else:
            time.sleep(min(_CANCEL_SLICE, remaining))


_sleep = _interruptible_sleep


def _jitter() -> float:
    """Return a multiplier applied to the exponential delay; tests pin it."""
    return random.uniform(0.75, 1.25)


def validate_reasoning_effort(value: str | None) -> str | None:
    """Return ``value`` if it is None or a supported OpenRouter reasoning effort."""
    if value is not None and value not in REASONING_EFFORTS:
        raise ValueError(f"reasoning_effort must be one of {', '.join(REASONING_EFFORTS)}")
    return value


class OpenRouterHTTPError(RuntimeError):
    """A non-2xx OpenRouter response with its status and a short body."""

    def __init__(
        self,
        status_code: int,
        body: str,
        elapsed_seconds: float,
        time_to_first_token_seconds: float | None = None,
        *,
        retry_after: float | None = None,
        context_overflow: bool | None = None,
    ) -> None:
        self.status_code = status_code
        self.body = body
        self.retry_after = retry_after
        self.is_context_overflow = (
            _is_overflow(status_code, body) if context_overflow is None else context_overflow
        )
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
    reasoning_effort: str | None = None
    provider_order: tuple[str, ...] = ()
    temperature: float | None = None
    top_p: float | None = None

    def __post_init__(self) -> None:
        validate_reasoning_effort(self.reasoning_effort)

    def provider_payload(self) -> dict[str, Any] | None:
        """Return the OpenRouter ``provider`` routing object, or None for default routing."""
        if not self.provider_order:
            return None
        return {"order": list(self.provider_order), "allow_fallbacks": False}

    def reasoning_payload(self) -> dict[str, Any]:
        """Return the OpenRouter ``reasoning`` object for this config."""
        if self.reasoning and self.reasoning_effort is not None:
            return {"effort": self.reasoning_effort}
        return {"enabled": self.reasoning}


@dataclass
class _ToolAcc:
    id: str = ""
    name: str = ""
    arguments: str = ""


@dataclass
class _StreamAcc:
    content: list[str] = field(default_factory=list)
    reasoning: list[str] = field(default_factory=list)
    reasoning_content: list[str] = field(default_factory=list)
    details: list[dict[str, Any]] = field(default_factory=list)
    calls: dict[int, _ToolAcc] = field(default_factory=dict)
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None


class OpenRouterModel:
    def __init__(self, config: ModelConfig, client: httpx.Client | None = None) -> None:
        self.config = config
        self._client = client or httpx.Client(timeout=config.timeout, follow_redirects=False)
        self._owns_client = client is None

    def complete(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition],
        *,
        on_content_delta: Callable[[str], None] | None = None,
        on_thought_delta: Callable[[str], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> ModelResponse:
        """Call OpenRouter with streaming deltas, retry-after backoff, and cancel checks."""
        apply_image_budget(messages)
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [_serialize_message(m) for m in messages],
            "reasoning": self.config.reasoning_payload(),
            "max_tokens": self.config.max_new_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if (provider := self.config.provider_payload()) is not None:
            payload["provider"] = provider
        if self.config.temperature is not None:
            payload["temperature"] = self.config.temperature
        if self.config.top_p is not None:
            payload["top_p"] = self.config.top_p
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]

        def is_cancelled() -> bool:
            ev = current_cancel_event()
            return (
                cancel_requested()
                or (should_cancel is not None and should_cancel())
                or (ev is not None and ev.is_set())
            )

        def sleep_cancelable(seconds: float) -> None:
            if seconds <= 0:
                return
            if is_cancelled():
                raise StreamCancelledError("cancelled during retry wait")
            if _sleep is not _interruptible_sleep:
                _sleep(seconds)
                if is_cancelled():
                    raise StreamCancelledError("cancelled during retry wait")
                return
            deadline = time.monotonic() + seconds
            while (remaining := deadline - time.monotonic()) > 0:
                if is_cancelled():
                    raise StreamCancelledError("cancelled during retry wait")
                _interruptible_sleep(min(_CANCEL_SLICE, remaining))

        def is_retryable(exc: BaseException) -> bool:
            if isinstance(exc, ModelCancelled) or is_cancelled():
                return False
            if isinstance(exc, OpenRouterHTTPError):
                return (not exc.is_context_overflow) and exc.status_code in _RETRY_STATUS
            return isinstance(exc, httpx.TimeoutException | httpx.TransportError)

        retrying = Retrying(
            stop=stop_after_attempt(_MAX_ATTEMPTS),
            retry=retry_if_exception(is_retryable),
            wait=_retry_wait,
            sleep=sleep_cancelable,
            reraise=True,
        )
        response, timing = retrying(
            self._complete_once, payload, on_content_delta, on_thought_delta, is_cancelled
        )
        try:
            message = response["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"Unexpected OpenRouter response shape: {response!r}") from exc
        if not isinstance(message, dict):
            raise TypeError(f"Expected response message object, got {type(message)}")
        parsed = _parse_response(message)
        parsed.finish_reason = _finish_reason_of(response)
        parsed.usage = parse_usage(response.get("usage"))
        parsed.timing = timing
        return parsed

    def list_models(self, *, refresh: bool = False) -> CatalogFetch:
        return fetch_model_catalog(self._client, self.config.api_key, bypass_cache=refresh)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def _complete_once(
        self,
        payload: dict[str, Any],
        on_content_delta: Callable[[str], None] | None,
        on_thought_delta: Callable[[str], None] | None,
        is_cancelled: Callable[[], bool],
    ) -> tuple[dict[str, Any], CallTiming]:
        if not self.config.api_key.strip():
            raise MissingOpenRouterKey(MISSING_KEY_MESSAGE)
        if is_cancelled():
            raise StreamCancelledError("cancelled before request")
        started = time.perf_counter()
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        t = self.config.timeout
        timeout = httpx.Timeout(connect=t, read=t, write=t, pool=t)
        stream = getattr(self._client, "stream", None)
        if callable(stream):
            with stream(
                "POST", OPENROUTER_URL, headers=headers, json=payload, timeout=timeout
            ) as resp:
                return _read_completion(
                    resp, started, on_content_delta, on_thought_delta, is_cancelled
                )
        resp = self._client.post(OPENROUTER_URL, headers=headers, json=payload)
        return _read_completion(resp, started, on_content_delta, on_thought_delta, is_cancelled)


def _retry_wait(state: RetryCallState) -> float:
    base = _RETRY_BASE_DELAY * (_RETRY_BACKOFF ** max(0, state.attempt_number - 1))
    backoff = float(base * _jitter())
    exc = state.outcome.exception() if state.outcome is not None else None
    retry_after = exc.retry_after if isinstance(exc, OpenRouterHTTPError) else None
    wait = retry_after if retry_after is not None and retry_after >= _MIN_RETRY_AFTER else backoff
    return min(wait, _RETRY_AFTER_CAP)


def _parse_retry_after(resp: Any) -> float | None:
    headers = getattr(resp, "headers", None)
    if headers is None or not hasattr(headers, "get"):
        return None
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip()
    try:
        value = float(text)
    except ValueError:
        value = None
    if value is not None:
        return value if value == value and value >= 0 else None
    try:
        target = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    if target.tzinfo is None:
        target = target.replace(tzinfo=UTC)
    return max(0.0, (target - datetime.now(UTC)).total_seconds())


def _is_overflow(status: int, text: str) -> bool:
    """True for a 400/413/422 whose full message says the prompt is too long."""
    if status not in _OVERFLOW_STATUS:
        return False
    lowered = text.lower()
    return any(marker in lowered for marker in _OVERFLOW_MARKERS) or bool(
        _OVERFLOW_COUNT_RE.search(lowered)
    )


def _full_error_text(resp: Any) -> str:
    """Return the untruncated error message, used only for overflow classification."""
    try:
        payload = resp.json()
    except Exception:
        payload = None
    if isinstance(payload, dict):
        err = payload.get("error")
        message = err.get("message") if isinstance(err, dict) else err
        if isinstance(message, str):
            return message
    text = getattr(resp, "text", "")
    return text if isinstance(text, str) else ""


def _http_error(status: int, resp: Any, elapsed: float, ttft: float | None) -> OpenRouterHTTPError:
    return OpenRouterHTTPError(
        status,
        _short_error_body(resp),
        elapsed,
        ttft,
        retry_after=_parse_retry_after(resp),
        context_overflow=_is_overflow(status, _full_error_text(resp)),
    )


def _finish_reason_of(response: dict[str, Any]) -> str | None:
    try:
        reason = response["choices"][0].get("finish_reason")
    except (KeyError, IndexError, TypeError, AttributeError):
        return None
    return reason if isinstance(reason, str) and reason else None


def _read_completion(
    resp: Any,
    started: float,
    on_content_delta: Callable[[str], None] | None,
    on_thought_delta: Callable[[str], None] | None,
    is_cancelled: Callable[[], bool],
) -> tuple[dict[str, Any], CallTiming]:
    raw_status = getattr(resp, "status_code", 200)
    status = raw_status if isinstance(raw_status, int) and not isinstance(raw_status, bool) else 200
    if status < 200 or status >= 300:
        if callable(read := getattr(resp, "read", None)):
            with contextlib.suppress(Exception):
                read()
        raise _http_error(
            status,
            resp,
            _elapsed(resp, started, use_response_elapsed=False),
            _num_opt(getattr(resp, "time_to_first_token_seconds", None)),
        )

    measured: list[float] = []
    acc = _StreamAcc()
    for payload in _iter_events(resp):
        if is_cancelled():
            raise StreamCancelledError("cancelled during stream")
        if isinstance(payload.get("error"), dict | str):
            raise _stream_error(payload, resp, started)
        if isinstance(payload.get("usage"), dict):
            acc.usage = payload["usage"]
        if isinstance(choices := payload.get("choices"), list):
            for choice in choices:
                if isinstance(choice, dict):
                    _apply_choice(
                        acc, choice, started, measured, on_content_delta, on_thought_delta
                    )
                    if choice.get("finish_reason") == "error":
                        detail = choice.get("error") or payload.get("error")
                        raise _stream_error({"error": detail}, resp, started)
    if not any((acc.content, acc.calls, acc.reasoning, acc.reasoning_content, acc.details)):
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
                "id": acc.calls[i].id,
                "function": {"name": acc.calls[i].name, "arguments": acc.calls[i].arguments},
            }
            for i in sorted(acc.calls)
        ]
    ttft = (
        _num_opt(resp.time_to_first_token_seconds)
        if hasattr(resp, "time_to_first_token_seconds")
        else (measured[0] if measured else None)
    )
    timing = CallTiming(
        http_status=status,
        elapsed_seconds=_elapsed(resp, started, use_response_elapsed=False),
        time_to_first_token_seconds=ttft,
    )
    choice_out = {"message": message, "finish_reason": acc.finish_reason}
    return {"choices": [choice_out], "usage": acc.usage}, timing


def _iter_events(resp: Any) -> Iterator[dict[str, Any]]:
    """Yield decoded JSON dicts from an SSE stream or JSON fallback response."""
    if isinstance(resp, httpx.Response):
        if "text/event-stream" in resp.headers.get("content-type", "").lower():
            for sse in EventSource(resp).iter_sse():
                data = sse.data.strip()
                if data == "[DONE]":
                    break
                if data:
                    yield from _parse_json_dict(data)
            return
        if isinstance(body := resp.json(), dict):
            yield body
        return

    lines = getattr(resp, "iter_lines", None)
    if not callable(lines):
        if isinstance(body := resp.json(), dict):
            yield body
        return
    for raw in lines():
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
        yield from _parse_json_dict(line)


def _parse_json_dict(line: str) -> Iterator[dict[str, Any]]:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Unexpected OpenRouter stream event: {line!r}") from exc
    if isinstance(payload, dict):
        yield payload


def _stream_error(payload: dict[str, Any], resp: Any, started: float) -> OpenRouterHTTPError:
    err, message, full, code = payload.get("error"), "", "", 502
    if isinstance(err, dict):
        if isinstance(err.get("message"), str):
            full = err["message"]
            message = full[:_ERROR_BODY_LIMIT]
        raw_code = err.get("code")
        if isinstance(raw_code, int) and not isinstance(raw_code, bool) and 400 <= raw_code <= 599:
            code = raw_code
    elif isinstance(err, str):
        full = err
        message = full[:_ERROR_BODY_LIMIT]
    return OpenRouterHTTPError(
        code,
        message or "request failed",
        _elapsed(resp, started, use_response_elapsed=False),
        _num_opt(getattr(resp, "time_to_first_token_seconds", None)),
        retry_after=_parse_retry_after(resp),
        context_overflow=_is_overflow(code, full),
    )


def _apply_choice(
    acc: _StreamAcc,
    choice: dict[str, Any],
    started: float,
    measured: list[float],
    on_content_delta: Callable[[str], None] | None,
    on_thought_delta: Callable[[str], None] | None,
) -> None:
    def mark() -> None:
        if not measured:
            measured.append(max(0.0, time.perf_counter() - started))

    reason = choice.get("finish_reason")
    if not (isinstance(reason, str) and reason):
        reason = choice.get("native_finish_reason")
    if isinstance(reason, str) and reason:
        acc.finish_reason = reason

    for key in ("delta", "message"):
        piece = choice.get(key)
        if not isinstance(piece, dict):
            continue
        if isinstance(content := piece.get("content"), str) and content:
            acc.content.append(content)
            mark()
            if on_content_delta is not None:
                on_content_delta(content)

        emitted_thought = False
        for rkey, target in (
            ("reasoning", acc.reasoning),
            ("reasoning_content", acc.reasoning_content),
        ):
            if isinstance(val := piece.get(rkey), str) and val:
                target.append(val)
                mark()
                if not emitted_thought and on_thought_delta is not None:
                    on_thought_delta(val)
                    emitted_thought = True

        if isinstance(details := piece.get("reasoning_details"), list):
            for item in details:
                if isinstance(item, dict):
                    _merge_detail(acc, item)
                    mark()
                    if not emitted_thought and on_thought_delta is not None:
                        for dkey in ("text", "summary", "content"):
                            if isinstance(dval := item.get(dkey), str) and dval:
                                on_thought_delta(dval)
                                break

        if isinstance(calls := piece.get("tool_calls"), list):
            for idx, call in enumerate(calls):
                if isinstance(call, dict):
                    _merge_tool_call(acc, call, idx)
                    mark()


def _merge_detail(acc: _StreamAcc, item: dict[str, Any]) -> None:
    index = item.get("index")
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
        cur = slot.get(key)
        slot[key] = (
            cur + value
            if key in ("text", "summary", "content")
            and isinstance(value, str)
            and isinstance(cur, str)
            else value
        )


def _merge_tool_call(acc: _StreamAcc, call: dict[str, Any], fallback: int) -> None:
    raw = call.get("index", fallback)
    idx = raw if isinstance(raw, int) and not isinstance(raw, bool) else fallback
    slot = acc.calls.setdefault(idx, _ToolAcc())
    if isinstance(call.get("id"), str) and call["id"]:
        slot.id = call["id"]
    if not isinstance(fn := call.get("function"), dict):
        return
    if isinstance(fn.get("name"), str) and fn["name"]:
        slot.name = fn["name"]
    args = fn.get("arguments")
    if isinstance(args, str):
        slot.arguments += args
    elif isinstance(args, dict) and not slot.arguments:
        slot.arguments = json.dumps(args)


@dataclass
class CatalogFetch:
    models: list[CatalogModel]
    error: str | None = None


@dataclass
class _CatalogCacheEntry:
    key_id: str
    stored_at: float
    models: list[CatalogModel]


_catalog_cache: _CatalogCacheEntry | None = None
_catalog_cache_lock = threading.Lock()


def _catalog_now() -> float:
    return time.perf_counter()


def clear_model_catalog_cache() -> None:
    global _catalog_cache
    with _catalog_cache_lock:
        _catalog_cache = None


def peek_model_catalog(api_key: str) -> list[CatalogModel] | None:
    if not api_key.strip():
        return None
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    with _catalog_cache_lock:
        entry = _catalog_cache
        if (
            entry is None
            or entry.key_id != key_hash
            or _catalog_now() - entry.stored_at >= _CATALOG_CACHE_SECONDS
        ):
            return None
        return list(entry.models)


def filter_model_catalog(models: Sequence[CatalogModel], query: str) -> list[CatalogModel]:
    """Return catalog rows whose id or display name matches ``query``."""
    needle = query.strip().casefold()
    if not needle:
        return list(models)
    return [
        m
        for m in models
        if needle in m.id.casefold() or (m.name is not None and needle in m.name.casefold())
    ]


def fetch_model_catalog(client: Any, api_key: str, *, bypass_cache: bool = False) -> CatalogFetch:
    """GET the OpenRouter models endpoint, following safe same-host pagination."""
    global _catalog_cache
    if not api_key.strip():
        raise MissingOpenRouterKey(MISSING_KEY_MESSAGE)
    if not bypass_cache and (cached := peek_model_catalog(api_key)) is not None:
        return CatalogFetch(models=cached)

    started = _catalog_now()
    deadline = started + _CATALOG_TIME_LIMIT
    headers = {"Authorization": f"Bearer {api_key}"}
    url: str | None = OPENROUTER_MODELS_URL
    seen: set[str] = set()
    models: list[CatalogModel] = []
    pages, error = 0, None
    while url is not None and url not in seen and pages < _CATALOG_PAGE_LIMIT:
        remaining = deadline - _catalog_now()
        if remaining <= 0:
            error = "model catalog timed out"
            break
        seen.add(url)
        pages += 1
        try:
            resp = client.get(url, headers=headers, follow_redirects=False, timeout=remaining)
            raw_s = getattr(resp, "status_code", 200)
            status = raw_s if isinstance(raw_s, int) and not isinstance(raw_s, bool) else 200
            elapsed = _elapsed(resp, started, use_response_elapsed=True)
            ttft = _num_opt(getattr(resp, "time_to_first_token_seconds", None))
            if status < 200 or status >= 300:
                raise OpenRouterHTTPError(status, _short_error_body(resp), elapsed, ttft)
            payload = resp.json()
            if not isinstance(payload, dict):
                raise TypeError(f"Expected OpenRouter response object, got {type(payload)}")
            page = parse_model_catalog(payload)
        except Exception as exc:
            if models:
                error = (
                    str(exc)
                    if isinstance(exc, OpenRouterHTTPError | MissingOpenRouterKey)
                    else f"{type(exc).__name__}: {exc}"[:300]
                )
                break
            raise
        models.extend(page)
        url = _next_catalog_page(payload, url)

    kept: dict[str, CatalogModel] = {}
    for model in models:
        cur = kept.get(model.id)
        if cur is None or (model.supports_tools and not cur.supports_tools):
            kept[model.id] = model
    deduped = list(kept.values())
    if error is None:
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        with _catalog_cache_lock:
            _catalog_cache = _CatalogCacheEntry(key_hash, _catalog_now(), list(deduped))
    return CatalogFetch(models=deduped, error=error)


def _next_catalog_page(payload: dict[str, Any], current: str) -> str | None:
    links = payload.get("links")
    raw = links.get("next") if isinstance(links, dict) else None
    if not isinstance(raw, str) or not raw.strip():
        return None
    joined = urljoin(current, raw.strip())
    parsed = urlparse(joined)
    if (
        parsed.scheme != "https"
        or parsed.hostname is None
        or parsed.hostname.casefold() != _OPENROUTER_HOST
        or parsed.port not in (None, 443)
    ):
        return None
    return joined


def parse_usage(raw: object) -> TokenUsage:
    if not isinstance(raw, dict):
        return TokenUsage()
    prompt = int(_num_opt(raw.get("prompt_tokens"), allow_neg=True) or 0)
    completion = int(_num_opt(raw.get("completion_tokens"), allow_neg=True) or 0)
    total_raw = raw.get("total_tokens")
    total = (
        int(_num_opt(total_raw, allow_neg=True) or 0)
        if total_raw is not None
        else prompt + completion
    )
    p_det, c_det = raw.get("prompt_tokens_details"), raw.get("completion_tokens_details")
    cached_src = (
        p_det["cached_tokens"]
        if isinstance(p_det, dict) and "cached_tokens" in p_det
        else raw.get("cached_tokens", raw.get("cache_read_input_tokens", 0))
    )
    reason_src = (
        c_det["reasoning_tokens"]
        if isinstance(c_det, dict) and "reasoning_tokens" in c_det
        else raw.get("reasoning_tokens", 0)
    )
    return TokenUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        cached_tokens=int(_num_opt(cached_src) or 0),
        reasoning_tokens=int(_num_opt(reason_src) or 0),
        cost=_num_opt(raw.get("cost"), allow_neg=True),
    )


def parse_model_catalog(payload: object) -> list[CatalogModel]:
    if not isinstance(payload, dict):
        raise ValueError("model catalog must be an object")
    data = payload.get("data")
    if not isinstance(data, list):
        raise ValueError("model catalog is missing data")
    models: list[CatalogModel] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        raw_id = item.get("id")
        if (
            not isinstance(raw_id, str)
            or _CONTROL_CHARS.search(raw_id)
            or _BIDI_CHARS.search(raw_id)
        ):
            continue
        if not (model_id := raw_id.strip()):
            continue
        raw_pricing = item.get("pricing")
        pricing: dict[str, Any] = raw_pricing if isinstance(raw_pricing, dict) else {}
        ctx = _num_opt(item.get("context_length"))
        supported = item.get("supported_parameters")
        models.append(
            CatalogModel(
                id=model_id,
                name=_sanitize_display(item.get("name")),
                description=_sanitize_display(item.get("description")),
                context_length=int(ctx) if ctx is not None else None,
                prompt_price=_as_price(pricing.get("prompt")),
                completion_price=_as_price(pricing.get("completion")),
                cache_read_price=_as_price(pricing.get("input_cache_read")),
                supports_tools=isinstance(supported, list) and "tools" in supported,
            )
        )
    return models


def _sanitize_display(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return _BIDI_CHARS.sub("", _CONTROL_CHARS.sub("", _ANSI_ESCAPE.sub("", value))).strip() or None


def _elapsed(resp: Any, started: float, *, use_response_elapsed: bool) -> float:
    if (explicit := _num_opt(getattr(resp, "elapsed_seconds", None))) is not None:
        return explicit
    if use_response_elapsed:
        elapsed = getattr(resp, "elapsed", None)
        if isinstance(elapsed, timedelta):
            return max(0.0, elapsed.total_seconds())
        if (val := _num_opt(elapsed)) is not None:
            return val
    return max(0.0, time.perf_counter() - started)


def _short_error_body(resp: Any) -> str:
    try:
        payload = resp.json()
    except Exception:
        payload = None
    if isinstance(payload, dict):
        err = payload.get("error")
        msg = err.get("message") if isinstance(err, dict) else err
        if isinstance(msg, str) and msg:
            return msg[:_ERROR_BODY_LIMIT]
    text = getattr(resp, "text", "")
    return (
        text.strip()[:_ERROR_BODY_LIMIT]
        if isinstance(text, str) and text.strip()
        else "request failed"
    )


def _serialize_message(message: ChatMessage) -> dict[str, Any]:
    data: dict[str, Any] = {"role": message.role.value}
    images = [] if message.role is Role.TOOL else message.images
    if images:
        parts: list[dict[str, Any]] = []
        if message.content is not None:
            parts.append({"type": "text", "text": message.content})
        for img in images:
            parts.append({"type": "text", "text": f"Image artifact: {img.path}"})
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{img.media_type};base64,{img.data}"},
                }
            )
        data["content"] = parts
    elif message.content is not None:
        data["content"] = message.content
    elif message.role is Role.ASSISTANT and not message.tool_calls:
        data["content"] = ""
    if message.tool_call_id is not None:
        data["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        data["tool_calls"] = [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
            }
            for c in message.tool_calls
        ]
    if message.reasoning:
        data["reasoning"] = message.reasoning
    if message.reasoning_details:
        data["reasoning_details"] = message.reasoning_details
    return data


def _json_type_name(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int | float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    return type(value).__name__


def _decode_arguments(raw: object) -> tuple[dict[str, Any], str | None]:
    """Decode tool-call arguments to an object, or ({}, reason) when impossible."""
    if raw is None:
        return {}, None
    if isinstance(raw, dict):
        return raw, None
    value: object = raw
    for _ in range(2):
        if not isinstance(value, str):
            break
        if not value.strip():
            return {}, None
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            return {}, f"arguments were not valid JSON ({exc.msg})."
    if isinstance(value, dict):
        return value, None
    return {}, f"arguments must be a JSON object, got {_json_type_name(value)}."


def _parse_response(message: dict[str, Any]) -> ModelResponse:
    content = message.get("content")
    if content is not None and not isinstance(content, str):
        raise TypeError(f"Expected string or null content, got {type(content)}")
    raw_calls = message.get("tool_calls", [])
    if not isinstance(raw_calls, list):
        raise TypeError(f"Expected list tool_calls, got {type(raw_calls)}")
    calls: list[ToolCall] = []
    argument_errors: dict[str, str] = {}
    for raw_call in raw_calls:
        if not isinstance(raw_call, dict) or not isinstance(fn := raw_call.get("function"), dict):
            raise ValueError(f"Unexpected OpenRouter tool call: {raw_call!r}")
        call_id = raw_call.get("id")
        name = fn.get("name")
        if not isinstance(call_id, str) or not call_id or not isinstance(name, str) or not name:
            raise ValueError(f"Unexpected OpenRouter tool call: {raw_call!r}")
        arguments, problem = _decode_arguments(fn.get("arguments"))
        if problem is not None:
            argument_errors[call_id] = problem
        calls.append(ToolCall(id=call_id, name=name, arguments=arguments, arguments_error=problem))

    details, detail_parts = [], []
    if isinstance(raw_details := message.get("reasoning_details"), list):
        for item in raw_details:
            if isinstance(item, dict):
                details.append(item)
                for key in ("text", "summary", "content"):
                    if isinstance(v := item.get(key), str) and v.strip():
                        detail_parts.append(v.strip())
                        break
            elif isinstance(item, str) and item.strip():
                detail_parts.append(item.strip())
    reasoning = (
        (r.strip() or None if isinstance(r := message.get("reasoning"), str) else None)
        or (rc.strip() or None if isinstance(rc := message.get("reasoning_content"), str) else None)
        or ("\n\n".join(detail_parts) if detail_parts else None)
    )
    return ModelResponse(
        content=content,
        tool_calls=calls,
        reasoning=reasoning,
        reasoning_details=details,
        argument_errors=argument_errors,
    )


def _num_opt(value: object, *, allow_neg: bool = False) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if allow_neg or value >= 0 else None


def _as_price(value: object) -> str | None:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        return None
    return str(value).strip() or None
