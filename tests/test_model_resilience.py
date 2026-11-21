from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from atlas.agent import model as model_module
from atlas.agent.contracts import ChatMessage, Role, ToolCall
from atlas.agent.model import (
    ModelConfig,
    OpenRouterHTTPError,
    OpenRouterModel,
    _parse_response,
    _serialize_message,
)


class Resp:
    def __init__(
        self,
        payload: dict[str, Any] | None = None,
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        lines: list[str] | None = None,
        text: str = "",
    ) -> None:
        self.payload = payload or {}
        self.status_code = status_code
        self.headers = headers or {}
        self.lines = lines
        self.text = text
        self.elapsed_seconds = 0.0

    def json(self) -> dict[str, Any]:
        return self.payload

    def iter_lines(self) -> list[str]:
        if self.lines is not None:
            return self.lines
        return [f"data: {json.dumps(self.payload)}", "data: [DONE]"]


class Opened:
    def __init__(self, resp: Resp) -> None:
        self.resp = resp

    def __enter__(self) -> Resp:
        return self.resp

    def __exit__(self, *_exc: object) -> None:
        return None


class Client:
    def __init__(self, responses: list[Resp]) -> None:
        self.responses = responses
        self.calls = 0

    def stream(self, *_a: Any, **_k: Any) -> Opened:
        resp = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return Opened(resp)


def _model(responses: list[Resp]) -> tuple[OpenRouterModel, Client]:
    client = Client(responses)
    model = OpenRouterModel(
        ModelConfig(model="v/m", api_key="k"),
        client=client,  # type: ignore[arg-type]
    )
    return model, client


def _ok(content: str = "hi", finish: str | None = "stop") -> Resp:
    choice: dict[str, Any] = {"message": {"content": content}}
    if finish:
        choice["finish_reason"] = finish
    return Resp({"choices": [choice]})


def _user() -> list[ChatMessage]:
    return [ChatMessage(role=Role.USER, content="hi")]


def _call(arguments: Any) -> dict[str, Any]:
    return {"tool_calls": [{"id": "c1", "function": {"name": "t", "arguments": arguments}}]}


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    out: list[float] = []
    monkeypatch.setattr(model_module, "_sleep", out.append)
    monkeypatch.setattr(model_module, "_jitter", lambda: 1.0)
    return out


@pytest.mark.parametrize("raw", [None, "", "   \n"])
def test_empty_arguments_become_empty_object(raw: Any) -> None:
    parsed = _parse_response(_call(raw))
    assert parsed.tool_calls[0].arguments == {}
    assert parsed.argument_errors == {}


def test_valid_arguments_unchanged() -> None:
    parsed = _parse_response(_call('{"a": 1}'))
    assert parsed.tool_calls[0].arguments == {"a": 1}
    assert _parse_response(_call({"b": 2})).tool_calls[0].arguments == {"b": 2}
    assert parsed.argument_errors == {}


def test_double_encoded_arguments_are_decoded() -> None:
    parsed = _parse_response(_call(json.dumps(json.dumps({"a": 1}))))
    assert parsed.tool_calls[0].arguments == {"a": 1}
    assert parsed.argument_errors == {}


def test_truncated_json_is_recorded_not_raised() -> None:
    parsed = _parse_response(_call('{"path": "a.py", "con'))
    assert parsed.tool_calls[0].arguments == {}
    assert parsed.argument_errors["c1"].startswith("arguments were not valid JSON (")


@pytest.mark.parametrize(
    "raw,kind", [("[1]", "array"), ("3", "number"), ("true", "boolean"), ("null", "null")]
)
def test_non_object_arguments_are_recorded(raw: str, kind: str) -> None:
    parsed = _parse_response(_call(raw))
    assert parsed.argument_errors["c1"] == f"arguments must be a JSON object, got {kind}."


def test_call_without_id_or_name_still_raises() -> None:
    with pytest.raises(ValueError):
        _parse_response({"tool_calls": [{"function": {"name": "t", "arguments": "{}"}}]})
    with pytest.raises(ValueError):
        _parse_response({"tool_calls": [{"id": "x", "function": {"arguments": "{}"}}]})


def test_finish_reason_from_stream_prefers_last_non_null() -> None:
    lines = [
        'data: {"choices":[{"delta":{"content":"a"},"finish_reason":null}]}',
        'data: {"choices":[{"delta":{"content":"b"},"finish_reason":"length"}]}',
        "data: [DONE]",
    ]
    model, _ = _model([Resp(lines=lines)])
    response = model.complete(_user(), [])
    assert response.content == "ab"
    assert response.finish_reason == "length"


def test_finish_reason_falls_back_to_native() -> None:
    lines = [
        'data: {"choices":[{"delta":{"content":"a"},"native_finish_reason":"STOP"}]}',
        "data: [DONE]",
    ]
    model, _ = _model([Resp(lines=lines)])
    assert model.complete(_user(), []).finish_reason == "STOP"


def test_finish_reason_from_plain_body_and_absent() -> None:
    model, _ = _model([_ok("x", "tool_calls")])
    assert model.complete(_user(), []).finish_reason == "tool_calls"
    model, _ = _model([_ok("x", None)])
    assert model.complete(_user(), []).finish_reason is None


def test_error_event_with_choices_raises(sleeps: list[float]) -> None:
    lines = [
        'data: {"choices":[{"delta":{"content":"half"}}],'
        '"error":{"code":400,"message":"provider broke"}}',
        "data: [DONE]",
    ]
    model, _ = _model([Resp(lines=lines)])
    with pytest.raises(OpenRouterHTTPError) as info:
        model.complete(_user(), [])
    assert info.value.status_code == 400


def test_finish_reason_error_raises_and_retryable_code_is_retried(sleeps: list[float]) -> None:
    lines = [
        'data: {"choices":[{"delta":{"content":"x"},"finish_reason":"error",'
        '"error":{"code":503,"message":"upstream"}}]}',
        "data: [DONE]",
    ]
    model, client = _model([Resp(lines=lines), _ok("fine")])
    assert model.complete(_user(), []).content == "fine"
    assert client.calls == 2


def test_finish_reason_error_without_detail_raises(sleeps: list[float]) -> None:
    lines = ['data: {"choices":[{"delta":{"content":"x"},"finish_reason":"error"}]}']
    model, _ = _model([Resp(lines=lines)])
    with pytest.raises(OpenRouterHTTPError):
        model.complete(_user(), [])


def _http(status: int, message: str, headers: dict[str, str] | None = None) -> Resp:
    return Resp({"error": {"message": message}}, status_code=status, headers=headers)


def test_context_overflow_classified_from_full_text(sleeps: list[float]) -> None:
    message = "x" * 300 + " This model's maximum context length is 1000 tokens."
    model, client = _model([_http(400, message)])
    with pytest.raises(OpenRouterHTTPError) as info:
        model.complete(_user(), [])
    assert info.value.is_context_overflow is True
    assert len(info.value.body) == 180
    assert "context" not in info.value.body
    assert client.calls == 1
    assert sleeps == []


def test_not_overflow_for_unrelated_400_and_429(sleeps: list[float]) -> None:
    for status, text in ((400, "bad schema"), (429, "context length is fine here")):
        model, _ = _model([_http(status, text)])
        with pytest.raises(OpenRouterHTTPError) as info:
            model.complete(_user(), [])
        assert info.value.is_context_overflow is False


def test_stream_error_overflow() -> None:
    lines = ['data: {"error":{"code":400,"message":"prompt is too long"}}']
    model, _ = _model([Resp(lines=lines)])
    with pytest.raises(OpenRouterHTTPError) as info:
        model.complete(_user(), [])
    assert info.value.is_context_overflow is True


def test_retry_after_honoured_and_capped(sleeps: list[float]) -> None:
    model, _ = _model([_http(429, "slow", {"Retry-After": "7"}), _ok()])
    model.complete(_user(), [])
    assert sleeps == [7.0]
    sleeps.clear()
    model, _ = _model([_http(503, "slow", {"retry-after": "500"}), _ok()])
    model.complete(_user(), [])
    assert sleeps == [30.0]


def test_non_numeric_retry_after_uses_backoff(sleeps: list[float]) -> None:
    model, _ = _model([_http(429, "slow", {"Retry-After": "Wed, 21 Oct"}), _ok()])
    model.complete(_user(), [])
    assert sleeps == [0.5]


def test_jittered_backoff_and_four_attempts(
    sleeps: list[float], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(model_module, "_jitter", lambda: 1.2)
    model, client = _model([_http(503, "down")])
    with pytest.raises(OpenRouterHTTPError):
        model.complete(_user(), [])
    assert client.calls == 4
    assert sleeps == pytest.approx([0.6, 1.2, 2.4])


@pytest.mark.parametrize("status", [400, 401, 403])
def test_final_statuses_are_not_retried(status: int, sleeps: list[float]) -> None:
    model, client = _model([_http(status, "no")])
    with pytest.raises(OpenRouterHTTPError):
        model.complete(_user(), [])
    assert client.calls == 1
    assert sleeps == []


def test_jitter_range() -> None:
    assert all(0.75 <= model_module._jitter() <= 1.25 for _ in range(50))


def test_empty_assistant_message_serializes_content_blank() -> None:
    data = _serialize_message(ChatMessage(role=Role.ASSISTANT))
    assert data == {"role": "assistant", "content": ""}
    with_calls = _serialize_message(
        ChatMessage(role=Role.ASSISTANT, tool_calls=[ToolCall(id="a", name="t", arguments={})])
    )
    assert "content" not in with_calls
    assert _serialize_message(ChatMessage(role=Role.ASSISTANT, content="x"))["content"] == "x"


def test_retry_wait_goes_through_the_sleep_hook(sleeps: list[float]) -> None:
    model, client = _model([_http(503, "down"), _http(503, "down"), _ok()])

    model.complete(_user(), [])

    assert client.calls == 3
    assert sleeps == [0.5, 1.0]


@pytest.mark.parametrize(
    "message",
    [
        "Your context limit was reached",
        "exceeds the context size of this model",
        "Requests exceed context for this model",
        "Prompt exceeds the maximum number of tokens allowed",
        "The sum of input length and max_tokens is too high",
        "Please reduce your input and try again",
        "The token count of your prompt (9000) exceeds the limit of 8000",
        "input of 300000 exceeds what this model accepts",
        "Input tokens exceed the configured limit",
    ],
)
@pytest.mark.parametrize("status", [400, 413, 422])
def test_more_overflow_wordings_are_classified(
    message: str, status: int, sleeps: list[float]
) -> None:
    model, client = _model([_http(status, message)])
    with pytest.raises(OpenRouterHTTPError) as info:
        model.complete(_user(), [])
    assert info.value.is_context_overflow is True
    assert client.calls == 1


@pytest.mark.parametrize(
    "status,message",
    [
        (400, "max_tokens must be a positive integer"),
        (400, "Unknown parameter: 'foo'"),
        (413, "request body too large"),
        (400, "The input is fine. Something else would exceed a quota"),
        (429, "token count exceeds the rate limit"),
    ],
)
def test_unrelated_errors_are_not_overflow(status: int, message: str, sleeps: list[float]) -> None:
    model, _ = _model([_http(status, message)])
    with pytest.raises(OpenRouterHTTPError) as info:
        model.complete(_user(), [])
    assert info.value.is_context_overflow is False


@pytest.mark.parametrize("raw", ["0", "0.1", "0.249"])
def test_tiny_retry_after_falls_back_to_the_backoff_delay(raw: str, sleeps: list[float]) -> None:
    model, _ = _model([_http(429, "slow", {"Retry-After": raw}), _ok()])

    model.complete(_user(), [])

    assert sleeps == [0.5]


def test_finish_reason_is_recorded_on_the_call(tmp_path: Path) -> None:
    from atlas.agent.artifacts import LocalArtifactStore
    from atlas.agent.contracts import ModelResponse, TokenUsage, default_registry
    from atlas.agent.runtime import Agent

    class _Model:
        def complete(self, messages: object, tools: object) -> ModelResponse:
            return ModelResponse(
                content="done",
                usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                finish_reason="length",
            )

    agent = Agent(_Model(), default_registry(), LocalArtifactStore(tmp_path))
    run = agent.run("hi")
    assert run.calls[0].finish_reason == "length"
