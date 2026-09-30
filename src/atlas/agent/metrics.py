"""Session totals derived from real completion usage and timing."""

from __future__ import annotations

from pydantic import BaseModel, Field

from atlas.agent.contracts import CallTiming, ModelCall, ModelResponse, TokenUsage

_RECENT_LIMIT = 8


class SessionTotals(BaseModel):
    """Running totals the TUI status line renders.

    ``context_limit`` stays unset until a catalog row supplies it. Nothing in
    this module substitutes a hardcoded window.
    """

    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float | None = None
    context_used: int = 0
    context_limit: int | None = None
    last_http_status: int | None = None
    last_elapsed_seconds: float | None = None
    last_ttft_seconds: float | None = None
    tokens_per_second: float | None = None
    recent: list[str] = Field(default_factory=list)


def observe_model_call(response: ModelResponse) -> ModelCall | None:
    """Return a call record when the response carried usage or timing."""
    usage = response.usage
    has_usage = (
        usage.prompt_tokens != 0
        or usage.completion_tokens != 0
        or usage.total_tokens != 0
        or usage.cost is not None
    )
    if response.timing is None and not has_usage:
        return None
    return ModelCall(usage=usage, timing=response.timing)


def tokens_per_second(completion_tokens: int, elapsed_seconds: float) -> float | None:
    """Completion tokens divided by the measured elapsed seconds."""
    if completion_tokens <= 0 or elapsed_seconds <= 0:
        return None
    return completion_tokens / elapsed_seconds


def apply_call(totals: SessionTotals, call: ModelCall) -> SessionTotals:
    """Fold one completion into ``totals`` and return the same object."""
    usage = call.usage
    totals.prompt_tokens += usage.prompt_tokens
    totals.completion_tokens += usage.completion_tokens
    totals.total_tokens += usage.total_tokens
    if usage.cost is not None:
        totals.cost = (totals.cost or 0.0) + usage.cost
    if usage.prompt_tokens > 0:
        totals.context_used = usage.prompt_tokens
    timing = call.timing
    if timing is not None:
        totals.last_http_status = timing.http_status
        totals.last_elapsed_seconds = timing.elapsed_seconds
        totals.last_ttft_seconds = timing.time_to_first_token_seconds
        totals.recent.append(_recent_label(timing))
        del totals.recent[:-_RECENT_LIMIT]
        totals.tokens_per_second = tokens_per_second(
            usage.completion_tokens, timing.elapsed_seconds
        )
    return totals


def apply_http_error(
    totals: SessionTotals,
    *,
    http_status: int,
    elapsed_seconds: float,
    time_to_first_token_seconds: float | None,
) -> SessionTotals:
    """Record a failed call without adding token totals."""
    timing = CallTiming(
        http_status=http_status,
        elapsed_seconds=elapsed_seconds,
        time_to_first_token_seconds=time_to_first_token_seconds,
    )
    return apply_call(totals, ModelCall(usage=TokenUsage(), timing=timing))


def format_tui_status(totals: SessionTotals) -> str:
    """One status line: context, tokens, spend, speed, and the latest HTTP call."""
    limit = "unknown" if totals.context_limit is None else str(totals.context_limit)
    model = totals.model or "unset"
    spend = "n/a" if totals.cost is None else _format_spend(totals.cost)
    speed = "n/a" if totals.tokens_per_second is None else f"{totals.tokens_per_second:.1f}"
    http = "—" if totals.last_http_status is None else str(totals.last_http_status)
    latency = (
        "n/a" if totals.last_elapsed_seconds is None else f"{totals.last_elapsed_seconds:.2f}s"
    )
    ttft = "n/a" if totals.last_ttft_seconds is None else f"{totals.last_ttft_seconds:.2f}s"
    recent = ", ".join(totals.recent) if totals.recent else "none"
    return (
        f"model {model}  context {totals.context_used}/{limit}  "
        f"tokens {totals.total_tokens} (prompt {totals.prompt_tokens} "
        f"completion {totals.completion_tokens})  "
        f"spend {spend}  {speed} tok/s  http {http}  latency {latency}  "
        f"ttft {ttft}  recent {recent}"
    )


def _format_spend(cost: float) -> str:
    text = f"{cost:.6f}".rstrip("0").rstrip(".")
    return f"${text}"


def _recent_label(timing: CallTiming) -> str:
    return f"{timing.http_status} {timing.elapsed_seconds:.2f}s"
