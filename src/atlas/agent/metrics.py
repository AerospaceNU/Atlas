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
    """Output tokens divided by that model request's duration.

    ``elapsed_seconds`` is the completion HTTP call only. Tool execution
    between calls is not part of it. A call with no output tokens does not
    produce a rate.
    """
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
        speed = tokens_per_second(usage.completion_tokens, timing.elapsed_seconds)
        if speed is not None:
            totals.tokens_per_second = speed
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


def format_token_count(value: int) -> str:
    """Three significant figures, in K, M, or B once the count reaches that unit.

    ``355`` stays ``355``, ``128000`` is ``128K``, ``1048576`` is ``1.05M``,
    and ``1500000000`` is ``1.50B``.
    """
    if value <= 0:
        return "0"
    rounded = _round_sig3(value)
    if rounded >= 1_000_000_000:
        return f"{_sig_coeff(rounded, 1_000_000_000)}B"
    if rounded >= 1_000_000:
        return f"{_sig_coeff(rounded, 1_000_000)}M"
    if rounded >= 1_000:
        return f"{_sig_coeff(rounded, 1_000)}K"
    return str(rounded)


def format_tui_status(totals: SessionTotals) -> str:
    """One status line: model, context window, tokens, spend, and output speed."""
    limit = "unknown" if totals.context_limit is None else format_token_count(totals.context_limit)
    model = totals.model or "unset"
    spend = "n/a" if totals.cost is None else _format_spend(totals.cost)
    speed = "n/a" if totals.tokens_per_second is None else f"{totals.tokens_per_second:.1f}"
    return (
        f"model {model}  context {format_token_count(totals.context_used)}/{limit}  "
        f"tokens {format_token_count(totals.total_tokens)}  "
        f"spend {spend}  {speed} tok/s"
    )


def _round_sig3(value: int) -> int:
    digits = _decimal_digits(value)
    if digits <= 3:
        return value
    factor = _pow10(digits - 3)
    head, rem = divmod(value, factor)
    rounded = head + int(rem * 2 >= factor)
    if rounded >= 1000:
        rounded //= 10
        factor *= 10
    return rounded * factor


def _pow10(exponent: int) -> int:
    result = 1
    for _ in range(exponent):
        result *= 10
    return result


def _decimal_digits(value: int) -> int:
    digits = 1
    while value >= 10:
        value //= 10
        digits += 1
    return digits


def _sig_coeff(rounded: int, unit: int) -> str:
    whole, frac = divmod(rounded, unit)
    if whole >= 100:
        return str(whole)
    if whole >= 10:
        return f"{whole}.{frac // (unit // 10)}"
    return f"{whole}.{frac // (unit // 100):02d}"


def _format_spend(cost: float) -> str:
    text = f"{cost:.6f}".rstrip("0").rstrip(".")
    return f"${text}"


def _recent_label(timing: CallTiming) -> str:
    return f"{timing.http_status} {timing.elapsed_seconds:.2f}s"
