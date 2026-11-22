"""Session totals derived from real completion usage and timing."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from pydantic import BaseModel, Field

from atlas.agent.contracts import CallTiming, ModelCall, ModelResponse, TokenUsage

_RECENT_LIMIT = 8
_MIN_GENERATION_SECONDS = 0.05
_PRICE_RE = re.compile(r"\+?(?:\d+\.?\d*|\.\d+)")


class SessionTotals(BaseModel):
    """Running totals tracked across a session."""

    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    cost: Decimal | None = None
    spend_known: bool = True
    estimated: bool = False
    context_used: int = 0
    context_limit: int | None = None
    last_http_status: int | None = None
    last_elapsed_seconds: float | None = None
    last_ttft_seconds: float | None = None
    tokens_per_second: float | None = None
    recent: list[str] = Field(default_factory=list)


def observe_model_call(response: ModelResponse) -> ModelCall | None:
    """Return a call record when the response carried usage or timing."""
    u = response.usage
    has_usage = any(
        (
            u.prompt_tokens,
            u.completion_tokens,
            u.total_tokens,
            u.cached_tokens,
            u.reasoning_tokens,
            u.cost is not None,
        )
    )
    if response.timing is None and not has_usage:
        return None
    return ModelCall(usage=u, timing=response.timing, finish_reason=response.finish_reason)


def tokens_per_second(completion_tokens: int, generation_seconds: float) -> float | None:
    """Output tokens divided by generation time after first token."""
    if completion_tokens <= 0 or generation_seconds <= _MIN_GENERATION_SECONDS:
        return None
    return completion_tokens / generation_seconds


def output_tokens_per_second(completion_tokens: int, timing: CallTiming | None) -> float | None:
    """Latest-call output speed measured after TTFT."""
    if timing is None or timing.time_to_first_token_seconds is None:
        return None
    return tokens_per_second(
        completion_tokens, timing.elapsed_seconds - timing.time_to_first_token_seconds
    )


def catalog_spend(
    prompt_tokens: int,
    completion_tokens: int,
    prompt_price: str | None,
    completion_price: str | None,
    *,
    cached_tokens: int = 0,
    cache_read_price: str | None = None,
) -> Decimal | None:
    """Price one call from catalog per-token rates."""
    prompt = _price_decimal(prompt_price)
    completion = _price_decimal(completion_price)
    if prompt is None or completion is None:
        return None
    cache = _price_decimal(cache_read_price)
    if cache is None:
        return (prompt * prompt_tokens) + (completion * completion_tokens)
    cached = max(0, cached_tokens)
    uncached = max(0, prompt_tokens - cached)
    return (prompt * uncached) + (cache * cached) + (completion * completion_tokens)


def reported_spend(totals: SessionTotals) -> float | None:
    """Dollars to publish, or None when any token-using call was unpriced."""
    if not totals.spend_known:
        return None
    amount = _decimal_cost(totals.cost)
    return 0.0 if amount is None else float(amount)


def apply_call(
    totals: SessionTotals,
    call: ModelCall,
    *,
    prompt_price: str | None = None,
    completion_price: str | None = None,
    cache_read_price: str | None = None,
) -> SessionTotals:
    """Fold one completion into ``totals`` and return the same object."""
    usage = call.usage
    totals.prompt_tokens += usage.prompt_tokens
    totals.completion_tokens += usage.completion_tokens
    totals.total_tokens += usage.total_tokens
    totals.cached_tokens += usage.cached_tokens
    totals.reasoning_tokens += usage.reasoning_tokens
    _note_spend(totals, usage, prompt_price, completion_price, cache_read_price)
    if usage.prompt_tokens > 0:
        totals.context_used = usage.prompt_tokens
    timing = call.timing
    if timing is not None:
        totals.last_http_status = timing.http_status
        totals.last_elapsed_seconds = timing.elapsed_seconds
        totals.last_ttft_seconds = timing.time_to_first_token_seconds
        totals.recent.append(f"{timing.http_status} {timing.elapsed_seconds:.2f}s")
        del totals.recent[:-_RECENT_LIMIT]
        speed = output_tokens_per_second(usage.completion_tokens, timing)
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
    """Format a token count to three significant figures with K/M/B suffix."""
    if value <= 0:
        return "0"
    digits = len(str(value))
    if digits <= 3:
        rounded = value
    else:
        factor = 10 ** (digits - 3)
        head, rem = divmod(value, factor)
        rounded = head + int(rem * 2 >= factor)
        if rounded >= 1000:
            rounded //= 10
            factor *= 10
        rounded *= factor
    for unit, suffix in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")):
        if rounded >= unit:
            whole, frac = divmod(rounded, unit)
            if whole >= 100:
                return f"{whole}{suffix}"
            if whole >= 10:
                return f"{whole}.{frac // (unit // 10)}{suffix}"
            return f"{whole}.{frac // (unit // 100):02d}{suffix}"
    return str(rounded)


def format_tui_status(totals: SessionTotals) -> str:
    """Format one compact status summary line for diagnostics and tests."""
    limit = "unknown" if totals.context_limit is None else format_token_count(totals.context_limit)
    model = totals.model or "unset"
    if not totals.spend_known:
        spend = "unknown"
    else:
        amount = _decimal_cost(totals.cost)
        if amount is None:
            spend = "$0"
        else:
            raw = format(amount, "f")
            if "." in raw:
                raw = raw.rstrip("0").rstrip(".")
            raw = "0" if raw in {"", "-0"} else raw
            spend = f"~${raw}" if totals.estimated else f"${raw}"
    speed = (
        "last n/a tok/s"
        if totals.tokens_per_second is None
        else f"last {totals.tokens_per_second:.1f} tok/s"
    )
    return (
        f"model {model}  context {format_token_count(totals.context_used)}/{limit}  "
        f"spend {spend}  {speed}"
    )


def _note_spend(
    totals: SessionTotals,
    usage: TokenUsage,
    prompt_price: str | None,
    completion_price: str | None,
    cache_read_price: str | None,
) -> None:
    if usage.cost is not None:
        current = _decimal_cost(totals.cost)
        amount = Decimal(str(usage.cost))
        totals.cost = amount if current is None else current + amount
        return
    if not (usage.prompt_tokens > 0 or usage.completion_tokens > 0 or usage.total_tokens > 0):
        return
    estimate = catalog_spend(
        usage.prompt_tokens,
        usage.completion_tokens,
        prompt_price,
        completion_price,
        cached_tokens=usage.cached_tokens,
        cache_read_price=cache_read_price,
    )
    if estimate is None or (usage.prompt_tokens == 0 and usage.completion_tokens == 0):
        totals.spend_known = False
        return
    totals.estimated = True
    current = _decimal_cost(totals.cost)
    totals.cost = estimate if current is None else current + estimate


def _decimal_cost(value: object) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, Decimal):
        return value
    if not isinstance(value, int | float | str):
        return None
    try:
        return Decimal(str(value))
    except InvalidOperation:
        return None


def _price_decimal(value: str | None) -> Decimal | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not _PRICE_RE.fullmatch(text):
        return None
    try:
        number = Decimal(text)
    except InvalidOperation:
        return None
    return number if number.is_finite() and number >= 0 else None
