"""Session totals derived from real completion usage and timing."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

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
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    cost: Decimal | None = None
    # False once any call used tokens but had neither a provider cost nor
    # catalog prices. The status line then says ``spend unknown``. A session
    # with no token-using calls stays known and renders ``$0``.
    spend_known: bool = True
    # True when any of the running total came from catalog prices rather
    # than a provider-reported cost. The status line then prefixes ``~``.
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
    usage = response.usage
    has_usage = (
        usage.prompt_tokens != 0
        or usage.completion_tokens != 0
        or usage.total_tokens != 0
        or usage.cached_tokens != 0
        or usage.reasoning_tokens != 0
        or usage.cost is not None
    )
    if response.timing is None and not has_usage:
        return None
    return ModelCall(usage=usage, timing=response.timing)


def tokens_per_second(completion_tokens: int, generation_seconds: float) -> float | None:
    """Output tokens divided by time after the first token.

    ``generation_seconds`` excludes time to first token and any tool
    execution between calls. A call with no output tokens does not produce
    a rate.
    """
    if completion_tokens <= 0 or generation_seconds <= 0:
        return None
    return completion_tokens / generation_seconds


def output_tokens_per_second(completion_tokens: int, timing: CallTiming | None) -> float | None:
    """Latest-call output speed, measured from the first token.

    The denominator is ``elapsed_seconds - time_to_first_token_seconds``.
    When the transport did not record that instant, the rate is left unset
    so the whole HTTP wait is not reported as generation speed. ``None``
    means a previously measured rate should stay on the status line.
    """
    if timing is None or timing.time_to_first_token_seconds is None:
        return None
    window = timing.elapsed_seconds - timing.time_to_first_token_seconds
    return tokens_per_second(completion_tokens, window)


def catalog_spend(
    prompt_tokens: int,
    completion_tokens: int,
    prompt_price: str | None,
    completion_price: str | None,
    *,
    cached_tokens: int = 0,
    cache_read_price: str | None = None,
) -> Decimal | None:
    """Price one call from catalog per-token rates.

    ``prompt_price`` and ``completion_price`` are dollars per token. The
    product is :class:`~decimal.Decimal` times the token counts. A missing,
    negative, or non-decimal prompt or completion rate leaves the call
    unpriced.

    Cached tokens use ``cache_read_price`` when the catalog row has one
    (OpenRouter ``pricing.input_cache_read``). The remainder of
    ``prompt_tokens`` stays at the prompt rate. When the catalog has no
    cache-read price, cached tokens are priced at the prompt rate, because
    ``prompt_tokens`` already includes them.
    """
    prompt = _price_decimal(prompt_price)
    completion = _price_decimal(completion_price)
    if prompt is None or completion is None:
        return None
    cache = _price_decimal(cache_read_price)
    if cache is None:
        # No cache-read price on this catalog row. Cached tokens are priced
        # at the prompt rate.
        return (prompt * prompt_tokens) + (completion * completion_tokens)
    cached = cached_tokens if cached_tokens > 0 else 0
    uncached = prompt_tokens - cached
    if uncached < 0:
        uncached = 0
    return (prompt * uncached) + (cache * cached) + (completion * completion_tokens)


def reported_spend(totals: SessionTotals) -> float | None:
    """Dollars to publish, or None when the session total is incomplete.

    A fresh session with no token-using calls reports ``0``. That is a known
    zero, not an unknown provider cost.
    """
    if not totals.spend_known:
        return None
    amount = _decimal_cost(totals.cost)
    if amount is None:
        return 0.0
    return float(amount)


def apply_call(
    totals: SessionTotals,
    call: ModelCall,
    *,
    prompt_price: str | None = None,
    completion_price: str | None = None,
    cache_read_price: str | None = None,
) -> SessionTotals:
    """Fold one completion into ``totals`` and return the same object.

    A provider ``cost`` is added as reported. When that field is missing and
    the call used tokens, the catalog per-token prices are applied and
    ``estimated`` is set so the status line can prefix ``~``. If those rates
    are missing too, the session spend becomes unknown. A session that has
    not used tokens stays at ``$0``. Output speed is the latest call that
    reported time to first token; a later call without that instant does not
    replace it.
    """
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
        totals.recent.append(_recent_label(timing))
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
    """One status line: model, context window, tokens, spend, and output speed.

    Prompt and completion counts are written beside the total once either is
    non-zero. Cached and reasoning counts are added only when they are
    non-zero. An empty session leaves those words off so the first frame
    matches the TUI default line. A session with no token-using calls shows
    ``spend $0``. An unpriced call shows ``spend unknown``. A catalog
    estimate shows ``spend ~$...`` so it is not mistaken for a provider
    cost. The speed is the latest call, labeled ``last``, and is ``n/a``
    until one is measured from the first token.
    """
    limit = "unknown" if totals.context_limit is None else format_token_count(totals.context_limit)
    model = totals.model or "unset"
    return (
        f"model {model}  context {format_token_count(totals.context_used)}/{limit}  "
        f"tokens {_token_field(totals)}  "
        f"spend {_spend_label(totals)}  {_speed_label(totals.tokens_per_second)}"
    )


def _token_field(totals: SessionTotals) -> str:
    total = format_token_count(totals.total_tokens)
    details: list[str] = []
    if totals.prompt_tokens or totals.completion_tokens:
        details.append(f"prompt {format_token_count(totals.prompt_tokens)}")
        details.append(f"completion {format_token_count(totals.completion_tokens)}")
    if totals.cached_tokens:
        details.append(f"cached {format_token_count(totals.cached_tokens)}")
    if totals.reasoning_tokens:
        details.append(f"reasoning {format_token_count(totals.reasoning_tokens)}")
    if not details:
        return total
    return f"{total} ({' '.join(details)})"


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


def _note_spend(
    totals: SessionTotals,
    usage: TokenUsage,
    prompt_price: str | None,
    completion_price: str | None,
    cache_read_price: str | None,
) -> None:
    if usage.cost is not None:
        _add_spend(totals, Decimal(str(usage.cost)))
        return
    used_tokens = usage.prompt_tokens > 0 or usage.completion_tokens > 0 or usage.total_tokens > 0
    if not used_tokens:
        return
    estimate = catalog_spend(
        usage.prompt_tokens,
        usage.completion_tokens,
        prompt_price,
        completion_price,
        cached_tokens=usage.cached_tokens,
        cache_read_price=cache_read_price,
    )
    # A total with no prompt or completion split cannot be priced from the
    # two catalog rates, so a zero product would still be a guess.
    if estimate is None or (usage.prompt_tokens == 0 and usage.completion_tokens == 0):
        totals.spend_known = False
        return
    totals.estimated = True
    _add_spend(totals, estimate)


def _add_spend(totals: SessionTotals, amount: Decimal) -> None:
    current = _decimal_cost(totals.cost)
    totals.cost = amount if current is None else current + amount


def _decimal_cost(value: object) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        return Decimal(str(value))
    except InvalidOperation:
        return None


def _price_decimal(value: str | None) -> Decimal | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not re.fullmatch(r"\+?(?:\d+\.?\d*|\.\d+)", text):
        return None
    try:
        number = Decimal(text)
    except InvalidOperation:
        return None
    if not number.is_finite() or number < 0:
        return None
    return number


def _spend_label(totals: SessionTotals) -> str:
    if not totals.spend_known:
        return "unknown"
    amount = _decimal_cost(totals.cost)
    if amount is None:
        return "$0"
    text = _format_spend(amount)
    if totals.estimated:
        return f"~{text}"
    return text


def _speed_label(speed: float | None) -> str:
    if speed is None:
        return "last n/a tok/s"
    return f"last {speed:.1f} tok/s"


def _format_spend(cost: Decimal) -> str:
    text = format(cost, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if text in {"", "-0"}:
        text = "0"
    return f"${text}"


def _recent_label(timing: CallTiming) -> str:
    return f"{timing.http_status} {timing.elapsed_seconds:.2f}s"
