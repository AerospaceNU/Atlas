from __future__ import annotations

import asyncio
import math
import random
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

from atlas.compile.product import SceneCatalog, SourceScenes
from atlas.data.base import DataPullClient, PullRequest, PullResult, Scene

# One search plus two retries. A 429 after that is recorded like any other failure.
_MAX_SEARCH_ATTEMPTS = 3
# Used when the response has no usable Retry-After. Attempt 1 waits 0.5s, then 1s.
_BACKOFF_SECONDS = 0.5
# Spread that wait by +/- 20% so sources do not retry in lockstep.
_JITTER_FRACTION = 0.2
# A huge or hostile Retry-After must not stall the catalog.
_MAX_BACKOFF_SECONDS = 60.0
# Wall-clock cap for one source, including retries. Others are not cancelled.
_SOURCE_BUDGET_SECONDS = 60.0


def _status_code(exc: BaseException) -> int | None:
    status = _as_status(getattr(exc, "status_code", None))
    if status is not None:
        return status
    response = getattr(exc, "response", None)
    return _as_status(getattr(response, "status_code", None))


def _as_status(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _is_rate_limited(exc: BaseException) -> bool:
    return _status_code(exc) == 429


def _apply_jitter(delay: float) -> float:
    """Return ``delay`` shifted by a fraction of itself, still inside the cap."""
    if delay <= 0.0 or not math.isfinite(delay):
        return 0.0
    span = delay * _JITTER_FRACTION
    jittered = delay + random.uniform(-span, span)
    if not math.isfinite(jittered):
        return min(delay, _MAX_BACKOFF_SECONDS)
    return min(_MAX_BACKOFF_SECONDS, max(0.0, jittered))


def _retry_delay(exc: BaseException, attempt: int) -> float:
    """Return the seconds to wait before the next search attempt.

    ``attempt`` is the 1-based number of the attempt that just failed.
    A parsed finite ``Retry-After`` is used as given, capped. The computed
    backoff is jittered.
    """
    header = _retry_after_seconds(exc)
    if header is None:
        return _apply_jitter(min(_BACKOFF_SECONDS * attempt, _MAX_BACKOFF_SECONDS))
    return min(header, _MAX_BACKOFF_SECONDS)


def _retry_after_seconds(exc: BaseException) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    getter = getattr(headers, "get", None)
    if not callable(getter):
        return None
    raw = getter("Retry-After")
    if raw is None:
        raw = getter("retry-after")
    if not isinstance(raw, str):
        return None
    return _parse_retry_after(raw.strip())


def _parse_retry_after(value: str) -> float | None:
    if not value:
        return None
    try:
        parsed = float(value)
    except ValueError:
        parsed = None
    if parsed is not None:
        # inf and nan would sleep forever or raise inside asyncio.sleep.
        if not math.isfinite(parsed):
            return None
        return max(0.0, parsed)
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError, OverflowError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - datetime.now(UTC)).total_seconds())


def _contains_url(text: str) -> bool:
    return "://" in text


def _reason_phrase(response: object) -> str | None:
    reason = getattr(response, "reason_phrase", None)
    if not isinstance(reason, str):
        return None
    text = reason.strip()
    if not text or _contains_url(text):
        return None
    return text


def _error_text(exc: Exception) -> str:
    """Return a status and reason, never the request URL or query string.

    ``str`` of an HTTP error includes the full URL, which can carry a key.
    """
    status = _status_code(exc)
    response = getattr(exc, "response", None)
    reason = _reason_phrase(response) if response is not None else None
    if status is not None and reason is not None:
        return f"{status} {reason}"
    if status is not None:
        return str(status)
    if response is not None or getattr(exc, "request", None) is not None:
        return type(exc).__name__
    message = str(exc)
    if _contains_url(message):
        return type(exc).__name__
    return f"{type(exc).__name__}: {message}"


_CURSOR_CLEARED = (
    "The stored next-page cursor was rejected and cleared. "
    "A pull without --resume starts at page one."
)


def _client_meta(client: DataPullClient) -> tuple[str, str]:
    collection = getattr(client, "collection", "")
    search_url = getattr(client, "search_url", "")
    if not isinstance(collection, str):
        collection = ""
    if not isinstance(search_url, str):
        search_url = ""
    return collection, search_url


def _cursor_was_rejected(exc: BaseException) -> bool:
    """True when the stored cursor itself failed and must not be tried again.

    HTTP 408 and 429 are left in place so a later resume can retry. Other
    4xx statuses, and a cursor this process refuses to send, are dead.
    """
    status = _status_code(exc)
    if status is not None:
        return 400 <= status <= 499 and status not in {408, 429}
    return isinstance(exc, ValueError)


def _stamp_source(name: str, collection: str, search_url: str, result: PullResult) -> SourceScenes:
    stamped: list[Scene] = []
    for scene in result.scenes:
        updates: dict[str, str] = {}
        if not scene.source:
            updates["source"] = name
        if not scene.collection and collection:
            updates["collection"] = collection
        stamped.append(scene.model_copy(update=updates) if updates else scene)
    return SourceScenes(
        source=name,
        collection=collection,
        scenes=stamped,
        next_cursor=result.next_cursor,
        next_cursor_unavailable=result.next_cursor_unavailable,
        search_url=search_url,
    )


def _failed_source(
    name: str,
    collection: str,
    search_url: str,
    page_cursor: str | None,
    failure: Exception,
) -> SourceScenes:
    rejected = bool(page_cursor) and _cursor_was_rejected(failure)
    error = _error_text(failure)
    if rejected:
        joiner = " " if error.endswith(".") else ". "
        error = f"{error}{joiner}{_CURSOR_CLEARED}"
    return SourceScenes(
        source=name,
        collection=collection,
        scenes=[],
        error=error,
        next_cursor=None if rejected else page_cursor,
        search_url=search_url,
        cursor_rejected=rejected,
    )


async def _search_attempts(
    name: str,
    collection: str,
    client: DataPullClient,
    request: PullRequest,
    page_cursor: str | None,
) -> SourceScenes:
    search_url = _client_meta(client)[1]
    failure: Exception | None = None
    for attempt in range(1, _MAX_SEARCH_ATTEMPTS + 1):
        try:
            if page_cursor:
                result = await client.search_from_cursor(request, page_cursor)
            else:
                result = await client.search(request)
        except Exception as exc:
            failure = exc
        else:
            return _stamp_source(name, collection, search_url, result)
        if attempt < _MAX_SEARCH_ATTEMPTS and _is_rate_limited(failure):
            await asyncio.sleep(_retry_delay(failure, attempt))
        else:
            break
    if failure is None:
        return SourceScenes(
            source=name,
            collection=collection,
            scenes=[],
            next_cursor=page_cursor,
            search_url=search_url,
        )
    return _failed_source(name, collection, search_url, page_cursor, failure)


async def _search_one(
    name: str,
    client: DataPullClient,
    request: PullRequest,
    *,
    page_cursor: str | None = None,
    resume: bool = False,
    unavailable: str | None = None,
) -> SourceScenes:
    collection, search_url = _client_meta(client)
    if resume and not page_cursor:
        return SourceScenes(
            source=name,
            collection=collection,
            scenes=[],
            next_cursor=None,
            next_cursor_unavailable=unavailable,
            search_url=search_url,
        )
    budget: asyncio.Timeout | None = None
    try:
        async with asyncio.timeout(_SOURCE_BUDGET_SECONDS) as budget:
            return await _search_attempts(name, collection, client, request, page_cursor)
    except TimeoutError as exc:
        if budget is not None and budget.expired():
            error = "TimeoutError: source search exceeded the time budget"
        else:
            error = _error_text(exc)
        return SourceScenes(
            source=name,
            collection=collection,
            scenes=[],
            error=error,
            next_cursor=page_cursor,
            search_url=search_url,
        )


async def aggregate(
    request: PullRequest,
    clients: Mapping[str, DataPullClient],
    *,
    on_source: Callable[[SourceScenes], None] | None = None,
    page_cursors: Mapping[str, str | None] | None = None,
    unavailable_cursors: Mapping[str, str] | None = None,
) -> SceneCatalog:
    """Search every source for `request` and collect results into one catalog.

    A failure in one source is captured on its `SourceScenes.error` rather than
    aborting the whole fan-out. ``on_source`` is called as each source finishes.

    A source that fails with HTTP 429 is retried, up to three attempts in all.
    The wait is the ``Retry-After`` header when that parses as a finite number
    of seconds or an HTTP date, otherwise half a second times the attempt
    number, jittered by 20 percent. The wait is never more than 60 seconds.
    Each source, including those waits, is limited to 60 seconds with
    ``asyncio.timeout``. Other sources keep running during that wait and if
    the budget runs out. Any other error is recorded on the first failure.
    An HTTP failure is stored as its status and reason phrase, with no
    request URL or query string. A 401 or 403 from Earthdata or CDSE while
    the credential is unset, and a FIRMS search with no map key, is stored
    as the missing variable instead of that provider status. The variables
    are ``EARTHDATA_TOKEN``, ``FIRMS_MAP_KEY``, and the CDSE pair
    ``CDSE_USERNAME`` and ``CDSE_PASSWORD``. The value is not included.

    ``page_cursors`` resumes a pull. A source whose cursor is a string is
    searched from that cursor. A source present in the map with no cursor is
    already finished and is not searched again, so page one is not repeated.
    Sources absent from the map start at page one. Omit the map for a fresh pull.
    A source with no cursor and an entry in ``unavailable_cursors`` is not
    finished: the reason is kept so resume can say the next page was not fetched.
    A resume that fails with HTTP 4xx, other than 408 and 429, or with a
    cursor this client refuses to send, clears that cursor. HTTP 5xx, 408,
    and 429 keep it.

    Args:
        request: Shared search window and area of interest.
        clients: Registry name to pull client.
        on_source: Called as each source finishes.
        page_cursors: Per-source next-page cursor. None starts every source
            at page one.
        unavailable_cursors: Per-source reason a next page exists but was not
            stored. Used when that source's cursor is absent.

    Returns:
        Catalog of scenes, errors, and the next cursor for each source.
    """

    async def run(name: str, client: DataPullClient) -> SourceScenes:
        if page_cursors is None:
            result = await _search_one(name, client, request)
        else:
            result = await _search_one(
                name,
                client,
                request,
                page_cursor=page_cursors.get(name),
                resume=name in page_cursors,
                unavailable=None if unavailable_cursors is None else unavailable_cursors.get(name),
            )
        if on_source is not None:
            on_source(result)
        return result

    sources = await asyncio.gather(*(run(name, client) for name, client in clients.items()))
    return SceneCatalog(request=request, sources=list(sources))
