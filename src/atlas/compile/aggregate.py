from __future__ import annotations

import asyncio
import math
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

from atlas.compile.product import SceneCatalog, SourceScenes
from atlas.data.base import DataPullClient, PullRequest, PullResult, Scene

# One search plus two retries. A 429 after that is recorded like any other failure.
_MAX_SEARCH_ATTEMPTS = 3
# Used when the response has no usable Retry-After. Attempt 1 waits 0.5s, then 1s.
_BACKOFF_SECONDS = 0.5
# A huge or hostile Retry-After must not stall the catalog.
_MAX_BACKOFF_SECONDS = 60.0


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


def _retry_delay(exc: BaseException, attempt: int) -> float:
    """Return the seconds to wait before the next search attempt.

    ``attempt`` is the 1-based number of the attempt that just failed.
    """
    header = _retry_after_seconds(exc)
    if header is None:
        header = _BACKOFF_SECONDS * attempt
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


def _stamp_source(name: str, collection: str, result: PullResult) -> SourceScenes:
    stamped: list[Scene] = []
    for scene in result.scenes:
        updates: dict[str, str] = {}
        if not scene.source:
            updates["source"] = name
        if not scene.collection and collection:
            updates["collection"] = collection
        stamped.append(scene.model_copy(update=updates) if updates else scene)
    return SourceScenes(source=name, collection=collection, scenes=stamped)


async def _search_one(name: str, client: DataPullClient, request: PullRequest) -> SourceScenes:
    collection = getattr(client, "collection", "")
    failure: Exception | None = None
    for attempt in range(1, _MAX_SEARCH_ATTEMPTS + 1):
        try:
            result = await client.search(request)
        except Exception as exc:
            failure = exc
        else:
            return _stamp_source(name, collection, result)
        if attempt < _MAX_SEARCH_ATTEMPTS and _is_rate_limited(failure):
            await asyncio.sleep(_retry_delay(failure, attempt))
        else:
            break
    if failure is None:
        return SourceScenes(source=name, collection=collection, scenes=[])
    return SourceScenes(
        source=name,
        collection=collection,
        scenes=[],
        error=f"{type(failure).__name__}: {failure}",
    )


async def aggregate(
    request: PullRequest,
    clients: Mapping[str, DataPullClient],
    *,
    on_source: Callable[[SourceScenes], None] | None = None,
) -> SceneCatalog:
    """Search every source for `request` and collect results into one catalog.

    A failure in one source is captured on its `SourceScenes.error` rather than
    aborting the whole fan-out. ``on_source`` is called as each source finishes.

    A source that fails with HTTP 429 is retried, up to three attempts in all.
    The wait is the ``Retry-After`` header when that parses as seconds or an
    HTTP date, otherwise half a second times the attempt number, and never more
    than 60 seconds. Other sources keep running during that wait. Any other
    error is recorded on the first failure.
    """

    async def run(name: str, client: DataPullClient) -> SourceScenes:
        result = await _search_one(name, client, request)
        if on_source is not None:
            on_source(result)
        return result

    sources = await asyncio.gather(*(run(name, client) for name, client in clients.items()))
    return SceneCatalog(request=request, sources=list(sources))
