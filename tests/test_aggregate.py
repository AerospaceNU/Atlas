from __future__ import annotations

import asyncio
import importlib
import math
from datetime import UTC, date, datetime

import httpx
import pytest

from atlas.compile.aggregate import aggregate
from atlas.compile.product import SourceScenes
from atlas.data.base import BBox, DataPullClient, PullRequest, PullResult, Scene

aggregate_mod = importlib.import_module("atlas.compile.aggregate")

AOI = BBox(west=-71.12, south=42.32, east=-71.02, north=42.40)


def _request() -> PullRequest:
    return PullRequest(
        start_date=date(2024, 7, 1),
        end_date=date(2024, 7, 3),
        bbox=AOI,
        limit=5,
    )


def _scene(scene_id: str) -> Scene:
    return Scene(
        id=scene_id,
        datetime=datetime(2024, 7, 1, tzinfo=UTC),
        bbox=AOI,
        platform="sentinel-2",
    )


def _rate_limit(retry_after: str | None = None) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://example.com/search")
    headers = {} if retry_after is None else {"Retry-After": retry_after}
    response = httpx.Response(429, headers=headers, request=request)
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        return exc
    raise AssertionError("expected HTTP 429")


class _ScriptedClient(DataPullClient):
    """Search script: each entry is a scene id or an exception to raise."""

    collection = "sentinel-2-l2a"

    def __init__(self, steps: list[str | Exception]) -> None:
        self._steps = list(steps)
        self.calls = 0

    async def search(self, request: PullRequest) -> PullResult:
        self.calls += 1
        step = self._steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return PullResult(request=request, scenes=[_scene(step)])


class _CodedError(RuntimeError):
    """Search failure that exposes ``status_code`` without an HTTP response."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"status {status_code}")
        self.status_code = status_code


class _StatusClient(DataPullClient):
    """Raises ``_CodedError`` once, then returns ``then`` when a later attempt runs."""

    def __init__(self, status_code: int, *, then: str | None = None) -> None:
        self._status_code = status_code
        self._then = then
        self.calls = 0

    async def search(self, request: PullRequest) -> PullResult:
        self.calls += 1
        if self.calls == 1 or self._then is None:
            raise _CodedError(self._status_code)
        return PullResult(request=request, scenes=[_scene(self._then)])


def _pin_jitter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the computed backoff deterministic. ``Retry-After`` is not jittered."""
    monkeypatch.setattr(aggregate_mod.random, "uniform", lambda _low, _high: 0.0)


def _patch_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record backoff delays and yield so sibling searches can finish."""
    delays: list[float] = []
    real_sleep = asyncio.sleep

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)
        await real_sleep(0)

    _pin_jitter(monkeypatch)
    monkeypatch.setattr(aggregate_mod.asyncio, "sleep", fake_sleep)
    return delays


@pytest.mark.asyncio
async def test_rate_limited_source_is_retried_and_keeps_its_scenes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delays = _patch_sleep(monkeypatch)
    limited = _ScriptedClient([_rate_limit(), "S2A_after_retry"])
    seen: list[SourceScenes] = []

    catalog = await aggregate(_request(), {"sentinel2": limited}, on_source=seen.append)

    assert limited.calls == 2
    assert delays == [0.5]
    assert len(seen) == 1
    assert seen[0].error is None
    assert [scene.id for scene in catalog.sources[0].scenes] == ["S2A_after_retry"]
    assert catalog.sources[0].scenes[0].source == "sentinel2"
    assert catalog.sources[0].scenes[0].collection == "sentinel-2-l2a"
    assert catalog.sources[0].error is None


@pytest.mark.asyncio
async def test_retry_does_not_abort_the_other_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delays: list[float] = []
    finished: list[str] = []
    limited = _ScriptedClient([_rate_limit(), _rate_limit(), _rate_limit()])
    healthy = _ScriptedClient(["S2B_ok"])

    def on_source(src: SourceScenes) -> None:
        finished.append(src.source)

    _pin_jitter(monkeypatch)
    real_sleep = asyncio.sleep

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)
        await real_sleep(0)
        assert "healthy" in finished

    monkeypatch.setattr(aggregate_mod.asyncio, "sleep", fake_sleep)

    catalog = await aggregate(
        _request(),
        {"limited": limited, "healthy": healthy},
        on_source=on_source,
    )

    by_name = {src.source: src for src in catalog.sources}
    assert limited.calls == 3
    assert delays == [0.5, 1.0]
    assert healthy.calls == 1
    assert finished.count("healthy") == 1
    assert finished.count("limited") == 1
    assert [scene.id for scene in by_name["healthy"].scenes] == ["S2B_ok"]
    assert by_name["healthy"].error is None
    assert by_name["limited"].scenes == []
    assert by_name["limited"].error is not None
    assert "HTTPStatusError" in by_name["limited"].error
    assert "429" in by_name["limited"].error


@pytest.mark.asyncio
async def test_retry_after_header_sets_the_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    delays = _patch_sleep(monkeypatch)
    limited = _ScriptedClient([_rate_limit("2"), "S2A_after_retry"])

    catalog = await aggregate(_request(), {"sentinel2": limited})

    assert delays == [2.0]
    assert [scene.id for scene in catalog.sources[0].scenes] == ["S2A_after_retry"]


@pytest.mark.asyncio
async def test_past_retry_after_date_waits_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    delays = _patch_sleep(monkeypatch)
    limited = _ScriptedClient([_rate_limit("Wed, 21 Oct 2015 07:28:00 GMT"), "S2A_after_retry"])

    await aggregate(_request(), {"sentinel2": limited})

    assert delays == [0.0]


@pytest.mark.asyncio
async def test_computed_backoff_includes_jitter(monkeypatch: pytest.MonkeyPatch) -> None:
    delays = _patch_sleep(monkeypatch)

    def uniform(low: float, high: float) -> float:
        assert low < 0.0 < high
        return high

    monkeypatch.setattr(aggregate_mod.random, "uniform", uniform)
    limited = _ScriptedClient([_rate_limit(), _rate_limit(), "S2A_after_retry"])

    catalog = await aggregate(_request(), {"sentinel2": limited})

    fraction = aggregate_mod._JITTER_FRACTION
    assert delays == [0.5 + 0.5 * fraction, 1.0 + 1.0 * fraction]
    assert [scene.id for scene in catalog.sources[0].scenes] == ["S2A_after_retry"]


@pytest.mark.asyncio
async def test_jitter_cannot_exceed_the_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    delays = _patch_sleep(monkeypatch)
    monkeypatch.setattr(aggregate_mod, "_BACKOFF_SECONDS", 100.0)
    monkeypatch.setattr(aggregate_mod.random, "uniform", lambda _low, high: high)
    limited = _ScriptedClient([_rate_limit(), "S2A_after_retry"])

    await aggregate(_request(), {"sentinel2": limited})

    assert delays == [aggregate_mod._MAX_BACKOFF_SECONDS]


@pytest.mark.asyncio
@pytest.mark.parametrize("header", ["inf", "+inf", "-inf", "nan", "NaN"])
async def test_non_finite_retry_after_falls_back_to_backoff(
    monkeypatch: pytest.MonkeyPatch, header: str
) -> None:
    delays = _patch_sleep(monkeypatch)
    limited = _ScriptedClient([_rate_limit(header), "S2A_after_retry"])

    catalog = await aggregate(_request(), {"sentinel2": limited})

    assert delays == [0.5]
    assert all(math.isfinite(delay) for delay in delays)
    assert [scene.id for scene in catalog.sources[0].scenes] == ["S2A_after_retry"]


@pytest.mark.asyncio
async def test_retry_after_is_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    delays = _patch_sleep(monkeypatch)
    limited = _ScriptedClient([_rate_limit("120"), "S2A_after_retry"])

    await aggregate(_request(), {"sentinel2": limited})

    assert delays == [60.0]


@pytest.mark.asyncio
async def test_status_code_attribute_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    delays = _patch_sleep(monkeypatch)
    client = _StatusClient(429, then="S2A_wrapped")

    catalog = await aggregate(_request(), {"sentinel2": client})

    assert client.calls == 2
    assert delays == [0.5]
    assert [scene.id for scene in catalog.sources[0].scenes] == ["S2A_wrapped"]


@pytest.mark.asyncio
async def test_other_errors_are_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    delays = _patch_sleep(monkeypatch)
    broken = _ScriptedClient([RuntimeError("boom"), "should-not-run"])
    healthy = _ScriptedClient(["S2B_ok"])
    server = _StatusClient(503, then="should-not-run")

    catalog = await aggregate(
        _request(),
        {"broken": broken, "healthy": healthy, "server": server},
    )

    by_name = {src.source: src for src in catalog.sources}
    assert delays == []
    assert broken.calls == 1
    assert server.calls == 1
    assert by_name["broken"].error == "RuntimeError: boom"
    assert by_name["server"].error is not None
    assert "503" in by_name["server"].error
    assert [scene.id for scene in by_name["healthy"].scenes] == ["S2B_ok"]
