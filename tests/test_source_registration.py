"""One gate for adding a source: search, a catalog row, and an atlas.data export."""

from __future__ import annotations

from datetime import date
from typing import Any, cast

import httpx
import pytest

import atlas.data as atlas_data
from atlas.cli.catalog import format_satellite, list_satellites
from atlas.data.base import BBox, PullRequest, SceneKind
from atlas.data.earth_search import EARTH_SEARCH_URL, EarthSearchSentinel2L2aClient
from atlas.data.registry import SOURCES, all_sources

SOURCE = "earthsearch_sentinel2_l2a"


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeHttp:
    """Stand-in for ``httpx.AsyncClient``. Records the search call and returns one page."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    async def post(self, url: str, **kwargs: Any) -> _FakeResponse:
        self.calls.append(("POST", url, kwargs))
        return _FakeResponse(self.payload)

    async def get(self, url: str, **kwargs: Any) -> _FakeResponse:
        self.calls.append(("GET", url, kwargs))
        return _FakeResponse(self.payload)

    async def aclose(self) -> None:
        return None


def _request() -> PullRequest:
    return PullRequest(
        start_date=date(2024, 7, 1),
        end_date=date(2024, 7, 3),
        bbox=BBox(west=-71.12, south=42.32, east=-71.02, north=42.40),
        limit=5,
    )


def _page() -> dict[str, Any]:
    return {
        "features": [
            {
                "id": "earthsearch-s2-1",
                "bbox": [-71.2, 42.2, -70.9, 42.5],
                "properties": {
                    "datetime": "2024-07-01T15:00:00Z",
                    "eo:cloud_cover": 8,
                    "platform": "sentinel-2b",
                },
                "assets": {
                    "visual": {
                        "href": "https://example.com/scene.tif",
                        "type": "image/tiff",
                        "roles": ["data"],
                    }
                },
            }
        ]
    }


@pytest.mark.asyncio
async def test_new_source_passes_registration() -> None:
    """A source is registered only when search, a catalog row, and an export agree."""
    cls = SOURCES[SOURCE]
    assert cls is EarthSearchSentinel2L2aClient

    opened = all_sources()
    try:
        assert isinstance(opened[SOURCE], cls)
    finally:
        for client in opened.values():
            close = getattr(client, "aclose", None)
            if close is not None:
                await close()

    request = _request()
    fake = _FakeHttp(_page())
    async with cls(client=cast(httpx.AsyncClient, fake)) as client:
        result = await client.search(request)

    assert result.request == request
    assert [scene.id for scene in result.scenes] == ["earthsearch-s2-1"]
    method, url, kwargs = fake.calls[0]
    assert method == "POST"
    assert url == EARTH_SEARCH_URL
    assert kwargs["json"]["collections"] == ["sentinel-2-l2a"]
    assert kwargs["json"]["bbox"] == request.bbox.as_list()

    sentinel2 = next(info for info in list_satellites() if info.name == "sentinel2")
    assert SOURCE in sentinel2.sources_by_kind[SceneKind.optical]
    rendered = format_satellite("sentinel2")
    optical = next(line for line in rendered.splitlines() if "optical" in line)
    assert SOURCE in optical

    assert atlas_data.EarthSearchSentinel2L2aClient is cls
    assert "EarthSearchSentinel2L2aClient" in atlas_data.__all__
