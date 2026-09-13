"""Live catalog evals. Run with: uv run pytest -m evals tests/test_live_catalogs.py"""

from __future__ import annotations

import os
from datetime import date, timedelta

import httpx
import pytest

from atlas.data.base import BBox, PullRequest, PullResult
from atlas.data.registry import sources

BOSTON = BBox(west=-71.12, south=42.32, east=-71.02, north=42.40)
LOS_ANGELES = BBox(west=-118.35, south=33.95, east=-118.15, north=34.12)
MILAN = BBox(west=9.05, south=45.40, east=9.25, north=45.55)
NORTH_ITALY = BBox(west=8.0, south=44.0, east=12.0, north=47.0)

today = date.today()
week = (today - timedelta(days=7), today)
days60 = (today - timedelta(days=60), today)
naip_years = (date(2021, 1, 1), date(2023, 12, 31))


def _req(bbox: BBox, start: date, end: date, *, limit: int = 5) -> PullRequest:
    return PullRequest(start_date=start, end_date=end, bbox=bbox, limit=limit)


async def _search(name: str, request: PullRequest) -> PullResult:
    client = sources(name)[name]
    try:
        return await client.search(request)
    finally:
        aclose = getattr(client, "aclose", None)
        if callable(aclose):
            await aclose()


@pytest.mark.evals
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source", "request_"),
    [
        ("hls_sentinel", _req(BOSTON, *days60)),
        ("hls_landsat", _req(BOSTON, *days60)),
        ("naip", _req(BOSTON, *naip_years)),
        ("cop_dem", _req(BOSTON, date(2000, 1, 1), today)),
        ("sentinel2", _req(BOSTON, *days60)),
        ("earthsearch_sentinel1_grd", _req(BOSTON, *days60)),
        ("cdse_sentinel3_olci", _req(MILAN, *days60)),
        ("cdse_sentinel5p_no2", _req(NORTH_ITALY, *days60)),
        ("gibs_modis_truecolor", _req(BOSTON, *week, limit=3)),
        ("gibs_viirs_truecolor", _req(BOSTON, *week, limit=3)),
        ("goes", _req(BOSTON, today - timedelta(days=1), today, limit=3)),
        ("goes", _req(LOS_ANGELES, today - timedelta(days=1), today, limit=3)),
    ],
)
async def test_live_search_returns_https_assets(source: str, request_: PullRequest) -> None:
    result = await _search(source, request_)
    assert result.scenes, (
        f"{source} returned no scenes for {request_.bbox} {request_.start_date}/{request_.end_date}"
    )
    scene = result.scenes[0]
    assert scene.id
    assert scene.datetime
    https_assets = [a for a in scene.assets.values() if a.href.startswith("https://")]
    assert https_assets, f"{source} scene {scene.id} has no https assets"


@pytest.mark.evals
@pytest.mark.asyncio
async def test_live_gibs_preview_is_a_png() -> None:
    result = await _search(
        "gibs_modis_truecolor", _req(BOSTON, today - timedelta(days=2), today, limit=1)
    )
    href = result.scenes[0].assets["rendered_preview"].href
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.get(href)
    resp.raise_for_status()
    assert resp.headers.get("content-type", "").startswith("image/png")
    assert resp.content[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.evals
@pytest.mark.asyncio
async def test_live_firms_accepts_arbitrary_bbox_when_keyed() -> None:
    if not os.environ.get("FIRMS_MAP_KEY"):
        pytest.skip("FIRMS_MAP_KEY is not set")
    # Detections are sparse; a successful parse (zero or more scenes) is enough.
    result = await _search(
        "firms_viirs",
        _req(BBox(west=-124.5, south=32.5, east=-114.0, north=42.0), *week, limit=20),
    )
    for scene in result.scenes:
        assert scene.properties.get("latitude") is not None
        assert scene.properties.get("longitude") is not None
