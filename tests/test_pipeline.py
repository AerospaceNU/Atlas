from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import cast

import httpx
import pytest

from atlas.compile.pipeline import consolidate
from atlas.compile.product import MosaicResult
from atlas.data.base import BBox, DataPullClient, PullRequest, PullResult, Scene, SceneKind
from atlas.data.sentinel2 import Sentinel2Client

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
        platform="terra",
        kind=SceneKind.browse,
        cloud_cover=10,
    )


class NonPlanetaryClient(DataPullClient):
    """A source that searches fine but cannot be mosaic-rendered."""

    collection = "MODIS_Terra_CorrectedReflectance_TrueColor"

    async def search(self, request: PullRequest) -> PullResult:
        return PullResult(request=request, scenes=[_scene("gibs:2024-07-01")])


class OfflinePlanetaryClient(Sentinel2Client):
    """A renderable Planetary Computer client whose search is offline."""

    def __init__(self) -> None:
        super().__init__(client=cast(httpx.AsyncClient, FakeHttpClient()))

    async def search(self, request: PullRequest) -> PullResult:
        return PullResult(request=request, scenes=[_scene("S2A_MSIL2A_20240701")])


class FakeHttpClient:
    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_unrenderable_source_is_reported_not_dropped(tmp_path: Path) -> None:
    product = await consolidate(
        _request(),
        {"gibs_modis_truecolor": NonPlanetaryClient()},
        out_dir=tmp_path,
        render=True,
    )

    assert product.mosaics == []
    assert [skip.source for skip in product.skipped_mosaics] == ["gibs_modis_truecolor"]
    reason = product.skipped_mosaics[0].reason
    assert "Planetary Computer" in reason
    assert "NonPlanetaryClient" in reason
    assert product.scene_ids["gibs_modis_truecolor"] == ["gibs:2024-07-01"]


@pytest.mark.parametrize(
    "error",
    [
        KeyError("searchid"),
        TypeError("Expected string searchid, got <class 'int'>"),
        httpx.HTTPError("boom"),
    ],
)
@pytest.mark.asyncio
async def test_render_failure_becomes_a_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    def boom(*_args: object, **_kwargs: object) -> MosaicResult:
        raise error

    monkeypatch.setattr("atlas.compile.pipeline.build_mosaic", boom)
    product = await consolidate(
        _request(),
        {"sentinel2": OfflinePlanetaryClient()},
        out_dir=tmp_path,
        render=True,
    )

    assert product.mosaics == []
    assert [skip.source for skip in product.skipped_mosaics] == ["sentinel2"]
    assert type(error).__name__ in product.skipped_mosaics[0].reason


@pytest.mark.asyncio
async def test_metadata_only_run_records_no_skips(tmp_path: Path) -> None:
    product = await consolidate(
        _request(),
        {"gibs_modis_truecolor": NonPlanetaryClient()},
        render=False,
    )
    assert product.mosaics == []
    assert product.skipped_mosaics == []
