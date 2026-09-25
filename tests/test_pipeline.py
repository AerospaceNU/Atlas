from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from atlas.compile.pipeline import consolidate
from atlas.data.base import BBox, DataPullClient, PullRequest, PullResult, Scene, SceneKind

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


@pytest.mark.asyncio
async def test_metadata_only_run_records_no_skips(tmp_path: Path) -> None:
    product = await consolidate(
        _request(),
        {"gibs_modis_truecolor": NonPlanetaryClient()},
        render=False,
    )
    assert product.mosaics == []
    assert product.skipped_mosaics == []
