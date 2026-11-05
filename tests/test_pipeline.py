from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime
from pathlib import Path
from typing import ClassVar, cast

import httpx
import pytest

from atlas.compile.pipeline import consolidate
from atlas.compile.product import MosaicResult
from atlas.data.base import BBox, DataPullClient, PullRequest, PullResult, Scene, SceneKind
from atlas.data.sentinel1 import Sentinel1Client
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
    scene_kind: ClassVar[SceneKind] = SceneKind.browse

    async def search(self, request: PullRequest) -> PullResult:
        return PullResult(request=request, scenes=[_scene("gibs:2024-07-01")])


class OfflinePlanetaryClient(Sentinel2Client):
    """A renderable Planetary Computer client whose search is offline."""

    scene_kind: ClassVar[SceneKind] = SceneKind.browse

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


def _kind_scene(scene_id: str, kind: SceneKind, cloud: float | None) -> Scene:
    return Scene(
        id=scene_id,
        datetime=datetime(2024, 7, 1, tzinfo=UTC),
        bbox=AOI,
        platform="sentinel",
        kind=kind,
        cloud_cover=cloud,
    )


class FixedClient(DataPullClient):
    """Returns fixed scenes and counts searches."""

    collection = "fixed"

    def __init__(self, *scenes: Scene) -> None:
        self.scenes = list(scenes)
        self.searches = 0

    async def search(self, request: PullRequest) -> PullResult:
        self.searches += 1
        return PullResult(request=request, scenes=self.scenes)


def _window(min_cloud_cover: float | None = None) -> PullRequest:
    return PullRequest(
        start_date=date(2024, 7, 1),
        end_date=date(2024, 7, 3),
        bbox=AOI,
        min_cloud_cover=min_cloud_cover,
    )


@pytest.mark.asyncio
async def test_cloudy_optical_falls_back_to_sentinel1() -> None:
    sar = FixedClient(_kind_scene("S1A", SceneKind.sar, None))
    product = await consolidate(
        _window(),
        {"sentinel2": FixedClient(_kind_scene("S2A", SceneKind.optical, 95))},
        render=False,
        sar_fallback=sar,
    )

    assert sar.searches == 1
    assert product.analysis_input is SceneKind.sar
    assert product.analysis_note is not None
    assert product.analysis_note.startswith("optical skipped for cloud")
    assert product.scene_ids == {"sentinel2": ["S2A"], "sentinel1": ["S1A"]}


@pytest.mark.asyncio
async def test_empty_optical_search_falls_back_after_cloud_filtering() -> None:
    sar = FixedClient(_kind_scene("S1A", SceneKind.sar, None))
    request = _window().model_copy(update={"max_cloud_cover": 20.0})

    product = await consolidate(
        request, {"sentinel2": FixedClient()}, render=False, sar_fallback=sar
    )

    assert sar.searches == 1
    assert product.analysis_input is SceneKind.sar
    assert product.scene_ids == {"sentinel1": ["S1A"]}
    assert "0%" in (product.analysis_note or "")
    assert "optical unavailable" in (product.analysis_note or "")


@pytest.mark.asyncio
async def test_failed_optical_search_uses_sar_without_claiming_cloud_was_observed() -> None:
    class FailedOptical(FixedClient):
        async def search(self, request: PullRequest) -> PullResult:
            raise RuntimeError("offline source unavailable")

    sar = FixedClient(_kind_scene("S1A", SceneKind.sar, None))
    product = await consolidate(
        _window(), {"sentinel2": FailedOptical()}, render=False, sar_fallback=sar
    )

    assert sar.searches == 1
    assert product.analysis_input is SceneKind.sar
    assert "optical unavailable" in (product.analysis_note or "")
    assert "skipped for cloud" not in (product.analysis_note or "")


@pytest.mark.parametrize("has_optical", [False, True])
@pytest.mark.asyncio
async def test_no_sar_results_do_not_claim_a_fallback(has_optical: bool) -> None:
    scenes = [_kind_scene("S2A", SceneKind.optical, 95)] if has_optical else []
    product = await consolidate(
        _window(), {"sentinel2": FixedClient(*scenes)}, render=False, sar_fallback=FixedClient()
    )

    assert product.analysis_input is (SceneKind.optical if has_optical else None)
    assert product.analysis_note is None
    assert product.scene_ids == ({"sentinel2": ["S2A"]} if has_optical else {})


@pytest.mark.asyncio
async def test_exactly_half_clear_coverage_does_not_fall_back() -> None:
    half = AOI.model_copy(update={"east": (AOI.west + AOI.east) / 2})
    optical = _kind_scene("S2A", SceneKind.optical, 20).model_copy(update={"bbox": half})
    sar = FixedClient(_kind_scene("unused", SceneKind.sar, None))

    product = await consolidate(
        _window(), {"sentinel2": FixedClient(optical)}, render=False, sar_fallback=sar
    )

    assert product.coverage["sentinel2"] == pytest.approx(0.5)
    assert sar.searches == 0
    assert product.analysis_input is SceneKind.optical


@pytest.mark.parametrize(("cap", "fallback_searches"), [(20.0, 1), (30.0, 0)])
@pytest.mark.asyncio
async def test_requested_cloud_cap_controls_clear_optical_coverage(
    cap: float, fallback_searches: int
) -> None:
    sar = FixedClient(_kind_scene("S1A", SceneKind.sar, None))
    request = _window().model_copy(update={"max_cloud_cover": cap})

    product = await consolidate(
        request,
        {"sentinel2": FixedClient(_kind_scene("S2A", SceneKind.optical, 25))},
        render=False,
        sar_fallback=sar,
    )

    assert sar.searches == fallback_searches
    assert product.analysis_input is (SceneKind.sar if fallback_searches else SceneKind.optical)


@pytest.mark.asyncio
async def test_empty_cloud_seeking_search_never_falls_back() -> None:
    sar = FixedClient(_kind_scene("S1A", SceneKind.sar, None))
    product = await consolidate(
        _window(min_cloud_cover=80),
        {"sentinel2": FixedClient()},
        render=False,
        sar_fallback=sar,
    )

    assert sar.searches == 0
    assert product.analysis_input is None


@pytest.mark.asyncio
async def test_sar_only_empty_search_does_not_add_a_fallback() -> None:
    class EmptySar(FixedClient):
        scene_kind: ClassVar[SceneKind] = SceneKind.sar

    sar = FixedClient(_kind_scene("unused", SceneKind.sar, None))
    product = await consolidate(
        _window(), {"sentinel1": EmptySar()}, render=False, sar_fallback=sar
    )

    assert sar.searches == 0
    assert product.scene_ids == {}


@pytest.mark.parametrize("cancel_search", [False, True])
@pytest.mark.asyncio
async def test_registry_fallback_is_closed_even_when_search_is_cancelled(
    monkeypatch: pytest.MonkeyPatch, cancel_search: bool
) -> None:
    class OwnedSar(FixedClient):
        closed = False

        async def search(self, request: PullRequest) -> PullResult:
            if cancel_search:
                raise asyncio.CancelledError
            return await super().search(request)

        async def aclose(self) -> None:
            self.closed = True

    sar = OwnedSar(_kind_scene("S1A", SceneKind.sar, None))
    requested: list[tuple[str, ...]] = []

    def factory(*names: str) -> dict[str, DataPullClient]:
        requested.append(names)
        return {"sentinel1": sar}

    monkeypatch.setattr("atlas.compile.pipeline.create_sources", factory)
    if cancel_search:
        with pytest.raises(asyncio.CancelledError):
            await consolidate(_window(), {"sentinel2": FixedClient()}, render=False)
    else:
        product = await consolidate(_window(), {"sentinel2": FixedClient()}, render=False)
        assert product.analysis_input is SceneKind.sar

    assert requested == [("sentinel1",)]
    assert sar.closed


@pytest.mark.parametrize("cloud", [5.0, None])
@pytest.mark.asyncio
async def test_clear_or_unknown_optical_stays_the_input(cloud: float | None) -> None:
    sar = FixedClient(_kind_scene("S1A", SceneKind.sar, None))
    product = await consolidate(
        _window(),
        {"naip": FixedClient(_kind_scene("naip", SceneKind.optical, cloud))},
        render=False,
        sar_fallback=sar,
    )

    assert sar.searches == 0
    assert product.analysis_input is SceneKind.optical
    assert product.analysis_note is None


@pytest.mark.asyncio
async def test_sar_already_in_the_join_is_used_without_a_new_search() -> None:
    fallback = FixedClient(_kind_scene("unused", SceneKind.sar, None))
    product = await consolidate(
        _window(),
        {
            "sentinel2": FixedClient(_kind_scene("S2A", SceneKind.optical, 90)),
            "earthsearch_sentinel1_grd": FixedClient(_kind_scene("S1_grd", SceneKind.sar, None)),
        },
        render=False,
        sar_fallback=fallback,
    )

    assert fallback.searches == 0
    assert product.analysis_input is SceneKind.sar
    assert product.analysis_note is not None
    assert "earthsearch_sentinel1_grd" in product.analysis_note


@pytest.mark.asyncio
async def test_cloud_seeking_request_never_falls_back() -> None:
    sar = FixedClient(_kind_scene("S1A", SceneKind.sar, None))
    product = await consolidate(
        _window(min_cloud_cover=80),
        {"sentinel2": FixedClient(_kind_scene("S2A", SceneKind.optical, 95))},
        render=False,
        sar_fallback=sar,
    )

    assert sar.searches == 0
    assert product.analysis_input is SceneKind.optical


@pytest.mark.asyncio
async def test_fallback_skips_optical_mosaic_for_cloud(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_mosaic(*_args: object, **kwargs: object) -> MosaicResult:
        return MosaicResult(
            source=str(kwargs["source"]),
            collection="c",
            search_id="s",
            n_tiles=1,
            width=1,
            height=1,
            path="p",
        )

    class OfflineSentinel1(Sentinel1Client):
        def __init__(self) -> None:
            super().__init__(client=cast(httpx.AsyncClient, FakeHttpClient()))

        async def search(self, request: PullRequest) -> PullResult:
            return PullResult(request=request, scenes=[_kind_scene("S1A", SceneKind.sar, None)])

    class CloudySentinel2(OfflinePlanetaryClient):
        scene_kind: ClassVar[SceneKind] = SceneKind.optical

        async def search(self, request: PullRequest) -> PullResult:
            return PullResult(request=request, scenes=[_kind_scene("S2A", SceneKind.optical, 99)])

    monkeypatch.setattr("atlas.compile.pipeline.build_mosaic", fake_mosaic)
    product = await consolidate(
        _window(),
        {"sentinel2": CloudySentinel2()},
        out_dir=tmp_path,
        sar_fallback=OfflineSentinel1(),
    )

    assert [m.source for m in product.mosaics] == ["sentinel1"]
    assert [s.source for s in product.skipped_mosaics] == ["sentinel2"]
    assert product.skipped_mosaics[0].reason.startswith("optical skipped for cloud")
