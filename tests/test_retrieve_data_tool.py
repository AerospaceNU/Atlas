"""The get_satellite_data tool: input checks, source choice, search-only, and download."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

import pytest
from pydantic import ValidationError

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import default_registry
from atlas.agent.tools.retrieve_data_tool import GetDataTool, GetDataToolInput
from atlas.cli.download import DownloadOutcome, DownloadReport
from atlas.data.base import BBox, DataPullClient, PullRequest, PullResult, Scene, SceneKind
from atlas.data.registry import SOURCES

AOI = BBox(west=-71.12, south=42.32, east=-71.02, north=42.40)
BOSTON = {"latitude": 42.36, "longitude": -71.06}
WINDOW = {"start_date": "2024-07-01", "end_date": "2024-07-07"}
MODULE = "atlas.agent.tools.retrieve_data_tool"


def _scene(**kwargs: object) -> Scene:
    body: dict[str, object] = {
        "id": "s",
        "datetime": datetime(2024, 7, 1, tzinfo=UTC),
        "bbox": AOI,
        "platform": "x",
        "source": "sentinel2",
    }
    body.update(kwargs)
    return Scene.model_validate(body)


class FakePullClient(DataPullClient):
    satellite: ClassVar[str] = "sentinel2"
    scene_kind: ClassVar[SceneKind] = SceneKind.optical

    def __init__(self, scenes: list[Scene] | None = None, error: str | None = None) -> None:
        self._scenes = scenes or []
        self._error = error
        self.requests: list[PullRequest] = []
        self.closed = False

    async def search(self, request: PullRequest) -> PullResult:
        self.requests.append(request)
        if self._error:
            raise RuntimeError(self._error)
        return PullResult(request=request, scenes=self._scenes)

    async def aclose(self) -> None:
        self.closed = True


def _use_clients(
    monkeypatch: pytest.MonkeyPatch, clients: dict[str, FakePullClient]
) -> list[tuple[str, ...]]:
    """Make the tool build ``clients`` instead of real ones. Returns the names it asked for."""
    asked: list[tuple[str, ...]] = []

    def fake_sources(*names: str) -> dict[str, FakePullClient]:
        asked.append(names)
        return clients

    monkeypatch.setattr(f"{MODULE}.sources", fake_sources)
    return asked


def _no_download(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fail(*_args: object, **_kwargs: object) -> DownloadReport:
        raise AssertionError("a search-only call must not download")

    monkeypatch.setattr(f"{MODULE}.download_assets", fail)


# input model


def test_point_and_bbox_are_both_accepted() -> None:
    GetDataToolInput.model_validate({"data_type": "optical", **WINDOW, **BOSTON})
    GetDataToolInput.model_validate({"data_type": "optical", **WINDOW, "bbox": AOI.model_dump()})


def test_point_and_bbox_together_are_rejected() -> None:
    with pytest.raises(ValidationError, match="not both"):
        GetDataToolInput.model_validate(
            {"data_type": "optical", **WINDOW, **BOSTON, "bbox": AOI.model_dump()}
        )


@pytest.mark.parametrize(
    "location",
    [{}, {"latitude": 42.36}, {"longitude": -71.06}],
    ids=["nothing", "latitude only", "longitude only"],
)
def test_missing_location_is_rejected(location: dict[str, float]) -> None:
    with pytest.raises(ValidationError, match="Cannot include nothing"):
        GetDataToolInput.model_validate({"data_type": "optical", **WINDOW, **location})


def test_zero_latitude_and_longitude_count_as_a_location() -> None:
    request = GetDataToolInput.model_validate(
        {"data_type": "optical", **WINDOW, "latitude": 0.0, "longitude": 0.0}
    )
    box = request.lat_long_to_bbox()
    assert box.contains(0.0, 0.0)


def test_out_of_range_latitude_is_rejected() -> None:
    with pytest.raises(ValidationError):
        GetDataToolInput.model_validate(
            {"data_type": "optical", **WINDOW, "latitude": 200.0, "longitude": 0.0}
        )


def test_unknown_data_type_is_rejected() -> None:
    with pytest.raises(ValidationError):
        GetDataToolInput.model_validate({"data_type": "radio", **WINDOW, **BOSTON})


# point to bounding box


def test_bbox_input_is_returned_unchanged() -> None:
    request = GetDataToolInput.model_validate(
        {"data_type": "optical", **WINDOW, "bbox": AOI.model_dump()}
    )
    assert request.lat_long_to_bbox() == AOI


def test_point_becomes_a_square_of_twice_the_radius() -> None:
    request = GetDataToolInput.model_validate(
        {"data_type": "optical", **WINDOW, **BOSTON, "radius_km": 2.0}
    )
    box = request.lat_long_to_bbox()

    km_per_lon_degree = 111.32 * math.cos(math.radians(42.36))
    width_km = (box.east - box.west) * km_per_lon_degree
    height_km = (box.north - box.south) * 111.32
    assert width_km == pytest.approx(4.0, rel=1e-6)
    assert height_km == pytest.approx(4.0, rel=1e-6)
    assert box.contains(-71.06, 42.36)


def test_point_at_a_pole_spans_every_longitude() -> None:
    request = GetDataToolInput.model_validate(
        {"data_type": "optical", **WINDOW, "latitude": 90.0, "longitude": 10.0}
    )
    box = request.lat_long_to_bbox()
    assert (box.west, box.east, box.north) == (-180.0, 180.0, 90.0)


# choosing sources


def test_tool_is_in_the_default_registry() -> None:
    assert "get_satellite_data" in {tool.name for tool in default_registry().definitions}


def test_data_type_picks_every_source_of_that_kind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked = _use_clients(monkeypatch, {"sentinel1": FakePullClient()})
    _no_download(monkeypatch)

    GetDataTool().run({"data_type": "sar", **WINDOW, **BOSTON}, LocalArtifactStore(tmp_path))

    expected = sorted(name for name, cls in SOURCES.items() if cls.scene_kind is SceneKind.sar)
    assert sorted(asked[0]) == expected


def test_named_sources_are_used_as_given(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    asked = _use_clients(monkeypatch, {"sentinel2": FakePullClient()})
    _no_download(monkeypatch)

    GetDataTool().run(
        {"data_type": "optical", **WINDOW, **BOSTON, "sources": ["sentinel2"]},
        LocalArtifactStore(tmp_path),
    )

    assert asked == [("sentinel2",)]


def test_unknown_source_is_a_value_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown source"):
        GetDataTool().run(
            {"data_type": "optical", **WINDOW, **BOSTON, "sources": ["not_a_source"]},
            LocalArtifactStore(tmp_path),
        )


@pytest.mark.parametrize(
    "arguments",
    [{"data_type": "landsurface"}, {"data_type": "optical", "sources": ["fldas"]}],
    ids=["landsurface", "named fldas"],
)
def test_fldas_is_refused(tmp_path: Path, arguments: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="fldas"):
        GetDataTool().run({**arguments, **WINDOW, **BOSTON}, LocalArtifactStore(tmp_path))


def test_end_before_start_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="end_date"):
        GetDataTool().run(
            {
                "data_type": "optical",
                "start_date": "2024-07-07",
                "end_date": "2024-07-01",
                **BOSTON,
            },
            LocalArtifactStore(tmp_path),
        )


# search only


def test_search_only_returns_the_estimate_and_scenes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = FakePullClient(
        [
            _scene(id="clear", cloud_cover=12.4),
            _scene(id="unknown", datetime=datetime(2024, 7, 3, tzinfo=UTC)),
        ]
    )
    _use_clients(monkeypatch, {"sentinel2": client})
    _no_download(monkeypatch)

    result = GetDataTool().run(
        {"data_type": "optical", **WINDOW, **BOSTON, "sources": ["sentinel2"]},
        LocalArtifactStore(tmp_path),
    )

    assert "Pull estimate" in result.text
    assert "scenes: 2" in result.text
    assert "sentinel2 clear 2024-07-01 cloud 12%" in result.text
    assert "sentinel2 unknown 2024-07-03 cloud n/a" in result.text
    assert result.artifacts == []
    assert client.closed


def test_search_sends_the_dates_box_and_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = FakePullClient()
    _use_clients(monkeypatch, {"sentinel2": client})
    _no_download(monkeypatch)

    GetDataTool().run(
        {
            "data_type": "optical",
            **WINDOW,
            "bbox": AOI.model_dump(),
            "sources": ["sentinel2"],
            "max_cloud_cover": 20,
            "limit": 5,
        },
        LocalArtifactStore(tmp_path),
    )

    sent = client.requests[0]
    assert (sent.start_date.isoformat(), sent.end_date.isoformat()) == (
        "2024-07-01",
        "2024-07-07",
    )
    assert sent.bbox == AOI
    assert sent.max_cloud_cover == 20
    assert sent.limit == 5


def test_search_only_lists_a_failed_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_clients(
        monkeypatch,
        {
            "sentinel2": FakePullClient([_scene(id="ok")]),
            "landsat9": FakePullClient(error="catalog is down"),
        },
    )
    _no_download(monkeypatch)

    result = GetDataTool().run(
        {"data_type": "optical", **WINDOW, **BOSTON, "sources": ["sentinel2", "landsat9"]},
        LocalArtifactStore(tmp_path),
    )

    assert "landsat9: RuntimeError: catalog is down" in result.text


# download


def test_download_saves_files_and_reports_each_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    good = FakePullClient([_scene(id="a"), _scene(id="b")])
    broken = FakePullClient(error="catalog is down")
    _use_clients(monkeypatch, {"sentinel2": good, "landsat9": broken})
    downloaded: list[dict[str, Any]] = []

    async def fake_download(scenes: list[Scene], out: Path, **kwargs: Any) -> DownloadReport:
        downloaded.append({"ids": [scene.id for scene in scenes], "out": out, **kwargs})
        path = out / "sentinel2" / "a_visual.tif"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"tif")
        return DownloadReport(
            outcomes=[
                DownloadOutcome(source="sentinel2", scene_id="a", status="saved", path=path),
                DownloadOutcome(
                    source="sentinel2",
                    scene_id="b",
                    status="no_asset",
                    error="no native-resolution asset",
                ),
            ]
        )

    monkeypatch.setattr(f"{MODULE}.download_assets", fake_download)
    store = LocalArtifactStore(tmp_path)

    result = GetDataTool().run(
        {
            "data_type": "optical",
            **WINDOW,
            **BOSTON,
            "sources": ["sentinel2", "landsat9"],
            "download": True,
        },
        store,
    )

    # only scenes from the source whose search worked are downloaded, cut to the box
    call = downloaded[0]
    assert call["ids"] == ["a", "b"]
    assert call["out"] == store.resolve("data")
    assert call["bbox"].contains(-71.06, 42.36)

    assert result.artifacts == ["data/sentinel2/a_visual.tif"]
    assert "sentinel2: 1 of 2 scenes saved" in result.text
    assert "  b: no native-resolution asset" in result.text
    assert "landsat9: search failed: RuntimeError: catalog is down" in result.text
    assert "saved: data/sentinel2/a_visual.tif" in result.text
    assert good.closed and broken.closed


def test_clients_are_closed_when_the_download_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = FakePullClient([_scene(id="a")])
    _use_clients(monkeypatch, {"sentinel2": client})

    async def explode(*_args: object, **_kwargs: object) -> DownloadReport:
        raise OSError("disk full")

    monkeypatch.setattr(f"{MODULE}.download_assets", explode)

    with pytest.raises(OSError, match="disk full"):
        GetDataTool().run(
            {
                "data_type": "optical",
                **WINDOW,
                **BOSTON,
                "sources": ["sentinel2"],
                "download": True,
            },
            LocalArtifactStore(tmp_path),
        )
    assert client.closed
