from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, date, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from pydantic import ValidationError

from atlas.cli import _attach_negative_option_values, _inject_data_command
from atlas.cli.data import add_data_parser, parse_polygon, run_data
from atlas.data.base import (
    MAX_POLYGON_VERTICES,
    BBox,
    DataPullClient,
    GeometryKind,
    Polygon,
    PullRequest,
    PullResult,
    Scene,
)
from atlas.data.sentinel2 import Sentinel2Client

SQUARE = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)]
CLOCKWISE = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)]
FIJI = [(178.0, -18.0), (-178.0, -18.0), (-178.0, -16.0), (178.0, -16.0), (178.0, -18.0)]


def test_square_envelope_contains_and_boundary() -> None:
    polygon = Polygon(ring=SQUARE)
    assert polygon.envelope() == BBox(west=0, south=0, east=1, north=1)
    assert polygon.contains(0.5, 0.5)
    assert polygon.contains(0.0, 0.5)
    assert polygon.contains(1.0, 1.0)
    assert not polygon.contains(2.0, 0.5)
    assert not polygon.contains(0.5, -0.1)


def test_clockwise_ring_is_stored_counterclockwise() -> None:
    polygon = Polygon(ring=CLOCKWISE)
    assert polygon.ring == SQUARE
    assert polygon.as_geojson()["coordinates"] == [list(map(list, SQUARE))]


def test_open_ring_is_rejected() -> None:
    with pytest.raises(ValidationError, match="start and end"):
        Polygon(ring=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])


def test_too_few_positions_are_rejected() -> None:
    with pytest.raises(ValidationError, match="at least four"):
        Polygon(ring=[(0.0, 0.0), (1.0, 0.0), (0.0, 0.0)])


def test_out_of_range_and_non_finite_positions_are_rejected() -> None:
    with pytest.raises(ValidationError, match="longitude"):
        Polygon(ring=[(0.0, 0.0), (181.0, 0.0), (181.0, 1.0), (0.0, 1.0), (0.0, 0.0)])
    with pytest.raises(ValidationError, match="latitude"):
        Polygon(ring=[(0.0, 0.0), (1.0, 91.0), (1.0, 0.0), (0.0, 0.0)])
    with pytest.raises(ValidationError, match="finite"):
        Polygon(ring=[(0.0, 0.0), (1.0, float("nan")), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)])


def test_zero_length_edge_and_zero_area_are_rejected() -> None:
    with pytest.raises(ValidationError, match="zero-length"):
        Polygon(ring=[(0.0, 0.0), (1.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)])
    with pytest.raises(ValidationError, match="no area"):
        Polygon(ring=[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (0.0, 0.0)])


def test_self_crossing_ring_is_rejected() -> None:
    with pytest.raises(ValidationError, match="crosses itself"):
        Polygon(ring=[(0.0, 0.0), (4.0, 1.0), (0.0, 1.0), (1.0, 0.0), (0.0, 0.0)])


def test_ring_that_wraps_a_pole_is_rejected() -> None:
    # The boundary goes the long way around a pole, so it does not close inside
    # one 360-degree window. A bbox catalog and a STAC antimeridian split both
    # need that window.
    assert Polygon.__doc__ is not None
    assert "wraps a pole" in Polygon.__doc__
    assert "360-degree window" in Polygon.__doc__
    with pytest.raises(ValidationError, match="wraps a pole"):
        Polygon(ring=[(0.0, 80.0), (120.0, 80.0), (-120.0, 80.0), (0.0, 80.0)])


def test_antimeridian_ring_uses_the_narrow_envelope() -> None:
    polygon = Polygon(ring=FIJI)
    assert polygon.envelope().as_list() == [178.0, -18.0, -178.0, -16.0]
    assert polygon.envelope().lon_span == 4
    assert polygon.contains(179.0, -17.0)
    assert polygon.contains(178.0, -17.0)
    assert not polygon.contains(0.0, -17.0)
    assert not polygon.contains(179.0, 0.0)


def test_geojson_polygon_and_holes() -> None:
    polygon = Polygon.from_geojson({"type": "Polygon", "coordinates": [SQUARE]})
    assert polygon.ring == SQUARE
    with pytest.raises(ValueError, match="holes"):
        Polygon.from_geojson(
            {
                "type": "Polygon",
                "coordinates": [
                    SQUARE,
                    [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8), (0.2, 0.2)],
                ],
            }
        )
    with pytest.raises(ValueError, match="must be a Polygon"):
        Polygon.from_geojson({"type": "MultiPolygon", "coordinates": [[SQUARE]]})


def test_pull_request_derives_bbox_and_rejects_a_ring_outside_it() -> None:
    request = PullRequest(
        start_date=date(2024, 7, 1),
        end_date=date(2024, 7, 7),
        polygon=Polygon(ring=SQUARE),
    )
    assert request.bbox == BBox(west=0, south=0, east=1, north=1)
    assert request.polygon is not None
    wider = PullRequest(
        start_date=date(2024, 7, 1),
        end_date=date(2024, 7, 7),
        bbox=BBox(west=-1, south=-1, east=2, north=2),
        polygon=Polygon(ring=SQUARE),
    )
    assert wider.bbox.west == -1
    bbox_only = PullRequest(
        start_date=date(2024, 7, 1),
        end_date=date(2024, 7, 7),
        bbox=BBox(west=0, south=0, east=1, north=1),
    )
    assert bbox_only.polygon is None
    with pytest.raises(ValidationError, match="outside bbox"):
        PullRequest(
            start_date=date(2024, 7, 1),
            end_date=date(2024, 7, 7),
            bbox=BBox(west=0, south=0, east=0.5, north=0.5),
            polygon=Polygon(ring=SQUARE),
        )


def test_pull_request_polygon_round_trips() -> None:
    request = PullRequest(
        start_date=date(2024, 7, 1),
        end_date=date(2024, 7, 7),
        polygon={"ring": FIJI},
    )
    restored = PullRequest.model_validate(request.model_dump(mode="json"))
    assert restored.polygon is not None
    assert restored.polygon.envelope().as_list() == [178.0, -18.0, -178.0, -16.0]
    assert restored.bbox == restored.polygon.envelope()


def test_parse_polygon_closes_an_open_ring_and_reads_geojson() -> None:
    opened = parse_polygon("0,0; 1,0; 1,1; 0,1")
    assert opened.ring == SQUARE
    raw = json.dumps({"type": "Polygon", "coordinates": [CLOCKWISE]})
    assert parse_polygon(raw).ring == SQUARE
    feature = json.dumps(
        {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [SQUARE]}}
    )
    assert parse_polygon(feature).ring == SQUARE
    with pytest.raises(ValueError, match="semicolons"):
        parse_polygon("0,0,1; 1,1")
    with pytest.raises(ValueError, match="crosses itself"):
        parse_polygon("0,0; 4,1; 0,1; 1,0; 0,0")


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeStac:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def post(self, url: str, **kwargs: Any) -> _FakeResponse:
        self.calls.append(kwargs["json"])
        return _FakeResponse({"features": []})

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_stac_search_omits_intersects_without_a_polygon() -> None:
    fake = _FakeStac()
    request = PullRequest(
        start_date=date(2024, 7, 1),
        end_date=date(2024, 7, 3),
        bbox=BBox(west=0, south=0, east=1, north=1),
        limit=2,
    )
    async with Sentinel2Client(client=cast(httpx.AsyncClient, fake)) as client:
        await client.search(request)
    assert "intersects" not in fake.calls[0]
    assert fake.calls[0]["bbox"] == [0, 0, 1, 1]


@pytest.mark.asyncio
async def test_stac_search_sends_polygon_intersects() -> None:
    fake = _FakeStac()
    request = PullRequest(
        start_date=date(2024, 7, 1),
        end_date=date(2024, 7, 3),
        polygon=Polygon(ring=SQUARE),
        limit=2,
    )
    async with Sentinel2Client(client=cast(httpx.AsyncClient, fake)) as client:
        result = await client.search(request)
    assert result.scenes == []
    body = fake.calls[0]
    assert body["bbox"] == [0.0, 0.0, 1.0, 1.0]
    assert body["intersects"] == {
        "type": "Polygon",
        "coordinates": [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]],
    }


class _Recording(DataPullClient):
    def __init__(self) -> None:
        self.requests: list[PullRequest] = []

    async def search(self, request: PullRequest) -> PullResult:
        self.requests.append(request)
        return PullResult(request=request, scenes=[])


def _parse_pull(tokens: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="atlas")
    sub = parser.add_subparsers(dest="group")
    add_data_parser(sub)
    return parser.parse_args(_attach_negative_option_values(_inject_data_command(tokens)))


def test_cli_polygon_without_coords_uses_the_envelope(tmp_path: Path) -> None:
    client = _Recording()
    args = _parse_pull(
        [
            "data",
            "sentinel2",
            "optical",
            "--date",
            "2024-07-01:2024-07-07",
            "--polygon",
            "-71.12,42.32;-71.02,42.32;-71.02,42.40;-71.12,42.40",
            "--out",
            str(tmp_path),
            "--search-only",
        ]
    )
    assert run_data(args, clients={"sentinel2": client}) == 0
    request = client.requests[0]
    assert request.polygon is not None
    assert request.polygon.contains(-71.07, 42.36)
    assert request.bbox == request.polygon.envelope()
    assert request.bbox.west == pytest.approx(-71.12)
    assert request.bbox.north == pytest.approx(42.40)


def test_cli_polygon_outside_coords_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = _parse_pull(
        [
            "data",
            "sentinel2",
            "optical",
            "--date",
            "2024-07-01:2024-07-07",
            "--coords",
            "0,0,0.2,0.2",
            "--polygon",
            "0,0;1,0;1,1;0,1",
            "--out",
            str(tmp_path),
            "--search-only",
        ]
    )
    assert run_data(args, clients={"sentinel2": _Recording()}) == 2
    assert "outside bbox" in capsys.readouterr().err


def _square_with_vertices(unique: int) -> list[tuple[float, float]]:
    extra = unique - 4
    bottom = [(index / (extra + 1), 0.0) for index in range(1, extra + 1)]
    ring = [(0.0, 0.0), *bottom, (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    ring.append(ring[0])
    return ring


def test_vertex_cap_rejects_before_the_intersection_check() -> None:
    assert Polygon(ring=_square_with_vertices(MAX_POLYGON_VERTICES)).ring[0] == (0.0, 0.0)
    with pytest.raises(ValidationError, match=str(MAX_POLYGON_VERTICES)):
        Polygon(ring=_square_with_vertices(MAX_POLYGON_VERTICES + 1))
    huge = [(float(index), 0.0) for index in range(10_000)]
    huge.append(huge[0])
    with pytest.raises(ValidationError, match="more than"):
        Polygon(ring=huge)


def test_dateline_ring_splits_into_a_multipolygon() -> None:
    polygon = Polygon(ring=FIJI)
    geometry = polygon.as_stac_geometry()
    assert geometry["type"] == "MultiPolygon"
    rings = geometry["coordinates"]
    assert len(rings) == 2
    lon_sets = [{point[0] for point in ring[0]} for ring in rings]
    assert any(178.0 in lons and 180.0 in lons for lons in lon_sets)
    assert any(-178.0 in lons and -180.0 in lons for lons in lon_sets)
    for ring in rings:
        assert ring[0] == ring[-1]
        for (x1, _y1), (x2, _y2) in pairwise(ring):
            assert abs(x1 - x2) < 180
    touching = [(170.0, -1.0), (180.0, -1.0), (180.0, 1.0), (170.0, 1.0), (170.0, -1.0)]
    assert Polygon(ring=touching).as_stac_geometry()["type"] == "Polygon"


@pytest.mark.asyncio
async def test_stac_search_sends_a_dateline_multipolygon() -> None:
    fake = _FakeStac()
    request = PullRequest(
        start_date=date(2024, 7, 1),
        end_date=date(2024, 7, 3),
        polygon=Polygon(ring=FIJI),
        limit=2,
    )
    async with Sentinel2Client(client=cast(httpx.AsyncClient, fake)) as client:
        await client.search(request)
    intersects = fake.calls[0]["intersects"]
    assert intersects["type"] == "MultiPolygon"
    assert intersects == Polygon(ring=FIJI).as_stac_geometry()
    assert fake.calls[0]["bbox"] == [178.0, -18.0, -178.0, -16.0]


def _scene(scene_id: str, bbox: BBox, **kwargs: object) -> Scene:
    body: dict[str, object] = {
        "id": scene_id,
        "datetime": datetime(2024, 7, 1, tzinfo=UTC),
        "bbox": bbox,
        "platform": "x",
    }
    body.update(kwargs)
    return Scene.model_validate(body)


TRIANGLE = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0)]


def test_every_source_drops_scenes_outside_the_polygon() -> None:
    polygon = Polygon(ring=TRIANGLE)
    request = PullRequest(
        start_date=date(2024, 7, 1),
        end_date=date(2024, 7, 7),
        bbox=BBox(west=0, south=0, east=1, north=1),
        polygon=polygon,
    )
    inside = _scene("inside", BBox(west=0, south=0, east=0.3, north=0.3))
    outside = _scene("outside", BBox(west=0.6, south=0.6, east=0.9, north=0.9))
    point_in = _scene(
        "point-in",
        BBox.from_point(0.1, 0.1),
        geometry_kind=GeometryKind.point,
        lon=0.1,
        lat=0.1,
    )
    point_out = _scene(
        "point-out",
        BBox.from_point(0.8, 0.8),
        geometry_kind=GeometryKind.point,
        lon=0.8,
        lat=0.8,
    )

    class _BoxClient(DataPullClient):
        async def search(self, request: PullRequest) -> PullResult:
            return PullResult(request=request, scenes=[inside, outside, point_in, point_out])

    result = asyncio.run(_BoxClient().search(request))
    assert [scene.id for scene in result.scenes] == ["inside", "point-in"]


def test_manifest_round_trips_polygon_and_source_bbox(tmp_path: Path) -> None:
    polygon = Polygon(ring=TRIANGLE)
    inside = _scene("inside", BBox(west=0, south=0, east=0.3, north=0.3))
    outside = _scene("outside", BBox(west=0.6, south=0.6, east=0.9, north=0.9))

    class _BoxClient(DataPullClient):
        def __init__(self, scenes: list[Scene]) -> None:
            self._scenes = scenes
            self.requests: list[PullRequest] = []

        async def search(self, request: PullRequest) -> PullResult:
            self.requests.append(request)
            return PullResult(request=request, scenes=self._scenes)

    kept = _BoxClient([inside, outside])
    missed = _BoxClient([outside])
    args = _parse_pull(
        [
            "data",
            "sentinel2",
            "optical",
            "--date",
            "2024-07-01:2024-07-07",
            "--coords",
            "0,0,1,1",
            "--polygon",
            "0,0;1,0;0,1",
            "--out",
            str(tmp_path),
            "--search-only",
        ]
    )
    assert run_data(args, clients={"sentinel2": kept, "modis": missed}) == 0
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    restored = PullRequest.model_validate(manifest["request"])
    assert restored.polygon is not None
    assert restored.polygon.ring == polygon.ring
    assert restored.bbox.as_list() == manifest["catalog_bbox"]
    assert Polygon.from_geojson(manifest["polygon"]).ring == restored.polygon.ring
    assert "only accept a bounding box" in manifest["catalog_bbox_note"]
    assert {item["source"] for item in manifest["sources"]} == {"sentinel2", "modis"}
    for item in manifest["sources"]:
        assert item["bbox"] == manifest["catalog_bbox"]
        assert item["bbox"] == restored.bbox.as_list()
    by_source = {item["source"]: item for item in manifest["sources"]}
    assert by_source["sentinel2"]["scene_ids"] == ["inside"]
    assert by_source["modis"]["scene_ids"] == []
    again = json.loads(json.dumps(manifest))
    assert PullRequest.model_validate(again["request"]).polygon == restored.polygon


def test_cli_requires_coords_or_polygon(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = _parse_pull(
        [
            "data",
            "sentinel2",
            "optical",
            "--date",
            "2024-07-01:2024-07-07",
            "--out",
            str(tmp_path),
            "--search-only",
        ]
    )
    assert run_data(args, clients={"sentinel2": _Recording()}) == 2
    assert "--polygon" in capsys.readouterr().err
