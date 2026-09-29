from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from atlas.data.base import Asset, BBox, GeometryKind, Scene, SceneKind
from atlas.data.stac import item_to_scene


def test_scene_kind_and_geometry_are_enums() -> None:
    scene = Scene(
        id="s",
        datetime=datetime(2024, 7, 1, tzinfo=UTC),
        bbox=BBox(west=-1, south=-1, east=1, north=1),
        platform="x",
        kind=SceneKind.sar,
        geometry_kind=GeometryKind.bbox,
    )
    assert scene.kind is SceneKind.sar
    with pytest.raises(ValidationError):
        Scene(
            id="s",
            datetime=datetime(2024, 7, 1, tzinfo=UTC),
            bbox=BBox(west=-1, south=-1, east=1, north=1),
            platform="x",
            kind="not-a-kind",  # type: ignore[arg-type]
        )


def test_from_corners_across_antimeridian_stays_narrow() -> None:
    box = BBox.from_corners(170, -10, -170, -20)
    assert box.as_list() == [170, -20, -170, -10]
    assert box.lon_span == 20


def test_from_corners_without_wrap_keeps_min_max() -> None:
    box = BBox.from_corners(-70.9, 42.5, -71.2, 42.2)
    assert box.as_list() == [-71.2, 42.2, -70.9, 42.5]


def test_asset_rejects_relative_and_s3_hrefs() -> None:
    assert Asset.try_new(href="./visual.tif") is None
    assert Asset.try_new(href="s3://bucket") is None
    assert Asset(href="https://example.com/a.tif").href.startswith("https://")


def test_point_requires_lon_lat() -> None:
    with pytest.raises(ValidationError):
        Scene(
            id="p",
            datetime=datetime(2024, 7, 1, tzinfo=UTC),
            bbox=BBox.from_point(0, 0),
            platform="x",
            geometry_kind=GeometryKind.point,
        )
    scene = Scene(
        id="p",
        datetime=datetime(2024, 7, 1, tzinfo=UTC),
        bbox=BBox.from_point(10, 20),
        platform="x",
        geometry_kind=GeometryKind.point,
        lon=10,
        lat=20,
        kind=SceneKind.detection,
    )
    assert scene.lon == 10


def test_inverted_interval_rejected_on_scene() -> None:
    with pytest.raises(ValidationError):
        Scene(
            id="s",
            datetime=datetime(2024, 7, 2, tzinfo=UTC),
            start_datetime=datetime(2024, 7, 2, tzinfo=UTC),
            end_datetime=datetime(2024, 7, 1, tzinfo=UTC),
            bbox=BBox(west=-1, south=-1, east=1, north=1),
            platform="x",
        )


def test_item_to_scene_point_without_bbox() -> None:
    feature = {
        "id": "pt",
        "geometry": {"type": "Point", "coordinates": [-71.08, 42.35]},
        "properties": {"datetime": "2024-07-01T13:42:00Z"},
        "assets": {"visual": {"href": "https://example.com/a.tif"}},
    }
    scene = item_to_scene(feature, kind=SceneKind.detection, collection="firms")
    assert scene is not None
    assert scene.geometry_kind is GeometryKind.point
    assert scene.lon == pytest.approx(-71.08)
    assert scene.kind is SceneKind.detection


def test_item_to_scene_keeps_3d_bbox_with_point() -> None:
    feature = {
        "id": "pt",
        "bbox": [-71, 42, 0, -70, 43, 1],
        "geometry": {"type": "Point", "coordinates": [-71.08, 42.35]},
        "properties": {"datetime": "2024-07-01T00:00:00Z"},
    }
    scene = item_to_scene(feature)
    assert scene is not None
    assert scene.bbox.as_list() == [-71, 42, -70, 43]
    assert scene.geometry_kind is GeometryKind.point
    assert scene.lon == pytest.approx(-71.08)
    assert scene.lat == pytest.approx(42.35)


def test_item_to_scene_keeps_3d_bbox_with_polygon() -> None:
    feature = {
        "id": "gedi",
        "bbox": [-71, 42, -120.5, -70, 43, 850.0],
        "geometry": {"type": "Polygon", "coordinates": []},
        "properties": {"datetime": "2024-07-01T00:00:00Z"},
    }
    scene = item_to_scene(feature, kind=SceneKind.lidar)
    assert scene is not None
    assert scene.bbox.as_list() == [-71, 42, -70, 43]
    assert scene.geometry_kind is GeometryKind.polygon


def test_item_to_scene_unreadable_bbox_falls_back_to_point() -> None:
    feature = {
        "id": "pt",
        "bbox": [-71, 42, -70],
        "geometry": {"type": "Point", "coordinates": [-71.08, 42.35]},
        "properties": {"datetime": "2024-07-01T00:00:00Z"},
    }
    scene = item_to_scene(feature)
    assert scene is not None
    assert scene.geometry_kind is GeometryKind.point
    assert scene.bbox.as_list() == pytest.approx([-71.09, 42.34, -71.07, 42.36])


def _cloud_item(cloud: object) -> dict[str, object]:
    return {
        "id": "item-1",
        "bbox": [-1, -1, 1, 1],
        "properties": {"datetime": "2024-07-01T00:00:00Z", "eo:cloud_cover": cloud},
    }


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("12.5", 12.5),
        (12.5, 12.5),
        (120, 100.0),
        (-3, 0.0),
        ("cloudy", None),
        ("nan", None),
        (True, None),
        (None, None),
    ],
)
def test_item_to_scene_cloud_cover_parsed_without_dropping_scene(
    raw: object, expected: float | None
) -> None:
    scene = item_to_scene(_cloud_item(raw))
    assert scene is not None
    assert scene.cloud_cover == expected


def test_item_to_scene_prefers_const_platform() -> None:
    feature = {
        "id": "item-1",
        "bbox": [-1, -1, 1, 1],
        "properties": {
            "datetime": "2024-07-01T00:00:00Z",
            "const:platform": "ISS",
            "platform": "other",
        },
    }
    scene = item_to_scene(feature, default_platform="GEDI02_A_002")
    assert scene is not None
    assert scene.platform == "ISS"


def test_item_to_scene_platform_falls_back_to_platform_then_default() -> None:
    feature = {
        "id": "item-1",
        "bbox": [-1, -1, 1, 1],
        "properties": {"datetime": "2024-07-01T00:00:00Z", "platform": ["landsat-9"]},
    }
    scene = item_to_scene(feature, default_platform="collection-id")
    assert scene is not None
    assert scene.platform == "landsat-9"
    feature["properties"] = {"datetime": "2024-07-01T00:00:00Z"}
    fallback = item_to_scene(feature, default_platform="collection-id")
    assert fallback is not None
    assert fallback.platform == "collection-id"


def test_item_to_scene_bad_point_does_not_raise() -> None:
    feature = {
        "id": "bad",
        "geometry": {"type": "Point", "coordinates": [181, 0]},
        "properties": {"datetime": "2024-07-01T00:00:00Z"},
    }
    assert item_to_scene(feature, kind=SceneKind.detection) is None


def test_item_to_scene_skips_bad_asset_keeps_scene() -> None:
    feature = {
        "id": "item-1",
        "bbox": [-71.2, 42.2, -70.9, 42.5],
        "properties": {"datetime": "2024-07-01T15:00:00Z", "platform": "s2"},
        "assets": {
            "bad": {"href": "./rel.tif"},
            "good": {"href": "https://example.com/a.tif", "roles": ["data"]},
        },
    }
    scene = item_to_scene(feature, collection="sentinel-2-l2a", kind=SceneKind.optical, gsd_m=10)
    assert scene is not None
    assert "good" in scene.assets
    assert "bad" not in scene.assets
    assert scene.gsd_m == 10
    assert scene.collection == "sentinel-2-l2a"


def test_item_to_scene_inverted_interval_returns_none() -> None:
    feature = {
        "id": "item-1",
        "bbox": [-1, -1, 1, 1],
        "properties": {
            "datetime": "2024-07-02T00:00:00Z",
            "start_datetime": "2024-07-02T00:00:00Z",
            "end_datetime": "2024-07-01T00:00:00Z",
        },
    }
    assert item_to_scene(feature) is None
