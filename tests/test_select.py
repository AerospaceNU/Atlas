from __future__ import annotations

from datetime import UTC, datetime

from atlas.compile.select import coverage_fraction, filter_cloud, representative_cloud, select
from atlas.data.base import BBox, GeometryKind, Scene, SceneKind

AOI = BBox(west=-71.12, south=42.32, east=-71.02, north=42.40)


def _scene(**kwargs: object) -> Scene:
    body: dict[str, object] = {
        "id": "s",
        "datetime": datetime(2024, 7, 1, tzinfo=UTC),
        "bbox": AOI,
        "platform": "x",
    }
    body.update(kwargs)
    return Scene.model_validate(body)


def test_filter_cloud_only_applies_to_optical() -> None:
    optical_clear = _scene(id="a", kind=SceneKind.optical, cloud_cover=5)
    optical_cloudy = _scene(id="b", kind=SceneKind.optical, cloud_cover=80)
    sar = _scene(id="c", kind=SceneKind.sar, cloud_cover=None)
    thermal = _scene(id="d", kind=SceneKind.thermal, cloud_cover=90)
    kept = filter_cloud([optical_clear, optical_cloudy, sar, thermal], 10)
    assert {s.id for s in kept} == {"a", "c", "d"}


def test_select_keeps_sar_when_cloud_capped() -> None:
    chosen = select(
        [
            _scene(id="s1", kind=SceneKind.sar, cloud_cover=None),
            _scene(id="opt", kind=SceneKind.optical, cloud_cover=80),
        ],
        max_cloud=10,
    )
    assert [s.id for s in chosen] == ["s1"]


def test_representative_cloud_ignores_non_optical() -> None:
    assert (
        representative_cloud(
            [
                _scene(kind=SceneKind.thermal, cloud_cover=1),
                _scene(kind=SceneKind.optical, cloud_cover=40),
            ]
        )
        == 40
    )


def test_coverage_skips_points_and_request_footprints() -> None:
    point = _scene(
        id="p",
        kind=SceneKind.detection,
        geometry_kind=GeometryKind.point,
        lon=-71.08,
        lat=42.35,
        bbox=BBox.from_point(-71.08, 42.35),
    )
    snapshot = _scene(id="g", kind=SceneKind.browse, footprint_is_request=True, bbox=AOI)
    lidar = _scene(id="l", kind=SceneKind.lidar, bbox=AOI)
    raster = _scene(
        id="r",
        kind=SceneKind.detection,
        bbox=BBox(west=-71.2, south=42.2, east=-70.9, north=42.5),
    )
    assert coverage_fraction([point, snapshot, lidar], AOI) == 0.0
    frac = coverage_fraction([raster], AOI)
    assert 0 < frac <= 1
