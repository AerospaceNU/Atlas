from __future__ import annotations

from datetime import UTC, datetime

from atlas.compile.select import (
    coverage_fraction,
    filter_cloud,
    filter_orbit,
    representative_cloud,
    select,
    select_per_period,
    select_scenes,
)
from atlas.data.base import BBox, GeometryKind, LimitMode, Scene, SceneKind

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


def _timeline() -> list[Scene]:
    return [
        _scene(id="mar", datetime=datetime(2024, 3, 1, tzinfo=UTC), cloud_cover=30),
        _scene(id="jan", datetime=datetime(2024, 1, 1, tzinfo=UTC), cloud_cover=40),
        _scene(id="apr", datetime=datetime(2024, 4, 1, tzinfo=UTC), cloud_cover=20),
        _scene(id="feb", datetime=datetime(2024, 2, 1, tzinfo=UTC), cloud_cover=10),
    ]


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


def test_select_limit_mode_first_keeps_earliest() -> None:
    chosen = select(_timeline(), limit=2, limit_mode=LimitMode.first)
    assert [s.id for s in chosen] == ["jan", "feb"]


def test_select_limit_mode_last_keeps_latest() -> None:
    chosen = select(_timeline(), limit=2, limit_mode=LimitMode.last)
    assert [s.id for s in chosen] == ["mar", "apr"]


def test_select_limit_mode_even_spreads_with_endpoints() -> None:
    chosen = select(_timeline(), limit=3, limit_mode=LimitMode.even)
    assert [s.id for s in chosen] == ["jan", "mar", "apr"]


def test_select_without_limit_returns_full_filtered_list() -> None:
    chosen = select(_timeline(), max_cloud=35)
    assert [s.id for s in chosen] == ["feb", "apr", "mar"]


def test_filter_cloud_minimum_keeps_only_cloudy_optical() -> None:
    optical_clear = _scene(id="a", kind=SceneKind.optical, cloud_cover=5)
    optical_cloudy = _scene(id="b", kind=SceneKind.optical, cloud_cover=85)
    unknown = _scene(id="c", kind=SceneKind.optical, cloud_cover=None)
    sar = _scene(id="d", kind=SceneKind.sar, cloud_cover=None)
    kept = filter_cloud([optical_clear, optical_cloudy, unknown, sar], None, min_cloud=80)
    assert {s.id for s in kept} == {"b", "c", "d"}


def test_filter_cloud_honors_both_bounds() -> None:
    scenes = [
        _scene(id="low", kind=SceneKind.optical, cloud_cover=10),
        _scene(id="mid", kind=SceneKind.optical, cloud_cover=70),
        _scene(id="high", kind=SceneKind.optical, cloud_cover=95),
    ]
    kept = filter_cloud(scenes, 90, min_cloud=60)
    assert [s.id for s in kept] == ["mid"]


def test_filter_cloud_at_bounds_is_inclusive() -> None:
    scenes = [
        _scene(id="floor", kind=SceneKind.optical, cloud_cover=80),
        _scene(id="ceiling", kind=SceneKind.optical, cloud_cover=90),
    ]
    kept = filter_cloud(scenes, 90, min_cloud=80)
    assert [s.id for s in kept] == ["floor", "ceiling"]


def test_select_forwards_cloud_minimum() -> None:
    chosen = select(
        [
            _scene(id="clear", kind=SceneKind.optical, cloud_cover=5),
            _scene(id="cloudy", kind=SceneKind.optical, cloud_cover=88),
        ],
        min_cloud=80,
    )
    assert [s.id for s in chosen] == ["cloudy"]


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


def test_coverage_of_antimeridian_aoi_by_matching_scene() -> None:
    pacific = BBox(west=170, south=-20, east=-170, north=-10)
    scene = _scene(id="fiji", kind=SceneKind.optical, bbox=pacific)
    assert coverage_fraction([scene], pacific) == 1.0


def test_coverage_of_antimeridian_aoi_by_scene_on_one_side() -> None:
    pacific = BBox(west=170, south=-20, east=-170, north=-10)
    east_half = _scene(id="west", bbox=BBox(west=-180, south=-20, east=-170, north=-10))
    assert coverage_fraction([east_half], pacific) == 0.5


def _at(id: str, month: int, day: int = 10, year: int = 2024, **kwargs: object) -> Scene:
    return _scene(id=id, datetime=datetime(year, month, day, tzinfo=UTC), **kwargs)


def test_per_month_keeps_the_clearest_acquisition_with_all_its_tiles() -> None:
    scenes = [_at(f"m{m}", m, cloud_cover=50 - m) for m in range(1, 13)]
    scenes += [_at("m3-low", 3, cloud_cover=10), _at("m3-late", 3, day=20, cloud_cover=5)]
    scenes[2] = _at("m3", 3, cloud_cover=30)
    picked = [s.id for s in select_per_period(scenes, "month")]
    # day 10: tiles m3 (30) and m3-low (10) average 20; day 20 (5) is clearer
    assert picked == [
        "m1",
        "m2",
        "m3-late",
        "m4",
        "m5",
        "m6",
        "m7",
        "m8",
        "m9",
        "m10",
        "m11",
        "m12",
    ]
    tied = [_at("a", 3, cloud_cover=10), _at("b", 3, cloud_cover=30)]
    assert [s.id for s in select_per_period(tied, "month")] == ["a", "b"]


def test_season_groups_december_with_the_next_january_and_february() -> None:
    scenes = [
        _at("dec", 12, year=2023, cloud_cover=5),
        _at("jan", 1, cloud_cover=3),
        _at("feb", 2, cloud_cover=9),
        _at("mar", 3, cloud_cover=1),
    ]
    assert [s.id for s in select_per_period(scenes, "season")] == ["jan", "mar"]
    assert [s.id for s in select_per_period(scenes, "year")] == ["dec", "mar"]


def test_per_period_keeps_sar_and_ranks_unknown_cloud_last() -> None:
    sar = _at("sar", 5, kind=SceneKind.sar)
    unknown = _at("unknown", 5, day=2, cloud_cover=None)
    known = _at("known", 5, day=3, cloud_cover=60)
    assert [s.id for s in select_per_period([sar], "month")] == ["sar"]
    assert [s.id for s in select_per_period([unknown, known], "month")] == ["known"]


def test_select_scenes_modes() -> None:
    scenes = [_at(f"m{m}", m, cloud_cover=m * 10) for m in range(1, 6)]
    assert [s.id for s in select_scenes(scenes, "clearest", limit=2)] == ["m1", "m2"]
    assert [s.id for s in select_scenes(scenes, "newest", limit=2)] == ["m5", "m4"]
    assert [s.id for s in select_scenes(scenes, "oldest", limit=2)] == ["m1", "m2"]
    assert [s.id for s in select_scenes(scenes, "even", limit=3)] == ["m1", "m3", "m5"]
    assert len(select_scenes(scenes, "per_month", limit=1)) == 5


def test_filter_orbit_drops_scenes_without_the_property() -> None:
    props = [
        {"sat:relative_orbit": 8, "sat:orbit_state": "descending"},
        {"sat:relative_orbit": "8", "sat:orbit_state": "ascending"},
        {"sat:relative_orbit": 9, "sat:orbit_state": "descending"},
        {},
    ]
    scenes = [_at(f"s{i}", 6, i + 1, kind=SceneKind.sar, properties=p) for i, p in enumerate(props)]
    kept, dropped = filter_orbit(scenes, "8:descending")
    assert [s.id for s in kept] == ["s0"] and len(dropped) == 3
    assert [s.id for s in filter_orbit(scenes, "8")[0]] == ["s0", "s1"]
    assert [s.id for s in filter_orbit(scenes, "desc")[0]] == ["s0", "s2"]
