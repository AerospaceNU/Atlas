"""The on-disk scene index: cards, local layers, offline queries, and versions."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest

from atlas.archive import (
    Availability,
    CloudStatus,
    IndexEntry,
    IndexVersionError,
    LookupStatus,
    SavedLayer,
    SceneIndex,
    SceneQuery,
    make_index_id,
)
from atlas.archive.geometry import footprint_intersects
from atlas.data.base import BBox, Scene, SceneKind
from atlas.data.stac import item_to_scene

AOI = BBox(west=-71.12, south=42.32, east=-71.02, north=42.40)
CHIP = BBox(west=-71.2, south=42.3, east=-71.0, north=42.45)
NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _square(west: float, south: float, east: float, north: float) -> dict[str, Any]:
    ring = [[west, south], [east, south], [east, north], [west, north], [west, south]]
    return {"type": "Polygon", "coordinates": [ring]}


def _feature(
    item_id: str = "S2A_MSIL2A_20240603",
    *,
    when: str = "2024-06-03T15:40:00Z",
    bounds: tuple[float, float, float, float] = (-71.5, 42.0, -70.5, 42.8),
    geometry: dict[str, Any] | None = None,
    cloud: object = 12.5,
    assets: dict[str, Any] | None = None,
    platform: str = "sentinel-2a",
) -> dict[str, Any]:
    props: dict[str, Any] = {"datetime": when, "platform": platform}
    if cloud is not None:
        props["eo:cloud_cover"] = cloud
    return {
        "id": item_id,
        "bbox": list(bounds),
        "geometry": geometry or _square(*bounds),
        "properties": props,
        "assets": assets
        if assets is not None
        else {
            "B04": {"href": "https://example.com/B04.tif", "type": "image/tiff", "roles": ["data"]},
            "visual": {"href": "https://example.com/visual.tif", "roles": ["visual"]},
            "thumbnail": {"href": "https://example.com/t.jpg", "roles": ["thumbnail"]},
        },
    }


def _scene(
    feature: dict[str, Any] | None = None,
    *,
    source: str = "sentinel2",
    collection: str = "sentinel-2-l2a",
) -> Scene:
    scene = item_to_scene(feature or _feature(), collection=collection, source=source)
    assert scene is not None
    return scene


def _index(root: Path) -> SceneIndex:
    return SceneIndex.at(root)


def _query(**kwargs: Any) -> SceneQuery:
    body: dict[str, Any] = {"bbox": AOI, "start": date(2024, 6, 1), "end": date(2024, 6, 30)}
    body.update(kwargs)
    return SceneQuery.model_validate(body)


def _file(root: Path, name: str = "sentinel2/S2A_visual.tif") -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"II*\x00")
    return path


# --- recording scenes -------------------------------------------------------


def test_upsert_twice_keeps_one_card_with_the_card_id(tmp_path: Path) -> None:
    with _index(tmp_path) as index:
        first = index.upsert([IndexEntry(_scene())], now=NOW)
        second = index.upsert([IndexEntry(_scene())], now=NOW)
        result = index.query(_query())
    expected = make_index_id("sentinel2", "sentinel-2-l2a", "S2A_MSIL2A_20240603")
    assert (first.added, second.added, second.updated) == (1, 0, 1)
    assert result.index_ids == [expected]
    assert second.index_ids == {("sentinel2", "S2A_MSIL2A_20240603"): expected}


def test_card_is_short_and_has_no_urls(tmp_path: Path) -> None:
    with _index(tmp_path) as index:
        index.upsert([IndexEntry(_scene())])
        result = index.query(_query())
    payload = result.to_json()
    assert "http" not in payload
    assert "eo:cloud_cover" not in payload  # raw STAC properties stay in the store
    assert "null" not in payload
    card = result.groups[0].cards[0]
    assert list(card.layers) == ["B04", "visual"]  # previews are not layers
    assert (card.cloud_cover, card.cloud_status) == (12.5, CloudStatus.value)
    assert card.satellite == "sentinel2"
    assert not card.has_local_data
    assert json.loads(payload)["groups"][0]["gap"] is False


def test_snapshot_drops_signed_query_strings(tmp_path: Path) -> None:
    assets = {"B04": {"href": "https://example.com/B04.tif?st=2024&sig=SECRET", "roles": ["data"]}}
    with _index(tmp_path) as index:
        index.upsert([IndexEntry(_scene(_feature(assets=assets)))])
    stored = sqlite3.connect(tmp_path / "index.sqlite").execute("SELECT scene_json FROM scenes")
    [(raw,)] = stored.fetchall()
    assert "SECRET" not in raw
    assert "https://example.com/B04.tif" in raw


def test_request_footprint_and_invalid_scenes_are_counted_not_indexed(tmp_path: Path) -> None:
    gibs = _scene().model_copy(update={"footprint_is_request": True, "id": "gibs-tile"})
    no_collection = _scene().model_copy(update={"collection": "", "id": "bare"})
    with _index(tmp_path) as index:
        stats = index.upsert([IndexEntry(gibs), IndexEntry(no_collection)])
        assert index.query(_query()).empty
    assert (stats.skipped_request_footprint, stats.skipped_invalid, stats.added) == (1, 1, 0)
    assert stats.index_ids == {}


# --- local layers -----------------------------------------------------------


def test_full_download_survives_a_later_search_only_pull(tmp_path: Path) -> None:
    saved = _file(tmp_path)
    with _index(tmp_path) as index:
        index.upsert([IndexEntry(_scene(), saved=SavedLayer("visual", saved))])
        index.upsert([IndexEntry(_scene())])
        [card] = index.query(_query(), workspace=tmp_path).groups[0].cards
    layer = card.layers["visual"]
    assert layer.availability is Availability.full and layer.window is None
    assert layer.locator is not None
    assert (layer.locator.scheme, layer.locator.key) == ("workspace", "sentinel2/S2A_visual.tif")
    assert card.layers["B04"].availability is Availability.absent
    assert card.has_local_data


def test_chip_records_its_window_and_a_full_download_replaces_it(tmp_path: Path) -> None:
    saved = _file(tmp_path)
    with _index(tmp_path) as index:
        index.upsert([IndexEntry(_scene(), saved=SavedLayer("visual", saved, CHIP))])
        [chip] = index.query(_query()).groups[0].cards
        index.upsert([IndexEntry(_scene(), saved=SavedLayer("visual", saved))])
        [full] = index.query(_query()).groups[0].cards
    assert chip.layers["visual"].availability is Availability.windowed
    assert chip.layers["visual"].window == CHIP
    assert full.layers["visual"].availability is Availability.full
    assert full.layers["visual"].window is None


def test_deleted_file_is_reported_missing_and_keeps_its_locator(tmp_path: Path) -> None:
    saved = _file(tmp_path)
    with _index(tmp_path) as index:
        index.upsert([IndexEntry(_scene(), saved=SavedLayer("visual", saved, CHIP))])
        saved.unlink()
        [card] = index.query(_query()).groups[0].cards
        saved.write_bytes(b"II*\x00")
        [back] = index.query(_query()).groups[0].cards
    assert card.layers["visual"].availability is Availability.missing
    assert card.layers["visual"].locator is not None
    assert card.layers["visual"].window == CHIP
    assert not card.has_local_data
    assert back.layers["visual"].availability is Availability.windowed


def test_locators_follow_the_workspace(tmp_path: Path) -> None:
    data = tmp_path / ".atlas" / "data"
    saved = _file(data)
    with _index(data) as index:
        index.upsert([IndexEntry(_scene(), saved=SavedLayer("visual", saved))])
        inside = index.query(_query(), workspace=tmp_path)
        outside = index.query(_query(), workspace=tmp_path / "elsewhere")
    inner = inside.groups[0].cards[0].layers["visual"].locator
    outer = outside.groups[0].cards[0].layers["visual"].locator
    assert inner is not None and outer is not None
    assert (inner.scheme, inner.key) == ("workspace", ".atlas/data/sentinel2/S2A_visual.tif")
    assert inside.index_dirs == [".atlas/data"]
    assert (outer.scheme, outer.key) == ("index", "sentinel2/S2A_visual.tif")
    assert outside.index_dirs == [str(data.resolve())]


def test_a_file_outside_the_index_folder_is_not_recorded_as_local(tmp_path: Path) -> None:
    stray = _file(tmp_path, "stray.tif")
    with _index(tmp_path / "data") as index:
        index.upsert([IndexEntry(_scene(), saved=SavedLayer("visual", stray))])
        [card] = index.query(_query()).groups[0].cards
    assert not card.has_local_data


def test_a_local_layer_stays_on_a_capped_card(tmp_path: Path) -> None:
    assets = {
        f"band_{n:03d}": {"href": f"https://example.com/{n}.tif", "roles": ["data"]}
        for n in range(40)
    }
    saved = _file(tmp_path, "sentinel2/last.tif")
    with _index(tmp_path) as index:
        index.upsert(
            [IndexEntry(_scene(_feature(assets=assets)), saved=SavedLayer("band_039", saved))]
        )
        index.upsert([IndexEntry(_scene(_feature(assets=assets)))])
        [card] = index.query(_query()).groups[0].cards
    assert card.layers["band_039"].availability is Availability.full
    assert card.layers_omitted == 40 - len(card.layers)


# --- versions and rebuilds --------------------------------------------------


def test_rebuild_keeps_index_ids_and_local_layers(tmp_path: Path) -> None:
    saved = _file(tmp_path)
    scenes = [_scene(_feature(f"S2A_{n}")) for n in range(3)]
    with _index(tmp_path) as index:
        index.upsert([IndexEntry(scene) for scene in scenes[1:]])
        index.upsert([IndexEntry(scenes[0], saved=SavedLayer("visual", saved, CHIP))])
        before = index.query(_query()).index_ids
        assert index.rebuild() == 3
        result = index.query(_query())
    assert result.index_ids == before
    chip = next(card for card in result.groups[0].cards if card.provider_id == "S2A_0")
    assert chip.layers["visual"].window == CHIP


def test_older_card_version_rebuilds_on_open(tmp_path: Path) -> None:
    saved = _file(tmp_path)
    with _index(tmp_path) as index:
        index.upsert([IndexEntry(_scene(), saved=SavedLayer("visual", saved))])
    db = sqlite3.connect(tmp_path / "index.sqlite")
    with db:
        [(raw,)] = db.execute("SELECT card_json FROM scenes").fetchall()
        stale = json.loads(raw)
        stale["layers"]["B04"]["roles"] = ["stale"]
        db.execute("UPDATE scenes SET card_json = ?", (json.dumps(stale),))
        db.execute("UPDATE meta SET value = '0' WHERE key = 'card_version'")
    db.close()
    with _index(tmp_path) as index:
        [card] = index.query(_query()).groups[0].cards
    assert card.layers["B04"].roles == ["data"]
    assert card.layers["visual"].availability is Availability.full


def test_newer_index_refuses_to_open(tmp_path: Path) -> None:
    _index(tmp_path).close()
    db = sqlite3.connect(tmp_path / "index.sqlite")
    with db:
        db.execute("UPDATE meta SET value = '99' WHERE key = 'schema_version'")
    db.close()
    with pytest.raises(IndexVersionError, match="newer Atlas"):
        _index(tmp_path)


def test_newer_index_is_left_unchanged(tmp_path: Path) -> None:
    db = sqlite3.connect(tmp_path / "index.sqlite")
    with db:
        db.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        db.execute("INSERT INTO meta VALUES ('schema_version', '99')")
    db.close()
    with pytest.raises(IndexVersionError):
        _index(tmp_path)
    db = sqlite3.connect(tmp_path / "index.sqlite")
    tables = {name for (name,) in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    mode = db.execute("PRAGMA journal_mode").fetchone()[0]
    db.close()
    assert tables == {"meta"}
    assert mode == "delete"


# --- place and time ---------------------------------------------------------


def test_tilted_footprint_whose_bbox_overlaps_but_shape_does_not(tmp_path: Path) -> None:
    # A diamond centred east of the AOI: its bbox covers the AOI, its shape does not.
    diamond = {
        "type": "Polygon",
        "coordinates": [
            [[-71.2, 42.6], [-70.4, 42.2], [-71.2, 41.8], [-72.0, 42.2], [-71.2, 42.6]]
        ],
    }
    feature = _feature(bounds=(-72.0, 41.8, -70.4, 42.6), geometry=diamond)
    aoi = BBox(west=-70.7, south=42.45, east=-70.5, north=42.55)
    with _index(tmp_path) as index:
        index.upsert([IndexEntry(_scene(feature))])
        assert index.query(_query(bbox=aoi)).empty
        inside = BBox(west=-71.3, south=42.1, east=-71.1, north=42.3)
        assert not index.query(_query(bbox=inside)).empty


def test_touching_edge_counts_as_a_match() -> None:
    bbox = BBox(west=-71.5, south=42.0, east=-71.12, north=42.8)
    footprint = _square(-71.5, 42.0, -71.12, 42.8)["coordinates"]
    assert footprint_intersects(footprint, bbox, AOI)
    assert footprint_intersects(None, bbox, AOI)


def test_antimeridian_scene_and_aoi(tmp_path: Path) -> None:
    # Longitudes past 180 are not valid GeoJSON; the ring is dropped and the wrapping bbox is used.
    unwrapped = [[179.5, -17], [180.5, -17], [180.5, -16], [179.5, -16], [179.5, -17]]
    feature = _feature(
        bounds=(179.5, -17.0, -179.5, -16.0),
        geometry={"type": "Polygon", "coordinates": [unwrapped]},
    )
    scene = _scene(feature)
    assert scene.footprint is None
    with _index(tmp_path) as index:
        index.upsert([IndexEntry(scene)])
        west_side = BBox(west=-179.9, south=-16.8, east=-179.7, north=-16.2)
        east_side = BBox(west=179.6, south=-16.8, east=179.8, north=-16.2)
        wrapping = BBox(west=179.9, south=-16.8, east=-179.9, north=-16.2)
        elsewhere = BBox(west=10, south=-16.8, east=11, north=-16.2)
        for aoi in (west_side, east_side, wrapping):
            assert not index.query(_query(bbox=aoi)).empty
        assert index.query(_query(bbox=elsewhere)).empty


def test_date_windows_are_inclusive_utc_days(tmp_path: Path) -> None:
    scenes = [
        _scene(_feature("last-second", when="2024-06-10T23:59:59Z")),
        _scene(_feature("next-day", when="2024-06-11T00:00:00Z")),
        _scene(_feature("offset", when="2024-06-11T03:00:00+05:00")),  # 22:00Z on the 10th
        _scene(_feature("first-second", when="2024-06-01T00:00:00Z")),
    ]
    with _index(tmp_path) as index:
        index.upsert([IndexEntry(scene) for scene in scenes])
        cards = index.query(_query(end=date(2024, 6, 10))).groups[0].cards
    assert [card.provider_id for card in cards] == ["first-second", "offset", "last-second"]


def test_on_dates_report_gaps_instead_of_substituting(tmp_path: Path) -> None:
    with _index(tmp_path) as index:
        index.upsert([IndexEntry(_scene())])
        result = index.query(_query(start=None, end=None, on=[date(2024, 6, 3), date(2024, 6, 4)]))
    before, after = result.groups
    assert len(before.cards) == 1 and not before.gap
    assert after.gap and after.cards == []
    assert not result.empty
    payload = json.loads(result.to_json())
    assert [group["gap"] for group in payload["groups"]] == [False, True]


def test_query_needs_exactly_one_kind_of_time_filter() -> None:
    with pytest.raises(ValueError, match="either start and end"):
        SceneQuery(bbox=AOI)
    with pytest.raises(ValueError, match="either start and end"):
        SceneQuery(bbox=AOI, start=date(2024, 6, 1), end=date(2024, 6, 2), on=[date(2024, 6, 1)])
    with pytest.raises(ValueError, match="both start and end"):
        SceneQuery(bbox=AOI, start=date(2024, 6, 1))


# --- cloud, filters, order, lookup ------------------------------------------


def _radar(item_id: str = "S1A_IW_GRDH") -> Scene:
    return _scene(_feature(item_id, cloud=None), source="sentinel1", collection="sentinel-1-rtc")


def test_cloud_threshold_drops_unknown_cloud_but_keeps_sensors_that_do_not_see_cloud(
    tmp_path: Path,
) -> None:
    radar = _radar().model_copy(update={"kind": SceneKind.sar})
    scenes = [
        _scene(_feature("clear", cloud=5)),
        _scene(_feature("cloudy", cloud=80)),
        _scene(_feature("unknown", cloud=None)),
    ]
    with _index(tmp_path) as index:
        index.upsert([IndexEntry(scene) for scene in scenes])
        index.upsert([IndexEntry(radar, satellite="sentinel1")])
        group = index.query(_query(max_cloud_cover=20)).groups[0]
    kept = {card.provider_id: card for card in group.cards}
    assert set(kept) == {"clear", "S1A_IW_GRDH"}
    assert group.excluded_unknown_cloud == 1
    assert kept["S1A_IW_GRDH"].cloud_status is CloudStatus.not_applicable


def test_cloud_order_keeps_the_clearest_scenes_under_the_limit(tmp_path: Path) -> None:
    clouds = {"day01": 60, "day02": None, "day03": 5, "day04": 20, "day05": 5}
    scenes = [
        _scene(_feature(item_id, when=f"2024-06-{item_id[-2:]}T15:00:00Z", cloud=cloud))
        for item_id, cloud in clouds.items()
    ]
    radar = _radar().model_copy(update={"kind": SceneKind.sar})
    with _index(tmp_path) as index:
        index.upsert([IndexEntry(scene) for scene in scenes])
        index.upsert([IndexEntry(radar, satellite="sentinel1")])
        clearest = index.query(_query(order="cloud", limit=3)).groups[0]
        everything = index.query(_query(order="cloud")).groups[0]
        oldest = index.query(_query(limit=3)).groups[0]
    assert [card.provider_id for card in clearest.cards] == ["day03", "day05", "day04"]
    assert clearest.matched == 6
    assert [card.provider_id for card in everything.cards][-2:] == ["S1A_IW_GRDH", "day02"]
    assert [card.provider_id for card in oldest.cards] == ["day01", "day02", "day03"]


def test_filters_limit_and_ordering(tmp_path: Path) -> None:
    scenes = [
        _scene(_feature(f"S2_{day:02d}", when=f"2024-06-{day:02d}T15:00:00Z")) for day in (9, 3, 6)
    ]
    landsat = _scene(
        _feature("LC09_x", platform="landsat-9"), source="landsat9", collection="landsat-c2-l2"
    )
    with _index(tmp_path) as index:
        index.upsert([IndexEntry(scene) for scene in [*scenes, landsat]])
        limited = index.query(_query(sources=["sentinel2"], limit=2)).groups[0]
        by_platform = index.query(_query(platforms=["Landsat-9"])).groups[0]
        by_satellite = index.query(_query(satellites=["Landsat9"])).groups[0]
    assert [card.provider_id for card in limited.cards] == ["S2_03", "S2_06"]
    assert limited.matched == 3
    assert [card.provider_id for card in by_platform.cards] == ["LC09_x"]
    assert [card.provider_id for card in by_satellite.cards] == ["LC09_x"]


def test_lookup_resolves_a_cited_id(tmp_path: Path) -> None:
    saved = _file(tmp_path)
    with _index(tmp_path) as index:
        stats = index.upsert([IndexEntry(_scene(), saved=SavedLayer("visual", saved))])
        [index_id] = stats.index_ids.values()
        found = index.lookup(index_id, workspace=tmp_path)
        missing = index.lookup("sc_" + "0" * 20)
    assert found.status is LookupStatus.found and found.card is not None
    assert found.card.index_id == index_id
    visual = found.card.layers["visual"].locator
    assert visual is not None and visual.scheme == "workspace"
    assert missing.status is LookupStatus.missing and missing.card is None


def test_multipolygon_footprint_and_crossing_ring() -> None:
    multi = _feature(
        geometry={
            "type": "MultiPolygon",
            "coordinates": [
                _square(-71.5, 42.0, -71.3, 42.2)["coordinates"],
                _square(-71.0, 42.5, -70.8, 42.7)["coordinates"],
            ],
        }
    )
    assert len(_scene(multi).footprint or []) == 2
    crossing = _feature(
        bounds=(179.5, -17.0, -179.5, -16.0),
        geometry={
            "type": "Polygon",
            "coordinates": [
                [[179.5, -17], [-179.5, -17], [-179.5, -16], [179.5, -16], [179.5, -17]]
            ],
        },
    )
    assert _scene(crossing).footprint is None
