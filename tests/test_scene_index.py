from __future__ import annotations

import json
import sqlite3
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest

from atlas.data.base import BBox, Scene
from atlas.data.stac import item_to_scene
from atlas.index import (
    MAX_CARD_BANDS,
    Availability,
    IndexEntry,
    IndexVersionError,
    SceneIndex,
    SceneQuery,
    make_index_id,
)
from atlas.index.geometry import footprint_intersects

AOI = BBox(west=-71.12, south=42.32, east=-71.02, north=42.40)
PC = "planetarycomputer.microsoft.com"
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
    catalog: str = PC,
    collection: str = "sentinel-2-l2a",
) -> Scene:
    scene = item_to_scene(feature or _feature(), collection=collection, source=source)
    assert scene is not None
    return scene.model_copy(update={"catalog": catalog})


def _index(tmp_path: Path) -> SceneIndex:
    return SceneIndex.at(tmp_path)


def _query(**kwargs: Any) -> SceneQuery:
    body: dict[str, Any] = {"bbox": AOI, "start": date(2024, 6, 1), "end": date(2024, 6, 30)}
    body.update(kwargs)
    return SceneQuery.model_validate(body)


def test_upsert_twice_keeps_one_card_with_a_stable_id(tmp_path: Path) -> None:
    with _index(tmp_path) as index:
        first = index.upsert([IndexEntry(_scene(), "sentinel2")], now=NOW)
        second = index.upsert([IndexEntry(_scene(), "sentinel2")], now=NOW)
        result = index.query(_query())
    assert (first.added, second.added, second.updated) == (1, 0, 1)
    assert result.index_ids == [make_index_id(PC, "sentinel-2-l2a", "S2A_MSIL2A_20240603")]


def test_same_item_found_via_two_sources_is_one_card(tmp_path: Path) -> None:
    feature = _feature("LC09_L2SP_012031_20240603", platform="landsat-9")
    with _index(tmp_path) as index:
        index.upsert(
            [
                IndexEntry(
                    _scene(feature, source="landsat8", collection="landsat-c2-l2"), "landsat8"
                ),
                IndexEntry(
                    _scene(feature, source="landsat9", collection="landsat-c2-l2"), "landsat9"
                ),
            ]
        )
        [card] = index.query(_query()).groups[0].cards
    assert card.found_via == ["landsat8", "landsat9"]
    assert card.catalog == PC


def test_card_is_short_and_has_no_urls(tmp_path: Path) -> None:
    with _index(tmp_path) as index:
        index.upsert([IndexEntry(_scene(), "sentinel2")])
        result = index.query(_query())
    payload = result.to_json()
    assert "http" not in payload
    assert "eo:cloud_cover" not in payload  # raw STAC properties stay in the store
    card = result.groups[0].cards[0]
    assert card.bands == ["B04", "visual"]  # browse assets are not bands
    assert card.cloud_cover == 12.5
    assert card.availability is Availability.found
    assert card.footprint_exact is True
    assert json.loads(payload)["groups"][0]["gap"] is False


def test_snapshot_drops_signed_query_strings(tmp_path: Path) -> None:
    assets = {"B04": {"href": "https://example.com/B04.tif?st=2024&sig=SECRET", "roles": ["data"]}}
    with _index(tmp_path) as index:
        index.upsert([IndexEntry(_scene(_feature(assets=assets)), "sentinel2")])
    stored = sqlite3.connect(tmp_path / "index.sqlite").execute("SELECT scene_json FROM scenes")
    [(raw,)] = stored.fetchall()
    assert "SECRET" not in raw
    assert "https://example.com/B04.tif" in raw


def test_rebuild_keeps_index_ids(tmp_path: Path) -> None:
    scenes = [_scene(_feature(f"S2A_{n}")) for n in range(3)]
    with _index(tmp_path) as index:
        index.upsert([IndexEntry(scene, "sentinel2") for scene in scenes])
        before = index.query(_query()).index_ids
        assert index.rebuild() == 3
        after = index.query(_query()).index_ids
    assert before == after
    assert len(after) == 3


def test_older_card_version_rebuilds_on_open(tmp_path: Path) -> None:
    with _index(tmp_path) as index:
        index.upsert([IndexEntry(_scene(), "sentinel2")])
    db = sqlite3.connect(tmp_path / "index.sqlite")
    with db:
        db.execute("UPDATE meta SET value = '0' WHERE key = 'card_version'")
        db.execute("UPDATE scenes SET bands = '[]', card_version = 0")
    db.close()
    with _index(tmp_path) as index:
        [card] = index.query(_query()).groups[0].cards
    assert card.bands == ["B04", "visual"]


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


def test_download_survives_later_search_only_pull(tmp_path: Path) -> None:
    saved = tmp_path / "sentinel2" / "S2A_visual.tif"
    saved.parent.mkdir()
    saved.write_bytes(b"II*\x00")
    with _index(tmp_path) as index:
        index.upsert([IndexEntry(_scene(), "sentinel2", local_path=saved)])
        index.upsert([IndexEntry(_scene(), "sentinel2")])
        [card] = index.query(_query(), cwd=tmp_path).groups[0].cards
    assert card.availability is Availability.downloaded
    assert card.local_path == "sentinel2/S2A_visual.tif"


def test_deleted_file_is_reported_missing(tmp_path: Path) -> None:
    saved = tmp_path / "S2A_visual.tif"
    saved.write_bytes(b"II*\x00")
    with _index(tmp_path) as index:
        index.upsert([IndexEntry(_scene(), "sentinel2", local_path=saved)])
        saved.unlink()
        [card] = index.query(_query()).groups[0].cards
    assert card.availability is Availability.missing


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
        index.upsert([IndexEntry(_scene(feature), "sentinel2")])
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
        index.upsert([IndexEntry(scene, "sentinel2")])
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
        index.upsert([IndexEntry(scene, "sentinel2") for scene in scenes])
        cards = index.query(_query(end=date(2024, 6, 10))).groups[0].cards
    assert [card.item_id for card in cards] == ["first-second", "offset", "last-second"]


def test_unknown_cloud_is_excluded_by_threshold_and_counted(tmp_path: Path) -> None:
    scenes = [
        _scene(_feature("clear", cloud=5)),
        _scene(_feature("cloudy", cloud=80)),
        _scene(_feature("unknown", cloud=None)),
    ]
    with _index(tmp_path) as index:
        index.upsert([IndexEntry(scene, "sentinel2") for scene in scenes])
        group = index.query(_query(max_cloud_cover=20)).groups[0]
        everything = index.query(_query()).groups[0]
    assert [card.item_id for card in group.cards] == ["clear"]
    assert group.excluded_unknown_cloud == 1
    unknown = next(card for card in everything.cards if card.item_id == "unknown")
    assert unknown.cloud_cover is None


def test_scene_without_bands_is_indexed(tmp_path: Path) -> None:
    with _index(tmp_path) as index:
        index.upsert([IndexEntry(_scene(_feature(assets={})), "sentinel2")])
        [card] = index.query(_query()).groups[0].cards
    assert card.bands == []


def test_bands_skip_metadata_and_list_common_bands_first(tmp_path: Path) -> None:
    def band(key: str, common: str | None = None, wavelength: float | None = None) -> Any:
        spec: dict[str, Any] = {"href": f"https://example.com/{key}.tif", "roles": ["data"]}
        if common is not None:
            spec["eo:bands"] = [{"common_name": common, "center_wavelength": wavelength}]
        return spec

    assets = {
        "AOT": band("AOT"),
        "B02": band("B02", "blue", 0.49),
        "B04": band("B04", "red", 0.665),
        "B08": band("B08", "nir", 0.842),
        "granule-metadata": {"href": "https://example.com/MTD.xml", "roles": ["metadata"]},
        "tilejson": {
            "href": "https://example.com/tilejson",
            "type": "application/json",
            "roles": ["tiles"],
        },
        "safe-manifest": {"href": "https://example.com/manifest.safe", "roles": ["metadata"]},
        "rendered_preview": {"href": "https://example.com/p.png", "roles": ["overview"]},
    }
    with _index(tmp_path) as index:
        index.upsert([IndexEntry(_scene(_feature(assets=assets)), "sentinel2")])
        [card] = index.query(_query()).groups[0].cards
    assert card.bands == ["B04", "B08", "AOT", "B02"]
    assert card.band_count == 4
    assert card.band_aliases == {"nir": "B08", "red": "B04"}


def test_card_band_list_is_capped(tmp_path: Path) -> None:
    assets: dict[str, Any] = {
        f"band_{n:03d}": {"href": f"https://example.com/{n}.tif", "roles": ["data"]}
        for n in range(MAX_CARD_BANDS + 16)
    }
    assets["z_red"] = {
        "href": "https://example.com/red.tif",
        "roles": ["data"],
        "eo:bands": [{"common_name": "red", "center_wavelength": 0.665}],
    }
    with _index(tmp_path) as index:
        index.upsert([IndexEntry(_scene(_feature(assets=assets)), "sentinel2")])
        [card] = index.query(_query()).groups[0].cards
    assert len(card.bands) == MAX_CARD_BANDS
    assert card.band_count == MAX_CARD_BANDS + 17
    assert card.bands[0] == "z_red"  # a common band survives the cut
    assert len(card.model_dump_json()) < 1024


def test_cloud_order_keeps_the_clearest_scenes_under_the_limit(tmp_path: Path) -> None:
    clouds = {"day01": 60, "day02": None, "day03": 5, "day04": 20, "day05": 5}
    scenes = [
        _scene(_feature(item_id, when=f"2024-06-{item_id[-2:]}T15:00:00Z", cloud=cloud))
        for item_id, cloud in clouds.items()
    ]
    with _index(tmp_path) as index:
        index.upsert([IndexEntry(scene, "sentinel2") for scene in scenes])
        clearest = index.query(_query(order="cloud", limit=3)).groups[0]
        everything = index.query(_query(order="cloud")).groups[0]
        oldest = index.query(_query(limit=3)).groups[0]
    assert [card.item_id for card in clearest.cards] == ["day03", "day05", "day04"]
    assert clearest.matched == 5
    assert [card.item_id for card in everything.cards][-1] == "day02"  # no cloud data: last
    assert [card.item_id for card in oldest.cards] == ["day01", "day02", "day03"]


def test_on_dates_report_gaps_instead_of_substituting(tmp_path: Path) -> None:
    with _index(tmp_path) as index:
        index.upsert([IndexEntry(_scene(), "sentinel2")])
        result = index.query(_query(start=None, end=None, on=[date(2024, 6, 3), date(2024, 6, 4)]))
    before, after = result.groups
    assert len(before.cards) == 1 and not before.gap
    assert after.gap and after.cards == []
    assert not result.empty
    payload = json.loads(result.to_json())
    assert [group["gap"] for group in payload["groups"]] == [False, True]


def test_request_footprint_products_are_not_indexed(tmp_path: Path) -> None:
    gibs = _scene().model_copy(update={"footprint_is_request": True, "id": "gibs-tile"})
    with _index(tmp_path) as index:
        stats = index.upsert([IndexEntry(gibs, "gibs")])
        assert index.query(_query()).empty
    assert stats.skipped_request_footprint == 1


def test_filters_limit_and_ordering(tmp_path: Path) -> None:
    scenes = [
        _scene(_feature(f"S2_{day:02d}", when=f"2024-06-{day:02d}T15:00:00Z")) for day in (9, 3, 6)
    ]
    landsat = _scene(
        _feature("LC09_x", platform="landsat-9"), source="landsat9", collection="landsat-c2-l2"
    )
    with _index(tmp_path) as index:
        index.upsert([IndexEntry(scene, "sentinel2") for scene in scenes])
        index.upsert([IndexEntry(landsat, "landsat9")])
        limited = index.query(_query(sources=["sentinel2"], limit=2)).groups[0]
        by_platform = index.query(_query(platforms=["Landsat-9"])).groups[0]
    assert [card.item_id for card in limited.cards] == ["S2_03", "S2_06"]
    assert limited.matched == 3
    assert [card.item_id for card in by_platform.cards] == ["LC09_x"]


def test_get_follows_aliases(tmp_path: Path) -> None:
    with _index(tmp_path) as index:
        index.upsert([IndexEntry(_scene(), "sentinel2")])
        [index_id] = index.query(_query()).index_ids
        index._db.execute("INSERT INTO id_aliases VALUES ('sc_old', ?)", (index_id,))
        card = index.get("sc_old")
    assert card is not None and card.index_id == index_id


def test_query_needs_exactly_one_kind_of_time_filter() -> None:
    with pytest.raises(ValueError, match="either start and end"):
        SceneQuery(bbox=AOI)
    with pytest.raises(ValueError, match="either start and end"):
        SceneQuery(bbox=AOI, start=date(2024, 6, 1), end=date(2024, 6, 2), on=[date(2024, 6, 1)])
    with pytest.raises(ValueError, match="both start and end"):
        SceneQuery(bbox=AOI, start=date(2024, 6, 1))


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
