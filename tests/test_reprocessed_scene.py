"""A reprocessed scene is found by collection, time, and footprint."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar

import httpx
import pytest

from atlas.cli.data import add_data_parser, run_data
from atlas.data.base import BBox, DataPullClient, PullRequest, PullResult, Scene, SceneKind
from atlas.data.scene_identity import SourceResolution, project_records, reconcile, stored_scenes
from atlas.data.stac import StacApiClient

AOI = BBox(west=0, south=0, east=1, north=1)
FAR = BBox(west=10, south=10, east=11, north=11)
# Adjacent tile: shares a sliver with the stored footprint, IoU well under 0.8.
NEIGHBOUR = BBox(west=0.9, south=0, east=1.9, north=1)
# Shares the east edge and nothing else. intersects() is true; area in common is 0.
EDGE = BBox(west=1, south=0, east=2, north=1)
CORNER = BBox(west=1, south=1, east=2, north=2)
DATELINE = BBox(west=179, south=-18, east=-179, north=-17)
DATELINE_EDGE = BBox(west=-179, south=-18, east=-178, north=-17)
WHEN = datetime(2024, 7, 1, 15, 30, tzinfo=UTC)
COLLECTION = "sentinel-2-l2a"
SEARCH = "https://catalog.example/search"
_EMPTY_WINDOW = "requested window is empty"


def _args(tmp_path: Path, *extra: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="group")
    add_data_parser(sub)
    return parser.parse_args(
        [
            "data",
            "sentinel2",
            "optical",
            "--date",
            "2024-07-01:2024-07-07",
            "--coords",
            "0,0,1,1",
            "--out",
            str(tmp_path),
            "--search-only",
            *extra,
        ]
    )


def _scene(
    scene_id: str,
    *,
    when: datetime = WHEN,
    bbox: BBox = AOI,
    collection: str = COLLECTION,
    properties: dict[str, Any] | None = None,
) -> Scene:
    return Scene(
        id=scene_id,
        source="sentinel2",
        collection=collection,
        datetime=when,
        bbox=bbox,
        platform="sentinel-2",
        properties=properties or {},
    )


class _Catalog(DataPullClient):
    """In-memory catalog. Id checks and place/time hits are scripted."""

    satellite = "sentinel2"
    scene_kind = SceneKind.optical

    def __init__(self) -> None:
        self.collection = COLLECTION
        self.search_url = SEARCH
        self.pages: list[Scene] = []
        self.present: dict[str, Scene] = {}
        self.candidates: list[Scene] = []
        self.resolve_calls: list[list[str]] = []
        self.find_calls: list[tuple[str, datetime, BBox]] = []

    async def search(self, request: PullRequest) -> PullResult:
        return PullResult(request=request, scenes=list(self.pages))

    async def search_from_cursor(self, request: PullRequest, page_cursor: str) -> PullResult:
        del page_cursor
        return PullResult(request=request, scenes=[])

    async def resolve_scene_ids(self, scene_ids: Sequence[str]) -> Mapping[str, Scene]:
        self.resolve_calls.append(list(scene_ids))
        return {
            scene_id: self.present[scene_id] for scene_id in scene_ids if scene_id in self.present
        }

    async def find_scenes_by_place_time(
        self,
        *,
        collection: str,
        when: datetime,
        footprint: BBox,
    ) -> list[Scene]:
        self.find_calls.append((collection, when, footprint))
        return list(self.candidates)


def _manifest(tmp_path: Path) -> dict[str, Any]:
    return json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))


def _source(manifest: dict[str, Any]) -> dict[str, Any]:
    return manifest["sources"][0]


def test_manifest_stores_collection_datetime_and_footprint(tmp_path: Path) -> None:
    catalog = _Catalog()
    catalog.pages = [_scene("scene-old")]
    assert run_data(_args(tmp_path), clients={"sentinel2": catalog}) == 0
    source = _source(_manifest(tmp_path))
    assert source["scene_ids"] == ["scene-old"]
    record = source["scenes"][0]
    assert record == {
        "id": "scene-old",
        "collection": COLLECTION,
        "datetime": "2024-07-01T15:30:00Z",
        "footprint": [0.0, 0.0, 1.0, 1.0],
    }
    assert source["replacements"] == []
    assert source["unresolved_scenes"] == []


def test_old_manifest_without_place_and_time_still_loads(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = {
        "request": {
            "start_date": "2024-07-01",
            "end_date": "2024-07-07",
            "bbox": {"west": 0, "south": 0, "east": 1, "north": 1},
            "polygon": None,
            "min_cloud_cover": None,
            "max_cloud_cover": None,
        },
        "sources": [
            {
                "source": "sentinel2",
                "collection": COLLECTION,
                "search_url": SEARCH,
                "scene_ids": ["legacy-scene"],
                "next_cursor": None,
            }
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    catalog = _Catalog()
    catalog.present = {"legacy-scene": _scene("legacy-scene")}
    assert run_data(_args(tmp_path, "--resume"), clients={"sentinel2": catalog}) == 0
    source = _source(_manifest(tmp_path))
    err = capsys.readouterr().err
    assert source["scene_ids"] == ["legacy-scene"]
    assert "no match" not in err
    assert "ambiguous" not in err
    assert "replaced scene" not in err
    assert source["scenes"][0]["id"] == "legacy-scene"
    assert source["scenes"][0]["collection"] == COLLECTION


def test_old_manifest_does_not_guess_when_the_id_is_gone(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = {
        "request": {
            "start_date": "2024-07-01",
            "end_date": "2024-07-07",
            "bbox": {"west": 0, "south": 0, "east": 1, "north": 1},
            "polygon": None,
        },
        "sources": [
            {
                "source": "sentinel2",
                "collection": COLLECTION,
                "search_url": SEARCH,
                "scene_ids": ["legacy-scene"],
                "next_cursor": None,
            }
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    catalog = _Catalog()
    catalog.candidates = [_scene("would-match")]
    assert run_data(_args(tmp_path, "--resume"), clients={"sentinel2": catalog}) == 0
    source = _source(_manifest(tmp_path))
    err = capsys.readouterr().err
    assert source["scene_ids"] == ["legacy-scene"]
    assert "would-match" not in source["scene_ids"]
    assert "legacy-scene" in err
    assert "no match" in err
    assert "replaced scene" not in err
    assert _EMPTY_WINDOW not in err
    assert source["failed"] == []
    assert catalog.find_calls == []


def test_resume_replaces_a_reprocessed_scene_and_reports_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    catalog = _Catalog()
    catalog.pages = [_scene("old-scene")]
    assert run_data(_args(tmp_path), clients={"sentinel2": catalog}) == 0
    capsys.readouterr()
    catalog.pages = []
    catalog.present = {}
    catalog.candidates = [
        _scene("new-scene", when=WHEN + timedelta(seconds=60)),
        _scene("other-collection", collection="landsat-c2-l2"),
        _scene("too-late", when=WHEN + timedelta(seconds=61)),
        _scene("elsewhere", bbox=FAR),
    ]
    assert run_data(_args(tmp_path, "--resume"), clients={"sentinel2": catalog}) == 0
    source = _source(_manifest(tmp_path))
    err = capsys.readouterr().err
    assert source["scene_ids"] == ["new-scene"]
    assert source["replacements"] == [{"old_id": "old-scene", "new_id": "new-scene"}]
    assert source["unresolved_scenes"] == []
    record = source["scenes"][0]
    assert record["id"] == "new-scene"
    assert record["collection"] == COLLECTION
    assert record["datetime"] == "2024-07-01T15:31:00Z"
    assert record["footprint"] == [0.0, 0.0, 1.0, 1.0]
    assert "replaced scene old-scene with new-scene" in err
    assert "other-collection" not in source["scene_ids"]
    assert "too-late" not in source["scene_ids"]
    assert "elsewhere" not in source["scene_ids"]
    assert catalog.resolve_calls == [["old-scene"]]
    assert len(catalog.find_calls) == 1
    collection, when, footprint = catalog.find_calls[0]
    assert collection == COLLECTION
    assert when == WHEN
    assert footprint.as_list() == [0.0, 0.0, 1.0, 1.0]


def test_refresh_replaces_a_reprocessed_scene_and_reports_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    catalog = _Catalog()
    catalog.pages = [_scene("old-scene")]
    assert run_data(_args(tmp_path), clients={"sentinel2": catalog}) == 0
    capsys.readouterr()
    catalog.pages = [_scene("new-scene", when=WHEN + timedelta(seconds=60))]
    catalog.present = {}
    catalog.candidates = [
        _scene("new-scene", when=WHEN + timedelta(seconds=60)),
        _scene("too-late", when=WHEN + timedelta(seconds=61)),
    ]
    assert run_data(_args(tmp_path), clients={"sentinel2": catalog}) == 0
    source = _source(_manifest(tmp_path))
    err = capsys.readouterr().err
    assert source["scene_ids"] == ["new-scene"]
    assert source["replacements"] == [{"old_id": "old-scene", "new_id": "new-scene"}]
    assert "replaced scene old-scene with new-scene" in err
    assert "old-scene" not in source["scene_ids"]


def test_no_match_is_reported_instead_of_an_empty_region(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    catalog = _Catalog()
    catalog.pages = [_scene("old-scene")]
    assert run_data(_args(tmp_path), clients={"sentinel2": catalog}) == 0
    capsys.readouterr()
    catalog.pages = []
    catalog.present = {}
    catalog.candidates = [
        _scene("too-late", when=WHEN + timedelta(seconds=61)),
        _scene("elsewhere", bbox=FAR),
        _scene("other-collection", collection="landsat-c2-l2"),
    ]
    assert run_data(_args(tmp_path), clients={"sentinel2": catalog}) == 0
    source = _source(_manifest(tmp_path))
    err = capsys.readouterr().err
    assert source["scene_ids"] == ["old-scene"]
    assert source["replacements"] == []
    assert source["unresolved_scenes"] == [
        {"id": "old-scene", "reason": "no match", "candidates": []}
    ]
    assert source["failed"] == []
    assert "scene old-scene has no match" in err
    assert _EMPTY_WINDOW not in err
    assert "too-late" not in source["scene_ids"]
    assert "replaced scene" not in err


def test_ambiguous_match_is_not_picked(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    catalog = _Catalog()
    catalog.pages = [_scene("old-scene")]
    assert run_data(_args(tmp_path), clients={"sentinel2": catalog}) == 0
    capsys.readouterr()
    catalog.pages = []
    catalog.present = {}
    catalog.candidates = [
        _scene("beta-scene", when=WHEN + timedelta(seconds=30)),
        _scene("alpha-scene"),
        _scene("too-late", when=WHEN + timedelta(seconds=61)),
    ]
    assert run_data(_args(tmp_path), clients={"sentinel2": catalog}) == 0
    source = _source(_manifest(tmp_path))
    err = capsys.readouterr().err
    assert source["scene_ids"] == ["old-scene"]
    assert source["replacements"] == []
    assert source["unresolved_scenes"] == [
        {
            "id": "old-scene",
            "reason": "ambiguous",
            "candidates": ["alpha-scene", "beta-scene"],
        }
    ]
    assert source["failed"] == []
    assert "scene old-scene is ambiguous (alpha-scene, beta-scene)" in err
    assert _EMPTY_WINDOW not in err
    assert "replaced scene" not in err
    assert "alpha-scene" not in source["scene_ids"]
    assert "beta-scene" not in source["scene_ids"]


def test_same_time_neighbour_tile_does_not_replace_the_original(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    catalog = _Catalog()
    catalog.pages = [_scene("old-scene")]
    assert run_data(_args(tmp_path), clients={"sentinel2": catalog}) == 0
    capsys.readouterr()
    catalog.pages = []
    catalog.present = {}
    catalog.candidates = [_scene("neighbour-tile", bbox=NEIGHBOUR)]
    assert AOI.intersects(NEIGHBOUR)
    assert AOI.iou(NEIGHBOUR) < 0.8
    assert run_data(_args(tmp_path, "--resume"), clients={"sentinel2": catalog}) == 0
    source = _source(_manifest(tmp_path))
    err = capsys.readouterr().err
    assert source["scene_ids"] == ["old-scene"]
    assert source["replacements"] == []
    assert source["unresolved_scenes"] == [
        {"id": "old-scene", "reason": "no match", "candidates": []}
    ]
    assert "neighbour-tile" not in source["scene_ids"]
    assert "scene old-scene has no match" in err
    assert "replaced scene" not in err
    assert _EMPTY_WINDOW not in err


def test_neighbour_beside_a_real_reprocess_does_not_make_it_ambiguous(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    catalog = _Catalog()
    catalog.pages = [_scene("old-scene")]
    assert run_data(_args(tmp_path), clients={"sentinel2": catalog}) == 0
    capsys.readouterr()
    catalog.pages = []
    catalog.present = {}
    catalog.candidates = [
        _scene("neighbour-tile", bbox=NEIGHBOUR),
        _scene("new-scene"),
    ]
    assert run_data(_args(tmp_path, "--resume"), clients={"sentinel2": catalog}) == 0
    source = _source(_manifest(tmp_path))
    err = capsys.readouterr().err
    assert source["scene_ids"] == ["new-scene"]
    assert source["replacements"] == [{"old_id": "old-scene", "new_id": "new-scene"}]
    assert "neighbour-tile" not in source["scene_ids"]
    assert "replaced scene old-scene with new-scene" in err
    assert "ambiguous" not in err


def test_edge_touching_boxes_do_not_match() -> None:
    assert AOI.intersects(EDGE)
    assert AOI.intersects(CORNER)
    assert AOI.iou(EDGE) == 0
    assert AOI.iou(CORNER) == 0
    stored = stored_scenes(
        {
            "scenes": [
                {
                    "id": "old-scene",
                    "collection": COLLECTION,
                    "datetime": "2024-07-01T15:30:00Z",
                    "footprint": AOI.as_list(),
                }
            ]
        }
    )
    resolution = reconcile(
        stored,
        {},
        {"old-scene": [_scene("edge-scene", bbox=EDGE), _scene("corner-scene", bbox=CORNER)]},
        collection=COLLECTION,
    )
    assert resolution.replacements == ()
    assert resolution.unresolved[0].reason == "no match"
    assert resolution.unresolved[0].candidates == ()


def test_antimeridian_footprint_matches_the_same_scene() -> None:
    assert DATELINE.west > DATELINE.east
    assert DATELINE.iou(DATELINE) == 1
    assert DATELINE.intersects(DATELINE_EDGE)
    assert DATELINE.iou(DATELINE_EDGE) == 0
    stored = stored_scenes(
        {
            "scenes": [
                {
                    "id": "old-scene",
                    "collection": COLLECTION,
                    "datetime": "2024-07-01T15:30:00Z",
                    "footprint": DATELINE.as_list(),
                }
            ]
        }
    )
    matched = reconcile(
        stored,
        {},
        {"old-scene": [_scene("new-scene", bbox=DATELINE), _scene("elsewhere", bbox=AOI)]},
        collection=COLLECTION,
    )
    assert [item.new_id for item in matched.replacements] == ["new-scene"]
    assert matched.unresolved == ()
    touched = reconcile(
        stored,
        {},
        {"old-scene": [_scene("edge-scene", bbox=DATELINE_EDGE)]},
        collection=COLLECTION,
    )
    assert touched.replacements == ()
    assert touched.unresolved[0].reason == "no match"


def test_time_window_accepts_60_seconds_and_rejects_61() -> None:
    stored = stored_scenes(
        {
            "scenes": [
                {
                    "id": "old-scene",
                    "collection": COLLECTION,
                    "datetime": "2024-07-01T15:30:00Z",
                    "footprint": AOI.as_list(),
                }
            ]
        }
    )
    for seconds, replaces in ((60, True), (-60, True), (61, False), (-61, False)):
        resolution = reconcile(
            stored,
            {},
            {"old-scene": [_scene("new-scene", when=WHEN + timedelta(seconds=seconds))]},
            collection=COLLECTION,
        )
        if replaces:
            assert [item.new_id for item in resolution.replacements] == ["new-scene"]
            assert resolution.unresolved == ()
        else:
            assert resolution.replacements == ()
            assert resolution.unresolved[0].reason == "no match"


def test_manifest_stores_the_stable_grid_key(tmp_path: Path) -> None:
    catalog = _Catalog()
    catalog.pages = [
        _scene(
            "scene-old",
            properties={
                "s2:mgrs_tile": "31TCJ",
                "grid:code": "MGRS-31TCJ",
                "sat:relative_orbit": 97,
            },
        )
    ]
    assert run_data(_args(tmp_path), clients={"sentinel2": catalog}) == 0
    record = _source(_manifest(tmp_path))["scenes"][0]
    assert record["grid_key"] == {"s2:mgrs_tile": "31TCJ"}


def test_grid_key_must_match() -> None:
    stored = stored_scenes(
        {
            "scenes": [
                {
                    "id": "old-scene",
                    "collection": COLLECTION,
                    "datetime": "2024-07-01T15:30:00Z",
                    "footprint": AOI.as_list(),
                    "grid_key": {"s2:mgrs_tile": "31TCJ"},
                }
            ]
        }
    )
    other_tile = reconcile(
        stored,
        {},
        {"old-scene": [_scene("other-tile", properties={"s2:mgrs_tile": "31TCK"})]},
        collection=COLLECTION,
    )
    assert other_tile.replacements == ()
    assert other_tile.unresolved[0].reason == "no match"
    same_tile = reconcile(
        stored,
        {},
        {"old-scene": [_scene("new-scene", properties={"s2:mgrs_tile": "31TCJ"})]},
        collection=COLLECTION,
    )
    assert [item.new_id for item in same_tile.replacements] == ["new-scene"]


def test_grid_key_priority() -> None:
    records = project_records(
        [
            _scene(
                "landsat",
                properties={
                    "grid:code": "WRS2-198030",
                    "landsat:wrs_path": 198,
                    "landsat:wrs_row": 30,
                    "sat:relative_orbit": 5,
                },
            ),
            _scene(
                "wrs-only",
                properties={"landsat:wrs_path": "44", "landsat:wrs_row": 34},
            ),
            _scene("orbit-only", properties={"sat:relative_orbit": 97}),
        ],
        [],
        SourceResolution.unchecked(),
        collection=COLLECTION,
        keep_stored=False,
    )
    assert records[0].as_manifest()["grid_key"] == {"grid:code": "WRS2-198030"}
    assert records[1].as_manifest()["grid_key"] == {
        "landsat:wrs_path": 44,
        "landsat:wrs_row": 34,
    }
    assert records[2].as_manifest()["grid_key"] == {"sat:relative_orbit": 97}


def test_newest_baseline_wins_only_when_the_scene_is_the_same() -> None:
    stored = stored_scenes(
        {
            "scenes": [
                {
                    "id": "old-scene",
                    "collection": COLLECTION,
                    "datetime": "2024-07-01T15:30:00Z",
                    "footprint": AOI.as_list(),
                    "grid_key": {"s2:mgrs_tile": "31TCJ"},
                }
            ]
        }
    )
    older = _scene(
        "older-scene",
        properties={"s2:mgrs_tile": "31TCJ", "s2:processing_baseline": "05.09"},
    )
    newer = _scene(
        "newer-scene",
        when=WHEN + timedelta(seconds=30),
        properties={
            "s2:mgrs_tile": "31TCJ",
            "s2:processing_baseline": "05.11",
            "updated": "2024-07-02T00:00:00Z",
        },
    )
    same = reconcile(stored, {}, {"old-scene": [older, newer]}, collection=COLLECTION)
    assert [item.new_id for item in same.replacements] == ["newer-scene"]
    assert same.unresolved == ()

    by_stamp = reconcile(
        stored,
        {},
        {
            "old-scene": [
                _scene(
                    "first",
                    properties={
                        "s2:mgrs_tile": "31TCJ",
                        "s2:processing_baseline": "05.09",
                        "updated": "2024-07-01T00:00:00Z",
                    },
                ),
                _scene(
                    "second",
                    properties={
                        "s2:mgrs_tile": "31TCJ",
                        "s2:processing_baseline": "05.09",
                        "created": "2024-07-03T00:00:00Z",
                    },
                ),
            ]
        },
        collection=COLLECTION,
    )
    assert [item.new_id for item in by_stamp.replacements] == ["second"]

    baseline_over_stamp = reconcile(
        stored,
        {},
        {
            "old-scene": [
                _scene(
                    "stale-baseline",
                    properties={
                        "s2:mgrs_tile": "31TCJ",
                        "s2:processing_baseline": "02.04",
                        "updated": "2024-08-01T00:00:00Z",
                    },
                ),
                _scene(
                    "fresh-baseline",
                    properties={
                        "s2:mgrs_tile": "31TCJ",
                        "s2:processing_baseline": "05.11",
                        "updated": "2024-07-01T00:00:00Z",
                    },
                ),
            ]
        },
        collection=COLLECTION,
    )
    assert [item.new_id for item in baseline_over_stamp.replacements] == ["fresh-baseline"]

    bare = stored_scenes(
        {
            "scenes": [
                {
                    "id": "old-scene",
                    "collection": COLLECTION,
                    "datetime": "2024-07-01T15:30:00Z",
                    "footprint": AOI.as_list(),
                }
            ]
        }
    )
    different = reconcile(
        bare,
        {},
        {
            "old-scene": [
                _scene(
                    "tile-a",
                    properties={"s2:mgrs_tile": "31TCJ", "s2:processing_baseline": "05.11"},
                ),
                _scene(
                    "tile-b",
                    properties={"s2:mgrs_tile": "31TCK", "s2:processing_baseline": "02.04"},
                ),
            ]
        },
        collection=COLLECTION,
    )
    assert different.replacements == ()
    assert different.unresolved[0].reason == "ambiguous"
    assert different.unresolved[0].candidates == ("tile-a", "tile-b")

    tied = reconcile(
        stored,
        {},
        {
            "old-scene": [
                _scene(
                    "tie-b",
                    properties={"s2:mgrs_tile": "31TCJ", "s2:processing_baseline": "05.09"},
                ),
                _scene(
                    "tie-a",
                    properties={"s2:mgrs_tile": "31TCJ", "s2:processing_baseline": "05.09"},
                ),
            ]
        },
        collection=COLLECTION,
    )
    assert tied.replacements == ()
    assert tied.unresolved[0].reason == "ambiguous"
    assert tied.unresolved[0].candidates == ("tie-a", "tie-b")

    wide = stored_scenes(
        {
            "scenes": [
                {
                    "id": "old-scene",
                    "collection": COLLECTION,
                    "datetime": "2024-07-01T15:30:00Z",
                    "footprint": [0.0, 0.0, 10.0, 1.0],
                    "grid_key": {"s2:mgrs_tile": "31TCJ"},
                }
            ]
        }
    )
    shifted = reconcile(
        wide,
        {},
        {
            "old-scene": [
                _scene(
                    "left",
                    bbox=BBox(west=0, south=0, east=8.5, north=1),
                    properties={"s2:mgrs_tile": "31TCJ", "s2:processing_baseline": "05.11"},
                ),
                _scene(
                    "right",
                    bbox=BBox(west=1.5, south=0, east=10, north=1),
                    properties={"s2:mgrs_tile": "31TCJ", "s2:processing_baseline": "02.04"},
                ),
            ]
        },
        collection=COLLECTION,
    )
    assert shifted.replacements == ()
    assert shifted.unresolved[0].reason == "ambiguous"
    assert shifted.unresolved[0].candidates == ("left", "right")


class _PlainCatalog(DataPullClient):
    """A source other than STAC. Id checks stay at the base-class default."""

    satellite = "sentinel2"
    scene_kind = SceneKind.optical

    def __init__(self) -> None:
        self.collection = COLLECTION
        self.search_url = SEARCH
        self.pages: list[Scene] = []

    async def search(self, request: PullRequest) -> PullResult:
        return PullResult(request=request, scenes=list(self.pages))


def test_non_stac_source_keeps_ids_unchecked(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    client = _PlainCatalog()
    client.pages = [_scene("old-scene")]
    assert run_data(_args(tmp_path), clients={"sentinel2": client}) == 0
    capsys.readouterr()
    client.pages = []
    assert run_data(_args(tmp_path, "--resume"), clients={"sentinel2": client}) == 0
    source = _source(_manifest(tmp_path))
    err = capsys.readouterr().err
    assert source["scene_ids"] == ["old-scene"]
    assert source["replacements"] == []
    assert source["unresolved_scenes"] == []
    assert "replaced scene" not in err
    assert "no match" not in err
    assert "ambiguous" not in err
    assert _EMPTY_WINDOW not in err


class _LocalStac(StacApiClient):
    search_url: ClassVar[str] = "https://stac.example/search"
    default_collection: ClassVar[str] = COLLECTION
    satellite: ClassVar[str] = "sentinel2"
    scene_kind: ClassVar[SceneKind] = SceneKind.optical


def _feature(scene_id: str) -> dict[str, Any]:
    return {
        "type": "Feature",
        "id": scene_id,
        "bbox": [0.0, 0.0, 1.0, 1.0],
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]],
        },
        "properties": {"datetime": "2024-07-01T15:31:00Z"},
        "assets": {},
    }


@pytest.mark.asyncio
async def test_stac_lookup_uses_ids_then_place_and_time() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        if "ids" in body:
            return httpx.Response(200, json={"type": "FeatureCollection", "features": []})
        return httpx.Response(
            200, json={"type": "FeatureCollection", "features": [_feature("new-scene")]}
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = _LocalStac(client=http)
    try:
        found = await client.resolve_scene_ids(["old-scene"])
        hits = await client.find_scenes_by_place_time(
            collection=COLLECTION, when=WHEN, footprint=AOI
        )
    finally:
        await http.aclose()
    assert found == {}
    assert [scene.id for scene in hits] == ["new-scene"]
    assert seen[0]["ids"] == ["old-scene"]
    assert seen[0]["collections"] == [COLLECTION]
    assert seen[1]["collections"] == [COLLECTION]
    assert seen[1]["bbox"] == [0.0, 0.0, 1.0, 1.0]
    assert seen[1]["datetime"] == "2024-07-01T15:29:00Z/2024-07-01T15:31:00Z"
