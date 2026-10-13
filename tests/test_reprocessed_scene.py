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
from atlas.data.stac import StacApiClient

AOI = BBox(west=0, south=0, east=1, north=1)
FAR = BBox(west=10, south=10, east=11, north=11)
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
) -> Scene:
    return Scene(
        id=scene_id,
        source="sentinel2",
        collection=collection,
        datetime=when,
        bbox=bbox,
        platform="sentinel-2",
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
