"""An empty date window must not present an older product as this request."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx

from atlas.cli.data import run_data
from atlas.data.base import Asset, BBox, DataPullClient, PullRequest, PullResult, Scene, SceneKind

AOI = BBox(west=-71.12, south=42.32, east=-71.02, north=42.40)
_MANIFEST_FIELDS = {
    "source",
    "collection",
    "bbox",
    "error",
    "n_scenes",
    "scene_ids",
    "saved",
    "skipped",
    "failed",
    "windows",
}


def _pull_args(tmp_path: Path, *, search_only: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        list_catalog=False,
        satellite="sentinel2",
        kind="optical",
        date="2024-07-08:2024-07-08",
        coords="-71.12,42.32,-71.02,42.40",
        out=tmp_path,
        limit=10,
        min_cloud_cover=None,
        max_cloud_cover=None,
        concurrency=2,
        search_only=search_only,
        asset=None,
        table=None,
        variables=None,
        points=None,
        polygon=None,
    )


def _scene() -> Scene:
    return Scene(
        id="today",
        source="keeps",
        datetime=datetime(2024, 7, 8, tzinfo=UTC),
        bbox=AOI,
        platform="sentinel-2",
        assets={
            "data": Asset(
                href="https://example.com/scene.bin",
                media_type="application/octet-stream",
                roles=["data"],
            )
        },
    )


class _Empty(DataPullClient):
    satellite = "sentinel2"
    scene_kind = SceneKind.optical

    async def search(self, request: PullRequest) -> PullResult:
        return PullResult(request=request, scenes=[])


class _Offline(DataPullClient):
    satellite = "sentinel2"
    scene_kind = SceneKind.optical

    async def search(self, request: PullRequest) -> PullResult:
        raise RuntimeError("sensor offline")


class _Publishing(DataPullClient):
    satellite = "sentinel2"
    scene_kind = SceneKind.optical

    async def search(self, request: PullRequest) -> PullResult:
        return PullResult(request=request, scenes=[_scene()])


def test_empty_window_fails_and_drops_the_older_chip(tmp_path: Path) -> None:
    chip = tmp_path / "sentinel2" / "chip_visual.tif"
    chip.parent.mkdir(parents=True)
    chip.write_bytes(b"yesterday")
    neighbor = tmp_path / "keeps" / "notes.txt"
    neighbor.parent.mkdir(parents=True)
    neighbor.write_text("leave-me", encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "example.com"
        return httpx.Response(200, content=b"today")

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        code = run_data(
            _pull_args(tmp_path),
            clients={"sentinel2": _Empty(), "keeps": _Publishing()},
            http=http,
        )
    finally:
        asyncio.run(http.aclose())
    assert code == 0
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    by_source = {item["source"]: item for item in manifest["sources"]}
    empty = by_source["sentinel2"]
    assert empty["n_scenes"] == 0
    assert empty["error"] is None
    assert empty["saved"] == []
    assert empty["skipped"] == []
    assert empty["failed"] == [{"id": "sentinel2-window", "error": "requested window is empty"}]
    assert set(empty) == _MANIFEST_FIELDS
    assert not chip.exists()
    assert neighbor.read_text(encoding="utf-8") == "leave-me"
    published = by_source["keeps"]
    assert published["failed"] == []
    assert published["saved"] == ["keeps/today_data.bin"]
    assert (tmp_path / "keeps" / "today_data.bin").read_bytes() == b"today"


def test_empty_window_search_only_records_the_failure_and_keeps_the_file(tmp_path: Path) -> None:
    chip = tmp_path / "sentinel2" / "chip_visual.tif"
    chip.parent.mkdir(parents=True)
    chip.write_bytes(b"yesterday")
    assert run_data(_pull_args(tmp_path, search_only=True), clients={"sentinel2": _Empty()}) == 0
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    source = manifest["sources"][0]
    assert source["failed"] == [{"id": "sentinel2-window", "error": "requested window is empty"}]
    assert source["saved"] == []
    assert set(source) == _MANIFEST_FIELDS
    assert chip.read_bytes() == b"yesterday"


def test_offline_source_does_not_present_the_older_chip(tmp_path: Path) -> None:
    chip = tmp_path / "sentinel2" / "chip_visual.tif"
    chip.parent.mkdir(parents=True)
    chip.write_bytes(b"yesterday")
    assert run_data(_pull_args(tmp_path), clients={"sentinel2": _Offline()}) == 1
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    source = manifest["sources"][0]
    assert source["error"] is not None
    assert "sensor offline" in source["error"]
    assert source["saved"] == []
    assert source["skipped"] == []
    assert source["failed"] == []
    assert not chip.exists()
    assert set(source) == _MANIFEST_FIELDS
