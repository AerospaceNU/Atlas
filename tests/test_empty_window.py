"""An empty date window must not present an older product as this request."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from atlas.cli.data import run_data
from atlas.data.base import Asset, BBox, DataPullClient, PullRequest, PullResult, Scene, SceneKind

AOI = BBox(west=-71.12, south=42.32, east=-71.02, north=42.40)
_MANIFEST_FIELDS = {
    "source",
    "collection",
    "bbox",
    "search_url",
    "next_cursor",
    "next_cursor_unavailable",
    "note",
    "error",
    "n_scenes",
    "scene_ids",
    "scenes",
    "replacements",
    "unresolved_scenes",
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


def _previous_manifest(
    out_dir: Path,
    *,
    source: str,
    saved: list[str],
    start: str = "2024-07-08",
    end: str = "2024-07-08",
    west: float = -71.12,
    south: float = 42.32,
    east: float = -71.02,
    north: float = 42.40,
) -> None:
    payload = {
        "request": {
            "start_date": start,
            "end_date": end,
            "bbox": {"west": west, "south": south, "east": east, "north": north},
            "polygon": None,
        },
        "sources": [{"source": source, "saved": saved, "skipped": []}],
    }
    (out_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def test_empty_window_sets_aside_only_the_listed_chip(tmp_path: Path) -> None:
    chip = tmp_path / "sentinel2" / "chip_visual.tif"
    chip.parent.mkdir(parents=True)
    chip.write_bytes(b"yesterday")
    notes = chip.parent / "notes.txt"
    notes.write_text("mine", encoding="utf-8")
    neighbor = tmp_path / "keeps" / "notes.txt"
    neighbor.parent.mkdir(parents=True)
    neighbor.write_text("leave-me", encoding="utf-8")
    _previous_manifest(tmp_path, source="sentinel2", saved=["sentinel2/chip_visual.tif"])

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
    assert chip.with_name(chip.name + ".stale").read_bytes() == b"yesterday"
    assert notes.read_text(encoding="utf-8") == "mine"
    assert neighbor.read_text(encoding="utf-8") == "leave-me"
    published = by_source["keeps"]
    assert published["failed"] == []
    assert published["saved"] == ["keeps/today_data.bin"]
    assert (tmp_path / "keeps" / "today_data.bin").read_bytes() == b"today"


def test_empty_window_leaves_another_dates_chip(tmp_path: Path) -> None:
    chip = tmp_path / "sentinel2" / "chip_visual.tif"
    chip.parent.mkdir(parents=True)
    chip.write_bytes(b"yesterday")
    _previous_manifest(
        tmp_path,
        source="sentinel2",
        saved=["sentinel2/chip_visual.tif"],
        start="2024-07-07",
        end="2024-07-07",
    )
    assert run_data(_pull_args(tmp_path), clients={"sentinel2": _Empty()}) == 0
    assert chip.read_bytes() == b"yesterday"
    assert not chip.with_name(chip.name + ".stale").exists()
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["sources"][0]["saved"] == []
    assert manifest["sources"][0]["failed"] == [
        {"id": "sentinel2-window", "error": "requested window is empty"}
    ]


def test_empty_window_search_only_records_the_failure_and_keeps_the_file(tmp_path: Path) -> None:
    chip = tmp_path / "sentinel2" / "chip_visual.tif"
    chip.parent.mkdir(parents=True)
    chip.write_bytes(b"yesterday")
    _previous_manifest(tmp_path, source="sentinel2", saved=["sentinel2/chip_visual.tif"])
    assert run_data(_pull_args(tmp_path, search_only=True), clients={"sentinel2": _Empty()}) == 0
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    source = manifest["sources"][0]
    assert source["failed"] == [{"id": "sentinel2-window", "error": "requested window is empty"}]
    assert source["saved"] == []
    assert set(source) == _MANIFEST_FIELDS
    assert chip.read_bytes() == b"yesterday"


def test_offline_source_keeps_the_older_chip(tmp_path: Path) -> None:
    chip = tmp_path / "sentinel2" / "chip_visual.tif"
    chip.parent.mkdir(parents=True)
    chip.write_bytes(b"yesterday")
    _previous_manifest(tmp_path, source="sentinel2", saved=["sentinel2/chip_visual.tif"])
    assert run_data(_pull_args(tmp_path), clients={"sentinel2": _Offline()}) == 1
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    source = manifest["sources"][0]
    assert source["error"] is not None
    assert "sensor offline" in source["error"]
    assert source["saved"] == []
    assert source["skipped"] == []
    assert source["failed"] == []
    assert chip.read_bytes() == b"yesterday"
    assert set(source) == _MANIFEST_FIELDS


def test_timeout_and_rate_limit_keep_the_older_chip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    aggregate_mod = importlib.import_module("atlas.compile.aggregate")
    monkeypatch.setattr(aggregate_mod, "_retry_delay", lambda _exc, _attempt: 0.0)
    chip = tmp_path / "sentinel2" / "chip_visual.tif"
    chip.parent.mkdir(parents=True)
    chip.write_bytes(b"yesterday")
    _previous_manifest(tmp_path, source="sentinel2", saved=["sentinel2/chip_visual.tif"])

    class _TimedOut(DataPullClient):
        satellite = "sentinel2"
        scene_kind = SceneKind.optical

        async def search(self, request: PullRequest) -> PullResult:
            raise TimeoutError("search timed out")

    assert run_data(_pull_args(tmp_path), clients={"sentinel2": _TimedOut()}) == 1
    assert chip.read_bytes() == b"yesterday"

    class _Limited(DataPullClient):
        satellite = "sentinel2"
        scene_kind = SceneKind.optical

        async def search(self, request: PullRequest) -> PullResult:
            response = httpx.Response(429, headers={"Retry-After": "0"})
            raise httpx.HTTPStatusError(
                "429",
                request=httpx.Request("GET", "https://example.com/search"),
                response=response,
            )

    assert run_data(_pull_args(tmp_path), clients={"sentinel2": _Limited()}) == 1
    assert chip.read_bytes() == b"yesterday"
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["sources"][0]["saved"] == []
    assert manifest["sources"][0]["error"]


def test_symlinked_out_dir_sets_the_chip_aside_inside_the_real_directory(tmp_path: Path) -> None:
    disk = tmp_path / "disk"
    disk.mkdir()
    parent = tmp_path / "parent-link"
    parent.symlink_to(disk, target_is_directory=True)
    real_out = disk / "real-out"
    real_out.mkdir()
    out = parent / "out-link"
    out.symlink_to(real_out, target_is_directory=True)
    chip = real_out / "sentinel2" / "chip_visual.tif"
    chip.parent.mkdir(parents=True)
    chip.write_bytes(b"yesterday")
    notes = chip.parent / "notes.txt"
    notes.write_text("mine", encoding="utf-8")
    secret = disk / "secret.txt"
    secret.write_text("secret", encoding="utf-8")
    _previous_manifest(real_out, source="sentinel2", saved=["sentinel2/chip_visual.tif"])
    assert run_data(_pull_args(out), clients={"sentinel2": _Empty()}) == 0
    assert not chip.exists()
    assert chip.with_name(chip.name + ".stale").read_bytes() == b"yesterday"
    assert notes.read_text(encoding="utf-8") == "mine"
    assert secret.read_text(encoding="utf-8") == "secret"
    assert (real_out / "manifest.json").is_file()


def test_product_symlink_is_unlinked_and_not_followed(tmp_path: Path) -> None:
    target = tmp_path / "outside.bin"
    target.write_bytes(b"elsewhere")
    chip = tmp_path / "sentinel2" / "chip_visual.tif"
    chip.parent.mkdir(parents=True)
    chip.symlink_to(target)
    _previous_manifest(tmp_path, source="sentinel2", saved=["sentinel2/chip_visual.tif"])
    assert run_data(_pull_args(tmp_path), clients={"sentinel2": _Empty()}) == 0
    assert not chip.exists()
    assert target.read_bytes() == b"elsewhere"
    assert not chip.with_name(chip.name + ".stale").exists()


def test_dotdot_source_does_not_escape_the_output_directory(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("secret", encoding="utf-8")
    notes = out / "sentinel2" / "notes.txt"
    notes.parent.mkdir(parents=True)
    notes.write_text("mine", encoding="utf-8")
    _previous_manifest(out, source="..", saved=["../secret.txt"])
    assert run_data(_pull_args(out), clients={"..": _Empty()}) == 0
    assert secret.read_text(encoding="utf-8") == "secret"
    assert notes.read_text(encoding="utf-8") == "mine"
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["sources"][0]["failed"] == [
        {"id": "..-window", "error": "requested window is empty"}
    ]
    assert set(tmp_path.iterdir()) == {out, secret}
