"""Pulls of different sources into one ``--out`` share one manifest."""

from __future__ import annotations

import argparse
import asyncio
import json
import threading
import time
from pathlib import Path
from typing import Any, ClassVar

import httpx
import pytest

from atlas.cli.data import (
    _manifest_lock,
    _products_for_request,
    _write_manifest,
    add_data_parser,
    run_data,
)
from atlas.cli.download import DownloadReport
from atlas.compile.product import SceneCatalog, SourceScenes
from atlas.data.base import BBox, PullRequest, SceneKind
from atlas.data.stac import StacApiClient

SEARCH = "https://stac.example/search"
WINDOW_A = "2024-07-01:2024-07-07"
WINDOW_B = "2024-08-01:2024-08-07"


class _SentinelStac(StacApiClient):
    search_url: ClassVar[str] = SEARCH
    default_collection: ClassVar[str] = "sentinel"
    satellite: ClassVar[str] = "sentinel2"
    scene_kind: ClassVar[SceneKind] = SceneKind.optical


class _LandsatStac(_SentinelStac):
    default_collection: ClassVar[str] = "landsat"


def _feature(scene_id: str) -> dict[str, Any]:
    ring = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]
    return {
        "type": "Feature",
        "id": scene_id,
        "bbox": [0.0, 0.0, 1.0, 1.0],
        "geometry": {"type": "Polygon", "coordinates": [ring]},
        "properties": {"datetime": "2024-07-01T00:00:00Z"},
        "assets": {"visual": {"href": f"https://stac.example/{scene_id}.tif", "roles": ["data"]}},
    }


def _page(scene_id: str, token: str | None) -> dict[str, Any]:
    links: list[dict[str, Any]] = []
    if token is not None:
        links.append(
            {
                "rel": "next",
                "method": "POST",
                "href": SEARCH,
                "merge": True,
                "body": {"token": token},
            }
        )
    return {"type": "FeatureCollection", "features": [_feature(scene_id)], "links": links}


class _Pages:
    """Sentinel pages by token; landsat is one page with no cursor."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        if isinstance(body.get("ids"), list):
            features = [_feature(scene_id) for scene_id in body["ids"]]
            return httpx.Response(200, json={"features": features, "links": []})
        token = body.get("token") if isinstance(body.get("token"), str) else None
        collection = body["collections"][0]
        self.calls.append((collection, token))
        if collection == "landsat":
            return httpx.Response(200, json=_page("land-1", None))
        pages = {None: ("scene-a", "page-2"), "page-2": ("scene-b", None)}
        scene_id, nxt = pages[token]
        return httpx.Response(200, json=_page(scene_id, nxt))


def _args(out: Path, window: str, *extra: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="atlas")
    sub = parser.add_subparsers(dest="group")
    add_data_parser(sub)
    tokens = ["data", "sentinel2", "optical", "--date", window, "--coords", "0,0,1,1"]
    return parser.parse_args([*tokens, "--out", str(out), "--search-only", *extra])


def _manifest(out: Path) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    return payload


def _row(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    return next(item for item in manifest["sources"] if item["source"] == name)


@pytest.fixture
def pages() -> _Pages:
    return _Pages()


@pytest.fixture
def http(pages: _Pages) -> Any:
    client = httpx.AsyncClient(transport=httpx.MockTransport(pages.handler))
    yield client
    asyncio.run(client.aclose())


def test_second_source_is_merged_not_replaced(tmp_path: Path, http: httpx.AsyncClient) -> None:
    assert (
        run_data(_args(tmp_path, WINDOW_A), clients={"sentinel2": _SentinelStac(client=http)}) == 0
    )
    assert run_data(_args(tmp_path, WINDOW_B), clients={"landsat8": _LandsatStac(client=http)}) == 0
    manifest = _manifest(tmp_path)
    assert [row["source"] for row in manifest["sources"]] == ["sentinel2", "landsat8"]
    sentinel = _row(manifest, "sentinel2")
    landsat = _row(manifest, "landsat8")
    assert sentinel["scene_ids"] == ["scene-a"]
    assert sentinel["request"]["start_date"] == "2024-07-01"
    assert landsat["scene_ids"] == ["land-1"]
    assert landsat["request"]["start_date"] == "2024-08-01"
    assert manifest["request"]["start_date"] == "2024-08-01"


def test_rerunning_a_source_replaces_only_its_row(tmp_path: Path, http: httpx.AsyncClient) -> None:
    sentinel = {"sentinel2": _SentinelStac(client=http)}
    assert run_data(_args(tmp_path, WINDOW_A), clients=sentinel) == 0
    assert run_data(_args(tmp_path, WINDOW_B), clients={"landsat8": _LandsatStac(client=http)}) == 0
    assert run_data(_args(tmp_path, WINDOW_B), clients=sentinel) == 0
    manifest = _manifest(tmp_path)
    assert [row["source"] for row in manifest["sources"]] == ["sentinel2", "landsat8"]
    assert _row(manifest, "sentinel2")["request"]["start_date"] == "2024-08-01"
    assert _row(manifest, "sentinel2")["scene_ids"] == ["scene-a"]


def test_resume_continues_one_source_beside_another(
    tmp_path: Path, http: httpx.AsyncClient, pages: _Pages
) -> None:
    sentinel = {"sentinel2": _SentinelStac(client=http)}
    assert run_data(_args(tmp_path, WINDOW_A), clients=sentinel) == 0
    assert run_data(_args(tmp_path, WINDOW_B), clients={"landsat8": _LandsatStac(client=http)}) == 0
    landsat_before = _row(_manifest(tmp_path), "landsat8")
    assert run_data(_args(tmp_path, WINDOW_A, "--resume"), clients=sentinel) == 0
    manifest = _manifest(tmp_path)
    assert pages.calls[-1] == ("sentinel", "page-2")
    assert _row(manifest, "sentinel2")["scene_ids"] == ["scene-a", "scene-b"]
    assert _row(manifest, "sentinel2")["next_cursor"] is None
    assert _row(manifest, "landsat8") == landsat_before


def test_resume_rejects_a_source_recorded_for_another_window(
    tmp_path: Path, http: httpx.AsyncClient, capsys: pytest.CaptureFixture[str]
) -> None:
    sentinel = {"sentinel2": _SentinelStac(client=http)}
    assert run_data(_args(tmp_path, WINDOW_A), clients=sentinel) == 0
    assert run_data(_args(tmp_path, WINDOW_B), clients={"landsat8": _LandsatStac(client=http)}) == 0
    recorded = (tmp_path / "manifest.json").read_text(encoding="utf-8")
    assert run_data(_args(tmp_path, WINDOW_B, "--resume"), clients=sentinel) == 2
    assert "resume query does not match the pull manifest" in capsys.readouterr().err
    assert (tmp_path / "manifest.json").read_text(encoding="utf-8") == recorded


def test_resume_still_rejects_a_source_with_no_row(
    tmp_path: Path, http: httpx.AsyncClient, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        run_data(_args(tmp_path, WINDOW_A), clients={"sentinel2": _SentinelStac(client=http)}) == 0
    )
    both = {"sentinel2": _SentinelStac(client=http), "landsat8": _LandsatStac(client=http)}
    assert run_data(_args(tmp_path, WINDOW_A, "--resume"), clients=both) == 2
    assert "resume sources do not match the pull manifest" in capsys.readouterr().err


def test_legacy_manifest_rows_keep_their_request(
    tmp_path: Path, http: httpx.AsyncClient, pages: _Pages
) -> None:
    sentinel = {"sentinel2": _SentinelStac(client=http)}
    assert run_data(_args(tmp_path, WINDOW_A), clients=sentinel) == 0
    legacy = _manifest(tmp_path)
    for row in legacy["sources"]:
        del row["request"]
    (tmp_path / "manifest.json").write_text(json.dumps(legacy) + "\n", encoding="utf-8")

    assert run_data(_args(tmp_path, WINDOW_B), clients={"landsat8": _LandsatStac(client=http)}) == 0
    merged = _manifest(tmp_path)
    assert _row(merged, "sentinel2")["request"]["start_date"] == "2024-07-01"
    assert _row(merged, "landsat8")["request"]["start_date"] == "2024-08-01"

    assert run_data(_args(tmp_path, WINDOW_A, "--resume"), clients=sentinel) == 0
    assert pages.calls[-1] == ("sentinel", "page-2")
    assert _row(_manifest(tmp_path), "sentinel2")["scene_ids"] == ["scene-a", "scene-b"]


def test_legacy_manifest_still_resumes_in_place(tmp_path: Path, http: httpx.AsyncClient) -> None:
    sentinel = {"sentinel2": _SentinelStac(client=http)}
    assert run_data(_args(tmp_path, WINDOW_A), clients=sentinel) == 0
    legacy = _manifest(tmp_path)
    for row in legacy["sources"]:
        del row["request"]
    (tmp_path / "manifest.json").write_text(json.dumps(legacy) + "\n", encoding="utf-8")
    assert run_data(_args(tmp_path, WINDOW_A, "--resume"), clients=sentinel) == 0
    assert _row(_manifest(tmp_path), "sentinel2")["scene_ids"] == ["scene-a", "scene-b"]


def test_unreadable_manifest_is_replaced(tmp_path: Path, http: httpx.AsyncClient) -> None:
    (tmp_path / "manifest.json").write_text("{not json", encoding="utf-8")
    assert (
        run_data(_args(tmp_path, WINDOW_A), clients={"sentinel2": _SentinelStac(client=http)}) == 0
    )
    assert [row["source"] for row in _manifest(tmp_path)["sources"]] == ["sentinel2"]


def _request(start: str, end: str) -> PullRequest:
    return PullRequest.model_validate(
        {"start_date": start, "end_date": end, "bbox": BBox(west=0, south=0, east=1, north=1)}
    )


def test_stale_products_follow_each_rows_own_window(tmp_path: Path) -> None:
    window_a = _request("2024-07-01", "2024-07-07")
    window_b = _request("2024-08-01", "2024-08-07")
    for name in ("sentinel2", "landsat8"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "chip.tif").write_bytes(b"chip")
    manifest = {
        "request": window_b.model_dump(mode="json"),
        "sources": [
            {
                "source": "sentinel2",
                "request": window_a.model_dump(mode="json"),
                "saved": ["sentinel2/chip.tif"],
            },
            {"source": "landsat8", "saved": ["landsat8/chip.tif"]},
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert _products_for_request(tmp_path, "sentinel2", window_a) == [
        tmp_path / "sentinel2" / "chip.tif"
    ]
    assert _products_for_request(tmp_path, "sentinel2", window_b) == []
    assert _products_for_request(tmp_path, "landsat8", window_b) == [
        tmp_path / "landsat8" / "chip.tif"
    ]
    assert _products_for_request(tmp_path, "landsat8", window_a) == []


def test_manifest_lock_serializes_writers(tmp_path: Path) -> None:
    order: list[str] = []

    def worker(name: str) -> None:
        with _manifest_lock(tmp_path):
            order.append(f"{name}-in")
            time.sleep(0.05)
            order.append(f"{name}-out")

    threads = [threading.Thread(target=worker, args=(name,)) for name in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert order in (
        ["a-in", "a-out", "b-in", "b-out"],
        ["b-in", "b-out", "a-in", "a-out"],
    )


def test_concurrent_source_writes_keep_every_row(tmp_path: Path) -> None:
    request = _request("2024-07-01", "2024-07-07")
    names = ["a", "b", "c", "d"]
    barrier = threading.Barrier(len(names))

    def worker(name: str) -> None:
        catalog = SceneCatalog(
            request=request,
            sources=[SourceScenes(source=name, collection=name)],
        )
        barrier.wait()
        _write_manifest(tmp_path, catalog, DownloadReport())

    threads = [threading.Thread(target=worker, args=(name,)) for name in names]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    manifest = _manifest(tmp_path)
    assert {row["source"] for row in manifest["sources"]} == set(names)
