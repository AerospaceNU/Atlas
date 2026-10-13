from __future__ import annotations

import argparse
import asyncio
import json
import os
import stat
import time
from datetime import date
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import urlparse

import httpx
import pytest

from atlas.cli.data import _write_manifest, add_data_parser, run_data
from atlas.cli.download import DownloadReport
from atlas.compile.aggregate import aggregate
from atlas.compile.product import SceneCatalog, SourceScenes
from atlas.data.base import BBox, DataPullClient, PullRequest, PullResult, SceneKind
from atlas.data.cdse import CdseSentinel3OlciClient
from atlas.data.cmr_stac import GediL2AClient, IceSat2Atl03Client
from atlas.data.digitalearth import DeAfricaS2Client, DeaS2ArdClient
from atlas.data.earth_search import EarthSearchSentinel1GrdClient
from atlas.data.inpe import Cbers4MuxClient
from atlas.data.sentinel2 import Sentinel2Client
from atlas.data.stac import StacApiClient
from atlas.data.usgs import UsgsLandsatC2L1Client

SEARCH = "https://stac.example/search"
SQUARE = "0,0;1,0;1,1;0,1"


class _DemoStac(StacApiClient):
    search_url: ClassVar[str] = SEARCH
    default_collection: ClassVar[str] = "demo"
    satellite: ClassVar[str] = "sentinel2"
    scene_kind: ClassVar[SceneKind] = SceneKind.optical


class _SentinelStac(_DemoStac):
    default_collection: ClassVar[str] = "sentinel"


class _LandsatStac(_DemoStac):
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


def _request() -> PullRequest:
    return PullRequest(
        start_date=date(2024, 7, 1),
        end_date=date(2024, 7, 7),
        bbox=BBox(west=0, south=0, east=1, north=1),
        limit=2,
    )


def _parse(tokens: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="atlas")
    sub = parser.add_subparsers(dest="group")
    add_data_parser(sub)
    return parser.parse_args(tokens)


def _pull_args(tmp_path: Path, *extra: str) -> argparse.Namespace:
    return _parse(
        [
            "data",
            "sentinel2",
            "optical",
            "--date",
            "2024-07-01:2024-07-07",
            "--coords",
            "0,0,1,1",
            "--polygon",
            SQUARE,
            "--out",
            str(tmp_path),
            "--search-only",
            *extra,
        ]
    )


def _page(scene_id: str, token: str | None, *, href: str | None = None) -> dict[str, Any]:
    links: list[dict[str, Any]] = []
    if token is not None:
        links.append(
            {
                "rel": "next",
                "method": "POST",
                "href": href or SEARCH,
                "merge": True,
                "body": {"token": token},
            }
        )
    elif href is not None:
        links.append({"rel": "next", "method": "GET", "href": href})
    return {"type": "FeatureCollection", "features": [_feature(scene_id)], "links": links}


class _TokenPages:
    """POST pages keyed by body token. ``None`` is page one."""

    def __init__(self, pages: dict[str | None, tuple[str, str | None]]) -> None:
        self.pages = pages
        self.calls: list[tuple[str, str | None, str | None]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        probe = _id_probe_features(body)
        if probe is not None:
            known = {scene_id for scene_id, _nxt in self.pages.values()}
            known.add("land-1")
            features = [_feature(scene_id) for scene_id in probe if scene_id in known]
            return httpx.Response(
                200, json={"type": "FeatureCollection", "features": features, "links": []}
            )
        token = body.get("token") if isinstance(body, dict) else None
        if token is not None and not isinstance(token, str):
            token = None
        collection = body["collections"][0] if isinstance(body, dict) else None
        self.calls.append((request.method, collection, token if isinstance(token, str) else None))
        if collection == "landsat":
            if token is not None:
                return httpx.Response(500, json={"detail": "landsat was resumed"})
            return httpx.Response(200, json=_page("land-1", None))
        scene_id, nxt = self.pages[token if isinstance(token, str) else None]
        return httpx.Response(200, json=_page(scene_id, nxt))


def _clients(pages: _TokenPages) -> tuple[httpx.AsyncClient, dict[str, StacApiClient]]:
    http = httpx.AsyncClient(transport=httpx.MockTransport(pages.handler))
    return http, {
        "sentinel2": _SentinelStac(client=http),
        "landsat8": _LandsatStac(client=http),
    }


def _source(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    return next(item for item in manifest["sources"] if item["source"] == name)


def _id_probe_features(body: object) -> list[str] | None:
    """Ids from a scene-identity lookup, which is not a page request."""
    if not isinstance(body, dict) or not isinstance(body.get("ids"), list):
        return None
    return [item for item in body["ids"] if isinstance(item, str)]


def _id_probe_response(request: httpx.Request) -> httpx.Response | None:
    body = json.loads(request.content) if request.content else None
    wanted = _id_probe_features(body)
    if wanted is None:
        return None
    return httpx.Response(
        200,
        json={
            "type": "FeatureCollection",
            "features": [_feature(scene_id) for scene_id in wanted],
            "links": [],
        },
    )


@pytest.mark.asyncio
async def test_post_token_is_the_cursor_and_page_one_omits_it() -> None:
    pages = _TokenPages({None: ("scene-a", "page-2")})
    http, clients = _clients(pages)
    try:
        result = await clients["sentinel2"].search(_request())
    finally:
        await http.aclose()
    cursor = json.loads(result.next_cursor or "")
    assert cursor["method"] == "POST"
    assert cursor["merge"] is True
    assert cursor["body"]["token"] == "page-2"
    assert [scene.id for scene in result.scenes] == ["scene-a"]
    assert pages.calls == [("POST", "sentinel", None)]


@pytest.mark.asyncio
async def test_token_wins_over_the_bare_search_url() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_page("scene-a", "page-2", href=SEARCH),
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        result = await _DemoStac(client=http).search(_request())
    finally:
        await http.aclose()
    cursor = json.loads(result.next_cursor or "")
    assert cursor["method"] == "POST"
    assert cursor["body"]["token"] == "page-2"
    assert cursor["href"] == SEARCH


@pytest.mark.asyncio
async def test_get_next_href_is_stored_and_resume_does_not_post() -> None:
    calls: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, str(request.url)))
        if request.method == "POST":
            return httpx.Response(200, json=_page("scene-a", None, href="?page=2"))
        return httpx.Response(200, json=_page("scene-b", None))

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = _DemoStac(client=http)
    try:
        first = await client.search(_request())
        assert json.loads(first.next_cursor or "") == {
            "href": f"{SEARCH}?page=2",
            "method": "GET",
        }
        second = await client.search_from_cursor(_request(), first.next_cursor or "")
        legacy = await client.search_from_cursor(_request(), f"{SEARCH}?page=2")
    finally:
        await http.aclose()
    assert [scene.id for scene in second.scenes] == ["scene-b"]
    assert [scene.id for scene in legacy.scenes] == ["scene-b"]
    assert second.next_cursor is None
    assert calls == [
        ("POST", SEARCH),
        ("GET", f"{SEARCH}?page=2"),
        ("GET", f"{SEARCH}?page=2"),
    ]


@pytest.mark.asyncio
async def test_cross_origin_and_bare_next_links_are_not_cursors() -> None:
    payloads = [
        _page("scene-a", None, href="https://evil.example/next"),
        _page("scene-a", None, href=SEARCH),
        {"type": "FeatureCollection", "features": [_feature("scene-a")]},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payloads.pop(0))

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = _DemoStac(client=http)
    try:
        cross = await client.search(_request())
        bare = await client.search(_request())
        missing = await client.search(_request())
    finally:
        await http.aclose()
    assert cross.next_cursor is None
    assert bare.next_cursor is None
    assert missing.next_cursor is None
    assert cross.next_cursor_unavailable is None
    assert bare.next_cursor_unavailable is None
    assert missing.next_cursor_unavailable is None


@pytest.mark.asyncio
async def test_resume_rejects_a_cursor_on_another_host() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"cursor host was fetched: {request.url}")

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(ValueError, match="host"):
            await _DemoStac(client=http).search_from_cursor(_request(), "https://evil.example/next")
    finally:
        await http.aclose()


@pytest.mark.asyncio
async def test_non_paging_resume_does_not_repeat_page_one() -> None:
    class _Once(DataPullClient):
        def __init__(self) -> None:
            self.calls = 0

        async def search(self, request: PullRequest) -> PullResult:
            self.calls += 1
            return PullResult(request=request, scenes=[])

    client = _Once()
    request = _request()
    resumed = await aggregate(request, {"sentinel2": client}, page_cursors={"sentinel2": "page-2"})
    assert client.calls == 0
    assert resumed.sources[0].scenes == []
    assert resumed.sources[0].next_cursor == "page-2"
    exhausted = await aggregate(request, {"sentinel2": client}, page_cursors={"sentinel2": None})
    assert client.calls == 0
    assert exhausted.sources[0].next_cursor is None
    await aggregate(request, {"sentinel2": client}, page_cursors={})
    assert client.calls == 1


def test_manifest_records_cursors_and_resume_continues(tmp_path: Path) -> None:
    pages = _TokenPages(
        {None: ("scene-a", "page-2"), "page-2": ("scene-b", "page-3"), "page-3": ("scene-c", None)}
    )
    http, clients = _clients(pages)
    try:
        assert run_data(_pull_args(tmp_path), clients=clients) == 0
        manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
        sentinel = _source(manifest, "sentinel2")
        landsat = _source(manifest, "landsat8")
        assert sentinel["next_cursor"]["body"]["token"] == "page-2"
        assert sentinel["next_cursor"]["merge"] is True
        assert sentinel["search_url"] == SEARCH
        assert landsat["search_url"] == SEARCH
        assert sentinel["scene_ids"] == ["scene-a"]
        assert landsat["next_cursor"] is None
        assert landsat["scene_ids"] == ["land-1"]
        assert sentinel["bbox"] == manifest["catalog_bbox"] == [0.0, 0.0, 1.0, 1.0]
        assert sentinel["windows"] == []
        assert manifest["polygon"]["type"] == "Polygon"
        assert "bounding box" in manifest["catalog_bbox_note"]
        sentinel["windows"] = [{"id": "scene-a", "bytes_read": 12, "granule_size": 80}]
        (tmp_path / "manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")

        assert run_data(_pull_args(tmp_path, "--resume"), clients=clients) == 0
        manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
        sentinel = _source(manifest, "sentinel2")
        assert sentinel["scene_ids"] == ["scene-a", "scene-b"]
        assert sentinel["n_scenes"] == 2
        assert sentinel["next_cursor"]["body"]["token"] == "page-3"
        assert sentinel["windows"] == [{"id": "scene-a", "bytes_read": 12, "granule_size": 80}]
        assert sentinel["bbox"] == [0.0, 0.0, 1.0, 1.0]
        assert _source(manifest, "landsat8")["scene_ids"] == ["land-1"]
        assert pages.calls[-1] == ("POST", "sentinel", "page-2")
        assert [call for call in pages.calls if call[1] == "landsat8" or call[1] == "landsat"] == [
            ("POST", "landsat", None)
        ]

        assert run_data(_pull_args(tmp_path, "--resume"), clients=clients) == 0
        manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
        sentinel = _source(manifest, "sentinel2")
        assert sentinel["scene_ids"] == ["scene-a", "scene-b", "scene-c"]
        assert sentinel["next_cursor"] is None
        assert sentinel["windows"][0]["bytes_read"] == 12
        assert sentinel["windows"][0]["granule_size"] == 80
        before = len(pages.calls)
        assert run_data(_pull_args(tmp_path, "--resume"), clients=clients) == 0
        assert len(pages.calls) == before
        again = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
        assert _source(again, "sentinel2")["scene_ids"] == ["scene-a", "scene-b", "scene-c"]
    finally:
        asyncio.run(http.aclose())


def test_failed_resume_keeps_the_cursor_and_prior_window(tmp_path: Path) -> None:
    calls: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        token = body.get("token")
        calls.append(token if isinstance(token, str) else None)
        if token:
            return httpx.Response(500, json={"detail": "unavailable"})
        return httpx.Response(200, json=_page("scene-a", "page-2"))

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    clients = {"sentinel2": _SentinelStac(client=http)}
    try:
        assert run_data(_pull_args(tmp_path), clients=clients) == 0
        manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
        manifest["sources"][0]["windows"] = [{"id": "scene-a", "bytes_read": 4, "granule_size": 9}]
        (tmp_path / "manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
        assert run_data(_pull_args(tmp_path, "--resume"), clients=clients) == 1
        manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
        source = manifest["sources"][0]
        assert calls == [None, "page-2"]
        assert source["scene_ids"] == ["scene-a"]
        assert source["next_cursor"]["body"]["token"] == "page-2"
        assert source["windows"] == [{"id": "scene-a", "bytes_read": 4, "granule_size": 9}]
        assert source["error"] is not None
        assert "500" in source["error"]
        assert "://" not in source["error"]
    finally:
        asyncio.run(http.aclose())


def test_resume_without_a_manifest_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run_data(_pull_args(tmp_path, "--resume"), clients={}) == 2
    assert "no pull manifest" in capsys.readouterr().err


def test_resume_rejects_a_different_query(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pages = _TokenPages({None: ("scene-a", "page-2")})
    http, clients = _clients(pages)
    try:
        assert run_data(_pull_args(tmp_path), clients=clients) == 0
        changed = _pull_args(tmp_path, "--resume")
        changed.coords = "0,0,2,2"
        before = len(pages.calls)
        assert run_data(changed, clients=clients) == 2
        assert len(pages.calls) == before
    finally:
        asyncio.run(http.aclose())
    assert "does not match" in capsys.readouterr().err


def test_fresh_pull_does_not_append_an_old_manifest(tmp_path: Path) -> None:
    pages = _TokenPages({None: ("scene-a", "page-2"), "page-2": ("scene-b", None)})
    http, clients = _clients(pages)
    try:
        assert run_data(_pull_args(tmp_path), clients=clients) == 0
        assert run_data(_pull_args(tmp_path), clients=clients) == 0
        manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
        assert _source(manifest, "sentinel2")["scene_ids"] == ["scene-a"]
        assert _source(manifest, "sentinel2")["next_cursor"]["body"]["token"] == "page-2"
    finally:
        asyncio.run(http.aclose())


def _write_back(tmp_path: Path, manifest: dict[str, Any]) -> None:
    (tmp_path / "manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")


def test_legacy_token_string_still_resumes(tmp_path: Path) -> None:
    pages = _TokenPages({None: ("scene-a", "page-2"), "page-2": ("scene-b", None)})
    http, clients = _clients(pages)
    try:
        assert run_data(_pull_args(tmp_path), clients=clients) == 0
        manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
        _source(manifest, "sentinel2")["next_cursor"] = "page-2"
        _write_back(tmp_path, manifest)
        assert run_data(_pull_args(tmp_path, "--resume"), clients=clients) == 0
        manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
        assert _source(manifest, "sentinel2")["scene_ids"] == ["scene-a", "scene-b"]
        assert pages.calls[-1] == ("POST", "sentinel", "page-2")
    finally:
        asyncio.run(http.aclose())


def test_rejected_resume_clears_the_cursor(tmp_path: Path) -> None:
    calls: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        probe = _id_probe_response(request)
        if probe is not None:
            return probe
        body = json.loads(request.content)
        token = body.get("token")
        calls.append(token if isinstance(token, str) else None)
        if token:
            return httpx.Response(400, json={"detail": "expired"})
        return httpx.Response(200, json=_page("scene-a", "page-2"))

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    clients = {"sentinel2": _SentinelStac(client=http)}
    try:
        assert run_data(_pull_args(tmp_path), clients=clients) == 0
        manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
        manifest["sources"][0]["windows"] = [{"id": "scene-a", "bytes_read": 4, "granule_size": 9}]
        _write_back(tmp_path, manifest)
        assert run_data(_pull_args(tmp_path, "--resume"), clients=clients) == 1
        manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
        source = manifest["sources"][0]
        assert calls == [None, "page-2"]
        assert source["next_cursor"] is None
        assert source["scene_ids"] == ["scene-a"]
        assert source["windows"] == [{"id": "scene-a", "bytes_read": 4, "granule_size": 9}]
        assert source["error"] is not None
        assert "400" in source["error"]
        assert "cleared" in source["error"]
        assert "page one" in source["error"]
        assert "://" not in source["error"]
        assert run_data(_pull_args(tmp_path, "--resume"), clients=clients) == 0
        assert calls == [None, "page-2"]
        again = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
        assert again["sources"][0]["next_cursor"] is None
    finally:
        asyncio.run(http.aclose())


def test_request_timeout_on_resume_keeps_the_cursor(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body.get("token"):
            return httpx.Response(408)
        return httpx.Response(200, json=_page("scene-a", "page-2"))

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    clients = {"sentinel2": _SentinelStac(client=http)}
    try:
        assert run_data(_pull_args(tmp_path), clients=clients) == 0
        assert run_data(_pull_args(tmp_path, "--resume"), clients=clients) == 1
        source = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))["sources"][0]
        assert source["next_cursor"]["body"]["token"] == "page-2"
        assert "408" in source["error"]
        assert "cleared" not in source["error"]
    finally:
        asyncio.run(http.aclose())


def test_rate_limited_resume_keeps_the_cursor(tmp_path: Path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        body = json.loads(request.content)
        if body.get("token"):
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(200, json=_page("scene-a", "page-2"))

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    clients = {"sentinel2": _SentinelStac(client=http)}
    try:
        assert run_data(_pull_args(tmp_path), clients=clients) == 0
        assert run_data(_pull_args(tmp_path, "--resume"), clients=clients) == 1
        source = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))["sources"][0]
        assert calls == 4
        assert source["next_cursor"]["body"]["token"] == "page-2"
        assert "429" in source["error"]
        assert "cleared" not in source["error"]
    finally:
        asyncio.run(http.aclose())


def test_signed_stored_cursor_is_cleared_and_not_fetched(tmp_path: Path) -> None:
    pages = _TokenPages({None: ("scene-a", "page-2")})
    http = httpx.AsyncClient(transport=httpx.MockTransport(pages.handler))
    clients = {"sentinel2": _SentinelStac(client=http)}
    try:
        assert run_data(_pull_args(tmp_path), clients=clients) == 0
        manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
        _source(manifest, "sentinel2")["next_cursor"] = f"{SEARCH}?sig=secret-value"
        _write_back(tmp_path, manifest)
        before = len(pages.calls)
        assert run_data(_pull_args(tmp_path, "--resume"), clients=clients) == 1
        assert len(pages.calls) == before
        source = _source(
            json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8")), "sentinel2"
        )
        assert source["next_cursor"] is None
        assert source["error"] is not None
        assert "signed" in source["error"]
        assert "cleared" in source["error"]
        assert "secret-value" not in source["error"]
        assert "://" not in source["error"]
    finally:
        asyncio.run(http.aclose())


def test_resume_rejects_a_different_source_set(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pages = _TokenPages({None: ("scene-a", "page-2")})
    http, clients = _clients(pages)
    try:
        assert run_data(_pull_args(tmp_path), clients=clients) == 0
        manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
        manifest["sources"] = [
            item for item in manifest["sources"] if item["source"] == "sentinel2"
        ]
        _write_back(tmp_path, manifest)
        recorded = (tmp_path / "manifest.json").read_text(encoding="utf-8")
        before = len(pages.calls)
        assert run_data(_pull_args(tmp_path, "--resume"), clients=clients) == 2
        assert len(pages.calls) == before
        assert (tmp_path / "manifest.json").read_text(encoding="utf-8") == recorded
    finally:
        asyncio.run(http.aclose())
    assert "resume sources do not match the pull manifest" in capsys.readouterr().err


def test_resume_rejects_a_different_collection(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pages = _TokenPages({None: ("scene-a", "page-2")})
    http, clients = _clients(pages)
    try:
        assert run_data(_pull_args(tmp_path), clients=clients) == 0
        manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
        _source(manifest, "sentinel2")["collection"] = "other"
        _write_back(tmp_path, manifest)
        recorded = (tmp_path / "manifest.json").read_text(encoding="utf-8")
        before = len(pages.calls)
        assert run_data(_pull_args(tmp_path, "--resume"), clients=clients) == 2
        assert len(pages.calls) == before
        assert (tmp_path / "manifest.json").read_text(encoding="utf-8") == recorded
    finally:
        asyncio.run(http.aclose())
    assert (
        "resume collection for sentinel2 does not match the pull manifest"
        in capsys.readouterr().err
    )


def test_resume_rejects_a_different_search_endpoint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pages = _TokenPages({None: ("scene-a", "page-2")})
    http, clients = _clients(pages)
    try:
        assert run_data(_pull_args(tmp_path), clients=clients) == 0
        manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
        _source(manifest, "sentinel2")["search_url"] = "https://other.example/search"
        _write_back(tmp_path, manifest)
        recorded = (tmp_path / "manifest.json").read_text(encoding="utf-8")
        before = len(pages.calls)
        assert run_data(_pull_args(tmp_path, "--resume"), clients=clients) == 2
        assert len(pages.calls) == before
        assert (tmp_path / "manifest.json").read_text(encoding="utf-8") == recorded
    finally:
        asyncio.run(http.aclose())
    err = capsys.readouterr().err
    assert "resume search endpoint for sentinel2 does not match the pull manifest" in err


_CATALOG_LINKS: list[tuple[type[StacApiClient], dict[str, Any], str]] = [
    (
        Sentinel2Client,
        {"method": "POST", "merge": True, "body": {"token": "pc-page"}},
        "merge",
    ),
    (
        EarthSearchSentinel1GrdClient,
        {"merge": True, "body": {"next": "es-page"}},
        "merge",
    ),
    (
        CdseSentinel3OlciClient,
        {"method": "POST", "merge": True, "body": {"token": "cdse-page"}},
        "merge",
    ),
    (UsgsLandsatC2L1Client, {"method": "GET", "href": "?page=2"}, "get"),
    (GediL2AClient, {"method": "GET", "href": "?page=2"}, "get"),
    (IceSat2Atl03Client, {"method": "GET", "href": "?page=2"}, "get"),
    (Cbers4MuxClient, {"method": "POST", "merge": True, "body": {"page": 2}}, "merge"),
    (DeaS2ArdClient, {"method": "GET", "href": "?page=2"}, "get"),
    (DeAfricaS2Client, {"method": "GET", "href": "?page=2"}, "get"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("client_cls", "link", "mode"),
    _CATALOG_LINKS,
    ids=[
        "planetary-computer",
        "earth-search",
        "cdse",
        "usgs",
        "cmr-lpcloud",
        "cmr-nsidc",
        "inpe",
        "dea",
        "de-africa",
    ],
)
async def test_catalog_next_link_is_resumed(
    client_cls: type[StacApiClient], link: dict[str, Any], mode: str
) -> None:
    search_url = client_cls.search_url
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if len(seen) == 1:
            fields = dict(link)
            href = fields.pop("href", search_url)
            return httpx.Response(
                200,
                json={
                    "type": "FeatureCollection",
                    "features": [_feature("scene-a")],
                    "links": [{"rel": "next", "href": href, **fields}],
                },
            )
        return httpx.Response(
            200,
            json={"type": "FeatureCollection", "features": [_feature("scene-b")], "links": []},
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = client_cls(client=http)
    try:
        first = await client.search(_request())
        assert first.next_cursor is not None
        stored = json.loads(first.next_cursor)
        if mode == "get":
            assert stored["method"] == "GET"
            assert stored["href"].startswith(search_url)
            assert "page=2" in stored["href"]
        else:
            assert stored["method"] == "POST"
            assert stored["merge"] is True
            assert stored["body"] == link["body"]
        second = await client.search_from_cursor(_request(), first.next_cursor)
    finally:
        await http.aclose()
    assert [scene.id for scene in second.scenes] == ["scene-b"]
    follow = seen[1]
    assert urlparse(str(follow.url)).netloc == urlparse(search_url).netloc
    if mode == "get":
        assert follow.method == "GET"
        assert "page=2" in str(follow.url)
        return
    assert follow.method == "POST"
    body = json.loads(follow.content)
    for key, value in link["body"].items():
        assert body[key] == value
    assert body["collections"] == [client.collection]


@pytest.mark.asyncio
async def test_merge_false_posts_only_the_link_body() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            seen.append(json.loads(request.content))
        if len(seen) == 1:
            return httpx.Response(
                200,
                json={
                    "type": "FeatureCollection",
                    "features": [_feature("scene-a")],
                    "links": [
                        {
                            "rel": "next",
                            "method": "POST",
                            "href": SEARCH,
                            "merge": False,
                            "body": {"token": "only-this", "limit": 1},
                        }
                    ],
                },
            )
        return httpx.Response(
            200, json={"type": "FeatureCollection", "features": [_feature("scene-b")], "links": []}
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = _DemoStac(client=http)
    try:
        first = await client.search(_request())
        stored = json.loads(first.next_cursor or "")
        assert stored["merge"] is False
        assert stored["body"] == {"token": "only-this", "limit": 1}
        second = await client.search_from_cursor(_request(), first.next_cursor or "")
    finally:
        await http.aclose()
    assert [scene.id for scene in second.scenes] == ["scene-b"]
    assert seen[1] == {"token": "only-this", "limit": 1}


@pytest.mark.asyncio
async def test_omitted_method_with_a_body_is_posted() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        if "continuation" in body:
            return httpx.Response(
                200,
                json={"type": "FeatureCollection", "features": [_feature("scene-b")], "links": []},
            )
        return httpx.Response(
            200,
            json={
                "type": "FeatureCollection",
                "features": [_feature("scene-a")],
                "links": [
                    {
                        "rel": "next",
                        "href": SEARCH,
                        "merge": True,
                        "body": {"continuation": "more", "page": 2},
                    }
                ],
            },
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = _DemoStac(client=http)
    try:
        first = await client.search(_request())
        stored = json.loads(first.next_cursor or "")
        assert stored["method"] == "POST"
        assert stored["body"] == {"continuation": "more", "page": 2}
        second = await client.search_from_cursor(_request(), first.next_cursor or "")
    finally:
        await http.aclose()
    assert [scene.id for scene in second.scenes] == ["scene-b"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    [
        "sig=secret-value",
        "signature=secret-value",
        "X-Amz-Signature=secret-value",
        "x-goog-signature=secret-value",
        "sv=2024-01-01&sig=secret-value",
        "skoid=secret-value",
        "AWSAccessKeyId=secret-value",
        "SecurityToken=secret-value",
        "token=page-2&sig=secret-value",
    ],
)
async def test_signed_next_href_is_not_stored(query: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_page("scene-a", None, href=f"?{query}"))

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        result = await _DemoStac(client=http).search(_request())
    finally:
        await http.aclose()
    assert result.next_cursor is None
    assert result.next_cursor_unavailable == "signed"
    assert "secret-value" not in json.dumps(result.model_dump(mode="json"))


@pytest.mark.asyncio
async def test_page_token_query_is_stored_and_fetched() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if request.method == "GET":
            return httpx.Response(200, json=_page("scene-b", None))
        return httpx.Response(200, json=_page("scene-a", None, href="?token=page-2"))

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = _DemoStac(client=http)
    try:
        first = await client.search(_request())
        assert json.loads(first.next_cursor or "")["href"] == f"{SEARCH}?token=page-2"
        second = await client.search_from_cursor(_request(), first.next_cursor or "")
    finally:
        await http.aclose()
    assert [scene.id for scene in second.scenes] == ["scene-b"]
    assert calls[1] == f"{SEARCH}?token=page-2"


@pytest.mark.asyncio
async def test_signed_href_is_dropped_when_the_body_can_continue() -> None:
    seen: list[tuple[str, str, dict[str, Any] | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        seen.append((request.method, str(request.url), body if isinstance(body, dict) else None))
        if len(seen) == 1:
            return httpx.Response(
                200,
                json={
                    "type": "FeatureCollection",
                    "features": [_feature("scene-a")],
                    "links": [
                        {
                            "rel": "next",
                            "method": "POST",
                            "merge": True,
                            "href": f"{SEARCH}?sig=secret-value",
                            "body": {"next": "page-2"},
                        }
                    ],
                },
            )
        return httpx.Response(
            200, json={"type": "FeatureCollection", "features": [_feature("scene-b")], "links": []}
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = _DemoStac(client=http)
    try:
        first = await client.search(_request())
        assert first.next_cursor is not None
        assert "secret-value" not in first.next_cursor
        assert "sig=" not in first.next_cursor
        stored = json.loads(first.next_cursor)
        assert "href" not in stored
        assert stored["body"] == {"next": "page-2"}
        second = await client.search_from_cursor(_request(), first.next_cursor)
    finally:
        await http.aclose()
    assert [scene.id for scene in second.scenes] == ["scene-b"]
    assert seen[1][0] == "POST"
    assert seen[1][1] == SEARCH
    assert seen[1][2] is not None
    assert seen[1][2]["next"] == "page-2"
    assert seen[1][2]["collections"] == ["demo"]


@pytest.mark.asyncio
async def test_resume_refuses_a_signed_cursor_without_fetching() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"signed cursor was fetched: {request.url}")

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = _DemoStac(client=http)
    signed = f"{SEARCH}?sig=secret-value"
    encoded = json.dumps({"href": signed, "method": "GET"}, separators=(",", ":"))
    try:
        with pytest.raises(ValueError, match="signed") as first:
            await client.search_from_cursor(_request(), signed)
        with pytest.raises(ValueError, match="signed") as second:
            await client.search_from_cursor(_request(), encoded)
    finally:
        await http.aclose()
    assert "secret-value" not in str(first.value)
    assert "://" not in str(first.value)
    assert "secret-value" not in str(second.value)
    assert "://" not in str(second.value)


def _sample_catalog(cursor: str | None) -> SceneCatalog:
    return SceneCatalog(
        request=_request(),
        sources=[
            SourceScenes(
                source="sentinel2",
                collection="sentinel",
                search_url=SEARCH,
                next_cursor=cursor,
            )
        ],
    )


def test_manifest_is_replaced_after_fsync(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    order: list[str] = []
    real_fsync = os.fsync
    real_replace = os.replace
    real_open = os.open

    def spy_fsync(fd: int) -> None:
        kind = "dir" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file"
        order.append(kind)
        real_fsync(fd)

    def spy_replace(src: str | Path, dst: str | Path) -> None:
        source = Path(src)
        assert source.parent == tmp_path
        assert source.name.startswith(".manifest.json.")
        assert source.name.endswith(".tmp")
        order.append("replace")
        real_replace(src, dst)

    def spy_open(
        path: str | Path, flags: int, mode: int = 0o777, *, dir_fd: int | None = None
    ) -> int:
        if str(path).endswith(".tmp"):
            assert flags & os.O_CREAT
            assert flags & os.O_EXCL
        if dir_fd is None:
            return real_open(path, flags, mode)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "fsync", spy_fsync)
    monkeypatch.setattr(os, "replace", spy_replace)
    monkeypatch.setattr(os, "open", spy_open)
    _write_manifest(tmp_path, _sample_catalog("page-2"), DownloadReport())
    assert order == ["file", "replace", "dir"]
    assert list(tmp_path.glob(".manifest.json.*.tmp")) == []
    written = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert written["sources"][0]["next_cursor"] == "page-2"


def test_data_help_warns_that_signed_pages_stay_unfetched(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = argparse.ArgumentParser(prog="atlas")
    sub = parser.add_subparsers(dest="group")
    add_data_parser(sub)
    with pytest.raises(SystemExit) as caught:
        parser.parse_args(["data", "--help"])
    assert caught.value.code == 0
    text = capsys.readouterr().out
    assert "signed" in text
    assert "unfetched" in text
    assert "warning" in text
    assert "exits 0" in text
    assert "exit status stays 0" in text


def test_stale_manifest_temp_files_are_removed(tmp_path: Path) -> None:
    stale = tmp_path / f".manifest.json.{os.getpid()}.tmp"
    stale.write_text("leftover", encoding="utf-8")
    old = time.time() - (15 * 60) - 1
    os.utime(stale, (old, old))
    other = tmp_path / "notes.tmp"
    other.write_text("keep", encoding="utf-8")
    _write_manifest(tmp_path, _sample_catalog("page-2"), DownloadReport())
    assert not stale.exists()
    assert other.read_text(encoding="utf-8") == "keep"
    assert (tmp_path / "manifest.json").is_file()
    assert list(tmp_path.glob(".manifest.json.*.tmp")) == []


def test_fresh_manifest_temp_survives_a_concurrent_write(tmp_path: Path) -> None:
    fresh = tmp_path / ".manifest.json.inflight.tmp"
    fresh.write_text("in-flight", encoding="utf-8")
    _write_manifest(tmp_path, _sample_catalog("page-2"), DownloadReport())
    assert fresh.read_text(encoding="utf-8") == "in-flight"
    assert (tmp_path / "manifest.json").is_file()


def test_manifest_mode_honors_the_umask(tmp_path: Path) -> None:
    previous = os.umask(0o027)
    try:
        _write_manifest(tmp_path, _sample_catalog("page-2"), DownloadReport())
        mode = (tmp_path / "manifest.json").stat().st_mode & 0o777
    finally:
        os.umask(previous)
    assert mode == (0o666 & ~0o027)


def test_signed_get_without_a_body_stays_unavailable_on_resume(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        probe = _id_probe_response(request)
        if probe is not None:
            return probe
        calls += 1
        return httpx.Response(200, json=_page("scene-a", None, href="?sig=secret-value"))

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    clients = {"sentinel2": _SentinelStac(client=http)}
    note = "The next page link was signed and was not stored. Later pages were not fetched."
    try:
        assert run_data(_pull_args(tmp_path), clients=clients) == 0
        manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
        source = manifest["sources"][0]
        assert source["scene_ids"] == ["scene-a"]
        assert source["next_cursor"] is None
        assert source["next_cursor_unavailable"] == "signed"
        assert source["note"] == note
        stored = (tmp_path / "manifest.json").read_text(encoding="utf-8")
        assert "secret-value" not in stored
        assert "sig=" not in stored
        first_err = capsys.readouterr().err
        assert "signed and was not stored" in first_err
        assert "not fetched" in first_err
        assert "secret-value" not in first_err
        assert run_data(_pull_args(tmp_path, "--resume"), clients=clients) == 0
        assert calls == 1
        again = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
        resumed = again["sources"][0]
        assert resumed["scene_ids"] == ["scene-a"]
        assert resumed["next_cursor"] is None
        assert resumed["next_cursor_unavailable"] == "signed"
        assert resumed["note"] == note
        err = capsys.readouterr().err
        assert f"warning: sentinel2: {note}" in err
        assert "secret-value" not in err
        assert "://" not in resumed["note"]
    finally:
        asyncio.run(http.aclose())


def test_failed_manifest_replace_keeps_the_previous_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_manifest(tmp_path, _sample_catalog("page-2"), DownloadReport())
    original = (tmp_path / "manifest.json").read_text(encoding="utf-8")
    real_replace = os.replace

    def boom(src: str | Path, dst: str | Path) -> None:
        if Path(dst).name == "manifest.json":
            raise OSError("replace failed")
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="replace failed"):
        _write_manifest(tmp_path, _sample_catalog("page-3"), DownloadReport())
    assert (tmp_path / "manifest.json").read_text(encoding="utf-8") == original
    assert list(tmp_path.glob(".manifest.json.*.tmp")) == []
