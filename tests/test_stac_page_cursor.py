from __future__ import annotations

import argparse
import asyncio
import json
from datetime import date
from pathlib import Path
from typing import Any, ClassVar

import httpx
import pytest

from atlas.cli.data import add_data_parser, run_data
from atlas.compile.aggregate import aggregate
from atlas.data.base import BBox, DataPullClient, PullRequest, PullResult, SceneKind
from atlas.data.stac import StacApiClient

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


@pytest.mark.asyncio
async def test_post_token_is_the_cursor_and_page_one_omits_it() -> None:
    pages = _TokenPages({None: ("scene-a", "page-2")})
    http, clients = _clients(pages)
    try:
        result = await clients["sentinel2"].search(_request())
    finally:
        await http.aclose()
    assert result.next_cursor == "page-2"
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
    assert result.next_cursor == "page-2"


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
        assert first.next_cursor == f"{SEARCH}?page=2"
        second = await client.search_from_cursor(_request(), first.next_cursor or "")
    finally:
        await http.aclose()
    assert [scene.id for scene in second.scenes] == ["scene-b"]
    assert second.next_cursor is None
    assert calls == [("POST", SEARCH), ("GET", f"{SEARCH}?page=2")]


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
        assert sentinel["next_cursor"] == "page-2"
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
        assert sentinel["next_cursor"] == "page-3"
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
        assert source["next_cursor"] == "page-2"
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
        assert _source(manifest, "sentinel2")["next_cursor"] == "page-2"
    finally:
        asyncio.run(http.aclose())
