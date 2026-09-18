from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, date, datetime
from pathlib import Path
from typing import ClassVar

import httpx
import pytest

from atlas.cli import _attach_negative_option_values, _inject_data_command, main
from atlas.cli.catalog import CatalogError, format_catalog, list_satellites, resolve_sources
from atlas.cli.data import add_data_parser, parse_coords, parse_date_range, run_data
from atlas.cli.download import BrowseAssetError, download_assets, pick_asset, scene_filename
from atlas.data.base import Asset, BBox, DataPullClient, PullRequest, PullResult, Scene, SceneKind
from atlas.data.registry import SOURCES

AOI = BBox(west=-71.12, south=42.32, east=-71.02, north=42.40)


def _scene(**kwargs: object) -> Scene:
    body: dict[str, object] = {
        "id": "s",
        "datetime": datetime(2024, 7, 1, tzinfo=UTC),
        "bbox": AOI,
        "platform": "x",
        "source": "sentinel2",
    }
    body.update(kwargs)
    return Scene.model_validate(body)


class FakePullClient(DataPullClient):
    satellite: ClassVar[str] = "sentinel2"
    scene_kind: ClassVar[SceneKind] = SceneKind.optical

    def __init__(self, scenes: list[Scene] | None = None, error: str | None = None) -> None:
        self._scenes = scenes or []
        self._error = error
        self.closed = False

    async def search(self, request: PullRequest) -> PullResult:
        if self._error:
            raise RuntimeError(self._error)
        return PullResult(request=request, scenes=self._scenes)

    async def aclose(self) -> None:
        self.closed = True


def test_data_list_prints_every_satellite(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["data", "--list"]) == 0
    out = capsys.readouterr().out
    for info in list_satellites():
        assert info.name in out
        for kind in info.kinds:
            assert kind.value in out
    assert "sentinel2" in out
    assert "landsat8" in out
    assert "himawari" in out
    assert "satellites" in out
    assert "SATELLITE" in out
    assert "KINDS" in out
    assert "dea_s2_ard" not in out
    assert "atlas <satellite> --list" in out


def test_data_with_no_args_lists_catalog(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["data"]) == 0
    out = capsys.readouterr().out
    for info in list_satellites():
        assert info.name in out


def test_every_source_declares_satellite_and_kind() -> None:
    missing: list[str] = []
    for name, cls in SOURCES.items():
        if not getattr(cls, "satellite", ""):
            missing.append(f"{name}: satellite")
        if not isinstance(getattr(cls, "scene_kind", None), SceneKind):
            missing.append(f"{name}: kind")
    assert missing == []


def test_list_catalog_picks_up_new_client() -> None:
    class ExtraClient(DataPullClient):
        satellite: ClassVar[str] = "sentinel2"
        scene_kind: ClassVar[SceneKind] = SceneKind.optical

        async def search(self, request: PullRequest) -> PullResult:
            raise NotImplementedError

    class NewSatClient(DataPullClient):
        satellite: ClassVar[str] = "newsat"
        scene_kind: ClassVar[SceneKind] = SceneKind.optical

        async def search(self, request: PullRequest) -> PullResult:
            raise NotImplementedError

    registry: dict[str, type[DataPullClient]] = {
        **SOURCES,
        "sentinel2_extra": ExtraClient,
        "newsat_extra": NewSatClient,
    }
    text = format_catalog(registry)
    assert "newsat" in text
    assert resolve_sources("sentinel2", "optical", registry) == ["sentinel2"]
    assert resolve_sources("newsat", "optical", registry) == ["newsat_extra"]
    sats = {info.name: info for info in list_satellites(registry)}
    assert SceneKind.optical in sats["sentinel2"].kinds
    assert "sentinel2_extra" in sats["sentinel2"].sources


def test_parse_date_and_coords() -> None:
    start, end = parse_date_range("2024-07-01:2024-07-07")
    assert start == date(2024, 7, 1)
    assert end == date(2024, 7, 7)
    slash_start, slash_end = parse_date_range("2024-07-01/2024-07-07")
    assert (slash_start, slash_end) == (start, end)
    box = parse_coords("-10.5, 51.0, -5.5, 55.5")
    assert box == BBox(west=-10.5, south=51.0, east=-5.5, north=55.5)
    with pytest.raises(ValueError, match="west,south,east,north"):
        parse_coords("1,2,3")
    with pytest.raises(ValueError, match="START:END"):
        parse_date_range("2024-07-01")


def test_scene_filename_sanitizes() -> None:
    assert scene_filename("S2A/tile:1", "visual", ".tif") == "S2A_tile_1_visual.tif"


def test_pick_asset_skips_browse_and_prefers_visual() -> None:
    scene = _scene(
        assets={
            "thumbnail": Asset(href="https://example.com/t.jpg", roles=["thumbnail"]),
            "rendered_preview": Asset(href="https://example.com/p.png", roles=["overview"]),
            "visual": Asset(
                href="https://example.com/visual.tif",
                media_type="image/tiff; application=geotiff",
                roles=["visual"],
            ),
        }
    )
    picked = pick_asset(scene)
    assert picked is not None
    key, asset = picked
    assert key == "visual"
    assert asset.href.endswith("visual.tif")
    browse_only = _scene(
        assets={"rendered_preview": Asset(href="https://example.com/p.png", roles=["overview"])}
    )
    assert pick_asset(browse_only) is None
    with pytest.raises(BrowseAssetError):
        pick_asset(scene, "thumbnail")
    with pytest.raises(BrowseAssetError):
        pick_asset(scene, "rendered_preview")
    assert pick_asset(scene, "missing") is None


def test_resolve_satellite_kind() -> None:
    assert resolve_sources("sentinel2", "optical") == ["sentinel2"]
    assert resolve_sources("s2", "l2a") == ["sentinel2"]
    assert resolve_sources("hls_sentinel", "optical") == ["hls_sentinel"]
    with pytest.raises(CatalogError, match="Available kinds"):
        resolve_sources("sentinel1", "optical")
    sar = resolve_sources("all", "sar")
    assert "sentinel1" in sar
    assert resolve_sources("landsat8", "optical") == ["landsat8"]
    assert resolve_sources("landsat9", "optical") == ["landsat9"]
    assert resolve_sources("alos", "sar") == ["alos_palsar"]
    assert resolve_sources("alos", "all") == ["alos_fnf", "alos_palsar"]


def test_unknown_satellite_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["data", "not-a-sat"]) == 2
    err = capsys.readouterr().err
    assert "Unknown satellite" in err


def test_satellite_list_is_scoped_to_that_satellite(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["alos", "--list"]) == 0
    out = capsys.readouterr().out
    assert "alos_fnf" in out
    assert "alos_palsar" in out
    assert "landcover" in out
    assert "sar" in out
    assert "sentinel2" not in out
    assert "modis" not in out
    assert "satellites  26" not in out


def test_bare_satellite_is_data_command(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["alos"]) == 0
    out = capsys.readouterr().out
    assert "alos" in out
    assert "sar" in out
    assert "landcover" in out
    assert "alos_palsar" in out
    assert main(["s2"]) == 0
    assert "optical" in capsys.readouterr().out


def test_negative_coords_are_not_parsed_as_flags() -> None:
    tokens = [
        "data",
        "sentinel2",
        "optical",
        "--date",
        "2024-07-01:2024-07-07",
        "--coords",
        "-71.12,42.32,-71.02,42.40",
        "--search-only",
    ]
    parser = argparse.ArgumentParser(prog="atlas")
    sub = parser.add_subparsers(dest="group")
    add_data_parser(sub)
    args = parser.parse_args(_attach_negative_option_values(_inject_data_command(tokens)))
    assert args.coords == "-71.12,42.32,-71.02,42.40"
    assert parse_coords(args.coords).west == -71.12


def test_asset_thumbnail_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        main(
            [
                "data",
                "sentinel2",
                "optical",
                "--date",
                "2024-07-01:2024-07-07",
                "--coords",
                "-71.12,42.32,-71.02,42.40",
                "--asset",
                "thumbnail",
            ]
        )
        == 2
    )
    assert "preview/thumbnail" in capsys.readouterr().err


def test_help_contains_example() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["data", "--help"])
    assert exc.value.code == 0


def test_help_example_text(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["data", "--help"])
    out = capsys.readouterr().out
    assert "atlas data sentinel2 optical --date" in out
    assert "--coords" in out
    assert "--no-tui" not in out
    assert "uv run" not in out


@pytest.mark.asyncio
async def test_download_assets_writes_raw_bytes(tmp_path: Path) -> None:
    payload = b"II*\x00fake-geotiff"

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://example.com/visual.tif"
        return httpx.Response(
            200,
            content=payload,
            headers={"content-type": "image/tiff; application=geotiff"},
        )

    scene = _scene(
        id="chip-1",
        assets={
            "rendered_preview": Asset(href="https://example.com/p.png", roles=["overview"]),
            "visual": Asset(
                href="https://example.com/visual.tif",
                media_type="image/tiff; application=geotiff",
                roles=["visual"],
            ),
        },
    )
    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http:
        saved = await download_assets([scene], tmp_path, http=http, clients={})
    path = saved["sentinel2"][0]
    assert path.name == "chip-1_visual.tif"
    assert path.read_bytes() == payload


@pytest.mark.asyncio
async def test_download_assets_isolates_sign_timeout(tmp_path: Path) -> None:
    payload = b"II*\x00ok"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=payload,
            headers={"content-type": "image/tiff; application=geotiff"},
        )

    visual = Asset(
        href="https://example.com/visual.tif",
        media_type="image/tiff; application=geotiff",
        roles=["visual"],
    )
    ok = _scene(id="ok-1", source="ok", assets={"visual": visual})
    bad = _scene(id="bad-1", source="bad", assets={"visual": visual})

    class TimeoutSignClient(FakePullClient):
        async def sign_href(self, href: str) -> str:
            raise httpx.ReadTimeout("sas")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http:
        saved = await download_assets(
            [bad, ok],
            tmp_path,
            http=http,
            clients={"bad": TimeoutSignClient(), "ok": FakePullClient()},
            concurrency=2,
        )
    assert saved["bad"] == []
    assert saved["ok"][0].read_bytes() == payload


@pytest.mark.asyncio
async def test_download_assets_skips_browse_only_scene(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"GET should not run: {request.url}")

    scene = _scene(
        id="chip-1",
        assets={"thumbnail": Asset(href="https://example.com/t.jpg", roles=["thumbnail"])},
    )
    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http:
        saved = await download_assets([scene], tmp_path, http=http, clients={})
    assert saved["sentinel2"] == []


def test_run_data_isolates_source_errors(tmp_path: Path) -> None:
    ok_scene = _scene(
        id="ok-1",
        source="ok",
        assets={
            "visual": Asset(
                href="https://example.com/visual.tif",
                media_type="image/tiff; application=geotiff",
            )
        },
    )
    clients = {
        "ok": FakePullClient(scenes=[ok_scene]),
        "bad": FakePullClient(error="boom"),
    }
    args = _pull_args(tmp_path, search_only=True)
    code = run_data(args, clients=clients)
    assert code == 0
    assert clients["ok"].closed is False  # caller-supplied clients are not closed
    manifest = (tmp_path / "manifest.json").read_text(encoding="utf-8")
    assert "ok-1" in manifest
    assert "boom" in manifest or "RuntimeError" in manifest


def test_all_sources_fail_exits_1(tmp_path: Path) -> None:
    clients = {"bad": FakePullClient(error="nope")}
    args = _pull_args(tmp_path, search_only=True)
    assert run_data(args, clients=clients) == 1


def test_search_only_does_not_get(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"GET should not run: {request.url}")

    scene = _scene(
        id="chip-1",
        source="ok",
        assets={
            "visual": Asset(
                href="https://example.com/visual.tif",
                media_type="image/tiff; application=geotiff",
            )
        },
    )
    clients = {"ok": FakePullClient(scenes=[scene])}
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    args = _pull_args(tmp_path, search_only=True)
    try:
        assert run_data(args, clients=clients, http=http) == 0
    finally:
        asyncio.run(http.aclose())
    assert not list(tmp_path.glob("**/*.png"))


def _pull_args(tmp_path: Path, *, search_only: bool) -> argparse.Namespace:
    return argparse.Namespace(
        list_catalog=False,
        satellite="sentinel2",
        kind="optical",
        date="2024-07-01:2024-07-07",
        coords="-71.12,42.32,-71.02,42.40",
        out=tmp_path,
        limit=10,
        max_cloud_cover=None,
        concurrency=2,
        search_only=search_only,
        asset=None,
    )
