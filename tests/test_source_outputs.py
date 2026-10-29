"""GIBS images and FIRMS detection tables written by ``atlas data``."""

from __future__ import annotations

import csv
import io
import json
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pytest
from PIL import Image

from atlas.cli.data import execute_pull
from atlas.cli.progress import LogReporter
from atlas.cli.source_outputs import write_detections_csv, writes_own_output
from atlas.data.base import (
    Asset,
    BBox,
    DataPullClient,
    PullRequest,
    PullResult,
    Scene,
    SceneKind,
)
from atlas.data.firms import FirmsClient, FirmsModisClient, _rows_to_scenes
from atlas.data.gibs import GibsModisTrueColorClient, GibsThermalAnomaliesClient

AOI = BBox(west=-71.19, south=42.04, east=-70.92, north=42.22)
VIIRS_CSV = (
    "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,"
    "satellite,instrument,confidence,version,bright_ti5,frp,daynight\n"
    "42.10,-71.00,330.1,0.4,0.4,2024-07-01,1342,N,VIIRS,n,2.0NRT,290.1,4.2,D\n"
    "42.05,-71.10,340.5,0.5,0.4,2024-07-01,0610,N,VIIRS,h,2.0NRT,291.0,9.8,N\n"
)
MODIS_CSV = (
    "latitude,longitude,brightness,scan,track,acq_date,acq_time,"
    "satellite,instrument,confidence,version,bright_t31,frp,daynight\n"
    "-10.5,-50.25,320.4,1.0,1.0,2024-07-01,1405,A,MODIS,77,6.1NRT,295.2,18.3,D\n"
)


def _request(days: int = 1) -> PullRequest:
    return PullRequest(
        start_date=date(2024, 7, 1),
        end_date=date(2024, 7, days),
        bbox=AOI,
        limit=5,
    )


def _png(alpha: int = 255) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGBA", (8, 4), (10, 120, 40, alpha)).save(buffer, format="PNG")
    return buffer.getvalue()


async def _pull(
    out: Path,
    clients: dict[str, DataPullClient],
    handler: httpx.MockTransport,
    request: PullRequest | None = None,
) -> int:
    async with httpx.AsyncClient(transport=handler) as http:
        return await execute_pull(
            source_names=list(clients),
            request=request or _request(),
            out_dir=out,
            search_only=False,
            concurrency=2,
            asset_name=None,
            reporter=LogReporter(io.StringIO()),
            clients=clients,
            http=http,
            assume_yes=True,
        )


def _manifest(out: Path) -> dict[str, dict[str, object]]:
    payload = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    return {row["source"]: row for row in payload["sources"]}


async def test_gibs_pull_saves_the_wms_image_for_each_day(tmp_path: Path) -> None:
    seen: list[httpx.URL] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url)
        return httpx.Response(200, content=_png(), headers={"content-type": "image/png"})

    out = tmp_path / "out"
    code = await _pull(
        out,
        {"gibs_modis_truecolor": GibsModisTrueColorClient()},
        httpx.MockTransport(handler),
        _request(days=2),
    )

    assert code == 0
    files = sorted((out / "gibs_modis_truecolor").iterdir())
    assert [path.name for path in files] == [
        "MODIS_Terra_CorrectedReflectance_TrueColor_2024-07-01_wms.png",
        "MODIS_Terra_CorrectedReflectance_TrueColor_2024-07-02_wms.png",
    ]
    with Image.open(files[0]) as image:
        assert image.format == "PNG"
    assert {url.params["TIME"] for url in seen} == {"2024-07-01", "2024-07-02"}
    assert all(url.params["BBOX"] == "42.04,-71.19,42.22,-70.92" for url in seen)
    row = _manifest(out)["gibs_modis_truecolor"]
    assert row["saved"] == [f"gibs_modis_truecolor/{path.name}" for path in files]
    assert row["failed"] == []


async def test_gibs_thermal_layer_is_saved_too(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_png(), headers={"content-type": "image/png"})

    out = tmp_path / "out"
    code = await _pull(
        out, {"gibs_thermal": GibsThermalAnomaliesClient()}, httpx.MockTransport(handler)
    )

    assert code == 0
    assert len(list((out / "gibs_thermal").glob("*.png"))) == 1


async def test_gibs_blank_or_error_document_is_a_failure_not_a_file(tmp_path: Path) -> None:
    bodies = iter(
        [
            httpx.Response(200, content=_png(alpha=0), headers={"content-type": "image/png"}),
            httpx.Response(
                200,
                content=b"<?xml version='1.0'?><ServiceExceptionReport/>",
                headers={"content-type": "application/vnd.ogc.se_xml"},
            ),
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return next(bodies)

    out = tmp_path / "out"
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        code = await execute_pull(
            source_names=["gibs_modis_truecolor"],
            request=_request(days=2),
            out_dir=out,
            search_only=False,
            concurrency=1,
            asset_name=None,
            reporter=LogReporter(io.StringIO()),
            clients={"gibs_modis_truecolor": GibsModisTrueColorClient()},
            http=http,
            assume_yes=True,
        )

    assert code == 0
    assert not list((out / "gibs_modis_truecolor").glob("*.png"))
    failed = _manifest(out)["gibs_modis_truecolor"]["failed"]
    assert isinstance(failed, list)
    errors = " ".join(str(item["error"]) for item in failed)
    assert "fully transparent" in errors
    assert "did not return an image" in errors


async def test_gibs_write_replaces_a_symlink_without_following_it(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_png(), headers={"content-type": "image/png"})

    out = tmp_path / "out"
    dest = out / "gibs_modis_truecolor"
    dest.mkdir(parents=True)
    target = tmp_path / "outside.txt"
    target.write_text("keep", encoding="utf-8")
    link = dest / "MODIS_Terra_CorrectedReflectance_TrueColor_2024-07-01_wms.png"
    link.symlink_to(target)

    await _pull(
        out, {"gibs_modis_truecolor": GibsModisTrueColorClient()}, httpx.MockTransport(handler)
    )

    assert target.read_text(encoding="utf-8") == "keep"
    assert not link.is_symlink()
    assert link.read_bytes() == _png()


async def test_gibs_failed_fetch_retires_another_aois_image(tmp_path: Path) -> None:
    out = tmp_path / "out"
    dest = out / "gibs_modis_truecolor"
    dest.mkdir(parents=True)
    earlier = dest / "MODIS_Terra_CorrectedReflectance_TrueColor_2024-07-01_wms.png"
    earlier.write_bytes(b"other-aoi")
    earlier_jpg = earlier.with_suffix(".jpg")
    earlier_jpg.write_bytes(b"other-aoi-jpg")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    code = await _pull(
        out, {"gibs_modis_truecolor": GibsModisTrueColorClient()}, httpx.MockTransport(handler)
    )

    assert code == 0
    assert not earlier.exists()
    assert not earlier_jpg.exists()
    assert earlier.with_name(earlier.name + ".stale").read_bytes() == b"other-aoi"
    assert earlier_jpg.with_name(earlier_jpg.name + ".stale").read_bytes() == b"other-aoi-jpg"
    row = _manifest(out)["gibs_modis_truecolor"]
    assert row["saved"] == []
    failed = row["failed"]
    assert isinstance(failed, list) and "GIBS request failed" in str(failed[0]["error"])


async def test_gibs_failure_survives_an_old_image_that_cannot_be_moved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    dest = out / "gibs_modis_truecolor"
    dest.mkdir(parents=True)
    (dest / "MODIS_Terra_CorrectedReflectance_TrueColor_2024-07-01_wms.png").write_bytes(b"x")

    def stuck(_path: Path) -> None:
        raise PermissionError("read-only")

    monkeypatch.setattr("atlas.cli.source_outputs.retire_product", stuck)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    code = await _pull(
        out, {"gibs_modis_truecolor": GibsModisTrueColorClient()}, httpx.MockTransport(handler)
    )

    assert code == 0
    failed = _manifest(out)["gibs_modis_truecolor"]["failed"]
    assert isinstance(failed, list)
    assert "older image not moved aside" in str(failed[0]["error"])


async def test_firms_pull_writes_one_csv_row_per_detection(tmp_path: Path) -> None:
    def firms(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=VIIRS_CSV)

    def downloads(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected download: {request.url}")

    out = tmp_path / "out"
    client = FirmsClient(
        map_key="test-key", client=httpx.AsyncClient(transport=httpx.MockTransport(firms))
    )
    code = await _pull(out, {"firms_viirs": client}, httpx.MockTransport(downloads))

    assert code == 0
    path = out / "firms_viirs" / "firms_viirs_detections.csv"
    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    assert len(rows) == 2
    assert [row["acq_datetime"] for row in rows] == ["2024-07-01T06:10:00Z", "2024-07-01T13:42:00Z"]
    first = rows[0]
    assert first["latitude"] == "42.05"
    assert first["longitude"] == "-71.1"
    assert first["confidence"] == "h"
    assert first["frp"] == "9.8"
    assert first["bright_ti4"] == "340.5"
    assert first["bright_ti5"] == "291.0"
    assert first["scan"] == "0.5"
    assert first["daynight"] == "N"
    assert first["instrument"] == "VIIRS"
    assert _manifest(out)["firms_viirs"]["saved"] == ["firms_viirs/firms_viirs_detections.csv"]
    assert "test-key" not in path.read_text(encoding="utf-8")


def test_firms_modis_rows_keep_brightness_columns(tmp_path: Path) -> None:
    scenes = _rows_to_scenes(
        MODIS_CSV,
        source=FirmsModisClient.source,
        platform=FirmsModisClient.platform_name,
        instrument=FirmsModisClient.instrument_name,
        kind=SceneKind.detection,
        gsd_m=1000.0,
    )
    scenes = [scene.model_copy(update={"source": "firms_modis"}) for scene in scenes]

    outcome = write_detections_csv(scenes, tmp_path, "firms_modis")

    assert outcome.status == "saved" and outcome.path is not None
    rows = list(csv.DictReader(outcome.path.read_text(encoding="utf-8").splitlines()))
    assert rows[0]["brightness"] == "320.4"
    assert rows[0]["bright_t31"] == "295.2"
    assert rows[0]["confidence"] == "77"
    assert rows[0]["bright_ti4"] == ""
    assert rows[0]["acq_datetime"] == "2024-07-01T14:05:00Z"


class _BrowseOnly(DataPullClient):
    async def search(self, request: PullRequest) -> PullResult:
        start = datetime(2024, 7, 1, tzinfo=UTC)
        scene = Scene(
            id="browse-only",
            platform="other",
            collection="other",
            datetime=start,
            bbox=AOI,
            assets={
                "rendered_preview": Asset(
                    href="https://example.com/preview.png",
                    media_type="image/png",
                    roles=["overview"],
                )
            },
        )
        return PullResult(request=request, scenes=[scene])


async def test_other_sources_still_refuse_browse_assets(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"browse asset fetched: {request.url}")

    out = tmp_path / "out"
    client = _BrowseOnly()
    assert not writes_own_output(client)
    code = await _pull(out, {"other": client}, httpx.MockTransport(handler))

    assert code == 0
    assert not (out / "other").exists()
    failed = _manifest(out)["other"]["failed"]
    assert failed == [{"id": "browse-only", "error": "no native-resolution asset"}]
