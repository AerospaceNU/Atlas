from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, date, datetime
from email.utils import format_datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from atlas.cli import download as download_module
from atlas.cli.data import _print_run_summary
from atlas.cli.download import (
    DownloadOutcome,
    DownloadReport,
    describe_error,
    download_assets,
    scene_filename,
)
from atlas.compile.product import SceneCatalog, SourceScenes
from atlas.data.base import Asset, BBox, DataPullClient, PullRequest, PullResult, Scene

AOI = BBox(west=-71.12, south=42.32, east=-71.02, north=42.40)
HREF = "https://example.com/granule.nc"


def _scene(scene_id: str = "granule", href: str = HREF) -> Scene:
    return Scene.model_validate(
        {
            "id": scene_id,
            "datetime": datetime(2024, 7, 1, tzinfo=UTC),
            "bbox": AOI,
            "platform": "x",
            "source": "goes",
            "assets": {"data": Asset(href=href, media_type="application/netcdf", roles=["data"])},
        }
    )


class _Stream(httpx.AsyncByteStream):
    """Body that sends ``first`` and then waits ``pause`` seconds per chunk."""

    def __init__(self, first: bytes, pause: float, rest: list[bytes] | None = None) -> None:
        self._first = first
        self._pause = pause
        self._rest = rest or []

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self._first
        for chunk in self._rest:
            await asyncio.sleep(self._pause)
            yield chunk
        await asyncio.sleep(self._pause)


async def _download(handler: httpx.MockTransport, tmp_path: Path, **kwargs: Any) -> DownloadReport:
    async with httpx.AsyncClient(transport=handler) as http:
        return await download_assets(
            [_scene()], tmp_path, http=http, clients={}, retry_backoff=0.0, **kwargs
        )


def test_scene_filename_drops_an_id_extension_that_repeats_the_suffix() -> None:
    goes = "OR_ABI-L2-CMIPF-M6C13_G19_s2026152_e2026152_c20261522359236.nc"
    assert (
        scene_filename(goes, "data", ".nc")
        == "OR_ABI-L2-CMIPF-M6C13_G19_s2026152_e2026152_c20261522359236_data.nc"
    )
    assert scene_filename("a.NC", "data", ".nc") == "a_data.nc"


def test_scene_filename_keeps_ids_whose_extension_differs() -> None:
    assert scene_filename("S2A/tile:1", "visual", ".tif") == "S2A_tile_1_visual.tif"
    assert scene_filename("granule.h5", "data", ".nc") == "granule.h5_data.nc"
    assert scene_filename("v1.0", "data", ".tif") == "v1.0_data.tif"
    assert scene_filename("x.bin", "data", ".bin") == "x.bin_data.bin"
    assert scene_filename(".nc", "data", ".nc") == ".nc_data.nc"


def test_describe_error_is_never_blank() -> None:
    assert describe_error(httpx.ReadTimeout("")) == "ReadTimeout"
    try:
        raise httpx.ReadTimeout("") from TimeoutError("socket read")
    except httpx.ReadTimeout as exc:
        assert describe_error(exc) == "ReadTimeout: socket read"
    assert describe_error(ValueError("bad")) == "ValueError: bad"


async def test_empty_read_timeout_is_recorded_with_text(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("", request=request)

    report = await _download(httpx.MockTransport(handler), tmp_path)
    outcome = report.outcomes[0]
    assert outcome.status == "failed"
    assert outcome.error == "ReadTimeout"


async def test_stalled_body_is_aborted_and_partial_removed(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            stream=_Stream(b"x" * 10, pause=5.0, rest=[b"y"]),
            headers={"content-type": "application/netcdf"},
        )

    report = await _download(
        httpx.MockTransport(handler), tmp_path, stall_seconds=0.1, stall_min_bytes=1024
    )
    outcome = report.outcomes[0]
    assert outcome.status == "failed"
    assert outcome.error is not None
    assert outcome.error.startswith("DownloadStallError: download stalled")
    assert "after 10 bytes" in outcome.error
    assert list((tmp_path / "goes").iterdir()) == []


async def test_trickling_body_is_aborted(tmp_path: Path) -> None:
    # Each chunk arrives well inside the window, but the window total stays tiny.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            stream=_Stream(b"x", pause=0.03, rest=[b"y"] * 50),
            headers={"content-type": "application/netcdf"},
        )

    report = await _download(
        httpx.MockTransport(handler), tmp_path, stall_seconds=0.2, stall_min_bytes=1024
    )
    assert report.outcomes[0].status == "failed"
    assert "stalled" in (report.outcomes[0].error or "")


async def test_steady_body_is_not_a_stall(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            stream=_Stream(b"a" * 64, pause=0.02, rest=[b"b" * 64] * 10),
            headers={"content-type": "application/netcdf"},
        )

    report = await _download(
        httpx.MockTransport(handler), tmp_path, stall_seconds=0.2, stall_min_bytes=64
    )
    assert report.outcomes[0].status == "saved"
    assert report.outcomes[0].path is not None
    assert report.outcomes[0].path.stat().st_size == 64 * 11


@pytest.mark.parametrize("status", [429, 500, 502, 503])
async def test_retryable_status_is_retried(tmp_path: Path, status: int) -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(status, headers={"retry-after": "0"})
        return httpx.Response(
            200, content=b"CDF\x01", headers={"content-type": "application/netcdf"}
        )

    report = await _download(httpx.MockTransport(handler), tmp_path)
    assert len(calls) == 3
    assert report.outcomes[0].status == "saved"
    files = list((tmp_path / "goes").iterdir())
    assert [path.name for path in files] == ["granule_data.nc"]
    assert files[0].read_bytes() == b"CDF\x01"


async def test_retries_are_bounded(tmp_path: Path) -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(503, text="busy")

    report = await _download(httpx.MockTransport(handler), tmp_path, attempts=3)
    assert len(calls) == 3
    outcome = report.outcomes[0]
    assert outcome.status == "failed"
    assert outcome.error is not None and "503" in outcome.error
    assert list((tmp_path / "goes").iterdir()) == []


@pytest.mark.parametrize("status", [400, 401, 403, 404])
async def test_other_client_errors_are_not_retried(tmp_path: Path, status: int) -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(status)

    report = await _download(httpx.MockTransport(handler), tmp_path)
    assert len(calls) == 1
    assert report.outcomes[0].status == "failed"
    assert str(status) in (report.outcomes[0].error or "")


async def test_retry_after_is_honored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    waits: list[float] = []
    real_sleep = asyncio.sleep

    async def fake_sleep(delay: float) -> None:
        waits.append(delay)
        await real_sleep(0)

    monkeypatch.setattr(download_module.asyncio, "sleep", fake_sleep)
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, headers={"retry-after": "7"})
        if len(calls) == 2:
            return httpx.Response(503, headers={"retry-after": "9999"})
        return httpx.Response(200, content=b"ok")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        report = await download_assets(
            [_scene()], tmp_path, http=http, clients={}, retry_backoff=1.0
        )
    assert report.outcomes[0].status == "saved"
    assert waits == [7.0, download_module.RETRY_AFTER_CAP_SECONDS]


def test_retry_delay_reads_an_http_date() -> None:
    future = datetime.now(UTC).replace(microsecond=0)
    when = format_datetime(future.fromtimestamp(future.timestamp() + 30, UTC), usegmt=True)
    response = httpx.Response(429, headers={"retry-after": when})
    delay = download_module._retry_delay(response, 1, 2.0)
    assert 20.0 <= delay <= 31.0
    junk = httpx.Response(429, headers={"retry-after": "soon"})
    assert download_module._retry_delay(junk, 2, 2.0) == 4.0


async def test_unexpected_client_error_fails_one_scene_only(tmp_path: Path) -> None:
    class Broken(DataPullClient):
        async def search(self, request: PullRequest) -> PullResult:
            raise NotImplementedError

        async def download_headers(self, href: str) -> dict[str, str]:
            raise RuntimeError("token helper exploded")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"ok")

    bad = _scene("bad-1").model_copy(update={"source": "bad"})
    ok = _scene("ok-1")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        report = await download_assets([bad, ok], tmp_path, http=http, clients={"bad": Broken()})
    assert report.for_source("bad")[0].error == "RuntimeError: token helper exploded"
    assert report.for_source("goes")[0].status == "saved"


def test_summary_lists_a_failure_without_error_text(capsys: pytest.CaptureFixture[str]) -> None:
    request = PullRequest(start_date=date(2026, 9, 24), end_date=date(2026, 9, 24), bbox=AOI)
    catalog = SceneCatalog(
        request=request,
        sources=[SourceScenes(source="goes_cmi", collection="goes-cmi", scenes=[])],
    )
    report = DownloadReport(
        outcomes=[
            DownloadOutcome(source="goes_cmi", scene_id="big", status="failed", error=None),
            DownloadOutcome(source="goes_cmi", scene_id="slow", status="failed", error="Stall"),
        ]
    )
    _print_run_summary(catalog, report, search_only=False)
    err = capsys.readouterr().err
    assert "goes_cmi big: failed (no error detail recorded)" in err
    assert "goes_cmi slow: Stall" in err
