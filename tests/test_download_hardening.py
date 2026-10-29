from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pytest

from atlas.cli import download as download_module
from atlas.cli.data import _write_manifest
from atlas.cli.download import (
    DownloadOutcome,
    describe_error,
    download_assets,
    legacy_scene_filename,
    partial_path,
    redact_secrets,
    scene_filename,
)
from atlas.compile.product import SceneCatalog, SourceScenes
from atlas.data.base import Asset, BBox, DataPullClient, PullRequest, PullResult, Scene
from atlas.data.himawari import _aoi_segments, _segment

AOI = BBox(west=-71.12, south=42.32, east=-71.02, north=42.40)
SIGNED = (
    "https://sentinel2l2a01.blob.core.windows.net/sentinel2-l2/T19/B04.tif"
    "?st=2026-06-01T00%3A00%3A00Z&se=2026-06-02T00%3A00%3A00Z&sp=rl&sv=2024-05-04"
    "&sr=c&skoid=abc&sig=SECRETSIGNATUREvalue%3D"
)
GOES_ID = "OR_ABI-L2-CMIPF-M6C13_G19_s2026152_e2026152_c20261522359236.nc"


def _scene(scene_id: str = "granule", href: str = "https://example.com/granule.nc") -> Scene:
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


class _Signer(DataPullClient):
    """Client whose ``sign_href`` goes through an injected HTTP transport."""

    def __init__(self, http: httpx.AsyncClient) -> None:
        self._http = http
        self.calls = 0

    async def search(self, request: PullRequest) -> PullResult:
        raise NotImplementedError

    async def sign_href(self, href: str) -> str:
        self.calls += 1
        response = await self._http.get("https://planetarycomputer.example/sign")
        response.raise_for_status()
        signed = response.json()["href"]
        assert isinstance(signed, str)
        return signed


@pytest.fixture
def waits(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    recorded: list[float] = []
    real_sleep = asyncio.sleep

    async def fake_sleep(delay: float) -> None:
        recorded.append(delay)
        await real_sleep(0)

    monkeypatch.setattr(download_module.asyncio, "sleep", fake_sleep)
    return recorded


# --- redaction ---------------------------------------------------------------


def test_redact_secrets_masks_sas_query_and_keeps_the_path() -> None:
    text = redact_secrets(f"Client error '403 Forbidden' for url '{SIGNED}'")
    assert "SECRETSIGNATURE" not in text
    assert "sig=" not in text and "se=" not in text and "st=" not in text
    assert "/sentinel2-l2/T19/B04.tif?***" in text


def test_redact_secrets_masks_firms_path_key_and_bare_params() -> None:
    url = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/0123456789abcdef0123456789abcdef/VIIRS/1,2,3,4/1"
    text = redact_secrets(f"failed {url} token=abc X-Amz-Signature=deadbeef Bearer eyJ.a.b")
    assert "0123456789abcdef" not in text
    assert "/api/area/csv/***/VIIRS/" in text
    assert "abc" not in text.replace("***", "")
    assert "deadbeef" not in text
    assert "eyJ" not in text


def test_redact_secrets_masks_url_user_info() -> None:
    text = redact_secrets("GET https://user:hunter2@host.example/file.nc failed")
    assert "hunter2" not in text
    assert "https://host.example/file.nc" in text


def test_describe_error_redacts_signed_url() -> None:
    request = httpx.Request("GET", SIGNED)
    response = httpx.Response(403, request=request)
    exc = httpx.HTTPStatusError("boom", request=request, response=response)
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as raised:
        exc = raised
    text = describe_error(exc)
    assert text.startswith("HTTPStatusError: ")
    assert "403" in text
    assert "SECRETSIGNATURE" not in text


def test_outcome_error_is_redacted_on_construction() -> None:
    outcome = DownloadOutcome(source="s", scene_id="x", status="failed", error=f"bad {SIGNED}")
    assert outcome.error is not None
    assert "SECRETSIGNATURE" not in outcome.error


async def test_signed_url_in_status_error_never_reaches_outcome_or_manifest(
    tmp_path: Path,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "planetarycomputer.example":
            return httpx.Response(200, json={"href": SIGNED})
        return httpx.Response(403, text="AuthenticationFailed")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        report = await download_assets(
            [_scene(href="https://sentinel2l2a01.blob.core.windows.net/x/B04.nc")],
            tmp_path,
            http=http,
            clients={"goes": _Signer(http)},
            retry_backoff=0.0,
        )
    outcome = report.outcomes[0]
    assert outcome.status == "failed"
    assert outcome.error is not None and "403" in outcome.error
    assert "SECRETSIGNATURE" not in outcome.error
    assert "sig=" not in outcome.error

    catalog = SceneCatalog(
        request=PullRequest(start_date=date(2024, 7, 1), end_date=date(2024, 7, 1), bbox=AOI),
        sources=[SourceScenes(source="goes", collection="c", scenes=[])],
    )
    _write_manifest(tmp_path, catalog, report)
    manifest = (tmp_path / "manifest.json").read_text()
    assert "403" in manifest
    assert "SECRETSIGNATURE" not in manifest


# --- signing retries ---------------------------------------------------------


@pytest.mark.parametrize("status", [429, 500, 503])
async def test_signing_retries_throttle_and_server_errors(
    tmp_path: Path, waits: list[float], status: int
) -> None:
    sign_calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "planetarycomputer.example":
            sign_calls.append(1)
            if len(sign_calls) < 4:
                return httpx.Response(status)
            return httpx.Response(200, json={"href": "https://example.com/signed.nc"})
        return httpx.Response(200, content=b"data")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        report = await download_assets(
            [_scene()], tmp_path, http=http, clients={"goes": _Signer(http)}
        )
    assert report.outcomes[0].status == "saved"
    assert len(sign_calls) == 4
    assert waits == [0.5, 1.0, 2.0]


async def test_signing_honors_retry_after_with_a_cap(tmp_path: Path, waits: list[float]) -> None:
    sign_calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "planetarycomputer.example":
            sign_calls.append(1)
            if len(sign_calls) == 1:
                return httpx.Response(429, headers={"retry-after": "3"})
            if len(sign_calls) == 2:
                return httpx.Response(429, headers={"retry-after": "600"})
            return httpx.Response(200, json={"href": "https://example.com/signed.nc"})
        return httpx.Response(200, content=b"data")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        report = await download_assets(
            [_scene()], tmp_path, http=http, clients={"goes": _Signer(http)}
        )
    assert report.outcomes[0].status == "saved"
    assert waits == [3.0, download_module.SIGN_WAIT_CAP_SECONDS]


async def test_signing_retries_are_bounded(tmp_path: Path, waits: list[float]) -> None:
    sign_calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sign_calls.append(1)
        return httpx.Response(429)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        report = await download_assets(
            [_scene()], tmp_path, http=http, clients={"goes": _Signer(http)}
        )
    assert len(sign_calls) == download_module.SIGN_ATTEMPTS
    assert len(waits) == download_module.SIGN_ATTEMPTS - 1
    assert max(waits) <= download_module.SIGN_WAIT_CAP_SECONDS
    assert report.outcomes[0].status == "failed"
    assert "429" in (report.outcomes[0].error or "")


@pytest.mark.parametrize("status", [400, 401, 403, 404])
async def test_signing_other_client_errors_are_not_retried(
    tmp_path: Path, waits: list[float], status: int
) -> None:
    sign_calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sign_calls.append(1)
        return httpx.Response(status)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        report = await download_assets(
            [_scene()], tmp_path, http=http, clients={"goes": _Signer(http)}
        )
    assert len(sign_calls) == 1
    assert waits == []
    assert report.outcomes[0].status == "failed"


async def test_signing_retries_a_transport_error(tmp_path: Path, waits: list[float]) -> None:
    sign_calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "planetarycomputer.example":
            sign_calls.append(1)
            if len(sign_calls) == 1:
                raise httpx.ConnectError("reset", request=request)
            return httpx.Response(200, json={"href": "https://example.com/signed.nc"})
        return httpx.Response(200, content=b"data")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        report = await download_assets(
            [_scene()], tmp_path, http=http, clients={"goes": _Signer(http)}
        )
    assert report.outcomes[0].status == "saved"
    assert waits == [0.5]


# --- partial files -----------------------------------------------------------


class _BrokenStream(httpx.AsyncByteStream):
    def __init__(self, seen: list[Path], directory: Path) -> None:
        self._seen = seen
        self._directory = directory

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield b"x" * 1024
        await asyncio.sleep(0)
        self._seen.extend(sorted(self._directory.iterdir()))
        raise httpx.ReadError("connection reset")


async def test_download_writes_a_partial_name_and_renames_on_success(tmp_path: Path) -> None:
    seen: list[Path] = []
    goes_dir = tmp_path / "goes"

    class _Watch(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield b"first"
            seen.extend(sorted(goes_dir.iterdir()))
            yield b"second"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=_Watch())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        report = await download_assets([_scene()], tmp_path, http=http, clients={})
    final = goes_dir / scene_filename("granule", "data", ".nc")
    assert seen == [partial_path(final)]
    assert report.outcomes[0].status == "saved"
    assert report.outcomes[0].path == final
    assert final.read_bytes() == b"firstsecond"
    assert sorted(goes_dir.iterdir()) == [final]


async def test_failed_download_leaves_no_file_and_is_retried(tmp_path: Path) -> None:
    seen: list[Path] = []
    goes_dir = tmp_path / "goes"

    def broken(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=_BrokenStream(seen, goes_dir))

    async with httpx.AsyncClient(transport=httpx.MockTransport(broken)) as http:
        report = await download_assets([_scene()], tmp_path, http=http, clients={})
    final = goes_dir / scene_filename("granule", "data", ".nc")
    assert report.outcomes[0].status == "failed"
    assert seen == [partial_path(final)]
    assert list(goes_dir.iterdir()) == []

    def ok(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"whole")

    async with httpx.AsyncClient(transport=httpx.MockTransport(ok)) as http:
        report = await download_assets([_scene()], tmp_path, http=http, clients={})
    assert report.outcomes[0].status == "saved"
    assert final.read_bytes() == b"whole"


async def test_leftover_partial_from_a_killed_run_is_not_skipped(tmp_path: Path) -> None:
    goes_dir = tmp_path / "goes"
    goes_dir.mkdir()
    final = goes_dir / scene_filename("granule", "data", ".nc")
    partial_path(final).write_bytes(b"truncated")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"complete")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        report = await download_assets([_scene()], tmp_path, http=http, clients={})
    assert report.outcomes[0].status == "saved"
    assert final.read_bytes() == b"complete"
    assert not partial_path(final).exists()


# --- legacy GOES names -------------------------------------------------------


async def test_legacy_goes_name_counts_as_present_and_is_renamed(tmp_path: Path) -> None:
    goes_dir = tmp_path / "goes"
    goes_dir.mkdir()
    legacy = goes_dir / legacy_scene_filename(GOES_ID, "data", ".nc")
    assert legacy.name.endswith(".nc_data.nc")
    legacy.write_bytes(b"old download")
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(200, content=b"new")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        report = await download_assets([_scene(GOES_ID)], tmp_path, http=http, clients={})
    current = goes_dir / scene_filename(GOES_ID, "data", ".nc")
    assert calls == []
    assert report.outcomes[0].status == "skipped"
    assert report.outcomes[0].path == current
    assert current.read_bytes() == b"old download"
    assert not legacy.exists()


async def test_empty_legacy_file_is_not_adopted(tmp_path: Path) -> None:
    goes_dir = tmp_path / "goes"
    goes_dir.mkdir()
    legacy = goes_dir / legacy_scene_filename(GOES_ID, "data", ".nc")
    legacy.write_bytes(b"")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"new")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        report = await download_assets([_scene(GOES_ID)], tmp_path, http=http, clients={})
    assert report.outcomes[0].status == "saved"
    assert (goes_dir / scene_filename(GOES_ID, "data", ".nc")).read_bytes() == b"new"


# --- Himawari segments -------------------------------------------------------


def test_wide_box_returns_segment_touched_near_sub_satellite_longitude() -> None:
    box = BBox(west=100.0, south=45.0, east=160.0, north=47.46)
    # The corners and edge midpoints all fall in segment 2; the north edge at
    # the sub-satellite longitude reaches segment 1.
    assert _segment(140.7, 47.46, "B03") == 1
    assert {_segment(lon, 47.46, "B03") for lon in (100.0, 130.0, 160.0)} == {2}
    assert _aoi_segments(box, "B03") == [1, 2]


def test_small_box_is_one_segment() -> None:
    box = BBox(west=139.60, south=35.60, east=139.90, north=35.80)
    assert _aoi_segments(box, "B03") == [2]


def test_box_off_the_disk_has_no_segments() -> None:
    box = BBox(west=-71.2, south=42.0, east=-70.9, north=42.2)
    assert _aoi_segments(box, "B03") == []


def test_tall_box_returns_every_segment_between_its_edges() -> None:
    box = BBox(west=139.0, south=-50.0, east=141.0, north=50.0)
    segments = _aoi_segments(box, "B03")
    assert segments == list(range(segments[0], segments[-1] + 1))
    assert segments[0] == 1 and segments[-1] == 10
