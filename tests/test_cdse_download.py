"""CDSE downloads use the OData https alternate and a short-lived bearer token."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from atlas.cli.download import download_assets, pick_asset
from atlas.data import cdse as cdse_module
from atlas.data.base import Asset, BBox, Scene
from atlas.data.cdse import CDSE_TOKEN_URL, CdseSentinel5PNo2Client
from atlas.data.credentials import CredentialRejectedError, MissingCredentialError
from atlas.data.stac import item_to_scene

ODATA = "https://download.dataspace.copernicus.eu/odata/v1/Products(abc)"


def _scene(scene_id: str) -> Scene:
    return Scene.model_validate(
        {
            "id": scene_id,
            "datetime": datetime(2026, 6, 10, tzinfo=UTC),
            "bbox": BBox(west=-72.0, south=41.0, east=-70.0, north=43.0),
            "platform": "sentinel-5p",
            "source": "sentinel5p_no2",
            "assets": {
                "data": Asset(
                    href=f"{ODATA}/Nodes({scene_id}.nc)/$value",
                    media_type="application/netcdf",
                    roles=["data"],
                )
            },
        }
    )


def _feature(assets: dict[str, dict[str, object]]) -> dict[str, object]:
    return {
        "id": "S3B_OL_1_EFR",
        "bbox": [-72.0, 41.0, -70.0, 43.0],
        "geometry": {"type": "Polygon", "coordinates": []},
        "properties": {"datetime": "2026-06-10T14:56:23Z"},
        "assets": assets,
    }


def _eodata(name: str, roles: list[str]) -> dict[str, object]:
    return {
        "href": f"s3://eodata/Sentinel-3/OLCI/S3B.SEN3/{name}.nc",
        "roles": roles,
        "alternate": {"https": {"href": f"{ODATA}/Nodes(S3B.SEN3)/Nodes({name}.nc)/$value"}},
    }


def test_s3_href_uses_the_https_alternate() -> None:
    scene = item_to_scene(_feature({"Oa01_radianceData": _eodata("Oa01_radiance", ["data"])}))
    assert scene is not None
    assert scene.assets["Oa01_radianceData"].href == (
        f"{ODATA}/Nodes(S3B.SEN3)/Nodes(Oa01_radiance.nc)/$value"
    )


def test_default_pick_skips_sentinel3_helper_layers_and_the_archive() -> None:
    scene = item_to_scene(
        _feature(
            {
                "product": {"href": f"{ODATA}/$value", "roles": ["data", "metadata", "archive"]},
                "tie-meteo": _eodata("tie_meteo", ["data"]),
                "quality-flags": _eodata("qualityFlags", ["data"]),
                "removed-pixels": _eodata("removed_pixels", ["data"]),
                "geo-coordinates": _eodata("geo_coordinates", ["data"]),
                "instrument-data": _eodata("instrument_data", ["data"]),
                "Oa01_radianceData": _eodata("Oa01_radiance", ["data"]),
            }
        )
    )
    assert scene is not None
    picked = pick_asset(scene)
    assert picked is not None
    assert picked[0] == "Oa01_radianceData"
    only_product = item_to_scene(
        _feature({"Product": {"href": f"{ODATA}/$value", "roles": ["data", "archive"]}})
    )
    assert only_product is not None
    picked = pick_asset(only_product)
    assert picked is not None
    assert picked[0] == "Product"


async def test_token_goes_only_to_the_download_host_and_is_reused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CDSE_USERNAME", "user@example.com")
    monkeypatch.setenv("CDSE_PASSWORD", "pw")
    forms: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == CDSE_TOKEN_URL
        forms.append(request.content)
        return httpx.Response(200, json={"access_token": "tok", "expires_in": 1800})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = CdseSentinel5PNo2Client(client=http)
        assert await client.download_headers(f"{ODATA}/$value") == {"Authorization": "Bearer tok"}
        assert await client.download_headers(f"{ODATA}/Nodes(x.nc)/$value") == {
            "Authorization": "Bearer tok"
        }
        zipper = "https://zipper.dataspace.copernicus.eu/odata/v1/Products(abc)/Nodes(x.nc)/$value"
        assert await client.download_headers(zipper) == {"Authorization": "Bearer tok"}
        assert await client.download_headers("https://eodata.s3.amazonaws.com/x.nc") == {}
        assert await client.download_headers("https://example.com/x.nc") == {}
        assert await client.download_headers(ODATA.replace("https", "http", 1)) == {}
    assert len(forms) == 1
    assert b"grant_type=password" in forms[0]
    assert b"client_id=cdse-public" in forms[0]


async def test_no_credentials_sends_no_token_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CDSE_USERNAME", raising=False)
    monkeypatch.delenv("CDSE_PASSWORD", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request expected")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = CdseSentinel5PNo2Client(client=http)
        with pytest.raises(MissingCredentialError) as excinfo:
            await client.download_headers(f"{ODATA}/$value")
        assert await client.download_headers("https://eodata.s3.amazonaws.com/x.nc") == {}
    assert "CDSE_USERNAME and CDSE_PASSWORD are not set" in str(excinfo.value)


async def test_one_unset_variable_is_named(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CDSE_USERNAME", "user@example.com")
    monkeypatch.setenv("CDSE_PASSWORD", "  ")
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(500))) as h:
        with pytest.raises(MissingCredentialError) as excinfo:
            await CdseSentinel5PNo2Client(client=h).download_headers(f"{ODATA}/$value")
    assert excinfo.value.names == ("CDSE_PASSWORD",)


async def test_missing_credentials_are_the_scene_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("CDSE_USERNAME", raising=False)
    monkeypatch.delenv("CDSE_PASSWORD", raising=False)
    gets: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        gets.append(str(request.url))
        return httpx.Response(401)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = CdseSentinel5PNo2Client(client=http)
        report = await download_assets(
            [_scene("s5p-1")], tmp_path, http=http, clients={"sentinel5p_no2": client}
        )
    outcome = report.outcomes[0]
    assert outcome.status == "failed"
    assert outcome.error is not None
    assert "CDSE_USERNAME and CDSE_PASSWORD are not set" in outcome.error
    assert gets == []


async def test_rejected_credentials_are_named(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CDSE_USERNAME", "user@example.com")
    monkeypatch.setenv("CDSE_PASSWORD", "wrong")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "invalid_grant"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = CdseSentinel5PNo2Client(client=http)
        with pytest.raises(CredentialRejectedError) as excinfo:
            await client.download_headers(f"{ODATA}/$value")
    assert "wrong" not in str(excinfo.value)
    assert "CDSE_USERNAME and CDSE_PASSWORD were rejected (HTTP 401)" in str(excinfo.value)


@pytest.mark.parametrize("status", [400, 401])
async def test_a_rejected_login_is_not_retried_for_later_scenes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, status: int
) -> None:
    monkeypatch.setenv("CDSE_USERNAME", "user@example.com")
    monkeypatch.setenv("CDSE_PASSWORD", "wrong")
    posts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == CDSE_TOKEN_URL:
            posts.append(1)
            return httpx.Response(status, json={"error": "invalid_grant"})
        raise AssertionError("no download expected")

    scenes = [_scene(f"s5p-{index}") for index in range(4)]
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = CdseSentinel5PNo2Client(client=http)
        report = await download_assets(
            scenes, tmp_path, http=http, clients={"sentinel5p_no2": client}
        )
    assert posts == [1]
    assert [outcome.status for outcome in report.outcomes] == ["failed"] * 4
    for outcome in report.outcomes:
        assert outcome.error is not None
        assert f"were rejected (HTTP {status})" in outcome.error


async def test_other_token_failures_are_not_remembered(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CDSE_USERNAME", "user@example.com")
    monkeypatch.setenv("CDSE_PASSWORD", "pw")
    replies = [
        httpx.Response(503),
        httpx.Response(400, json={"error": "invalid_request"}),
        httpx.Response(200, json={"access_token": "tok", "expires_in": 600}),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return replies.pop(0)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = CdseSentinel5PNo2Client(client=http)
        for _ in range(2):
            with pytest.raises(httpx.HTTPStatusError):
                await client.download_headers(f"{ODATA}/$value")
        assert await client.download_headers(f"{ODATA}/$value") == {"Authorization": "Bearer tok"}


async def test_token_is_refreshed_before_a_retry_window_can_outlast_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CDSE_USERNAME", "user@example.com")
    monkeypatch.setenv("CDSE_PASSWORD", "pw")
    now = [1000.0]
    monkeypatch.setattr(cdse_module.time, "monotonic", lambda: now[0])
    issued: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        issued.append(1)
        return httpx.Response(200, json={"access_token": f"tok{len(issued)}", "expires_in": 600})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = CdseSentinel5PNo2Client(client=http)
        assert await client.download_headers(f"{ODATA}/$value") == {"Authorization": "Bearer tok1"}
        # 420 s of the token's life left: still reused.
        now[0] += 180.0
        assert await client.download_headers(f"{ODATA}/$value") == {"Authorization": "Bearer tok1"}
        # 300 s left could run out across GET retries, so a new token is fetched.
        now[0] += 120.0
        assert await client.download_headers(f"{ODATA}/$value") == {"Authorization": "Bearer tok2"}
