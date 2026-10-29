"""CDSE downloads use the OData https alternate and a short-lived bearer token."""

from __future__ import annotations

import httpx
import pytest

from atlas.cli.download import pick_asset
from atlas.data.cdse import CDSE_TOKEN_URL, CdseSentinel5PNo2Client
from atlas.data.stac import item_to_scene

ODATA = "https://download.dataspace.copernicus.eu/odata/v1/Products(abc)"


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
        assert await client.download_headers("https://eodata.s3.amazonaws.com/x.nc") == {}
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
        assert await client.download_headers(f"{ODATA}/$value") == {}


async def test_rejected_credentials_are_an_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CDSE_USERNAME", "user@example.com")
    monkeypatch.setenv("CDSE_PASSWORD", "wrong")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "invalid_grant"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = CdseSentinel5PNo2Client(client=http)
        with pytest.raises(httpx.HTTPStatusError) as excinfo:
            await client.download_headers(f"{ODATA}/$value")
    assert "wrong" not in str(excinfo.value)
