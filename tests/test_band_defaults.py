from __future__ import annotations

import json
from datetime import date
from typing import Any

import httpx
import pytest

from atlas.cli.download import pick_asset
from atlas.data.base import BBox, PullRequest
from atlas.data.cdse import CdseSentinel3OlciClient
from atlas.data.pc_more import GoesCmiClient
from atlas.data.stac import StacApiClient

_TIF = "image/tiff; application=geotiff; profile=cloud-optimized"


def _band(name: str, common: str, wavelength: float) -> list[dict[str, Any]]:
    return [{"name": name, "common_name": common, "center_wavelength": wavelength}]


def _feature(assets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "type": "Feature",
        "id": "item-1",
        "bbox": [-72.0, 41.0, -70.0, 43.0],
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[-72, 41], [-70, 41], [-70, 43], [-72, 43], [-72, 41]]],
        },
        "properties": {"datetime": "2026-06-10T14:56:23Z"},
        "assets": {
            key: {"href": f"https://example.test/{key}", **spec} for key, spec in assets.items()
        },
    }


def _goes_assets(*, web_mercator: bool) -> dict[str, dict[str, Any]]:
    assets: dict[str, dict[str, Any]] = {
        "C01_2km": {"type": _TIF, "eo:bands": _band("ABI Band 1", "blue", 0.47)},
        "C02_2km": {"type": _TIF, "eo:bands": _band("ABI Band 2", "red", 0.64)},
        "C03_1km": {
            "type": _TIF,
            "roles": ["data"],
            "eo:bands": _band("ABI Band 3", "nir09", 0.87),
        },
        "C03_2km": {"type": _TIF, "eo:bands": _band("ABI Band 3", "nir09", 0.87)},
        "C02_0.5km": {
            "type": _TIF,
            "roles": ["data"],
            "eo:bands": _band("ABI Band 2", "red", 0.64),
        },
        "CMIP_C02-nc": {
            "type": "application/netcdf",
            "roles": ["data"],
            "eo:bands": _band("ABI Band 2", "red", 0.64),
        },
    }
    if web_mercator:
        for number, common, wavelength in (
            (1, "blue", 0.47),
            (2, "red", 0.64),
            (3, "nir09", 0.87),
        ):
            assets[f"C0{number}_2km_wm"] = {
                "type": _TIF,
                "eo:bands": _band(f"ABI Band {number}", common, wavelength),
            }
    return assets


def _olci_assets() -> dict[str, dict[str, Any]]:
    assets: dict[str, dict[str, Any]] = {
        "product": {"type": "application/zip", "roles": ["data", "metadata", "archive"]},
        "thumbnail": {"type": "image/jpeg", "roles": ["thumbnail"]},
    }
    for number in range(1, 22):
        assets[f"Oa{number:02d}_radianceData"] = {
            "type": "application/netcdf",
            "roles": ["data"],
        }
    return assets


def _request() -> PullRequest:
    return PullRequest(
        bbox=BBox(west=-71.19, south=42.04, east=-70.92, north=42.22),
        start_date=date(2026, 6, 1),
        end_date=date(2026, 6, 10),
        limit=1,
    )


def _transport(feature: dict[str, Any]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        payload = {"type": "FeatureCollection", "features": [feature], "links": []}
        if "ids" in body:
            payload["features"] = [feature] if feature["id"] in body["ids"] else []
        return httpx.Response(200, json=payload)

    return httpx.MockTransport(handler)


async def _search_and_resolve(
    cls: type[StacApiClient], feature: dict[str, Any]
) -> list[dict[str, str]]:
    """Band aliases from ``search`` and from the id lookup, in that order."""
    http = httpx.AsyncClient(transport=_transport(feature))
    async with cls(client=http) as client:
        result = await client.search(_request())
        resolved = await client.resolve_scene_ids([feature["id"]])
    await http.aclose()
    return [result.scenes[0].band_aliases, resolved[feature["id"]].band_aliases]


async def test_olci_default_pick_is_the_665nm_radiance_band() -> None:
    feature = _feature(_olci_assets())
    for aliases in await _search_and_resolve(CdseSentinel3OlciClient, feature):
        assert aliases == {
            "red": "Oa08_radianceData",
            "green": "Oa06_radianceData",
            "nir": "Oa17_radianceData",
        }

    http = httpx.AsyncClient(transport=_transport(feature))
    async with CdseSentinel3OlciClient(client=http) as client:
        scene = (await client.search(_request())).scenes[0]
    await http.aclose()
    picked = pick_asset(scene)
    assert picked is not None
    assert picked[0] == "Oa08_radianceData"
    named = pick_asset(scene, "nir")
    assert named is not None
    assert named[0] == "Oa17_radianceData"


async def test_olci_alias_is_skipped_when_the_item_lacks_that_band() -> None:
    assets = _olci_assets()
    del assets["Oa08_radianceData"]
    aliases, _ = await _search_and_resolve(CdseSentinel3OlciClient, _feature(assets))
    assert "red" not in aliases
    assert aliases["nir"] == "Oa17_radianceData"


async def test_goes_cmi_default_pick_is_the_2km_web_mercator_red_band() -> None:
    feature = _feature(_goes_assets(web_mercator=True))
    for aliases in await _search_and_resolve(GoesCmiClient, feature):
        assert aliases["red"] == "C02_2km_wm"
        assert aliases["nir"] == "C03_2km_wm"
        assert aliases["blue"] == "C01_2km_wm"

    http = httpx.AsyncClient(transport=_transport(feature))
    async with GoesCmiClient(client=http) as client:
        scene = (await client.search(_request())).scenes[0]
    await http.aclose()
    picked = pick_asset(scene)
    assert picked is not None
    assert picked[0] == "C02_2km_wm"
    # The full-resolution band stays reachable by its literal key.
    literal = pick_asset(scene, "C02_0.5km")
    assert literal is not None
    assert literal[0] == "C02_0.5km"


async def test_goes_cmi_falls_back_to_the_2km_geos_band_without_web_mercator() -> None:
    feature = _feature(_goes_assets(web_mercator=False))
    aliases, _ = await _search_and_resolve(GoesCmiClient, feature)
    assert aliases["red"] == "C02_2km"
    assert aliases["nir"] == "C03_2km"
    assert aliases["blue"] == "C01_2km"


@pytest.mark.parametrize("cls", [GoesCmiClient, CdseSentinel3OlciClient])
async def test_items_without_the_pinned_bands_keep_their_aliases(
    cls: type[StacApiClient],
) -> None:
    feature = _feature({"other": {"type": _TIF, "eo:bands": _band("x", "red", 0.665)}})
    aliases, _ = await _search_and_resolve(cls, feature)
    assert aliases == {"red": "other"}
