from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any, cast

import httpx
import pytest

from atlas.data.base import BBox, PullRequest
from atlas.data.cdse import CdseSentinel3OlciClient
from atlas.data.cop_dem import CopDemGlo30Client
from atlas.data.earth_search import EarthSearchSentinel1GrdClient
from atlas.data.firms import FirmsClient
from atlas.data.gibs import GibsModisTrueColorClient
from atlas.data.goes import GOES_EAST_BUCKET, GoesClient
from atlas.data.hls_sentinel import HLSSentinelClient
from atlas.data.naip import NaipClient
from atlas.data.registry import SOURCES
from atlas.data.sentinel2 import Sentinel2Client


class FakeResponse:
    def __init__(
        self,
        *,
        json_payload: Any = None,
        text: str = "",
        content: bytes = b"",
        status_code: int = 200,
    ) -> None:
        self._json = json_payload
        self.text = text
        self.content = content
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> Any:
        return self._json


class FakeAsyncClient:
    def __init__(
        self,
        *,
        json_payload: Any = None,
        text: str = "",
        content: bytes = b"",
    ) -> None:
        self.json_payload = json_payload
        self.text = text
        self.content = content
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    async def post(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append(("POST", url, kwargs))
        return FakeResponse(json_payload=self.json_payload)

    async def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append(("GET", url, kwargs))
        return FakeResponse(json_payload=self.json_payload, text=self.text, content=self.content)

    async def aclose(self) -> None:
        return None


BOSTON = BBox(west=-71.12, south=42.32, east=-71.02, north=42.40)


def _request(**kwargs: Any) -> PullRequest:
    body = {
        "start_date": date(2024, 7, 1),
        "end_date": date(2024, 7, 3),
        "bbox": BOSTON,
        "limit": 5,
    }
    body.update(kwargs)
    return PullRequest(**body)


def _stac_payload(*, platform: str | None = "sentinel-2b") -> dict[str, Any]:
    props: dict[str, Any] = {
        "datetime": "2024-07-01T15:00:00Z",
        "eo:cloud_cover": 12,
    }
    if platform is not None:
        props["platform"] = platform
    return {
        "features": [
            {
                "id": "item-1",
                "bbox": [-71.2, 42.2, -70.9, 42.5],
                "properties": props,
                "assets": {
                    "visual": {
                        "href": "https://example.com/scene.tif",
                        "type": "image/tiff",
                        "roles": ["data"],
                    }
                },
            }
        ]
    }


def _client(cls: type, fake: FakeAsyncClient) -> Any:
    return cls(client=cast(httpx.AsyncClient, fake))


@pytest.mark.asyncio
async def test_hls_and_naip_post_stac_and_parse_items() -> None:
    fake = FakeAsyncClient(json_payload=_stac_payload())
    async with _client(HLSSentinelClient, fake) as client:
        result = await client.search(_request(max_cloud_cover=20))
    assert [s.id for s in result.scenes] == ["item-1"]
    assert result.scenes[0].assets["visual"].href.startswith("https://")
    posted = fake.calls[0]
    assert posted[0] == "POST"
    assert "planetarycomputer" in posted[1]
    body = posted[2]["json"]
    assert body["collections"] == ["hls2-s30"]
    assert body["query"] == {"eo:cloud_cover": {"lte": 20}}

    naip_fake = FakeAsyncClient(json_payload=_stac_payload(platform=None))
    async with _client(NaipClient, naip_fake) as client:
        naip = await client.search(
            _request(start_date=date(2021, 1, 1), end_date=date(2023, 12, 31))
        )
    assert naip.scenes[0].platform == "naip"


@pytest.mark.asyncio
async def test_stac_rewrites_s3_asset_hrefs_to_https() -> None:
    payload = _stac_payload()
    payload["features"][0]["assets"]["visual"]["href"] = "s3://sentinel-s1-l1c/path/file.tiff"
    fake = FakeAsyncClient(json_payload=payload)
    async with _client(EarthSearchSentinel1GrdClient, fake) as client:
        result = await client.search(_request())
    assert result.scenes[0].assets["visual"].href == (
        "https://sentinel-s1-l1c.s3.amazonaws.com/path/file.tiff"
    )


@pytest.mark.asyncio
async def test_cop_dem_omits_datetime_filter() -> None:
    fake = FakeAsyncClient(json_payload=_stac_payload(platform="TanDEM-X"))
    async with _client(CopDemGlo30Client, fake) as client:
        result = await client.search(_request())
    assert result.scenes[0].id == "item-1"
    assert "datetime" not in fake.calls[0][2]["json"]


@pytest.mark.asyncio
async def test_earth_search_and_cdse_use_their_endpoints() -> None:
    fake = FakeAsyncClient(json_payload=_stac_payload(platform="sentinel-1a"))
    async with _client(EarthSearchSentinel1GrdClient, fake) as client:
        await client.search(_request())
    assert fake.calls[0][1] == "https://earth-search.aws.element84.com/v1/search"
    assert fake.calls[0][2]["json"]["collections"] == ["sentinel-1-grd"]

    cdse_fake = FakeAsyncClient(json_payload=_stac_payload(platform="sentinel-3a"))
    async with _client(CdseSentinel3OlciClient, cdse_fake) as client:
        await client.search(_request())
    assert "dataspace.copernicus.eu" in cdse_fake.calls[0][1]
    assert cdse_fake.calls[0][2]["json"]["collections"] == ["sentinel-3-olci-1-efr-ntc"]


@pytest.mark.asyncio
async def test_existing_sentinel2_client_still_targets_planetary_computer() -> None:
    fake = FakeAsyncClient(json_payload=_stac_payload())
    async with _client(Sentinel2Client, fake) as client:
        await client.search(_request())
    assert fake.calls[0][1].startswith("https://planetarycomputer.microsoft.com/")


@pytest.mark.asyncio
async def test_gibs_builds_wms_preview_per_day() -> None:
    fake = FakeAsyncClient()
    async with _client(GibsModisTrueColorClient, fake) as client:
        result = await client.search(_request())
    assert [s.id.split(":")[-1] for s in result.scenes] == [
        "2024-07-01",
        "2024-07-02",
        "2024-07-03",
    ]
    href = result.scenes[0].assets["rendered_preview"].href
    assert "MODIS_Terra_CorrectedReflectance_TrueColor" in href
    assert "TIME=2024-07-01" in href
    assert "42.32" in href
    assert fake.calls == []


@pytest.mark.asyncio
async def test_firms_parses_csv_and_requires_key() -> None:
    csv_body = (
        "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,"
        "satellite,instrument,confidence,version,bright_ti5,frp,daynight\n"
        "42.35,-71.08,330.1,0.4,0.4,2024-07-01,1342,N,VIIRS,n,2.0NRT,290.1,4.2,D\n"
    )
    fake = FakeAsyncClient(text=csv_body)
    with pytest.raises(ValueError, match="FIRMS_MAP_KEY"):
        await FirmsClient(map_key="", client=cast(httpx.AsyncClient, fake)).search(_request())

    async with FirmsClient(map_key="abc", client=cast(httpx.AsyncClient, fake)) as client:
        result = await client.search(_request())
    assert len(result.scenes) == 1
    scene = result.scenes[0]
    assert scene.datetime == datetime(2024, 7, 1, 13, 42, tzinfo=UTC)
    assert scene.bbox.west < -71.08 < scene.bbox.east
    assert "area/csv/abc/VIIRS_SNPP_NRT" in fake.calls[0][1]


@pytest.mark.asyncio
async def test_goes_lists_netcdf_from_s3() -> None:
    xml = """<?xml version="1.0"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <Contents>
    <Key>ABI-L2-MCMIPC/2024/183/15/OR_ABI-L2-MCMIPC-M6_G19_s20241831501182_e20241831503556_c20241831504063.nc</Key>
  </Contents>
</ListBucketResult>
"""
    fake = FakeAsyncClient(text=xml)
    async with _client(GoesClient, fake) as client:
        result = await client.search(
            _request(limit=1, start_date=date(2024, 7, 1), end_date=date(2024, 7, 1))
        )
    assert len(result.scenes) == 1
    assert (
        result.scenes[0]
        .assets["data"]
        .href.startswith(f"https://{GOES_EAST_BUCKET}.s3.amazonaws.com/")
    )
    assert result.scenes[0].assets["data"].href.endswith(".nc")
    assert "ABI-L2-MCMIPC/2024/183/" in fake.calls[0][1]


def test_new_sources_are_registered() -> None:
    for name in (
        "hls_landsat",
        "hls_sentinel",
        "naip",
        "cop_dem",
        "earthsearch_sentinel1_grd",
        "cdse_sentinel3_olci",
        "cdse_sentinel5p_no2",
        "firms_viirs",
        "gibs_modis_truecolor",
        "gibs_viirs_truecolor",
        "goes",
    ):
        assert name in SOURCES
