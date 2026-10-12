"""Dateline AOIs for the clients that cannot send west > east in one request."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Any, cast
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from atlas.compile.aggregate import aggregate
from atlas.data.base import BBox, PullRequest
from atlas.data.firms import FirmsClient
from atlas.data.gibs import GibsModisTrueColorClient
from atlas.data.goes import GOES_EAST_BUCKET, GOES_WEST_BUCKET, GoesClient
from atlas.data.himawari import HIMAWARI_BUCKET, HimawariClient

# Fiji. west > east means the box crosses 180°.
DATELINE = BBox(west=178.0, south=-18.0, east=-178.0, north=-16.0)
# Same longitude cut, but north of every geostationary disk in this package.
OFF_DISK = BBox(west=170.0, south=70.0, east=-170.0, north=80.0)
DAY = date(2024, 7, 1)


def _request(bbox: BBox, **kwargs: Any) -> PullRequest:
    body: dict[str, Any] = {
        "start_date": DAY,
        "end_date": DAY,
        "bbox": bbox,
        "limit": 5,
    }
    body.update(kwargs)
    return PullRequest(**body)


class FakeResponse:
    def __init__(self, *, text: str = "", status_code: int = 200) -> None:
        self.text = text
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("GET", "https://example.test/dateline")
            response = httpx.Response(self.status_code, request=request)
            response.raise_for_status()


class FakeAsyncClient:
    def __init__(self, *, text: str = "", status_code: int = 200) -> None:
        self.text = text
        self.status_code = status_code
        self.calls: list[str] = []

    async def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append(url)
        return FakeResponse(text=self.text, status_code=self.status_code)

    async def aclose(self) -> None:
        return None


def _s3_listing(
    *, keys: Sequence[str] = (), prefixes: Sequence[str] = (), next_token: str | None = None
) -> str:
    body = "".join(f"<Contents><Key>{key}</Key></Contents>" for key in keys)
    body += "".join(
        f"<CommonPrefixes><Prefix>{prefix}</Prefix></CommonPrefixes>" for prefix in prefixes
    )
    if next_token:
        body += f"<IsTruncated>true</IsTruncated><NextContinuationToken>{next_token}</NextContinuationToken>"
    return (
        '<?xml version="1.0"?>'
        f'<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">{body}</ListBucketResult>'
    )


class S3Fake(FakeAsyncClient):
    def __init__(self, listings: dict[str, str]) -> None:
        super().__init__()
        self.listings = listings

    async def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append(url)
        query = parse_qs(urlparse(url).query)
        key = query["prefix"][0]
        if "continuation-token" in query:
            key += "@" + query["continuation-token"][0]
        return FakeResponse(text=self.listings.get(key, _s3_listing()))


def _wms_box(href: str) -> tuple[float, float, float, float]:
    raw = parse_qs(urlparse(href).query)["BBOX"][0]
    south, west, north, east = (float(item) for item in raw.split(","))
    return south, west, north, east


def _firms_box(url: str) -> tuple[float, float, float, float]:
    segment = urlparse(url).path.split("/")[6]
    west, south, east, north = (float(item) for item in segment.split(","))
    return west, south, east, north


def _hotspot(lon: float) -> str:
    return (
        "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,"
        "satellite,instrument,confidence,version,bright_ti5,frp,daynight\n"
        f"-17.0,{lon},330.1,0.4,0.4,2024-07-01,1342,N,VIIRS,n,2.0NRT,290.1,4.2,D\n"
    )


class FirmsFake(FakeAsyncClient):
    def __init__(self, bodies: dict[str, tuple[int, str]]) -> None:
        super().__init__()
        self.bodies = bodies

    async def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append(url)
        for bbox, (status, text) in self.bodies.items():
            if f"/{bbox}/" in url:
                return FakeResponse(text=text, status_code=status)
        raise AssertionError(f"unexpected FIRMS url {url}")


def test_antimeridian_parts_cuts_a_dateline_box_at_180() -> None:
    parts = DATELINE.antimeridian_parts()
    assert [(part.west, part.south, part.east, part.north) for part in parts] == [
        (178.0, -18.0, 180.0, -16.0),
        (-180.0, -18.0, -178.0, -16.0),
    ]
    plain = BBox(west=-71.12, south=42.32, east=-71.02, north=42.40)
    assert plain.antimeridian_parts() == [plain]


def test_antimeridian_parts_rejects_a_cut_with_no_width() -> None:
    with pytest.raises(ValueError, match="no width"):
        BBox(west=180.0, south=0.0, east=-180.0, north=1.0).antimeridian_parts()


@pytest.mark.asyncio
async def test_gibs_dateline_aoi_is_two_wms_requests() -> None:
    fake = FakeAsyncClient()
    async with GibsModisTrueColorClient(client=cast(httpx.AsyncClient, fake)) as client:
        result = await client.search(_request(DATELINE))
    assert fake.calls == []
    assert [scene.id.rsplit(":", 1)[-1] for scene in result.scenes] == ["east", "west"]
    boxes = [_wms_box(scene.assets["rendered_preview"].href) for scene in result.scenes]
    assert boxes == [(-18.0, 178.0, -16.0, 180.0), (-18.0, -180.0, -16.0, -178.0)]
    assert all(scene.bbox.west <= scene.bbox.east for scene in result.scenes)
    assert [scene.properties["antimeridian_part"] for scene in result.scenes] == ["east", "west"]
    assert all(scene.footprint_is_request for scene in result.scenes)


@pytest.mark.asyncio
async def test_gibs_dateline_without_width_is_a_source_error() -> None:
    empty = BBox(west=180.0, south=-1.0, east=-180.0, north=1.0)
    async with GibsModisTrueColorClient(
        client=cast(httpx.AsyncClient, FakeAsyncClient())
    ) as client:
        with pytest.raises(ValueError, match="no width"):
            await client.search(_request(empty))


@pytest.mark.asyncio
async def test_firms_dateline_aoi_merges_both_sides() -> None:
    fake = FirmsFake(
        {
            "178.0,-18.0,180.0,-16.0": (200, _hotspot(179.0)),
            "-180.0,-18.0,-178.0,-16.0": (200, _hotspot(-179.0)),
        }
    )
    async with FirmsClient(map_key="abc", client=cast(httpx.AsyncClient, fake)) as client:
        result = await client.search(_request(DATELINE))
    assert [_firms_box(url) for url in fake.calls] == [
        (178.0, -18.0, 180.0, -16.0),
        (-180.0, -18.0, -178.0, -16.0),
    ]
    assert sorted(scene.lon for scene in result.scenes if scene.lon is not None) == [-179.0, 179.0]


@pytest.mark.asyncio
async def test_firms_error_document_is_a_source_error() -> None:
    fake = FirmsFake(
        {
            "178.0,-18.0,180.0,-16.0": (200, _hotspot(179.0)),
            "-180.0,-18.0,-178.0,-16.0": (200, "Invalid area coordinates"),
        }
    )
    async with FirmsClient(map_key="abc", client=cast(httpx.AsyncClient, fake)) as client:
        with pytest.raises(ValueError, match="error document"):
            await client.search(_request(DATELINE))
    catalog = await aggregate(
        _request(DATELINE),
        {"firms_viirs": FirmsClient(map_key="abc", client=cast(httpx.AsyncClient, fake))},
    )
    assert catalog.sources[0].scenes == []
    assert catalog.sources[0].error == (
        "ValueError: FIRMS returned an error document instead of hotspot CSV"
    )


@pytest.mark.asyncio
async def test_firms_unpublished_day_is_still_skipped() -> None:
    fake = FirmsFake(
        {
            "178.0,-18.0,180.0,-16.0": (400, "Invalid date"),
            "-180.0,-18.0,-178.0,-16.0": (400, "Invalid date"),
        }
    )
    async with FirmsClient(map_key="abc", client=cast(httpx.AsyncClient, fake)) as client:
        result = await client.search(_request(DATELINE))
    assert result.scenes == []
    assert len(fake.calls) == 2


def _goes_listing() -> dict[str, str]:
    prefix = "ABI-L2-MCMIPF/2024/183/"
    name = "OR_ABI-L2-MCMIPF-M6_G18_s20241831501182_e20241831503556_c20241831504063.nc"
    return {prefix: _s3_listing(keys=[prefix + "15/" + name])}


@pytest.mark.asyncio
async def test_goes_dateline_aoi_lists_the_west_full_disk() -> None:
    fake = S3Fake(_goes_listing())
    async with GoesClient(client=cast(httpx.AsyncClient, fake)) as client:
        result = await client.search(_request(DATELINE))
    assert result.scenes
    assert {scene.platform for scene in result.scenes} == {"GOES-West"}
    assert all(GOES_WEST_BUCKET in scene.assets["data"].href for scene in result.scenes)
    assert all("ABI-L2-MCMIPF" in url for url in fake.calls)
    assert all(GOES_EAST_BUCKET not in url for url in fake.calls)


@pytest.mark.asyncio
async def test_goes_dateline_outside_the_disk_stays_empty() -> None:
    fake = S3Fake({})
    async with GoesClient(client=cast(httpx.AsyncClient, fake)) as client:
        result = await client.search(_request(OFF_DISK))
    assert result.scenes == []
    assert fake.calls == []


@pytest.mark.asyncio
async def test_goes_dateline_listing_failure_is_a_source_error() -> None:
    fake = FakeAsyncClient(status_code=500)
    async with GoesClient(client=cast(httpx.AsyncClient, fake)) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await client.search(_request(DATELINE))
    assert fake.calls
    catalog = await aggregate(
        _request(DATELINE),
        {"goes": GoesClient(client=cast(httpx.AsyncClient, fake))},
    )
    assert catalog.sources[0].scenes == []
    assert catalog.sources[0].error == "500 Internal Server Error"


def _himawari_listing() -> dict[str, str]:
    day = "AHI-L1b-FLDK/2024/07/01/"
    slot = f"{day}0010/"
    name = "HS_H09_20240701_0010_B01_FLDK_R10_S0110.DAT.bz2"
    return {
        day: _s3_listing(prefixes=[slot]),
        slot: _s3_listing(keys=[slot + name]),
    }


@pytest.mark.asyncio
async def test_himawari_dateline_aoi_lists_the_slot_once() -> None:
    fake = S3Fake(_himawari_listing())
    async with HimawariClient(client=cast(httpx.AsyncClient, fake)) as client:
        result = await client.search(_request(DATELINE, limit=1))
    assert len(result.scenes) == 1
    assert result.scenes[0].bbox.west > result.scenes[0].bbox.east
    day_lists = [url for url in fake.calls if "2024/07/01/" in url and "delimiter=" in url]
    assert len(day_lists) == 1
    assert all(HIMAWARI_BUCKET in url for url in fake.calls)


@pytest.mark.asyncio
async def test_himawari_dateline_outside_the_disk_stays_empty() -> None:
    fake = S3Fake({})
    async with HimawariClient(client=cast(httpx.AsyncClient, fake)) as client:
        result = await client.search(_request(OFF_DISK))
    assert result.scenes == []
    assert fake.calls == []


@pytest.mark.asyncio
async def test_himawari_dateline_listing_failure_is_a_source_error() -> None:
    fake = FakeAsyncClient(status_code=500)
    async with HimawariClient(client=cast(httpx.AsyncClient, fake)) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await client.search(_request(DATELINE))
    catalog = await aggregate(
        _request(DATELINE),
        {"himawari": HimawariClient(client=cast(httpx.AsyncClient, fake))},
    )
    assert catalog.sources[0].scenes == []
    assert catalog.sources[0].error == "500 Internal Server Error"
