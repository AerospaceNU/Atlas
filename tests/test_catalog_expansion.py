from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, date, datetime
from typing import Any, cast
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

import atlas.data as atlas_data
from atlas.compile.select import coverage_fraction
from atlas.data.base import BBox, GeometryKind, PullRequest, SceneKind
from atlas.data.cdse import CdseSentinel3OlciClient
from atlas.data.cop_dem import CopDemGlo30Client
from atlas.data.earth_search import EarthSearchSentinel1GrdClient
from atlas.data.firms import FirmsClient
from atlas.data.gibs import GibsModisTrueColorClient
from atlas.data.goes import GOES_EAST_BUCKET, GoesClient
from atlas.data.himawari import HIMAWARI_BUCKET, HimawariClient
from atlas.data.hls_sentinel import HLSSentinelClient
from atlas.data.maxar_opendata import MaxarOpenDataClient
from atlas.data.naip import NaipClient
from atlas.data.registry import SOURCES
from atlas.data.sentinel2 import Sentinel2Client
from atlas.data.usgs import UsgsLandsatC2L1Client


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


def _s3_listing(
    *, keys: Sequence[str] = (), prefixes: Sequence[str] = (), next_token: str | None = None
) -> str:
    body = "".join(f"<Contents><Key>{k}</Key></Contents>" for k in keys)
    body += "".join(f"<CommonPrefixes><Prefix>{p}</Prefix></CommonPrefixes>" for p in prefixes)
    if next_token:
        body += f"<IsTruncated>true</IsTruncated><NextContinuationToken>{next_token}</NextContinuationToken>"
    return (
        '<?xml version="1.0"?>'
        f'<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">{body}</ListBucketResult>'
    )


class S3Fake(FakeAsyncClient):
    """Answers ListObjectsV2 by ``prefix`` (``prefix@token`` for later pages)."""

    def __init__(self, listings: dict[str, str]) -> None:
        super().__init__()
        self.listings = listings

    async def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append(("GET", url, kwargs))
        query = parse_qs(urlparse(url).query)
        key = query["prefix"][0]
        if "continuation-token" in query:
            key += "@" + query["continuation-token"][0]
        return FakeResponse(text=self.listings.get(key, _s3_listing()))


BOSTON = BBox(west=-71.12, south=42.32, east=-71.02, north=42.40)
TOKYO = BBox(west=139.6, south=35.6, east=139.9, north=35.8)


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
    assert result.scenes[0].kind is SceneKind.browse
    assert result.scenes[0].footprint_is_request is True
    assert result.scenes[0].properties["synthetic"] is True
    assert "MODIS_Terra_CorrectedReflectance_TrueColor" in href
    assert "TIME=2024-07-01" in href
    assert "42.32" in href
    assert fake.calls == []
    assert coverage_fraction(result.scenes, BOSTON) == 0.0


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
    assert scene.geometry_kind is GeometryKind.point
    assert scene.kind is SceneKind.detection
    assert scene.lon == pytest.approx(-71.08)
    assert "area/csv/abc/VIIRS_SNPP_NRT" in fake.calls[0][1]


@pytest.mark.asyncio
async def test_goes_lists_netcdf_from_s3() -> None:
    day = "ABI-L2-MCMIPC/2024/183/"
    name = "OR_ABI-L2-MCMIPC-M6_G19_s2024183{hhmm}182_e20241831503556_c20241831504063.nc"
    fake = S3Fake(
        {
            day: _s3_listing(
                keys=[
                    day + "15/" + name.format(hhmm="1501"),
                    day + "15/" + name.format(hhmm="1506"),
                ],
                next_token="page-2",
            ),
            f"{day}@page-2": _s3_listing(keys=[day + "16/" + name.format(hhmm="1601")]),
        }
    )
    async with _client(GoesClient, fake) as client:
        result = await client.search(
            _request(limit=2, start_date=date(2024, 7, 1), end_date=date(2024, 7, 1))
        )
    assert [s.datetime.strftime("%H%M") for s in result.scenes] == ["1506", "1601"]
    scene = result.scenes[0]
    assert scene.assets["data"].href.startswith(f"https://{GOES_EAST_BUCKET}.s3.amazonaws.com/")
    assert scene.assets["data"].href.endswith(".nc")
    assert scene.kind is SceneKind.optical
    assert scene.bbox.as_list() != BOSTON.as_list()
    assert "max-keys=1" not in fake.calls[0][1]
    assert "continuation-token=page-2" in fake.calls[1][1]


@pytest.mark.asyncio
async def test_goes_skips_aoi_off_the_disk() -> None:
    fake = S3Fake({})
    async with _client(GoesClient, fake) as client:
        result = await client.search(_request(bbox=TOKYO))
    assert result.scenes == []
    assert fake.calls == []


def test_new_sources_are_registered() -> None:
    for name in (
        "hls_landsat",
        "hls_sentinel",
        "naip",
        "cop_dem",
        "cop_dem_90",
        "nasadem",
        "aster",
        "alos_palsar",
        "landsat_c2_l1",
        "pc_modis_14a1",
        "goes_cmi",
        "esa_worldcover",
        "io_lulc_annual",
        "mrms_qpe_24h",
        "usgs_landsat_l1",
        "gedi",
        "icesat2_atl03",
        "smap_l3",
        "dea_s2_ard",
        "deafrica_s2",
        "cbers4_mux",
        "amazonia1_wfi",
        "maxar_opendata",
        "firms_viirs_noaa21",
        "firms_landsat",
        "gibs_night_lights",
        "gibs_flood_3day",
        "gibs_snow",
        "himawari",
        "cdse_sentinel3_slstr_lst",
        "cdse_sentinel5p_ch4",
        "earthsearch_sentinel1_grd",
        "goes",
    ):
        assert name in SOURCES


@pytest.mark.asyncio
async def test_usgs_and_worldcover_stac_targets() -> None:
    fake = FakeAsyncClient(json_payload=_stac_payload(platform="landsat-8"))
    async with _client(UsgsLandsatC2L1Client, fake) as client:
        await client.search(_request())
    assert fake.calls[0][1] == "https://landsatlook.usgs.gov/stac-server/search"
    assert fake.calls[0][2]["json"]["collections"] == ["landsat-c2l1"]


@pytest.mark.asyncio
async def test_himawari_lists_ahi_slots() -> None:
    day = "AHI-L1b-FLDK/2024/07/01/"
    slot = f"{day}0010/"
    name = "HS_H09_20240701_0010_{band}_FLDK_R10_{seg}.DAT.bz2"
    fake = S3Fake(
        {
            day: _s3_listing(prefixes=[f"{day}0000/", slot]),
            slot: _s3_listing(
                keys=[slot + name.format(band="B01", seg="S0110")],
                next_token="page-2",
            ),
            f"{slot}@page-2": _s3_listing(keys=[slot + name.format(band="B03", seg="S0510")]),
        }
    )
    async with _client(HimawariClient, fake) as client:
        result = await client.search(
            _request(bbox=TOKYO, limit=1, start_date=date(2024, 7, 1), end_date=date(2024, 7, 1))
        )
    assert len(result.scenes) == 1
    scene = result.scenes[0]
    assert scene.datetime == datetime(2024, 7, 1, 0, 10, tzinfo=UTC)
    assert scene.kind is SceneKind.optical
    assert set(scene.assets) == {"B01_S0110", "B03_S0510"}
    assert all(
        a.href.startswith(f"https://{HIMAWARI_BUCKET}.s3.amazonaws.com/")
        for a in scene.assets.values()
    )
    assert "delimiter=/" in fake.calls[0][1]
    assert "max-keys=1" not in " ".join(url for _, url, _ in fake.calls)
    assert "continuation-token=page-2" in fake.calls[-1][1]


@pytest.mark.asyncio
async def test_himawari_skips_aoi_off_the_disk() -> None:
    fake = S3Fake({})
    async with _client(HimawariClient, fake) as client:
        result = await client.search(_request())
    assert result.scenes == []
    assert fake.calls == []


@pytest.mark.asyncio
async def test_himawari_disk_reaches_across_the_antimeridian() -> None:
    fake = S3Fake({})
    samoa = BBox(west=-172.8, south=-14.1, east=-171.4, north=-13.4)
    async with _client(HimawariClient, fake) as client:
        await client.search(_request(bbox=samoa))
    assert fake.calls


@pytest.mark.asyncio
async def test_maxar_walks_event_catalog() -> None:
    root = {
        "links": [{"rel": "child", "href": "./Event/collection.json"}],
    }
    event = {
        "extent": {
            "spatial": {"bbox": [[-71.2, 42.2, -70.9, 42.5]]},
            "temporal": {"interval": [["2024-07-01T00:00:00Z", "2024-07-03T00:00:00Z"]]},
        },
        "links": [{"rel": "child", "href": "./acq/collection.json"}],
    }
    acq = {
        "extent": {
            "spatial": {"bbox": [[-71.2, 42.2, -70.9, 42.5]]},
            "temporal": {"interval": [["2024-07-01T00:00:00Z", "2024-07-03T00:00:00Z"]]},
        },
        "links": [{"rel": "item", "href": "./item.json"}],
    }
    item = _stac_payload(platform="worldview-3")["features"][0]

    class CatalogFake(FakeAsyncClient):
        async def get(self, url: str, **kwargs: Any) -> FakeResponse:
            self.calls.append(("GET", url, kwargs))
            if url.endswith("catalog.json"):
                return FakeResponse(json_payload=root)
            if url.endswith("Event/collection.json"):
                return FakeResponse(json_payload=event)
            if url.endswith("acq/collection.json"):
                return FakeResponse(json_payload=acq)
            return FakeResponse(json_payload=item)

    fake = CatalogFake()
    async with MaxarOpenDataClient(client=cast(httpx.AsyncClient, fake), max_events=5) as client:
        result = await client.search(_request())
    assert result.scenes[0].id == "item-1"
    assert result.scenes[0].platform == "worldview-3"


def _maxar_collection(bbox: list[float], links: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "extent": {
            "spatial": {"bbox": [bbox]},
            "temporal": {"interval": [["2024-07-01T00:00:00Z", "2024-07-03T00:00:00Z"]]},
        },
        "links": links,
    }


@pytest.mark.asyncio
async def test_maxar_cap_counts_matching_events_only() -> None:
    far_away = [10.0, 10.0, 11.0, 11.0]
    events = [f"Far{i}" for i in range(3)] + ["Boston"]
    root = {"links": [{"rel": "child", "href": f"./{e}/collection.json"} for e in events]}
    payloads: dict[str, Any] = {"catalog.json": root}
    for event in events:
        box = [-71.2, 42.2, -70.9, 42.5] if event == "Boston" else far_away
        payloads[f"{event}/collection.json"] = _maxar_collection(
            box, [{"rel": "child", "href": "./acq/collection.json"}]
        )
        payloads[f"{event}/acq/collection.json"] = _maxar_collection(
            box, [{"rel": "item", "href": "./item.json"}]
        )
    payloads["Boston/acq/item.json"] = _stac_payload(platform="worldview-3")["features"][0]

    class CatalogFake(FakeAsyncClient):
        async def get(self, url: str, **kwargs: Any) -> FakeResponse:
            self.calls.append(("GET", url, kwargs))
            for suffix, payload in payloads.items():
                if url.endswith(f"/{suffix}"):
                    return FakeResponse(json_payload=payload)
            return FakeResponse(json_payload={}, status_code=404)

    fake = CatalogFake()
    async with MaxarOpenDataClient(client=cast(httpx.AsyncClient, fake), max_events=1) as client:
        result = await client.search(_request())
    assert [s.id for s in result.scenes] == ["item-1"]


@pytest.mark.asyncio
async def test_maxar_drops_items_outside_request_when_extent_is_missing() -> None:
    root = {"links": [{"rel": "child", "href": "./Event/collection.json"}]}
    event = {"links": [{"rel": "child", "href": "./acq/collection.json"}]}
    acq = {"links": [{"rel": "item", "href": "./item.json"}]}
    item = _stac_payload(platform="worldview-3")["features"][0]
    item["bbox"] = [10.0, 10.0, 11.0, 11.0]

    class CatalogFake(FakeAsyncClient):
        async def get(self, url: str, **kwargs: Any) -> FakeResponse:
            self.calls.append(("GET", url, kwargs))
            if url.endswith("catalog.json"):
                return FakeResponse(json_payload=root)
            if url.endswith("Event/collection.json"):
                return FakeResponse(json_payload=event)
            if url.endswith("acq/collection.json"):
                return FakeResponse(json_payload=acq)
            return FakeResponse(json_payload=item)

    async with MaxarOpenDataClient(client=cast(httpx.AsyncClient, CatalogFake())) as client:
        result = await client.search(_request())
    assert result.scenes == []


def test_bbox_intersects_handles_the_antimeridian() -> None:
    fiji = BBox(west=177.0, south=-19.0, east=-178.0, north=-16.0)
    assert fiji.intersects(BBox(west=179.0, south=-18.0, east=179.5, north=-17.0))
    assert fiji.intersects(BBox(west=-179.0, south=-18.0, east=-178.5, north=-17.0))
    assert not fiji.intersects(BBox(west=0.0, south=-18.0, east=1.0, north=-17.0))
    assert not fiji.intersects(BBox(west=178.0, south=0.0, east=179.0, north=1.0))


@pytest.mark.asyncio
async def test_cloud_minimum_and_maximum_reach_the_stac_query() -> None:
    fake = FakeAsyncClient(json_payload=_stac_payload())
    async with _client(Sentinel2Client, fake) as client:
        await client.search(_request(min_cloud_cover=80))
    assert fake.calls[0][2]["json"]["query"] == {"eo:cloud_cover": {"gte": 80}}

    both = FakeAsyncClient(json_payload=_stac_payload())
    async with _client(Sentinel2Client, both) as client:
        await client.search(_request(min_cloud_cover=60, max_cloud_cover=90))
    assert both.calls[0][2]["json"]["query"] == {"eo:cloud_cover": {"gte": 60, "lte": 90}}

    neither = FakeAsyncClient(json_payload=_stac_payload())
    async with _client(Sentinel2Client, neither) as client:
        await client.search(_request())
    assert "query" not in neither.calls[0][2]["json"]


def test_cloud_minimum_may_not_exceed_maximum() -> None:
    with pytest.raises(ValueError, match="min_cloud_cover"):
        _request(min_cloud_cover=80, max_cloud_cover=20)


def test_registry_classes_are_all_publicly_exported() -> None:
    exported = set(atlas_data.__all__)
    registered = {cls.__name__ for cls in SOURCES.values()}
    assert not registered - exported, "registered clients missing from __all__"
    assert all(hasattr(atlas_data, name) for name in exported)


def test_dea_covers_both_sentinel2_satellites() -> None:
    collections = {
        SOURCES[name].default_collection  # type: ignore[attr-defined]
        for name in ("dea_s2_ard", "dea_s2b_ard")
    }
    assert collections == {"ga_s2am_ard_3", "ga_s2bm_ard_3"}
