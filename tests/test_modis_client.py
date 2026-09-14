from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any, cast

import httpx
import pytest

from atlas.data import MODISVegetationClient
from atlas.data.base import BBox, PullRequest
from atlas.data.modis_base import MODISClient
from atlas.data.registry import SOURCES


def _gpolygon_item() -> dict[str, Any]:
    """LPCLOUD-shaped UMM-G: GPolygons only, HTTPS GET DATA then s3 duplicate."""
    return {
        "meta": {"concept-id": "G-VEG-1"},
        "umm": {
            "GranuleUR": "MOD13A2.A2024177.h12v04.061",
            "TemporalExtent": {
                "RangeDateTime": {
                    "BeginningDateTime": "2024-06-25T00:00:00.000Z",
                    "EndingDateTime": "2024-07-10T23:59:59.000Z",
                }
            },
            "SpatialExtent": {
                "HorizontalSpatialDomain": {
                    "Geometry": {
                        "GPolygons": [
                            {
                                "Boundary": {
                                    "Points": [
                                        {"Longitude": -78.2083, "Latitude": 39.7858},
                                        {"Longitude": -65.0781, "Latitude": 39.8411},
                                        {"Longitude": -77.7506, "Latitude": 50.0754},
                                        {"Longitude": -93.3822, "Latitude": 49.9972},
                                        {"Longitude": -78.2083, "Latitude": 39.7858},
                                    ]
                                }
                            }
                        ]
                    }
                }
            },
            "RelatedUrls": [
                {
                    "Type": "GET DATA",
                    "URL": (
                        "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13A2.hdf"
                    ),
                    "Description": "HTTPS HDF",
                },
                {
                    "Type": "GET DATA VIA DIRECT ACCESS",
                    "URL": "s3://lp-prod-protected/MOD13A2.hdf",
                },
                {
                    "Type": "GET RELATED VISUALIZATION",
                    "URL": (
                        "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-public/BROWSE.jpg"
                    ),
                },
            ],
            "AdditionalAttributes": [
                {"Name": "HORIZONTALTILENUMBER", "Values": ["12"]},
                {"Name": "VERTICALTILENUMBER", "Values": ["04"]},
            ],
        },
    }


class FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self.payload


class FakeAsyncClient:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls: list[dict[str, Any]] = []

    async def get(self, _url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append(kwargs)
        return FakeResponse(self.payload)

    async def aclose(self) -> None:
        return None


def test_modis_client_requires_subclass() -> None:
    with pytest.raises(ValueError, match="short_name"):
        MODISClient()


def _client(payload: dict[str, Any] | None = None) -> MODISVegetationClient:
    fake = FakeAsyncClient(payload or {"items": []})
    return MODISVegetationClient(client=cast(httpx.AsyncClient, fake))


def test_gpolygons_bbox_and_https_data_not_s3() -> None:
    client = _client()
    scene = client._granule_to_scene(_gpolygon_item())
    assert scene is not None
    assert scene.id == "G-VEG-1"
    assert scene.datetime == datetime(2024, 6, 25, tzinfo=UTC)
    assert scene.bbox.west == pytest.approx(-93.3822)
    assert scene.bbox.east == pytest.approx(-65.0781)
    assert scene.bbox.south == pytest.approx(39.7858)
    assert scene.bbox.north == pytest.approx(50.0754)
    assert scene.assets["data"].href.startswith("https://")
    assert scene.assets["data"].href.endswith(".hdf")
    assert "s3://" not in scene.assets["data"].href
    assert scene.assets["browse"].href.endswith(".jpg")
    assert scene.properties["HORIZONTALTILENUMBER"] == "12"
    assert scene.cloud_cover is None
    assert scene.platform == "MODIS"
    assert scene.geometry_kind.value == "polygon"
    assert scene.kind.value == "optical"
    assert scene.gsd_m == 1000.0
    assert scene.collection == "MOD13A2"


def test_bounding_rectangles_preferred_when_present() -> None:
    item = {
        "meta": {"concept-id": "G-RECT"},
        "umm": {
            "TemporalExtent": {"SingleDateTime": "2020-01-01T00:00:00Z"},
            "SpatialExtent": {
                "HorizontalSpatialDomain": {
                    "Geometry": {
                        "BoundingRectangles": [
                            {
                                "WestBoundingCoordinate": -71.2,
                                "SouthBoundingCoordinate": 42.2,
                                "EastBoundingCoordinate": -70.9,
                                "NorthBoundingCoordinate": 42.5,
                            }
                        ]
                    }
                }
            },
            "Platforms": [{"ShortName": "Terra", "Instruments": [{"ShortName": "MODIS"}]}],
            "CloudCover": 0.0,
        },
    }
    scene = _client()._granule_to_scene(item)
    assert scene is not None
    assert scene.bbox.as_list() == [-71.2, 42.2, -70.9, 42.5]
    assert scene.platform == "Terra"
    assert scene.instrument == "MODIS"
    assert scene.cloud_cover == 0.0
    assert scene.datetime == datetime(2020, 1, 1, tzinfo=UTC)
    assert scene.geometry_kind.value == "bbox"


def test_empty_bounding_rectangles_and_missing_polygons_are_skipped() -> None:
    item = {
        "meta": {"concept-id": "G-EMPTY"},
        "umm": {
            "TemporalExtent": {"RangeDateTime": {"BeginningDateTime": "2020-01-01T00:00:00Z"}},
            "SpatialExtent": {"HorizontalSpatialDomain": {"Geometry": {"BoundingRectangles": []}}},
        },
    }
    scene = _client()._granule_to_scene(item)
    assert scene is None


def test_unparseable_datetime_is_skipped() -> None:
    item = {
        "meta": {"concept-id": "G-BAD-TIME"},
        "umm": {
            "TemporalExtent": {"RangeDateTime": {"BeginningDateTime": "not-a-date"}},
            "SpatialExtent": {
                "HorizontalSpatialDomain": {
                    "Geometry": {
                        "BoundingRectangles": [
                            {
                                "WestBoundingCoordinate": -1,
                                "SouthBoundingCoordinate": -1,
                                "EastBoundingCoordinate": 1,
                                "NorthBoundingCoordinate": 1,
                            }
                        ]
                    }
                }
            },
        },
    }
    scene = _client()._granule_to_scene(item)
    assert scene is None


@pytest.mark.asyncio
async def test_search_skips_unusable_granules_and_forwards_cloud_filter() -> None:
    payload = {"items": [_gpolygon_item(), {"umm": {}}, "ignore"]}
    fake = FakeAsyncClient(payload)
    request = PullRequest(
        start_date=date(2024, 7, 1),
        end_date=date(2024, 7, 31),
        bbox=BBox(west=-71.2, south=42.2, east=-70.9, north=42.5),
        max_cloud_cover=20,
        limit=5,
    )
    async with MODISVegetationClient(client=cast(httpx.AsyncClient, fake)) as client:
        result = await client.search(request)

    assert [scene.id for scene in result.scenes] == ["G-VEG-1"]
    params = fake.calls[0]["params"]
    assert params["short_name"] == "MOD13A2"
    assert params["cloud_cover"] == "0,20"
    assert params["page_size"] == 5
    assert client.collection == "MOD13A2"


def test_modis_sources_are_registered() -> None:
    assert set(SOURCES) >= {
        "modis_active_fire",
        "modis_land_cover",
        "modis_surface",
        "modis_vegetation",
    }
    assert SOURCES["modis_vegetation"] is MODISVegetationClient
