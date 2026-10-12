"""Every registered source must search, appear in the catalog, and be exported."""

from __future__ import annotations

from datetime import date
from typing import Any, cast

import httpx
import numpy as np
import pytest

import atlas.data as atlas_data
from atlas.cli.catalog import (
    SatelliteInfo,
    client_kind,
    client_satellite,
    format_satellite,
    list_satellites,
)
from atlas.data.base import BBox, DataPullClient, PullRequest, Scene
from atlas.data.earth_search import EarthSearchSentinel2L2aClient, l2a_reflectance_calibration
from atlas.data.indices import CALIBRATION_PRESETS
from atlas.data.registry import SOURCES, all_sources

EARTH_SEARCH_SEARCH = "https://earth-search.aws.element84.com/v1/search"
# Asset keys on an Earth Search v1 Sentinel-2 L2A item. `nir` is B08.
_BAND_ASSETS = ("blue", "green", "red", "nir", "nir08", "swir16", "swir22")


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeHttp:
    """Stand-in for ``httpx.AsyncClient``. Records the search call and returns one page."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    async def post(self, url: str, **kwargs: Any) -> _FakeResponse:
        self.calls.append(("POST", url, kwargs))
        return _FakeResponse(self.payload)

    async def get(self, url: str, **kwargs: Any) -> _FakeResponse:
        self.calls.append(("GET", url, kwargs))
        return _FakeResponse(self.payload)

    async def aclose(self) -> None:
        return None


def _request() -> PullRequest:
    return PullRequest(
        start_date=date(2024, 7, 1),
        end_date=date(2024, 7, 3),
        bbox=BBox(west=-71.12, south=42.32, east=-71.02, north=42.40),
        limit=5,
    )


def _reflectance(name: str, wavelength: float, common_name: str, offset: float) -> dict[str, Any]:
    return {
        "href": f"https://example.com/{name}.tif",
        "type": "image/tiff; application=geotiff; profile=cloud-optimized",
        "roles": ["data", "reflectance"],
        "eo:bands": [{"name": name, "common_name": common_name, "center_wavelength": wavelength}],
        "raster:bands": [{"nodata": 0, "data_type": "uint16", "scale": 0.0001, "offset": offset}],
    }


def _page(*, boa_offset_applied: bool, offset: float) -> dict[str, Any]:
    return {
        "features": [
            {
                "id": "earthsearch-s2-1",
                "bbox": [-71.2, 42.2, -70.9, 42.5],
                "properties": {
                    "datetime": "2024-07-01T15:00:00Z",
                    "eo:cloud_cover": 8,
                    "platform": "sentinel-2b",
                    "earthsearch:boa_offset_applied": boa_offset_applied,
                },
                "assets": {
                    "blue": _reflectance("blue", 0.49, "blue", offset),
                    "green": _reflectance("green", 0.56, "green", offset),
                    "red": _reflectance("red", 0.665, "red", offset),
                    "nir": _reflectance("nir", 0.842, "nir", offset),
                    "nir08": _reflectance("nir08", 0.865, "nir08", offset),
                    "swir16": _reflectance("swir16", 1.61, "swir16", offset),
                    "swir22": _reflectance("swir22", 2.19, "swir22", offset),
                    "visual": {
                        "href": "https://example.com/visual.tif",
                        "type": "image/tiff; application=geotiff; profile=cloud-optimized",
                        "roles": ["visual"],
                        "eo:bands": [
                            {"name": "red", "common_name": "red", "center_wavelength": 0.665},
                            {"name": "green", "common_name": "green", "center_wavelength": 0.56},
                            {"name": "blue", "common_name": "blue", "center_wavelength": 0.49},
                        ],
                    },
                },
            }
        ]
    }


def _assert_source_registered(
    name: str,
    cls: type[DataPullClient],
    opened: dict[str, DataPullClient],
    satellites: dict[str, SatelliteInfo],
    rendered: dict[str, str],
) -> None:
    """Search client, catalog row, and ``atlas.data`` export must agree."""
    assert isinstance(opened[name], cls)
    assert getattr(atlas_data, cls.__name__) is cls
    assert cls.__name__ in atlas_data.__all__
    satellite = client_satellite(cls)
    kind = client_kind(cls)
    assert name in satellites[satellite].sources_by_kind[kind]
    line = next(row for row in rendered[satellite].splitlines() if kind.value in row)
    assert name in line.replace(",", " ").split()


@pytest.mark.asyncio
async def test_every_registry_entry_passes_registration() -> None:
    """A source is registered only when search, a catalog row, and an export agree."""
    opened = all_sources()
    satellites = {info.name: info for info in list_satellites()}
    rendered = {name: format_satellite(name) for name in satellites}
    try:
        assert opened.keys() == SOURCES.keys()
        for name, cls in SOURCES.items():
            _assert_source_registered(name, cls, opened, satellites, rendered)
    finally:
        for client in opened.values():
            close = getattr(client, "aclose", None)
            if close is not None:
                await close()


@pytest.mark.asyncio
async def test_earthsearch_sentinel2_l2a_reads_band_assets() -> None:
    request = _request()
    fake = _FakeHttp(_page(boa_offset_applied=True, offset=-0.1))
    async with EarthSearchSentinel2L2aClient(client=cast(httpx.AsyncClient, fake)) as client:
        result = await client.search(request)

    assert result.request == request
    assert [scene.id for scene in result.scenes] == ["earthsearch-s2-1"]
    method, url, kwargs = fake.calls[0]
    assert method == "POST"
    assert url == EARTH_SEARCH_SEARCH
    assert kwargs["json"]["collections"] == ["sentinel-2-l2a"]
    assert kwargs["json"]["bbox"] == request.bbox.as_list()

    scene = result.scenes[0]
    for name in _BAND_ASSETS:
        assert name in scene.assets
    assert scene.band_aliases == {
        "red": "red",
        "green": "green",
        "nir": "nir08",
        "swir22": "swir22",
    }
    assert scene.assets["red"].scale == pytest.approx(0.0001)
    assert scene.assets["red"].offset == pytest.approx(-0.1)
    assert scene.assets["swir22"].href.endswith("/swir22.tif")
    # The item still publishes the s2-l2a offset. The flag says it was applied.
    harmonized = l2a_reflectance_calibration(scene, "red")
    assert harmonized.scale == pytest.approx(CALIBRATION_PRESETS["s2-l2a"].scale)
    assert harmonized.offset == pytest.approx(0.0)


def _scene(*, boa_offset_applied: bool, offset: float) -> Scene:
    scene = Scene.model_validate(
        {
            "id": "earthsearch-s2-1",
            "datetime": "2024-07-01T15:00:00Z",
            "bbox": {"west": -71.2, "south": 42.2, "east": -70.9, "north": 42.5},
            "platform": "sentinel-2b",
            "properties": {"earthsearch:boa_offset_applied": boa_offset_applied},
            "assets": {
                "red": {
                    "href": "https://example.com/red.tif",
                    "roles": ["data"],
                    "scale": 0.0001,
                    "offset": offset,
                }
            },
        }
    )
    return scene


def test_harmonized_earth_search_does_not_apply_the_s2_l2a_offset_twice() -> None:
    # v1 items keep raster:bands offset -0.1 after earthsearch:boa_offset_applied.
    # The s2-l2a preset is that same -0.1, so using it shifts reflectance again.
    preset = CALIBRATION_PRESETS["s2-l2a"]
    calibration = l2a_reflectance_calibration(
        _scene(boa_offset_applied=True, offset=-0.1),
        "red",
    )
    assert calibration.scale == pytest.approx(preset.scale)
    assert calibration.offset == pytest.approx(0.0)
    stored = np.array([2000.0, 6000.0])
    once = stored * calibration.scale + calibration.offset
    twice = stored * preset.scale + preset.offset
    assert once == pytest.approx([0.2, 0.6])
    assert twice == pytest.approx([0.1, 0.5])


def test_unharmonized_earth_search_matches_the_s2_l2a_preset() -> None:
    preset = CALIBRATION_PRESETS["s2-l2a"]
    calibration = l2a_reflectance_calibration(
        _scene(boa_offset_applied=False, offset=-0.1),
        "red",
    )
    assert calibration.scale == pytest.approx(preset.scale)
    assert calibration.offset == pytest.approx(preset.offset)


def test_published_zero_offset_is_left_alone() -> None:
    # Collection item assets and pre-baseline scenes publish offset 0.
    preset = CALIBRATION_PRESETS["s2-l2a"]
    calibration = l2a_reflectance_calibration(
        _scene(boa_offset_applied=False, offset=0.0),
        "red",
    )
    assert calibration.scale == pytest.approx(preset.scale)
    assert calibration.offset == pytest.approx(0.0)
