"""NASA GIBS WMS snapshots. One scene per calendar day in the request window."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import ClassVar, Self
from urllib.parse import urlencode

import httpx

from atlas.data.base import (
    Asset,
    DataPullClient,
    PullRequest,
    PullResult,
    Scene,
    SceneKind,
)

GIBS_WMS = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"


class GibsClient(DataPullClient):
    layer: ClassVar[str] = ""
    platform_name: ClassVar[str] = "GIBS"
    instrument_name: ClassVar[str | None] = None
    image_size: ClassVar[int] = 512
    scene_kind: ClassVar[SceneKind] = SceneKind.browse

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        timeout: float = 30.0,
    ) -> None:
        if not self.layer:
            raise ValueError(f"{type(self).__name__} has no layer; use a concrete subclass.")
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = client is None

    @property
    def collection(self) -> str:
        return self.layer

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.aclose()

    async def search(self, request: PullRequest) -> PullResult:
        days = _dates_inclusive(request.start_date, request.end_date)[: request.limit]
        scenes: list[Scene] = []
        for day in days:
            start = datetime(day.year, day.month, day.day, tzinfo=UTC)
            end = datetime(day.year, day.month, day.day, 23, 59, 59, tzinfo=UTC)
            scene = Scene.try_new(
                id=f"{self.layer}:{day.isoformat()}",
                collection=self.layer,
                kind=self.scene_kind,
                datetime=start,
                start_datetime=start,
                end_datetime=end,
                bbox=request.bbox,
                footprint_is_request=True,
                platform=self.platform_name,
                instrument=self.instrument_name,
                assets={
                    "rendered_preview": Asset(
                        href=_wms_url(self.layer, day, request, self.image_size),
                        media_type="image/png",
                        title=self.layer,
                        roles=["overview", "visual"],
                    )
                },
                properties={"layer": self.layer, "time": day.isoformat()},
            )
            if scene is not None:
                scenes.append(scene)
        return PullResult(request=request, scenes=scenes)


class GibsModisTrueColorClient(GibsClient):
    layer: ClassVar[str] = "MODIS_Terra_CorrectedReflectance_TrueColor"
    platform_name: ClassVar[str] = "Terra"
    instrument_name: ClassVar[str] = "MODIS"


class GibsViirsTrueColorClient(GibsClient):
    layer: ClassVar[str] = "VIIRS_SNPP_CorrectedReflectance_TrueColor"
    platform_name: ClassVar[str] = "Suomi-NPP"
    instrument_name: ClassVar[str] = "VIIRS"


class GibsNightLightsClient(GibsClient):
    layer: ClassVar[str] = "VIIRS_SNPP_DayNightBand_At_Sensor_Radiance"
    platform_name: ClassVar[str] = "Suomi-NPP"
    instrument_name: ClassVar[str] = "VIIRS"


class GibsBlackMarbleClient(GibsClient):
    layer: ClassVar[str] = "VIIRS_Black_Marble"
    platform_name: ClassVar[str] = "Suomi-NPP"
    instrument_name: ClassVar[str] = "VIIRS"


class GibsFlood3DayClient(GibsClient):
    layer: ClassVar[str] = "MODIS_Combined_Flood_3-Day"
    platform_name: ClassVar[str] = "Terra/Aqua"
    instrument_name: ClassVar[str] = "MODIS"


class GibsSnowCoverClient(GibsClient):
    layer: ClassVar[str] = "MODIS_Terra_NDSI_Snow_Cover"
    platform_name: ClassVar[str] = "Terra"
    instrument_name: ClassVar[str] = "MODIS"


class GibsThermalAnomaliesClient(GibsClient):
    layer: ClassVar[str] = "VIIRS_NOAA20_Thermal_Anomalies_375m_All"
    platform_name: ClassVar[str] = "NOAA-20"
    instrument_name: ClassVar[str] = "VIIRS"
    scene_kind: ClassVar[SceneKind] = SceneKind.thermal


class GibsAquaTrueColorClient(GibsClient):
    layer: ClassVar[str] = "MODIS_Aqua_CorrectedReflectance_TrueColor"
    platform_name: ClassVar[str] = "Aqua"
    instrument_name: ClassVar[str] = "MODIS"


def _dates_inclusive(start: date, end: date) -> list[date]:
    days: list[date] = []
    cursor = start
    while cursor <= end:
        days.append(cursor)
        cursor += timedelta(days=1)
    return days


def _wms_url(layer: str, day: date, request: PullRequest, size: int) -> str:
    bbox = request.bbox
    # WMS 1.3.0 + EPSG:4326 uses lat,lon axis order.
    params = {
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetMap",
        "LAYERS": layer,
        "TIME": day.isoformat(),
        "CRS": "EPSG:4326",
        "BBOX": f"{bbox.south},{bbox.west},{bbox.north},{bbox.east}",
        "WIDTH": str(size),
        "HEIGHT": str(size),
        "FORMAT": "image/png",
        "TRANSPARENT": "TRUE",
    }
    return f"{GIBS_WMS}?{urlencode(params)}"
