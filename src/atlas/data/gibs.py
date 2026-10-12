from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import ClassVar, Self
from urllib.parse import urlencode

import httpx

from atlas.data.base import (
    Asset,
    BBox,
    DataPullClient,
    PullRequest,
    PullResult,
    Scene,
    SceneKind,
)

GIBS_WMS = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"


class GibsClient(DataPullClient):
    """NASA GIBS WMS browse scenes, one GetMap per day.

    A box that crosses the antimeridian is two GetMap requests, one on each
    side of ±180°. WMS 1.3 with EPSG:4326 requires west <= east. One BBOX with
    west greater than east is rejected by GIBS or drawn the long way around
    the globe, which is an empty or wrapped image of the wrong hemisphere.
    ``limit`` caps days, and both sides of a selected day are kept together.
    """

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
        """Build one WMS preview per day and per side of the antimeridian.

        No HTTP request is made, so a day with no imagery still gets a scene (it
        renders blank). Scenes are marked ``synthetic`` and never count toward
        coverage. A dateline box that cannot be cut into a west-before-east
        BBOX raises, so the source fails instead of publishing a wrapped URL.
        """
        parts = request.bbox.antimeridian_parts()
        split = request.bbox.west > request.bbox.east
        days = _dates_inclusive(request.start_date, request.end_date)[: request.limit]
        scenes: list[Scene] = []
        for day in days:
            start = datetime(day.year, day.month, day.day, tzinfo=UTC)
            end = datetime(day.year, day.month, day.day, 23, 59, 59, tzinfo=UTC)
            for part in parts:
                side = _dateline_side(part) if split else None
                scene_id = f"{self.layer}:{day.isoformat()}"
                if side is not None:
                    scene_id = f"{scene_id}:{side}"
                properties: dict[str, str | bool] = {
                    "layer": self.layer,
                    "time": day.isoformat(),
                    "synthetic": True,
                }
                if side is not None:
                    properties["antimeridian_part"] = side
                scene = Scene.try_new(
                    id=scene_id,
                    collection=self.layer,
                    kind=self.scene_kind,
                    datetime=start,
                    start_datetime=start,
                    end_datetime=end,
                    bbox=part,
                    footprint_is_request=True,
                    platform=self.platform_name,
                    instrument=self.instrument_name,
                    assets={
                        "rendered_preview": Asset(
                            href=_wms_url(self.layer, day, part, self.image_size),
                            media_type="image/png",
                            title=self.layer,
                            roles=["overview", "visual"],
                        )
                    },
                    properties=properties,
                )
                if scene is None:
                    raise ValueError(
                        "GIBS could not build a dateline preview for "
                        f"{day.isoformat()} west={part.west} east={part.east}"
                    )
                scenes.append(scene)
        return PullResult(request=request, scenes=scenes)


class GibsModisTrueColorClient(GibsClient):
    satellite: ClassVar[str] = "modis"
    layer: ClassVar[str] = "MODIS_Terra_CorrectedReflectance_TrueColor"
    platform_name: ClassVar[str] = "Terra"
    instrument_name: ClassVar[str] = "MODIS"


class GibsViirsTrueColorClient(GibsClient):
    satellite: ClassVar[str] = "viirs"
    layer: ClassVar[str] = "VIIRS_SNPP_CorrectedReflectance_TrueColor"
    platform_name: ClassVar[str] = "Suomi-NPP"
    instrument_name: ClassVar[str] = "VIIRS"


class GibsNightLightsClient(GibsClient):
    satellite: ClassVar[str] = "viirs"
    layer: ClassVar[str] = "VIIRS_SNPP_DayNightBand_At_Sensor_Radiance"
    platform_name: ClassVar[str] = "Suomi-NPP"
    instrument_name: ClassVar[str] = "VIIRS"


class GibsBlackMarbleClient(GibsClient):
    satellite: ClassVar[str] = "viirs"
    layer: ClassVar[str] = "VIIRS_Black_Marble"
    platform_name: ClassVar[str] = "Suomi-NPP"
    instrument_name: ClassVar[str] = "VIIRS"


class GibsFlood3DayClient(GibsClient):
    satellite: ClassVar[str] = "modis"
    layer: ClassVar[str] = "MODIS_Combined_Flood_3-Day"
    platform_name: ClassVar[str] = "Terra/Aqua"
    instrument_name: ClassVar[str] = "MODIS"


class GibsSnowCoverClient(GibsClient):
    satellite: ClassVar[str] = "modis"
    layer: ClassVar[str] = "MODIS_Terra_NDSI_Snow_Cover"
    platform_name: ClassVar[str] = "Terra"
    instrument_name: ClassVar[str] = "MODIS"


class GibsThermalAnomaliesClient(GibsClient):
    layer: ClassVar[str] = "VIIRS_NOAA20_Thermal_Anomalies_375m_All"
    platform_name: ClassVar[str] = "NOAA-20"
    instrument_name: ClassVar[str] = "VIIRS"
    satellite: ClassVar[str] = "viirs"
    scene_kind: ClassVar[SceneKind] = SceneKind.thermal


class GibsAquaTrueColorClient(GibsClient):
    satellite: ClassVar[str] = "modis"
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


def _dateline_side(part: BBox) -> str:
    """``east`` for the side that ends at 180°, ``west`` for the side from -180°."""
    if part.east == 180.0:
        return "east"
    return "west"


def _wms_url(layer: str, day: date, bbox: BBox, size: int) -> str:
    if bbox.west > bbox.east:
        raise ValueError(
            f"GIBS WMS BBOX requires west <= east; got west={bbox.west} east={bbox.east}"
        )
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
