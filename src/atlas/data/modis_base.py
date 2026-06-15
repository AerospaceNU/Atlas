from __future__ import annotations

from datetime import datetime
from typing import Any, ClassVar, Self

import httpx

from atlas.data.base import (
    Asset,
    BBox,
    DataPullClient,
    PullRequest,
    PullResult,
    Scene,
)

CMR_SEARCH_URL = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"


class MODISClient(DataPullClient):
    short_name: ClassVar[str] = ""
    version: ClassVar[str] = "061"

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        timeout: float = 60.0,
    ) -> None:
        if not self.short_name:
            raise ValueError(f"{type(self).__name__} has no short_name; use a concrete subclass.")
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = client is None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.aclose()

    async def search(self, request: PullRequest) -> PullResult:
        params: dict[str, Any] = {
            "short_name": self.short_name,
            "version": self.version,
            "bounding_box": (
                f"{request.bbox.west},{request.bbox.south},{request.bbox.east},{request.bbox.north}"
            ),
            "temporal": (
                f"{request.start_date.isoformat()}T00:00:00Z,"
                f"{request.end_date.isoformat()}T23:59:59Z"
            ),
            "page_size": request.limit,
        }

        resp = await self._client.get(CMR_SEARCH_URL, params=params)
        resp.raise_for_status()
        payload = resp.json()

        scenes = [self._granule_to_scene(item) for item in payload.get("items", [])]
        return PullResult(request=request, scenes=scenes)

    def _granule_to_scene(self, item: dict[str, Any]) -> Scene:
        meta: dict[str, Any] = item.get("meta", {})
        umm: dict[str, Any] = item.get("umm", {})

        time_range = umm.get("TemporalExtent", {}).get("RangeDateTime", {})
        raw_dt = time_range.get("BeginningDateTime") or time_range.get("EndingDateTime", "")
        try:
            scene_dt = datetime.fromisoformat(raw_dt.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            scene_dt = datetime.utcnow()

        bb = (
            umm.get("SpatialExtent", {})
            .get("HorizontalSpatialDomain", {})
            .get("Geometry", {})
            .get("BoundingRectangles", [{}])[0]
        )
        bbox = BBox(
            west=float(bb.get("WestBoundingCoordinate", -180)),
            south=float(bb.get("SouthBoundingCoordinate", -90)),
            east=float(bb.get("EastBoundingCoordinate", 180)),
            north=float(bb.get("NorthBoundingCoordinate", 90)),
        )

        platforms = umm.get("Platforms", [{}])
        platform_name: str = platforms[0].get("ShortName", "MODIS") if platforms else "MODIS"
        instruments = platforms[0].get("Instruments", [{}]) if platforms else [{}]
        instrument_name: str | None = instruments[0].get("ShortName") if instruments else None

        assets: dict[str, Asset] = {}
        for related in umm.get("RelatedUrls", []):
            url: str = related.get("URL", "")
            url_type: str = related.get("Type", "")
            if "DATA" in url_type or "GET DATA" in url_type:
                key = url.rstrip("/").split("/")[-1].rsplit(".", 1)[0][:64] or "data"
                assets[key] = Asset(
                    href=url,
                    media_type=_guess_media_type(url),
                    title=related.get("Description") or key,
                    roles=["data"],
                )

        return Scene(
            id=meta.get("concept-id", "unknown"),
            datetime=scene_dt,
            bbox=bbox,
            platform=platform_name,
            instrument=instrument_name,
            cloud_cover=None,
            assets=assets,
            properties={
                "short_name": self.short_name,
                "version": self.version,
            },
        )


def _guess_media_type(url: str) -> str:
    lower = url.lower()
    if lower.endswith((".hdf", ".he4")):
        return "application/x-hdf"
    if lower.endswith((".nc", ".nc4")):
        return "application/x-netcdf"
    if lower.endswith((".tif", ".tiff")):
        return "image/tiff"
    return "application/octet-stream"
