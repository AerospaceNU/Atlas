from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, ClassVar, Self

import httpx

from atlas.data.base import (
    Asset,
    BBox,
    DataPullClient,
    GeometryKind,
    PullRequest,
    PullResult,
    Scene,
    SceneKind,
)

CMR_SEARCH_URL = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"

_TILE_ATTRS = frozenset({"HORIZONTALTILENUMBER", "VERTICALTILENUMBER", "TileID"})


class MODISClient(DataPullClient):
    short_name: ClassVar[str] = ""
    version: ClassVar[str] = "061"
    satellite: ClassVar[str] = "modis"
    scene_kind: ClassVar[SceneKind] = SceneKind.optical
    nominal_gsd_m: ClassVar[float | None] = None

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

    @property
    def collection(self) -> str:
        return self.short_name

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
        if request.max_cloud_cover is not None and self.scene_kind in {
            SceneKind.optical,
            SceneKind.browse,
        }:
            params["cloud_cover"] = f"0,{request.max_cloud_cover:g}"

        resp = await self._client.get(CMR_SEARCH_URL, params=params)
        resp.raise_for_status()
        payload = resp.json()
        if not isinstance(payload, dict):
            raise TypeError(f"Expected CMR object, got {type(payload)}")

        scenes: list[Scene] = []
        for item in payload.get("items") or []:
            if not isinstance(item, dict):
                continue
            scene = self._granule_to_scene(item)
            if scene is not None:
                scenes.append(scene)
        return PullResult(request=request, scenes=scenes)

    def _granule_to_scene(self, item: dict[str, Any]) -> Scene | None:
        raw_meta = item.get("meta")
        raw_umm = item.get("umm")
        meta: dict[str, Any] = raw_meta if isinstance(raw_meta, dict) else {}
        umm: dict[str, Any] = raw_umm if isinstance(raw_umm, dict) else {}

        scene_dt, start_dt, end_dt = _scene_times(umm)
        parsed_bbox = _scene_bbox_and_kind(umm)
        scene_id = meta.get("concept-id") or umm.get("GranuleUR")
        if scene_dt is None or parsed_bbox is None or not isinstance(scene_id, str) or not scene_id:
            return None
        bbox, geometry_kind = parsed_bbox

        platform_name, instrument_name = _platform(umm)
        return Scene.try_new(
            id=scene_id,
            collection=self.short_name,
            kind=self.scene_kind,
            datetime=scene_dt,
            start_datetime=start_dt,
            end_datetime=end_dt,
            bbox=bbox,
            geometry_kind=geometry_kind,
            gsd_m=self.nominal_gsd_m,
            platform=platform_name,
            instrument=instrument_name,
            cloud_cover=_cloud_cover(umm),
            assets=_https_data_assets(umm),
            properties=_properties(self.short_name, self.version, umm),
        )


def _scene_times(umm: dict[str, Any]) -> tuple[datetime | None, datetime | None, datetime | None]:
    temporal = umm.get("TemporalExtent")
    if not isinstance(temporal, dict):
        return None, None, None
    range_dt = temporal.get("RangeDateTime")
    start = end = None
    if isinstance(range_dt, dict):
        start = _parse_datetime(range_dt.get("BeginningDateTime"))
        end = _parse_datetime(range_dt.get("EndingDateTime"))
    nominal = start or _parse_datetime(temporal.get("SingleDateTime")) or end
    return nominal, start, end


def _parse_datetime(raw: object) -> datetime | None:
    if not isinstance(raw, str) or not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def _scene_bbox_and_kind(umm: dict[str, Any]) -> tuple[BBox, GeometryKind] | None:
    spatial = umm.get("SpatialExtent")
    if not isinstance(spatial, dict):
        return None
    domain = spatial.get("HorizontalSpatialDomain")
    if not isinstance(domain, dict):
        return None
    geometry = domain.get("Geometry")
    if not isinstance(geometry, dict):
        return None
    rect = _bbox_from_rectangles(geometry)
    if rect is not None:
        return rect, GeometryKind.bbox
    poly = _bbox_from_polygons(geometry)
    if poly is not None:
        return poly, GeometryKind.polygon
    return None


def _bbox_from_rectangles(geometry: dict[str, Any]) -> BBox | None:
    rects = geometry.get("BoundingRectangles")
    if not isinstance(rects, list) or not rects:
        return None
    bb = rects[0]
    if not isinstance(bb, dict):
        return None
    try:
        return BBox(
            west=float(bb["WestBoundingCoordinate"]),
            south=float(bb["SouthBoundingCoordinate"]),
            east=float(bb["EastBoundingCoordinate"]),
            north=float(bb["NorthBoundingCoordinate"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _bbox_from_polygons(geometry: dict[str, Any]) -> BBox | None:
    polygons = geometry.get("GPolygons")
    if not isinstance(polygons, list):
        return None
    lons: list[float] = []
    lats: list[float] = []
    for poly in polygons:
        if not isinstance(poly, dict):
            continue
        boundary = poly.get("Boundary")
        points = boundary.get("Points") if isinstance(boundary, dict) else None
        if not isinstance(points, list):
            continue
        for point in points:
            if not isinstance(point, dict):
                continue
            try:
                lons.append(float(point["Longitude"]))
                lats.append(float(point["Latitude"]))
            except (KeyError, TypeError, ValueError):
                continue
    if not lons or not lats:
        return None
    return BBox(west=min(lons), south=min(lats), east=max(lons), north=max(lats))


def _platform(umm: dict[str, Any]) -> tuple[str, str | None]:
    platforms = umm.get("Platforms")
    if not isinstance(platforms, list) or not platforms or not isinstance(platforms[0], dict):
        return "MODIS", None
    first = platforms[0]
    name = first.get("ShortName")
    platform_name = name if isinstance(name, str) and name else "MODIS"
    instruments = first.get("Instruments")
    if not isinstance(instruments, list) or not instruments or not isinstance(instruments[0], dict):
        return platform_name, None
    instrument = instruments[0].get("ShortName")
    instrument_name = instrument if isinstance(instrument, str) and instrument else None
    return platform_name, instrument_name


def _cloud_cover(umm: dict[str, Any]) -> float | None:
    value = umm.get("CloudCover")
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _https_data_assets(umm: dict[str, Any]) -> dict[str, Asset]:
    assets: dict[str, Asset] = {}
    related = umm.get("RelatedUrls")
    if not isinstance(related, list):
        return assets
    for entry in related:
        if not isinstance(entry, dict):
            continue
        url = entry.get("URL")
        url_type = entry.get("Type")
        if not isinstance(url, str) or not url.startswith("https://"):
            continue
        if url_type == "GET DATA" and "data" not in assets:
            asset = Asset.try_new(
                href=url,
                media_type=_guess_media_type(url),
                title=_asset_title(entry, "data"),
                roles=["data"],
            )
            if asset is not None:
                assets["data"] = asset
        elif url_type == "GET RELATED VISUALIZATION" and "browse" not in assets:
            asset = Asset.try_new(
                href=url,
                media_type=_guess_media_type(url),
                title=_asset_title(entry, "browse"),
                roles=["overview"],
            )
            if asset is not None:
                assets["browse"] = asset
    return assets


def _asset_title(entry: dict[str, Any], default: str) -> str:
    description = entry.get("Description")
    return description if isinstance(description, str) and description else default


def _properties(short_name: str, version: str, umm: dict[str, Any]) -> dict[str, Any]:
    props: dict[str, Any] = {"short_name": short_name, "version": version}
    attributes = umm.get("AdditionalAttributes")
    if not isinstance(attributes, list):
        return props
    for attr in attributes:
        if not isinstance(attr, dict):
            continue
        name = attr.get("Name")
        values = attr.get("Values")
        if name in _TILE_ATTRS and isinstance(values, list) and values:
            props[str(name)] = values[0]
    return props


def _guess_media_type(url: str) -> str:
    path = url.split("?", 1)[0].lower()
    if path.endswith((".hdf", ".he4")):
        return "application/x-hdf"
    if path.endswith((".nc", ".nc4")):
        return "application/x-netcdf"
    if path.endswith((".tif", ".tiff")):
        return "image/tiff"
    if path.endswith((".jpg", ".jpeg")):
        return "image/jpeg"
    if path.endswith(".png"):
        return "image/png"
    return "application/octet-stream"
