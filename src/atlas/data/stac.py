"""Generic STAC Item Search client used by Planetary Computer, Earth Search, and CDSE."""

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

_CLOUD_QUERY_KINDS = frozenset({SceneKind.optical, SceneKind.browse})


class StacApiClient(DataPullClient):
    search_url: ClassVar[str] = ""
    default_collection: ClassVar[str] = ""
    include_datetime: ClassVar[bool] = True
    scene_kind: ClassVar[SceneKind] = SceneKind.optical
    nominal_gsd_m: ClassVar[float | None] = None

    def __init__(
        self,
        *,
        collection: str | None = None,
        client: httpx.AsyncClient | None = None,
        timeout: float = 30.0,
    ) -> None:
        if not self.search_url:
            raise ValueError(f"{type(self).__name__} has no search_url")
        if collection is None and not self.default_collection:
            raise ValueError(
                f"{type(self).__name__} has no default_collection; pass `collection=...`"
            )
        self.collection = collection or self.default_collection
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
        body: dict[str, Any] = {
            "collections": [self.collection],
            "bbox": request.bbox.as_list(),
            "limit": request.limit,
        }
        if self.include_datetime:
            body["datetime"] = (
                f"{request.start_date.isoformat()}T00:00:00Z/"
                f"{request.end_date.isoformat()}T23:59:59Z"
            )
        query = self._build_query_filter(request)
        if query:
            body["query"] = query

        resp = await self._client.post(self.search_url, json=body)
        resp.raise_for_status()
        payload = resp.json()
        if not isinstance(payload, dict):
            raise TypeError(f"Expected STAC FeatureCollection, got {type(payload)}")

        scenes: list[Scene] = []
        for feature in payload.get("features") or []:
            if not isinstance(feature, dict):
                continue
            scene = item_to_scene(
                feature,
                default_platform=self.collection,
                collection=self.collection,
                kind=self.scene_kind,
                gsd_m=self.nominal_gsd_m,
            )
            if scene is not None:
                scenes.append(scene)
        return PullResult(request=request, scenes=scenes)

    def _cloud_query(self, request: PullRequest) -> dict[str, Any] | None:
        if request.max_cloud_cover is None or self.scene_kind not in _CLOUD_QUERY_KINDS:
            return None
        return {"eo:cloud_cover": {"lte": request.max_cloud_cover}}

    def _build_query_filter(self, request: PullRequest) -> dict[str, Any] | None:
        return self._cloud_query(request)


def item_to_scene(
    feature: dict[str, Any],
    *,
    default_platform: str = "unknown",
    source: str = "",
    collection: str = "",
    kind: SceneKind = SceneKind.optical,
    gsd_m: float | None = None,
) -> Scene | None:
    scene_id = feature.get("id")
    if not isinstance(scene_id, str) or not scene_id:
        return None
    props = feature.get("properties")
    if not isinstance(props, dict):
        props = {}
    scene_dt = _stac_datetime(props.get("datetime") or props.get("start_datetime"))
    if scene_dt is None:
        return None

    parsed = _parse_geometry(feature)
    if parsed is None:
        return None
    bbox, geometry_kind, lon, lat = parsed

    assets: dict[str, Asset] = {}
    raw_assets = feature.get("assets")
    if isinstance(raw_assets, dict):
        for name, spec in raw_assets.items():
            if not isinstance(spec, dict) or not isinstance(spec.get("href"), str):
                continue
            roles = spec.get("roles")
            asset = Asset.try_new(
                href=_http_href(spec["href"]),
                media_type=spec.get("type") if isinstance(spec.get("type"), str) else None,
                title=spec.get("title") if isinstance(spec.get("title"), str) else None,
                roles=list(roles) if isinstance(roles, list) else [],
            )
            if asset is not None:
                assets[str(name)] = asset

    instruments = props.get("instruments")
    instrument: str | None = None
    if isinstance(instruments, list) and instruments and isinstance(instruments[0], str):
        instrument = instruments[0]

    cloud = props.get("eo:cloud_cover")
    cloud_cover = (
        float(cloud) if isinstance(cloud, int | float) and not isinstance(cloud, bool) else None
    )
    item_gsd = props.get("gsd")
    resolved_gsd = (
        float(item_gsd)
        if isinstance(item_gsd, int | float) and not isinstance(item_gsd, bool)
        else gsd_m
    )

    return Scene.try_new(
        id=scene_id,
        source=source,
        collection=collection,
        kind=kind,
        datetime=scene_dt,
        start_datetime=_stac_datetime(props.get("start_datetime")),
        end_datetime=_stac_datetime(props.get("end_datetime")),
        bbox=bbox,
        geometry_kind=geometry_kind,
        lon=lon,
        lat=lat,
        gsd_m=resolved_gsd,
        platform=_stac_platform(props, default_platform),
        instrument=instrument,
        cloud_cover=cloud_cover,
        assets=assets,
        properties=props,
    )


def _parse_geometry(
    feature: dict[str, Any],
) -> tuple[BBox, GeometryKind, float | None, float | None] | None:
    geom = feature.get("geometry") if isinstance(feature.get("geometry"), dict) else None
    coords = geom.get("coordinates") if isinstance(geom, dict) else None
    geom_type = geom.get("type") if isinstance(geom, dict) else None

    bbox_key = feature.get("bbox")
    if isinstance(bbox_key, list):
        bbox = _stac_bbox(feature)
        if bbox is None:
            return None
        if geom_type == "Point" and isinstance(coords, list) and len(coords) >= 2:
            try:
                return bbox, GeometryKind.point, float(coords[0]), float(coords[1])
            except (TypeError, ValueError):
                return bbox, GeometryKind.bbox, None, None
        if geom_type in {"Polygon", "MultiPolygon"}:
            return bbox, GeometryKind.polygon, None, None
        return bbox, GeometryKind.bbox, None, None

    if geom_type == "Point" and isinstance(coords, list) and len(coords) >= 2:
        try:
            lon, lat = float(coords[0]), float(coords[1])
        except (TypeError, ValueError, IndexError):
            return None
        bbox = BBox.try_new(
            west=max(-180.0, lon - 0.01),
            south=max(-90.0, lat - 0.01),
            east=min(180.0, lon + 0.01),
            north=min(90.0, lat + 0.01),
        )
        if bbox is None:
            return None
        return bbox, GeometryKind.point, lon, lat
    return None


def _stac_datetime(raw: object) -> datetime | None:
    if not isinstance(raw, str) or not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00").replace(" ", "T", 1))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def _stac_bbox(feature: dict[str, Any]) -> BBox | None:
    bbox_list = feature.get("bbox")
    if not isinstance(bbox_list, list) or len(bbox_list) != 4:
        return None
    try:
        return BBox.try_new(
            west=float(bbox_list[0]),
            south=float(bbox_list[1]),
            east=float(bbox_list[2]),
            north=float(bbox_list[3]),
        )
    except (TypeError, ValueError):
        return None


def _http_href(href: str) -> str:
    """Expose S3 object URLs over HTTPS so callers can fetch without an s3 client."""
    if not href.startswith("s3://"):
        return href
    rest = href.removeprefix("s3://")
    bucket, _, key = rest.partition("/")
    if not bucket or not key:
        return href
    return f"https://{bucket}.s3.amazonaws.com/{key}"


def _stac_platform(props: dict[str, Any], default: str) -> str:
    raw = props.get("platform")
    if isinstance(raw, str) and raw:
        return raw
    if isinstance(raw, list) and raw and isinstance(raw[0], str) and raw[0]:
        return raw[0]
    return default
