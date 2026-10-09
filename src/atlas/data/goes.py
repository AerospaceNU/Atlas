from __future__ import annotations

import re
from datetime import UTC, date, datetime, timedelta
from typing import ClassVar, Self

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
from atlas.data.s3 import list_keys

_START_RE = re.compile(r"_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})")

GOES_EAST_BUCKET = "noaa-goes19"
GOES_WEST_BUCKET = "noaa-goes18"

# Rough CONUS in WGS84. Outside this, list full-disk MCMIPF instead of CONUS MCMIPC.
_CONUS = BBox(west=-125.0, south=24.0, east=-66.0, north=50.0)
_EAST_DISK = BBox(west=-135.0, south=-50.0, east=-15.0, north=50.0)
_WEST_DISK = BBox(west=-180.0, south=-50.0, east=-105.0, north=60.0)


class GoesClient(DataPullClient):
    """Pick GOES-East or West from the AOI longitude; list ABI MCMIP NetCDFs.

    Scene footprints are the whole CONUS or full-disk sector, not the AOI.
    """

    satellite: ClassVar[str] = "goes"
    scene_kind: ClassVar[SceneKind] = SceneKind.optical

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = client is None

    @property
    def collection(self) -> str:
        return "goes-abi-mcmip"

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.aclose()

    async def search(self, request: PullRequest) -> PullResult:
        bucket = _bucket_for_bbox(request.bbox)
        product = _product_for_bbox(request.bbox)
        if not _footprint(bucket, product).intersects(request.bbox):
            return PullResult(request=request, scenes=[])
        hours = set(_hours_in_range(request.start_date, request.end_date))
        days = sorted({(year, doy) for year, doy, _ in hours})
        scenes: dict[str, Scene] = {}
        for year, doy in reversed(days):
            if request.limit is not None and len(scenes) >= request.limit:
                break
            for key in await list_keys(self._client, bucket, f"{product}/{year}/{doy:03d}/"):
                scene = _key_to_scene(bucket, key, product)
                # Keep finished hours only.
                if scene is not None and _hour_key(scene.datetime) in hours:
                    scenes.setdefault(scene.id, scene)
        newest = sorted(scenes.values(), key=lambda s: s.datetime)
        if request.limit is not None:
            newest = newest[-request.limit :]
        return PullResult(request=request, scenes=newest)


def _bucket_for_bbox(bbox: BBox) -> str:
    lon = (bbox.west + bbox.east) / 2
    return GOES_EAST_BUCKET if lon >= -105.0 else GOES_WEST_BUCKET


def _product_for_bbox(bbox: BBox) -> str:
    inside = (
        bbox.west >= _CONUS.west
        and bbox.east <= _CONUS.east
        and bbox.south >= _CONUS.south
        and bbox.north <= _CONUS.north
    )
    return "ABI-L2-MCMIPC" if inside else "ABI-L2-MCMIPF"


def _hours_in_range(
    start: date,
    end: date,
    *,
    now: datetime | None = None,
) -> list[tuple[int, int, int]]:
    """UTC hours in ``[start, end]`` that have already finished (GOES prefixes exist)."""
    cursor = datetime(start.year, start.month, start.day, tzinfo=UTC)
    stop = datetime(end.year, end.month, end.day, 23, tzinfo=UTC)
    latest = now or datetime.now(UTC)
    hours: list[tuple[int, int, int]] = []
    while cursor <= stop:
        if cursor + timedelta(hours=1) <= latest:
            hours.append((cursor.year, cursor.timetuple().tm_yday, cursor.hour))
        cursor += timedelta(hours=1)
    return hours


def _hour_key(when: datetime) -> tuple[int, int, int]:
    return (when.year, when.timetuple().tm_yday, when.hour)


def _footprint(bucket: str, product: str) -> BBox:
    if product.endswith("C"):
        return _CONUS
    return _EAST_DISK if bucket == GOES_EAST_BUCKET else _WEST_DISK


def _key_to_scene(bucket: str, key: str, product: str) -> Scene | None:
    match = _START_RE.search(key)
    if match is None:
        return None
    year, doy, hour, minute, second = (int(p) for p in match.groups())
    try:
        scene_dt = datetime(year, 1, 1, tzinfo=UTC) + timedelta(
            days=doy - 1, hours=hour, minutes=minute, seconds=second
        )
    except ValueError:
        return None
    href = f"https://{bucket}.s3.amazonaws.com/{key}"
    return Scene.try_new(
        id=key.rsplit("/", 1)[-1],
        collection=product,
        kind=SceneKind.optical,
        datetime=scene_dt,
        bbox=_footprint(bucket, product),
        gsd_m=2000.0,
        platform="GOES-East" if bucket == GOES_EAST_BUCKET else "GOES-West",
        instrument="ABI",
        assets={
            "data": Asset(
                href=href,
                media_type="application/x-netcdf",
                title=key.rsplit("/", 1)[-1],
                roles=["data"],
            )
        },
        properties={"bucket": bucket, "key": key},
    )
