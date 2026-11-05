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

# Operational windows (inclusive) for the public NOAA buckets. A window that
# spans a handover lists both buckets, so a pull across the change still gets
# imagery. GOES-19 became GOES-East on 2025-04-07; GOES-18 became GOES-West on
# 2023-01-04; GOES-17 became operational as GOES-West on 2019-02-12.
_EAST_WINDOWS: tuple[tuple[str, date, date], ...] = (
    ("noaa-goes16", date(2017, 12, 18), date(2025, 4, 6)),
    (GOES_EAST_BUCKET, date(2025, 4, 7), date.max),
)
_WEST_WINDOWS: tuple[tuple[str, date, date], ...] = (
    ("noaa-goes15", date(2012, 1, 1), date(2019, 2, 11)),
    ("noaa-goes17", date(2019, 2, 12), date(2023, 1, 3)),
    (GOES_WEST_BUCKET, date(2023, 1, 4), date.max),
)

# Rough CONUS in WGS84. Outside this, list full-disk MCMIPF instead of CONUS MCMIPC.
_CONUS = BBox(west=-125.0, south=24.0, east=-66.0, north=50.0)
_EAST_DISK = BBox(west=-135.0, south=-50.0, east=-15.0, north=50.0)
_WEST_DISK = BBox(west=-180.0, south=-50.0, east=-105.0, north=60.0)


class GoesClient(DataPullClient):
    """Pick GOES-East or West from the AOI longitude; list ABI MCMIP NetCDFs.

    Scene footprints are the whole CONUS or full-disk sector, not the AOI.

    A box that crosses the antimeridian is cut at ±180° before the satellite
    is chosen. The midpoint of the uncut box is not a point inside the AOI, so
    it selects a disk that misses the Pacific and the search returns no
    scenes. Each side is listed only when it meets that side's disk. An HTTP
    failure while listing is raised, so the source is an error rather than an
    empty catalog.
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
        targets = _list_targets(request.bbox, request.start_date, request.end_date)
        if not targets:
            return PullResult(request=request, scenes=[])
        hours = set(_hours_in_range(request.start_date, request.end_date))
        merged: dict[str, Scene] = {}
        for bucket, side, product in targets:
            for scene in await _search_product(self._client, request, bucket, side, product, hours):
                merged.setdefault(scene.id, scene)
        newest = sorted(merged.values(), key=lambda s: s.datetime)
        if request.limit is not None:
            newest = newest[-request.limit :]
        return PullResult(request=request, scenes=newest)


def _list_targets(
    bbox: BBox,
    start: date,
    end: date,
) -> list[tuple[str, str, str]]:
    """Bucket, side, and product for each side of ``bbox`` that meets a GOES disk.

    A wrapping box is cut first. Sides that miss both disks are skipped, which
    is an empty result for an AOI outside GOES coverage, not an error. A side
    whose dates have no public bucket is an error: that window is not served,
    rather than an empty catalog that reads like no imagery.

    Raises:
        ValueError: The AOI meets a disk but no bucket covers the date window.
    """
    targets: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for part in bbox.antimeridian_parts():
        side = _side_for_bbox(part)
        product = _product_for_bbox(part)
        if not _footprint(side, product).intersects(part):
            continue
        buckets = _buckets_for(side, start, end)
        if not buckets:
            raise ValueError(
                f"GOES has no public {side} bucket for {start.isoformat()}..{end.isoformat()}"
            )
        for bucket in buckets:
            target = (bucket, side, product)
            if target not in seen:
                seen.add(target)
                targets.append(target)
    return targets


def _buckets_for(side: str, start: date, end: date) -> list[str]:
    """Public buckets on ``side`` whose operational window meets the request."""
    windows = _EAST_WINDOWS if side == "east" else _WEST_WINDOWS
    return [bucket for bucket, first, last in windows if first <= end and start <= last]


async def _search_product(
    client: httpx.AsyncClient,
    request: PullRequest,
    bucket: str,
    side: str,
    product: str,
    hours: set[tuple[int, int, int]],
) -> list[Scene]:
    days = sorted({(year, doy) for year, doy, _hour in hours})
    scenes: dict[str, Scene] = {}
    for year, doy in reversed(days):
        if request.limit is not None and len(scenes) >= request.limit:
            break
        for key in await list_keys(client, bucket, f"{product}/{year}/{doy:03d}/"):
            scene = _key_to_scene(bucket, side, key, product)
            # Keep finished hours only.
            if scene is not None and _hour_key(scene.datetime) in hours:
                scenes.setdefault(scene.id, scene)
    newest = sorted(scenes.values(), key=lambda s: s.datetime)
    if request.limit is not None:
        newest = newest[-request.limit :]
    return newest


def _require_unwrapped(bbox: BBox) -> None:
    if bbox.west > bbox.east:
        raise ValueError(
            f"GOES disk selection needs a non-wrapping box; got west={bbox.west} east={bbox.east}"
        )


def _side_for_bbox(bbox: BBox) -> str:
    """``east`` or ``west`` disk that contains the centre of ``bbox``."""
    _require_unwrapped(bbox)
    lon = (bbox.west + bbox.east) / 2
    return "east" if lon >= -105.0 else "west"


def _product_for_bbox(bbox: BBox) -> str:
    _require_unwrapped(bbox)
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


def _footprint(side: str, product: str) -> BBox:
    if product.endswith("C"):
        return _CONUS
    return _EAST_DISK if side == "east" else _WEST_DISK


def _key_to_scene(bucket: str, side: str, key: str, product: str) -> Scene | None:
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
        bbox=_footprint(side, product),
        gsd_m=2000.0,
        platform="GOES-East" if side == "east" else "GOES-West",
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
