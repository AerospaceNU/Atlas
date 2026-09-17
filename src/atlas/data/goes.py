"""NOAA GOES ABI Cloud and Moisture Imagery on AWS Open Data (no sign-in)."""

from __future__ import annotations

import re
from datetime import UTC, date, datetime, timedelta
from typing import ClassVar, Self
from xml.etree import ElementTree

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

_S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
_START_RE = re.compile(r"_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})")

GOES_EAST_BUCKET = "noaa-goes19"
GOES_WEST_BUCKET = "noaa-goes18"

# Rough CONUS in WGS84. Outside this, list full-disk MCMIPF instead of CONUS MCMIPC.
_CONUS = BBox(west=-125.0, south=24.0, east=-66.0, north=50.0)
_EAST_DISK = BBox(west=-135.0, south=-50.0, east=-15.0, north=50.0)
_WEST_DISK = BBox(west=-180.0, south=-50.0, east=-105.0, north=60.0)


class GoesClient(DataPullClient):
    """Pick GOES-East or West from the AOI longitude; list recent ABI MCMIP NetCDFs."""

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
        hours = _hours_in_range(request.start_date, request.end_date)[-request.limit :]
        scenes: list[Scene] = []
        for year, doy, hour in hours:
            if len(scenes) >= request.limit:
                break
            prefix = f"{product}/{year}/{doy:03d}/{hour:02d}/"
            url = f"https://{bucket}.s3.amazonaws.com/?list-type=2&prefix={prefix}&max-keys=1"
            resp = await self._client.get(url)
            resp.raise_for_status()
            for key in _parse_keys(resp.text):
                scene = _key_to_scene(bucket, key, product)
                if scene is not None:
                    scenes.append(scene)
                    if len(scenes) >= request.limit:
                        break
        return PullResult(request=request, scenes=scenes)


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


def _hours_in_range(start: date, end: date) -> list[tuple[int, int, int]]:
    cursor = datetime(start.year, start.month, start.day, tzinfo=UTC)
    stop = datetime(end.year, end.month, end.day, 23, tzinfo=UTC)
    hours: list[tuple[int, int, int]] = []
    while cursor <= stop:
        hours.append((cursor.year, cursor.timetuple().tm_yday, cursor.hour))
        cursor += timedelta(hours=1)
    return hours


def _parse_keys(xml_text: str) -> list[str]:
    try:
        root = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError:
        return []
    keys: list[str] = []
    for node in root.findall("s3:Contents/s3:Key", _S3_NS):
        if node.text:
            keys.append(node.text)
    return keys


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
