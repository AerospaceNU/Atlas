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
from atlas.data.s3 import list_keys, list_prefixes

# Himawari-9 sits at 140.7°E; the usable disk runs about ±60° around it, across 180°.
_HIMAWARI_DISK = BBox(west=80.0, south=-60.0, east=-160.0, north=60.0)

_FILE_RE = re.compile(r"HS_H09_(\d{8})_(\d{4})_(B\d{2})_FLDK_R\d{2}_(S\d{4})")
_SLOT_RE = re.compile(r"/(\d{4})/$")
HIMAWARI_BUCKET = "noaa-himawari9"
HIMAWARI_PRODUCT = "AHI-L1b-FLDK"
# Slots this recent may still be uploading band/segment files.
_SETTLE = timedelta(hours=3)


class HimawariClient(DataPullClient):
    """List AHI full-disk slots. Imagery covers the Asia-Pacific disk, not the Americas.

    One scene is one 10-minute slot; its assets are every band/segment file in it.
    """

    satellite: ClassVar[str] = "himawari"
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
        return HIMAWARI_PRODUCT

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.aclose()

    async def search(self, request: PullRequest) -> PullResult:
        if not _HIMAWARI_DISK.intersects(request.bbox):
            return PullResult(request=request, scenes=[])
        cutoff = datetime.now(UTC) - _SETTLE
        scenes: list[Scene] = []
        for day in reversed(_days_inclusive(request.start_date, request.end_date)):
            day_prefix = f"{HIMAWARI_PRODUCT}/{day.year}/{day.month:02d}/{day.day:02d}/"
            for slot_prefix in reversed(
                await list_prefixes(self._client, HIMAWARI_BUCKET, day_prefix)
            ):
                if len(scenes) >= request.limit:
                    break
                when = _slot_time(day, slot_prefix)
                if when is None or when > cutoff:
                    continue
                keys = await list_keys(self._client, HIMAWARI_BUCKET, slot_prefix)
                scene = _slot_to_scene(when, slot_prefix, keys)
                if scene is not None:
                    scenes.append(scene)
            if len(scenes) >= request.limit:
                break
        scenes.reverse()
        return PullResult(request=request, scenes=scenes)


def _days_inclusive(start: date, end: date) -> list[date]:
    return [start + timedelta(days=n) for n in range((end - start).days + 1)]


def _slot_time(day: date, slot_prefix: str) -> datetime | None:
    match = _SLOT_RE.search(slot_prefix)
    if match is None:
        return None
    try:
        hhmm = datetime.strptime(match.group(1), "%H%M")
    except ValueError:
        return None
    return datetime(day.year, day.month, day.day, hhmm.hour, hhmm.minute, tzinfo=UTC)


def _slot_to_scene(when: datetime, slot_prefix: str, keys: list[str]) -> Scene | None:
    assets: dict[str, Asset] = {}
    for key in keys:
        match = _FILE_RE.search(key)
        if match is None:
            continue
        name = key.rsplit("/", 1)[-1]
        assets[f"{match.group(3)}_{match.group(4)}"] = Asset(
            href=f"https://{HIMAWARI_BUCKET}.s3.amazonaws.com/{key}",
            media_type="application/octet-stream",
            title=name,
            roles=["data"],
        )
    if not assets:
        return None
    return Scene.try_new(
        id=f"HS_H09_{when:%Y%m%d_%H%M}_FLDK",
        collection=HIMAWARI_PRODUCT,
        kind=SceneKind.optical,
        datetime=when,
        bbox=_HIMAWARI_DISK,
        gsd_m=2000.0,
        platform="Himawari-9",
        instrument="AHI",
        assets=assets,
        properties={"bucket": HIMAWARI_BUCKET, "prefix": slot_prefix, "files": len(assets)},
    )
