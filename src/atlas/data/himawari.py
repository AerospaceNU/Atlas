"""Himawari-9 AHI full-disk on NOAA Open Data (AWS, no sign-in)."""

from __future__ import annotations

import re
from datetime import UTC, date, datetime, timedelta
from typing import Self
from xml.etree import ElementTree

import httpx

from atlas.data.base import Asset, DataPullClient, PullRequest, PullResult, Scene

_S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
_TIME_RE = re.compile(r"HS_H09_(\d{8})_(\d{4})_")
HIMAWARI_BUCKET = "noaa-himawari9"
HIMAWARI_PRODUCT = "AHI-L1b-FLDK"


class HimawariClient(DataPullClient):
    """List recent AHI full-disk slots. Imagery covers the Asia-Pacific disk, not the Americas."""

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
        cutoff = datetime.now(UTC) - timedelta(hours=3)
        slots = [
            slot
            for slot in _ten_minute_slots(request.start_date, request.end_date)
            if slot <= cutoff
        ][-request.limit :]
        scenes: list[Scene] = []
        for when in slots:
            if len(scenes) >= request.limit:
                break
            prefix = (
                f"{HIMAWARI_PRODUCT}/{when.year}/{when.month:02d}/{when.day:02d}/"
                f"{when.strftime('%H%M')}/"
            )
            url = f"https://{HIMAWARI_BUCKET}.s3.amazonaws.com/?list-type=2&prefix={prefix}&max-keys=1"
            resp = await self._client.get(url)
            resp.raise_for_status()
            for key in _parse_keys(resp.text):
                scene = _key_to_scene(key, request)
                if scene is not None:
                    scenes.append(scene)
        return PullResult(request=request, scenes=scenes)


def _ten_minute_slots(start: date, end: date) -> list[datetime]:
    cursor = datetime(start.year, start.month, start.day, tzinfo=UTC)
    stop = datetime(end.year, end.month, end.day, 23, 50, tzinfo=UTC)
    slots: list[datetime] = []
    while cursor <= stop:
        slots.append(cursor)
        cursor += timedelta(minutes=10)
    return slots


def _parse_keys(xml_text: str) -> list[str]:
    try:
        root = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError:
        return []
    return [node.text for node in root.findall("s3:Contents/s3:Key", _S3_NS) if node.text]


def _key_to_scene(key: str, request: PullRequest) -> Scene | None:
    match = _TIME_RE.search(key)
    if match is None:
        return None
    try:
        scene_dt = datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M").replace(
            tzinfo=UTC
        )
    except ValueError:
        return None
    href = f"https://{HIMAWARI_BUCKET}.s3.amazonaws.com/{key}"
    return Scene(
        id=key.rsplit("/", 1)[-1],
        datetime=scene_dt,
        bbox=request.bbox,
        platform="Himawari-9",
        instrument="AHI",
        cloud_cover=None,
        assets={
            "data": Asset(
                href=href,
                media_type="application/octet-stream",
                title=key.rsplit("/", 1)[-1],
                roles=["data"],
            )
        },
        properties={"bucket": HIMAWARI_BUCKET, "key": key},
    )
