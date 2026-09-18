from __future__ import annotations

from datetime import date, datetime
from typing import Any, ClassVar, Self
from urllib.parse import urljoin

import httpx

from atlas.data.base import BBox, DataPullClient, PullRequest, PullResult, Scene, SceneKind
from atlas.data.stac import item_to_scene

MAXAR_ROOT = "https://maxar-opendata.s3.amazonaws.com/events/catalog.json"


class MaxarOpenDataClient(DataPullClient):
    satellite: ClassVar[str] = "maxar"
    scene_kind: ClassVar[SceneKind] = SceneKind.optical

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        timeout: float = 60.0,
        max_events: int = 40,
    ) -> None:
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = client is None
        self._max_events = max_events

    @property
    def collection(self) -> str:
        return "maxar-opendata"

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.aclose()

    async def search(self, request: PullRequest) -> PullResult:
        root = await self._get_json(MAXAR_ROOT)
        scenes: list[Scene] = []
        scanned = 0
        for link in root.get("links") or []:
            if not isinstance(link, dict) or link.get("rel") != "child":
                continue
            href = link.get("href")
            if not isinstance(href, str):
                continue
            event_url = urljoin(MAXAR_ROOT, href)
            scanned += 1
            if scanned > self._max_events or len(scenes) >= request.limit:
                break
            event = await self._get_json(event_url)
            if not _extent_overlaps(event.get("extent"), request):
                continue
            for scene in await self._scenes_from_event(event_url, event, request):
                scenes.append(scene)
                if len(scenes) >= request.limit:
                    break
        return PullResult(request=request, scenes=scenes)

    async def _scenes_from_event(
        self, event_url: str, event: dict[str, Any], request: PullRequest
    ) -> list[Scene]:
        scenes: list[Scene] = []
        for link in event.get("links") or []:
            if not isinstance(link, dict) or link.get("rel") != "child":
                continue
            href = link.get("href")
            if not isinstance(href, str):
                continue
            acq = await self._get_json(urljoin(event_url, href))
            if not _extent_overlaps(acq.get("extent"), request):
                continue
            for item_link in acq.get("links") or []:
                if not isinstance(item_link, dict) or item_link.get("rel") != "item":
                    continue
                item_href = item_link.get("href")
                if not isinstance(item_href, str):
                    continue
                item_url = urljoin(urljoin(event_url, href), item_href)
                item = await self._get_json(item_url)
                _absolutize_asset_hrefs(item, item_url)
                scene = item_to_scene(
                    item,
                    default_platform="maxar",
                    collection="maxar-opendata",
                    kind=SceneKind.optical,
                    gsd_m=0.5,
                )
                if scene is not None:
                    scenes.append(scene)
                if len(scenes) >= request.limit:
                    return scenes
        return scenes

    async def _get_json(self, url: str) -> dict[str, Any]:
        resp = await self._client.get(url)
        resp.raise_for_status()
        payload = resp.json()
        return payload if isinstance(payload, dict) else {}


def _absolutize_asset_hrefs(item: dict[str, Any], base_url: str) -> None:
    assets = item.get("assets")
    if not isinstance(assets, dict):
        return
    for spec in assets.values():
        if isinstance(spec, dict) and isinstance(spec.get("href"), str):
            spec["href"] = urljoin(base_url, spec["href"])


def _extent_overlaps(extent: object, request: PullRequest) -> bool:
    if not isinstance(extent, dict):
        return True
    spatial = extent.get("spatial")
    boxes = spatial.get("bbox") if isinstance(spatial, dict) else None
    spatial_ok = (
        not isinstance(boxes, list)
        or not boxes
        or any(
            isinstance(box, list) and len(box) >= 4 and _bbox_overlap(box, request.bbox)
            for box in boxes
        )
    )
    temporal = extent.get("temporal")
    intervals = temporal.get("interval") if isinstance(temporal, dict) else None
    temporal_ok = (
        not isinstance(intervals, list)
        or not intervals
        or any(_interval_overlap(iv, request.start_date, request.end_date) for iv in intervals)
    )
    return spatial_ok and temporal_ok


def _bbox_overlap(box: list[object], bbox: BBox) -> bool:
    try:
        west = float(str(box[0]))
        south = float(str(box[1]))
        east = float(str(box[2]))
        north = float(str(box[3]))
    except (TypeError, ValueError, IndexError):
        return False
    return not (east < bbox.west or west > bbox.east or north < bbox.south or south > bbox.north)


def _interval_overlap(interval: object, start: date, end: date) -> bool:
    if not isinstance(interval, list) or not interval:
        return True
    raw_a = interval[0] if interval else None
    raw_b = interval[1] if len(interval) > 1 else None
    try:
        a = (
            date.min
            if not raw_a
            else datetime.fromisoformat(str(raw_a).replace("Z", "+00:00")).date()
        )
        b = (
            date.max
            if not raw_b
            else datetime.fromisoformat(str(raw_b).replace("Z", "+00:00")).date()
        )
    except ValueError:
        return True
    return not (b < start or a > end)
