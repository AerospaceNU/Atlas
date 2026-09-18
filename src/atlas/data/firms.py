from __future__ import annotations

import csv
import io
import os
from datetime import UTC, date, datetime, timedelta
from typing import Any, ClassVar, Self

import httpx

from atlas.data.base import (
    BBox,
    DataPullClient,
    GeometryKind,
    PullRequest,
    PullResult,
    Scene,
    SceneKind,
)

FIRMS_AREA_URL = (
    "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{source}/{bbox}/{days}/{date}"
)
_MAX_SPAN_DAYS = 10


class FirmsClient(DataPullClient):
    """Map FIRMS hotspot rows to point scenes. Requires ``FIRMS_MAP_KEY``."""

    source: ClassVar[str] = "VIIRS_SNPP_NRT"
    platform_name: ClassVar[str] = "Suomi-NPP"
    instrument_name: ClassVar[str] = "VIIRS"
    satellite: ClassVar[str] = "viirs"
    scene_kind: ClassVar[SceneKind] = SceneKind.detection
    nominal_gsd_m: ClassVar[float | None] = 375.0

    def __init__(
        self,
        *,
        map_key: str | None = None,
        client: httpx.AsyncClient | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._map_key = map_key if map_key is not None else os.environ.get("FIRMS_MAP_KEY", "")
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = client is None

    @property
    def collection(self) -> str:
        return self.source

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.aclose()

    async def search(self, request: PullRequest) -> PullResult:
        if not self._map_key:
            raise ValueError("FIRMS_MAP_KEY is not set")
        bbox = f"{request.bbox.west},{request.bbox.south},{request.bbox.east},{request.bbox.north}"
        scenes: list[Scene] = []
        start_date, end_date = _nrt_window(request.start_date, request.end_date)
        if start_date > end_date:
            return PullResult(request=request, scenes=[])
        for start, days in _chunk_days(start_date, end_date):
            if len(scenes) >= request.limit:
                break
            url = FIRMS_AREA_URL.format(
                key=self._map_key,
                source=self.source,
                bbox=bbox,
                days=days,
                date=start.isoformat(),
            )
            resp = await self._client.get(url)
            if resp.status_code == 400:
                # NRT rejects "today" / future dates depending on FIRMS' calendar.
                continue
            resp.raise_for_status()
            for scene in _rows_to_scenes(
                resp.text,
                source=self.source,
                platform=self.platform_name,
                instrument=self.instrument_name,
                kind=self.scene_kind,
                gsd_m=self.nominal_gsd_m,
            ):
                scenes.append(scene)
                if len(scenes) >= request.limit:
                    break
        return PullResult(request=request, scenes=scenes)


class FirmsViirsNoaa20Client(FirmsClient):
    satellite: ClassVar[str] = "viirs"
    source: ClassVar[str] = "VIIRS_NOAA20_NRT"
    platform_name: ClassVar[str] = "NOAA-20"


class FirmsViirsNoaa21Client(FirmsClient):
    satellite: ClassVar[str] = "viirs"
    source: ClassVar[str] = "VIIRS_NOAA21_NRT"
    platform_name: ClassVar[str] = "NOAA-21"


class FirmsModisClient(FirmsClient):
    satellite: ClassVar[str] = "modis"
    source: ClassVar[str] = "MODIS_NRT"
    platform_name: ClassVar[str] = "Terra/Aqua"
    instrument_name: ClassVar[str] = "MODIS"
    nominal_gsd_m: ClassVar[float | None] = 1000.0


class FirmsLandsatClient(FirmsClient):
    satellite: ClassVar[str] = "landsat"
    source: ClassVar[str] = "LANDSAT_NRT"
    platform_name: ClassVar[str] = "Landsat"
    instrument_name: ClassVar[str] = "OLI"
    nominal_gsd_m: ClassVar[float | None] = 30.0


def _nrt_window(start: date, end: date, *, today_utc: date | None = None) -> tuple[date, date]:
    """Clamp NRT queries so the end date is not FIRMS' un-published 'today'."""
    latest = (today_utc or datetime.now(UTC).date()) - timedelta(days=1)
    return start, min(end, latest)


def _chunk_days(start: date, end: date) -> list[tuple[date, int]]:
    """FIRMS area queries allow at most 10 days, anchored at ``date`` going backward."""
    chunks: list[tuple[date, int]] = []
    cursor = end
    while cursor >= start:
        span = min(_MAX_SPAN_DAYS, (cursor - start).days + 1)
        chunks.append((cursor, span))
        cursor = cursor - timedelta(days=span)
    return chunks


def _rows_to_scenes(
    text: str,
    *,
    source: str,
    platform: str,
    instrument: str,
    kind: SceneKind,
    gsd_m: float | None,
) -> list[Scene]:
    body = text.strip()
    if not body or body.lower().startswith("invalid") or "<html" in body.lower():
        return []
    reader = csv.DictReader(io.StringIO(body))
    scenes: list[Scene] = []
    for row in reader:
        scene = _row_to_scene(
            row,
            source=source,
            platform=platform,
            instrument=instrument,
            kind=kind,
            gsd_m=gsd_m,
        )
        if scene is not None:
            scenes.append(scene)
    return scenes


def _row_to_scene(
    row: dict[str, Any],
    *,
    source: str,
    platform: str,
    instrument: str,
    kind: SceneKind,
    gsd_m: float | None,
) -> Scene | None:
    try:
        lat = float(row["latitude"])
        lon = float(row["longitude"])
    except (KeyError, TypeError, ValueError):
        return None
    acq_date = str(row.get("acq_date") or "")
    acq_time = str(row.get("acq_time") or "0").zfill(4)
    try:
        scene_dt = datetime.strptime(acq_date + acq_time, "%Y-%m-%d%H%M").replace(tzinfo=UTC)
    except ValueError:
        return None
    bbox = BBox.try_new(
        west=max(-180.0, lon - 0.01),
        south=max(-90.0, lat - 0.01),
        east=min(180.0, lon + 0.01),
        north=min(90.0, lat + 0.01),
    )
    if bbox is None:
        return None
    return Scene.try_new(
        id=f"{source}:{lat:.4f}:{lon:.4f}:{acq_date}T{acq_time}",
        collection=source,
        kind=kind,
        datetime=scene_dt,
        bbox=bbox,
        geometry_kind=GeometryKind.point,
        lon=lon,
        lat=lat,
        gsd_m=gsd_m,
        platform=str(row.get("satellite") or platform),
        instrument=instrument,
        properties={
            "source": source,
            "frp": row.get("frp"),
            "confidence": row.get("confidence"),
            "daynight": row.get("daynight"),
            "bright_ti4": row.get("bright_ti4"),
            "latitude": lat,
            "longitude": lon,
        },
    )
