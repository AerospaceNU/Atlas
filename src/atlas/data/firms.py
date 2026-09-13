"""NASA FIRMS active-fire detections (CSV area API)."""

from __future__ import annotations

import csv
import io
import os
from datetime import UTC, date, datetime, timedelta
from typing import Any, ClassVar, Self

import httpx

from atlas.data.base import BBox, DataPullClient, PullRequest, PullResult, Scene

FIRMS_AREA_URL = (
    "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{source}/{bbox}/{days}/{date}"
)
_MAX_SPAN_DAYS = 10


class FirmsClient(DataPullClient):
    """Map FIRMS hotspot rows to point scenes. Requires ``FIRMS_MAP_KEY``."""

    source: ClassVar[str] = "VIIRS_SNPP_NRT"
    platform_name: ClassVar[str] = "Suomi-NPP"
    instrument_name: ClassVar[str] = "VIIRS"

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
        for start, days in _chunk_days(request.start_date, request.end_date):
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
            resp.raise_for_status()
            for scene in _rows_to_scenes(
                resp.text,
                source=self.source,
                platform=self.platform_name,
                instrument=self.instrument_name,
            ):
                scenes.append(scene)
                if len(scenes) >= request.limit:
                    break
        return PullResult(request=request, scenes=scenes)


class FirmsViirsNoaa20Client(FirmsClient):
    source: ClassVar[str] = "VIIRS_NOAA20_NRT"
    platform_name: ClassVar[str] = "NOAA-20"


class FirmsViirsNoaa21Client(FirmsClient):
    source: ClassVar[str] = "VIIRS_NOAA21_NRT"
    platform_name: ClassVar[str] = "NOAA-21"


class FirmsModisClient(FirmsClient):
    source: ClassVar[str] = "MODIS_NRT"
    platform_name: ClassVar[str] = "Terra/Aqua"
    instrument_name: ClassVar[str] = "MODIS"


class FirmsLandsatClient(FirmsClient):
    source: ClassVar[str] = "LANDSAT_NRT"
    platform_name: ClassVar[str] = "Landsat"
    instrument_name: ClassVar[str] = "OLI"


def _chunk_days(start: date, end: date) -> list[tuple[date, int]]:
    """FIRMS area queries allow at most 10 days, anchored at ``date`` going backward."""
    chunks: list[tuple[date, int]] = []
    cursor = end
    while cursor >= start:
        span = min(_MAX_SPAN_DAYS, (cursor - start).days + 1)
        chunks.append((cursor, span))
        cursor = cursor - timedelta(days=span)
    return chunks


def _rows_to_scenes(text: str, *, source: str, platform: str, instrument: str) -> list[Scene]:
    body = text.strip()
    if not body or body.lower().startswith("invalid") or "<html" in body.lower():
        return []
    reader = csv.DictReader(io.StringIO(body))
    scenes: list[Scene] = []
    for row in reader:
        scene = _row_to_scene(row, source=source, platform=platform, instrument=instrument)
        if scene is not None:
            scenes.append(scene)
    return scenes


def _row_to_scene(
    row: dict[str, Any], *, source: str, platform: str, instrument: str
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
    pad = 0.01
    try:
        bbox = BBox(
            west=max(-180.0, lon - pad),
            south=max(-90.0, lat - pad),
            east=min(180.0, lon + pad),
            north=min(90.0, lat + pad),
        )
    except ValueError:
        return None
    frp = row.get("frp")
    confidence = row.get("confidence")
    scene_id = f"{source}:{lat:.4f}:{lon:.4f}:{acq_date}T{acq_time}"
    return Scene(
        id=scene_id,
        datetime=scene_dt,
        bbox=bbox,
        platform=str(row.get("satellite") or platform),
        instrument=instrument,
        cloud_cover=None,
        assets={},
        properties={
            "source": source,
            "frp": frp,
            "confidence": confidence,
            "daynight": row.get("daynight"),
            "bright_ti4": row.get("bright_ti4"),
            "latitude": lat,
            "longitude": lon,
        },
    )
