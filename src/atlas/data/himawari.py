from __future__ import annotations

import math
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

# CGMS normalized geostationary projection for Himawari-9 (JMA HSD user's guide).
_SUB_LON = 140.7
_R_EQ = 6378.137
_R_POL = 6356.7523
_SAT_DIST = 42164.0  # Satellite distance from Earth's centre, km.
_SEGMENTS = 10
# Full-disk line count by band. B03 is 0.5 km; B01, B02, and B04 are 1 km; the rest are 2 km.
_FULL_DISK_LINES = {"B01": 11000, "B02": 11000, "B03": 22000, "B04": 11000}
_DEFAULT_LINES = 5500
# AOI segment search grid: sample spacing in degrees, and a cap per axis.
_SAMPLE_STEP_DEG = 0.25
_MAX_SAMPLES = 240
# Line scaling factor (LFAC) per line count; LOFF is the disk centre line.
_LFAC = {22000: 81865099, 11000: 40932549, 5500: 20466275}
# Common band name to AHI band: 0.47, 0.51, 0.64, and 0.86 µm.
_ALIAS_BANDS = {"blue": "B01", "green": "B02", "red": "B03", "nir": "B04"}


class HimawariClient(DataPullClient):
    """List AHI full-disk slots. Imagery covers the Asia-Pacific disk, not the Americas.

    One scene is one 10-minute slot; its assets are every band/segment file in it.
    Each file is one of ten line strips of the disk. ``band_aliases`` maps
    blue/green/red/nir (B01-B04) to the strip holding the AOI centre, found with
    the CGMS geostationary projection, so a default pick reads rows over the AOI.
    ``properties["aoi_segments"]`` lists every B03 strip the AOI touches.
    The disk crosses the antimeridian, from about 80°E through 180° to 160°W.

    A dateline AOI is cut at ±180° and the slot is listed once when either side
    meets the disk. Both sides share the same full-disk prefix, so a second
    request would repeat the slot. The scene footprint stays the wrapping disk.
    An HTTP failure while listing is raised rather than reported as no imagery.
    An AOI that misses the disk is an empty result, not an error.
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
        """List settled full-disk slots when the AOI meets the Himawari disk.

        A dateline AOI is tested one side at a time. The listing itself is not
        split: the prefix does not depend on longitude.
        """
        if not _intersects_disk(request.bbox):
            return PullResult(request=request, scenes=[])
        cutoff = datetime.now(UTC) - _SETTLE
        scenes: list[Scene] = []
        for day in reversed(_days_inclusive(request.start_date, request.end_date)):
            day_prefix = f"{HIMAWARI_PRODUCT}/{day.year}/{day.month:02d}/{day.day:02d}/"
            for slot_prefix in reversed(
                await list_prefixes(self._client, HIMAWARI_BUCKET, day_prefix)
            ):
                if request.limit is not None and len(scenes) >= request.limit:
                    break
                when = _slot_time(day, slot_prefix)
                if when is None or when > cutoff:
                    continue
                keys = await list_keys(self._client, HIMAWARI_BUCKET, slot_prefix)
                scene = _slot_to_scene(when, slot_prefix, keys, request.bbox)
                if scene is not None:
                    scenes.append(scene)
            if request.limit is not None and len(scenes) >= request.limit:
                break
        scenes.reverse()
        return PullResult(request=request, scenes=scenes)


def _intersects_disk(bbox: BBox) -> bool:
    """True when any non-wrapping side of ``bbox`` meets the Himawari disk."""
    return any(_HIMAWARI_DISK.intersects(part) for part in bbox.antimeridian_parts())


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


def _bbox_center(bbox: BBox) -> tuple[float, float]:
    """Centre of ``bbox``; a box across the antimeridian is centred across 180°."""
    lon = bbox.west + bbox.lon_span / 2
    if lon > 180:
        lon -= 360
    return lon, (bbox.south + bbox.north) / 2


def _disk_line(lon: float, lat: float, lines: int) -> float | None:
    """Fractional full-disk line (1 = north) of a point, or ``None`` off the visible disk.

    Uses the CGMS normalized geostationary projection with the Himawari-9 LFAC
    and LOFF for a band with ``lines`` full-disk lines.
    """
    lat_r = math.radians(lat)
    dlon = math.radians(lon - _SUB_LON)
    c_lat = math.atan((_R_POL / _R_EQ) ** 2 * math.tan(lat_r))
    ecc2 = 1 - (_R_POL / _R_EQ) ** 2
    rl = _R_POL / math.sqrt(1 - ecc2 * math.cos(c_lat) ** 2)
    r1 = _SAT_DIST - rl * math.cos(c_lat) * math.cos(dlon)
    r2 = -rl * math.cos(c_lat) * math.sin(dlon)
    r3 = rl * math.sin(c_lat)
    # The point faces the satellite only when the satellite sits above its horizon.
    if _SAT_DIST * (_SAT_DIST - r1) - r2**2 - r3**2 * (_R_EQ / _R_POL) ** 2 < 0:
        return None
    rn = math.sqrt(r1**2 + r2**2 + r3**2)
    y = math.degrees(math.asin(-r3 / rn))
    loff = lines / 2 + 0.5
    line = loff + y * 2**-16 * _LFAC[lines]
    return line if 1 <= line < lines + 1 else None


def _segment(lon: float, lat: float, band: str) -> int | None:
    """Full-disk segment (1 = northernmost of 10 equal line strips) holding a point."""
    lines = _FULL_DISK_LINES.get(band, _DEFAULT_LINES)
    line = _disk_line(lon, lat, lines)
    if line is None:
        return None
    return min(int((line - 1) // (lines // _SEGMENTS)) + 1, _SEGMENTS)


def _aoi_segments(aoi: BBox, band: str) -> list[int]:
    """Every segment between the northernmost and southernmost visible line of ``aoi``.

    Along a meridian the disk line moves monotonically with latitude, but
    along a parallel it peaks at the sub-satellite longitude, not at a box
    corner. The box is therefore sampled on a grid at most
    ``_SAMPLE_STEP_DEG`` apart that always includes the sub-satellite
    longitude when the box spans it. The line range is continuous over the
    visible part of the box, so all segments from the lowest to the highest
    sampled one are returned. Near the limb the visible edge is resolved to
    the grid step. A box crossing the antimeridian is tested per side.
    """
    lines = _FULL_DISK_LINES.get(band, _DEFAULT_LINES)
    low: float | None = None
    high: float | None = None
    for part in aoi.antimeridian_parts():
        for lon in _sample_lons(part):
            for lat in _sample_axis(part.south, part.north):
                line = _disk_line(lon, lat, lines)
                if line is None:
                    continue
                low = line if low is None else min(low, line)
                high = line if high is None else max(high, line)
    if low is None or high is None:
        return []
    strip = lines // _SEGMENTS
    first = min(int((low - 1) // strip) + 1, _SEGMENTS)
    last = min(int((high - 1) // strip) + 1, _SEGMENTS)
    return list(range(first, last + 1))


def _sample_axis(start: float, end: float) -> list[float]:
    """Evenly spaced values from ``start`` to ``end``, at most ``_SAMPLE_STEP_DEG`` apart."""
    span = end - start
    count = min(max(math.ceil(span / _SAMPLE_STEP_DEG), 1), _MAX_SAMPLES)
    return [start + span * index / count for index in range(count + 1)]


def _sample_lons(part: BBox) -> list[float]:
    """Longitude samples of one non-wrapping box, plus the sub-satellite longitude inside it."""
    lons = _sample_axis(part.west, part.east)
    for candidate in (_SUB_LON, _SUB_LON - 360.0):
        if part.west < candidate < part.east:
            lons.append(candidate)
    return lons


def _band_aliases(assets: dict[str, Asset], aoi: BBox) -> dict[str, str]:
    """Point each common band name at the segment file holding the AOI centre.

    A band is left out when the centre is off the visible disk or that segment
    file is missing from the slot, so an alias never names a strip that misses
    the AOI.
    """
    lon, lat = _bbox_center(aoi)
    aliases: dict[str, str] = {}
    for name, band in _ALIAS_BANDS.items():
        segment = _segment(lon, lat, band)
        if segment is None:
            continue
        key = f"{band}_S{segment:02d}{_SEGMENTS}"
        if key in assets:
            aliases[name] = key
    return aliases


def _slot_to_scene(when: datetime, slot_prefix: str, keys: list[str], aoi: BBox) -> Scene | None:
    """One full-disk slot as a scene whose band aliases name the AOI's segment files."""
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
        band_aliases=_band_aliases(assets, aoi),
        properties={
            "bucket": HIMAWARI_BUCKET,
            "prefix": slot_prefix,
            "files": len(assets),
            "aoi_segments": _aoi_segments(aoi, "B03"),
        },
    )
