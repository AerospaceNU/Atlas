from __future__ import annotations

import csv
import io
from datetime import UTC, date, datetime, timedelta
from typing import Any, ClassVar, Self
from urllib.parse import quote

import httpx

from atlas.agent.layout import FIRMS_MAP_KEY, quiet_http_client_logs, resolve_credential
from atlas.data.base import (
    BBox,
    DataPullClient,
    GeometryKind,
    PullRequest,
    PullResult,
    Scene,
    SceneKind,
)
from atlas.data.credentials import (
    FIRMS_CREDENTIALS,
    MissingCredentialError,
    raise_for_missing_credential,
)

FIRMS_AREA_URL = (
    "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{source}/{bbox}/{days}/{date}"
)
_MAX_SPAN_DAYS = 10


class FirmsClient(DataPullClient):
    """Map FIRMS hotspot rows to point scenes. Requires ``FIRMS_MAP_KEY``.

    An explicit ``map_key`` wins. Otherwise the session value wins over the
    environment, which wins over ``~/.atlas/keys``. A missing key raises
    ``MissingCredentialError``. The message names ``FIRMS_MAP_KEY``, says to
    set it with ``/key firms`` or the env var, and does not include a value.
    A blank key is missing. A 401 or 403 with a key still present says that
    variable was rejected and includes the status, not the value. The key is
    scrubbed from any HTTP error URL.

    The key is percent-encoded as one URL path segment. httpx errors are
    re-raised without that URL, and the httpx logger stays at WARNING so an
    INFO request line cannot print the key.

    The area API takes ``west,south,east,north`` and requires west <= east.
    A dateline AOI is two area requests, one on each side of ±180°, and the
    rows are merged. An error document is raised so a rejected box is a source
    error rather than an empty scene list. HTTP 400 on one area request is
    skipped: NRT uses it when the requested day is not published yet. When
    every request was answered 400, nothing was searched, so that is a
    source error too. An empty window must be a CSV with no rows, or a pull
    would retire the detections table of the same window.
    """

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
        quiet_http_client_logs()
        if map_key is not None:
            self._map_key = map_key.strip()
        else:
            self._map_key = (resolve_credential(FIRMS_MAP_KEY) or "").strip()
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
        """Request each side of a dateline AOI and merge the hotspot rows.

        Raises:
            MissingCredentialError: ``FIRMS_MAP_KEY`` is unset. The message
                names that variable, says how to set it, and does not include
                a value.
            CredentialRejectedError: FIRMS returned 401 or 403 and the map
                key is present. The message names the variable and the status.
            ValueError: The box cannot be cut at 180°, FIRMS returns an
                error document instead of CSV, or FIRMS rejected every area
                request with HTTP 400.
        """
        if not self._map_key:
            raise MissingCredentialError(FIRMS_MAP_KEY)
        parts = request.bbox.antimeridian_parts()
        scenes: list[Scene] = []
        seen: set[str] = set()
        sent = rejected = 0
        start_date, end_date = _nrt_window(request.start_date, request.end_date)
        if start_date > end_date:
            return PullResult(request=request, scenes=[])
        for part in parts:
            if _at_limit(len(scenes), request.limit):
                break
            bbox = _area_param(part)
            for start, days in _chunk_days(start_date, end_date):
                # The row loop's break does not leave this loop. A later chunk
                # would send another area request and append more hotspots.
                if _at_limit(len(scenes), request.limit):
                    break
                url = FIRMS_AREA_URL.format(
                    key=quote(self._map_key, safe=""),
                    source=self.source,
                    bbox=bbox,
                    days=days,
                    date=start.isoformat(),
                )
                resp = await self._get(url)
                sent += 1
                raise_for_missing_credential(
                    resp,
                    FIRMS_CREDENTIALS,
                    satisfied=FIRMS_CREDENTIALS if self._map_key else (),
                )
                if resp.status_code == 400:
                    # NRT rejects "today" / future dates depending on FIRMS' calendar.
                    rejected += 1
                    continue
                self._raise_for_status(resp)
                filled = False
                for scene in _rows_to_scenes(
                    resp.text,
                    source=self.source,
                    platform=self.platform_name,
                    instrument=self.instrument_name,
                    kind=self.scene_kind,
                    gsd_m=self.nominal_gsd_m,
                ):
                    if scene.id in seen:
                        continue
                    seen.add(scene.id)
                    scenes.append(scene)
                    if _at_limit(len(scenes), request.limit):
                        filled = True
                        break
                if filled:
                    break
        if sent and rejected == sent:
            raise ValueError(
                f"FIRMS rejected all {sent} area request(s) for this window (HTTP 400); "
                "the dates may not be published yet"
            )
        return PullResult(request=request, scenes=scenes)

    async def _get(self, url: str) -> httpx.Response:
        """GET ``url`` and re-raise httpx failures without the map key."""
        try:
            return await self._client.get(url)
        except httpx.HTTPError as exc:
            raise _scrubbed_http_error(exc, self._map_key) from None

    def _raise_for_status(self, response: httpx.Response) -> None:
        """Raise a status error whose message and URL omit the map key."""
        try:
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise _scrubbed_http_error(exc, self._map_key) from None


def _scrub_secret(text: str, key: str) -> str:
    """Replace ``key`` and its percent-encoded form with ``***``."""
    if not key:
        return text
    encoded = quote(key, safe="")
    scrubbed = text.replace(encoded, "***")
    if encoded != key:
        scrubbed = scrubbed.replace(key, "***")
    return scrubbed


def _scrubbed_request(request: httpx.Request, key: str) -> httpx.Request:
    """Return a request whose URL no longer contains ``key``."""
    return httpx.Request(request.method, _scrub_secret(str(request.url), key))


def _scrubbed_http_error(exc: httpx.HTTPError, key: str) -> httpx.HTTPError:
    """Rebuild ``exc`` so its message and request URL omit ``key``.

    The new error is not chained to ``exc``. A traceback of the rebuilt error
    does not include the original message, which carries the FIRMS URL.
    """
    message = _scrub_secret(str(exc), key)
    original = getattr(exc, "_request", None)
    request = _scrubbed_request(original, key) if isinstance(original, httpx.Request) else None
    if isinstance(exc, httpx.HTTPStatusError):
        status_request = request if request is not None else _scrubbed_request(exc.request, key)
        response = httpx.Response(exc.response.status_code, request=status_request)
        return httpx.HTTPStatusError(message, request=status_request, response=response)
    if isinstance(exc, httpx.RequestError):
        return type(exc)(message, request=request)
    return httpx.HTTPError(message)


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


def _at_limit(count: int, limit: int | None) -> bool:
    return limit is not None and count >= limit


def _area_param(bbox: BBox) -> str:
    """FIRMS area path segment. West must not pass east."""
    if bbox.west > bbox.east:
        raise ValueError(f"FIRMS area expects west <= east; got west={bbox.west} east={bbox.east}")
    return f"{bbox.west},{bbox.south},{bbox.east},{bbox.north}"


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
    if not body:
        return []
    lowered = body.lower()
    if lowered.startswith("invalid") or "<html" in lowered:
        raise ValueError("FIRMS returned an error document instead of hotspot CSV")
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
        properties=_row_properties(row, source=source, lat=lat, lon=lon),
    )


def _row_properties(row: dict[str, Any], *, source: str, lat: float, lon: float) -> dict[str, Any]:
    """Keep every FIRMS column so a detection table loses nothing.

    VIIRS rows carry ``bright_ti4``/``bright_ti5``; MODIS rows carry
    ``brightness``/``bright_t31``. Column values stay as FIRMS sent them,
    except ``latitude`` and ``longitude``, which are the parsed floats.
    ``frp``, ``confidence``, ``daynight``, and ``bright_ti4`` are always
    present (``None`` when the product has no such column).
    """
    properties: dict[str, Any] = {
        key: value for key, value in row.items() if isinstance(key, str) and key
    }
    for key in ("frp", "confidence", "daynight", "bright_ti4"):
        properties.setdefault(key, None)
    properties["source"] = source
    properties["latitude"] = lat
    properties["longitude"] = lon
    return properties
