from __future__ import annotations

import asyncio
import csv
import math
import os
import re
from bisect import bisect_left, bisect_right
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Self
from urllib.parse import quote

import httpx
from dotenv import load_dotenv

from atlas.data.base import (
    Asset,
    BBox,
    DataPullClient,
    GeometryKind,
    PullRequest,
    PullResult,
    Scene,
    SceneKind,
)

CMR_SEARCH_URL = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"

SHORT_NAME = "FLDAS_NOAH01_C_GL_M"
VERSION = "001"
GSD_M = 11132.0
FILL_VALUE = -9999.0

# The monthly Noah grid stops at 60S; nothing south of that parallel has cells.
GRID_SOUTH_LAT = -60.0

# Nearest-cell lookup tolerates half a cell; beyond that the point is off-grid.
_HALF_CELL = 0.5
_AXIS_EPSILON = 1e-6

# Monthly Noah 0.1° grid. The axis read asks only for the first and last
# centers; the gateway often returns 502 when the whole axis is requested.
_X_LAST = 3599
_Y_LAST = 1499
_RETRY_STATUSES = frozenset({502, 503, 504})

DEFAULT_VARIABLES: tuple[str, ...] = (
    "SoilMoi00_10cm_tavg",
    "SoilMoi10_40cm_tavg",
    "SoilMoi40_100cm_tavg",
    "SoilMoi100_200cm_tavg",
    "Evap_tavg",
    "Qs_tavg",
    "Qsb_tavg",
    "SWE_inst",
    "SnowDepth_inst",
    "RadT_tavg",
)

KNOWN_VARIABLES: frozenset[str] = frozenset(
    {
        *DEFAULT_VARIABLES,
        "LWdown_f_tavg",
        "Lwnet_tavg",
        "Psurf_f_tavg",
        "Qair_f_tavg",
        "Qg_tavg",
        "Qh_tavg",
        "Qle_tavg",
        "Rainf_f_tavg",
        "SWdown_f_tavg",
        "SnowCover_inst",
        "Snowf_tavg",
        "Swnet_tavg",
        "Tair_f_tavg",
        "Wind_f_tavg",
        "SoilTemp00_10cm_tavg",
        "SoilTemp10_40cm_tavg",
        "SoilTemp40_100cm_tavg",
        "SoilTemp100_200cm_tavg",
    }
)

MISSING_TOKEN_MESSAGE = (
    "EARTHDATA_TOKEN needs to be set in .env for this session. "
    "Add it to the EARTHDATA_TOKEN line in the Atlas repo .env and run this command again."
)
_AUTH_HELP = (
    "EARTHDATA_TOKEN from .env was refused by Earthdata (401 or 403). "
    "At https://urs.earthdata.nasa.gov open Applications, Authorized Apps, "
    "Approve More Applications, and approve both Hyrax in the cloud and "
    "NASA GESDISC DATA ARCHIVE. Then run this command again."
)

_POINT_LON_KEYS = ("longitude", "lon", "x")
_POINT_LAT_KEYS = ("latitude", "lat", "y")


class FldasError(RuntimeError):
    """Raised when a granule cannot be read or a response cannot be parsed."""


class FldasAuthError(FldasError):
    """Raised when the OPeNDAP host demands an Earthdata login."""


class FldasRequestError(FldasError, ValueError):
    """Raised for a request the FLDAS grid cannot serve (a usage error)."""


@dataclass(frozen=True)
class _Axis:
    """One discovered 1-D grid axis."""

    name: str
    values: tuple[float, ...]

    @property
    def step(self) -> float:
        return (self.values[-1] - self.values[0]) / (len(self.values) - 1)


@dataclass(frozen=True)
class _Array:
    """One parsed DAP2 array: row-major values plus the dimension sizes."""

    name: str
    dims: tuple[int, ...]
    values: tuple[float, ...]


@dataclass(frozen=True)
class _Cell:
    """A row to emit: the reported lon/lat plus the grid index it reads from."""

    lon: float
    lat: float
    x: int
    y: int


@dataclass
class TrainingTable:
    """A CSV training table. Rows are already formatted as CSV strings."""

    columns: tuple[str, ...]
    rows: list[list[str]] = field(default_factory=list)
    path: Path | None = None
    skipped: list[str] = field(default_factory=list)

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    def write_csv(self, path: Path) -> Path:
        r"""Write the table to ``path`` (utf-8, ``\n`` endings), overwriting it."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(list(self.columns))
            writer.writerows(self.rows)
        self.path = path
        return path


def resolve_variables(variables: Sequence[str] | None) -> tuple[str, ...]:
    """Validate a variable selection, defaulting to the training columns.

    Raises:
        FldasRequestError: if a name is not a FLDAS variable.
    """
    if variables is None:
        return DEFAULT_VARIABLES
    names = tuple(name.strip() for name in variables if name.strip())
    if not names:
        raise FldasRequestError("no FLDAS variables requested")
    unknown = [name for name in names if name not in KNOWN_VARIABLES]
    if unknown:
        raise FldasRequestError(
            f"unknown FLDAS variable(s): {', '.join(unknown)}. "
            f"Known: {', '.join(sorted(KNOWN_VARIABLES))}"
        )
    return names


def parse_points_csv(path: Path) -> list[tuple[float, float]]:
    """Read a points CSV with a header into ``(longitude, latitude)`` pairs.

    The header may use longitude/latitude, lon/lat, or x/y, in any case.

    Raises:
        FldasRequestError: if the file is unreadable, headerless, or malformed.
    """
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = [row for row in csv.reader(handle) if any(cell.strip() for cell in row)]
    except OSError as exc:
        raise FldasRequestError(f"cannot read points file {path}: {exc}") from exc
    if not rows:
        raise FldasRequestError(f"points file {path} is empty")
    header = [cell.strip().lower() for cell in rows[0]]
    lon_index = _point_column(header, _POINT_LON_KEYS, path)
    lat_index = _point_column(header, _POINT_LAT_KEYS, path)
    points: list[tuple[float, float]] = []
    for number, row in enumerate(rows[1:], start=2):
        try:
            lon = float(row[lon_index])
            lat = float(row[lat_index])
        except (IndexError, ValueError) as exc:
            raise FldasRequestError(
                f"points file {path} row {number} is not a lon,lat pair: {exc}"
            ) from exc
        if not math.isfinite(lon) or not math.isfinite(lat):
            raise FldasRequestError(f"points file {path} row {number} is not a finite lon,lat pair")
        points.append((lon, lat))
    if not points:
        raise FldasRequestError(f"points file {path} has a header but no data rows")
    return points


def _point_column(header: list[str], keys: Sequence[str], path: Path) -> int:
    for key in keys:
        if key in header:
            return header.index(key)
    raise FldasRequestError(
        f"points file {path} needs a column named {' or '.join(keys)}; got: {', '.join(header)}"
    )


class FldasClient(DataPullClient):
    """FLDAS monthly Noah land-surface model, which is extracted over OPeNDAP.

    ``search`` is CMR-only and needs no credentials. ``extract_training_table``
    reads one hyper-slab per granule for every requested variable and cell
    together, so the 0.1° global NetCDF is never downloaded. The OPeNDAP axes
    are read once per extract and every granule reuses them, so nothing depends
    on hardcoded axis direction or endpoints.
    """

    short_name: ClassVar[str] = SHORT_NAME
    version: ClassVar[str] = VERSION
    satellite: ClassVar[str] = "fldas"
    scene_kind: ClassVar[SceneKind] = SceneKind.landsurface
    nominal_gsd_m: ClassVar[float | None] = GSD_M

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        timeout: float = 120.0,
        token: str | None = None,
        username: str | None = None,
        password: str | None = None,
        table_path: Path | None = None,
    ) -> None:
        load_repo_env()
        self._client = client or httpx.AsyncClient(
            timeout=timeout, follow_redirects=True, trust_env=True
        )
        self._owns_client = client is None
        self._token = token or os.environ.get("EARTHDATA_TOKEN") or None
        self._username = username or os.environ.get("EARTHDATA_USERNAME") or None
        self._password = password or os.environ.get("EARTHDATA_PASSWORD") or None
        self.table_path = table_path

    @property
    def collection(self) -> str:
        return self.short_name

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.aclose()

    async def search(self, request: PullRequest) -> PullResult:
        """Search CMR for FLDAS granules (metadata only, no credentials)."""
        reject_wrapping_bbox(request.bbox)
        params: dict[str, Any] = {
            "short_name": self.short_name,
            "version": self.version,
            "bounding_box": (
                f"{request.bbox.west},{request.bbox.south},{request.bbox.east},{request.bbox.north}"
            ),
            "temporal": (
                f"{request.start_date.isoformat()}T00:00:00Z,"
                f"{request.end_date.isoformat()}T23:59:59Z"
            ),
            "page_size": request.limit,
        }
        resp = await self._client.get(CMR_SEARCH_URL, params=params)
        resp.raise_for_status()
        payload = resp.json()
        if not isinstance(payload, dict):
            raise TypeError(f"Expected a CMR object, got {type(payload)}")
        scenes: list[Scene] = []
        for item in payload.get("items") or []:
            if not isinstance(item, dict):
                continue
            scene = _granule_to_scene(self.short_name, self.version, item)
            if scene is not None:
                scenes.append(scene)
        return PullResult(request=request, scenes=scenes)

    async def extract_training_table(
        self,
        request: PullRequest,
        scenes: Sequence[Scene],
        *,
        points: Sequence[tuple[float, float]] | None = None,
        variables: Sequence[str] | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> TrainingTable:
        """Subset every granule once and return the training rows.

        One OPeNDAP request per granule covers all variables and all cells; the
        constraint window is the smallest axis span the cells need. A granule
        that still fails after the request retries is skipped. Months that
        succeeded are kept, and the table is rewritten after each one when
        ``table_path`` is set. An authentication failure stops the run.

        Raises:
            FldasRequestError: if the request cannot be served by the grid.
            FldasAuthError: if Earthdata credentials are missing or rejected.
            FldasError: if every granule failed.
        """
        reject_wrapping_bbox(request.bbox)
        names = resolve_variables(variables)
        if not scenes:
            raise FldasError("no FLDAS granules to extract; the search returned none")
        if points is not None and not points:
            raise FldasRequestError("no points to extract; the points file had no rows")
        if points is not None:
            for lon, lat in points:
                if not request.bbox.contains(lon, lat):
                    raise FldasRequestError(f"point ({lon}, {lat}) is outside the request bbox")
        if not self._token:
            raise FldasAuthError(MISSING_TOKEN_MESSAGE)

        base_url = _opendap_href(scenes[0])
        x_axis, y_axis = await self._read_axes(base_url)
        cells = _plan_cells(request.bbox, x_axis, y_axis, points)
        x_lo = min(cell.x for cell in cells)
        x_hi = max(cell.x for cell in cells)
        y_lo = min(cell.y for cell in cells)
        y_hi = max(cell.y for cell in cells)

        columns = _columns(names, with_cells=points is not None)
        rows: list[list[str]] = []
        skipped: list[str] = []
        ordered = sorted(scenes, key=lambda scene: (scene.datetime, scene.id))
        for index, scene in enumerate(ordered, start=1):
            stamp = _utc(scene.datetime).date().isoformat()
            try:
                arrays = await self._read_granule(scene, names, x_lo, x_hi, y_lo, y_hi)
            except FldasAuthError:
                raise
            except (FldasError, httpx.HTTPError, OSError) as exc:
                skipped.append(f"{stamp} ({scene.id}): {exc}")
                if on_progress is not None:
                    on_progress(index, len(ordered))
                continue
            for cell in cells:
                row = [_number(cell.lon), _number(cell.lat)]
                if points is not None:
                    row.extend((_number(x_axis.values[cell.x]), _number(y_axis.values[cell.y])))
                row.append(stamp)
                row.extend(
                    _cell_text(_array_value(arrays, name, cell, x_lo, x_hi, y_lo, y_hi))
                    for name in names
                )
                rows.append(row)
            if self.table_path is not None and rows:
                TrainingTable(columns=columns, rows=rows, skipped=list(skipped)).write_csv(
                    self.table_path
                )
            if on_progress is not None:
                on_progress(index, len(ordered))

        table = TrainingTable(columns=columns, rows=rows, skipped=skipped)
        if not rows:
            raise FldasError(_skip_note(skipped))
        return table

    async def _read_axes(self, base_url: str) -> tuple[_Axis, _Axis]:
        """Read the first and last center of each axis, then rebuild the grid.

        A full ``X,Y`` projection makes the cloud gateway read the whole file
        and often return 502. Two centers are enough for this regular 0.1° grid.
        A response that already lists every center is kept as-is.
        """
        url = _opendap_ascii_url(base_url, _axis_constraint())
        arrays = parse_dap2(await self._get_text(url), "grid axes")
        return (
            _expand_sampled_axis(_axis(arrays, "X"), _X_LAST),
            _expand_sampled_axis(_axis(arrays, "Y"), _Y_LAST),
        )

    async def _read_granule(
        self,
        scene: Scene,
        names: Sequence[str],
        x_lo: int,
        x_hi: int,
        y_lo: int,
        y_hi: int,
    ) -> dict[str, _Array]:
        parts = [f"{name}[0:0][{y_lo}:{y_hi}][{x_lo}:{x_hi}]" for name in names]
        parts.extend((f"X[{x_lo}:{x_hi}]", f"Y[{y_lo}:{y_hi}]"))
        url = _opendap_ascii_url(_opendap_href(scene), ",".join(parts))
        return parse_dap2(await self._get_text(url), scene.id)

    async def _get_text(self, url: str) -> str:
        last_status = 0
        for attempt in range(3):
            headers, auth = self._request_auth()
            try:
                resp = await self._client.get(url, headers=headers, auth=auth)
            except httpx.TimeoutException:
                if attempt < 2:
                    await asyncio.sleep(0.5 * (attempt + 1))
                    continue
                raise FldasError("Earthdata OPeNDAP timed out. Run the command again.") from None
            if resp.status_code == 401 or _looks_like_html(resp.text):
                raise FldasAuthError(_AUTH_HELP)
            if resp.status_code == 403 and not _transient_archive(resp.text):
                raise FldasAuthError(_AUTH_HELP)
            if resp.status_code in _RETRY_STATUSES or (
                resp.status_code == 403 and _transient_archive(resp.text)
            ):
                last_status = resp.status_code
                if attempt < 2:
                    await asyncio.sleep(0.5 * (attempt + 1))
                    continue
                raise FldasError(
                    f"Earthdata OPeNDAP returned {resp.status_code}. Run the command again."
                )
            resp.raise_for_status()
            text = resp.text
            if _looks_like_html(text):
                raise FldasAuthError(_AUTH_HELP)
            return text
        raise FldasError(f"Earthdata OPeNDAP returned {last_status}. Run the command again.")

    def _request_auth(self) -> tuple[dict[str, str], httpx.Auth | None]:
        """Bearer token first, then basic auth; netrc stays with the http client."""
        if self._token:
            return {"Authorization": f"Bearer {self._token}"}, None
        if self._username and self._password:
            return {}, httpx.BasicAuth(self._username, self._password)
        return {}, None


def load_repo_env() -> None:
    """Load the Atlas repo ``.env`` without overriding variables already set."""
    root = _atlas_repo_root()
    if root is not None:
        load_dotenv(root / ".env", override=False)
        return
    load_dotenv(override=False)


def _atlas_repo_root() -> Path | None:
    """Walk the working directory and this file's parents for the Atlas checkout."""
    seen: set[Path] = set()
    for start in (Path.cwd().resolve(), Path(__file__).resolve()):
        for directory in (start, *start.parents):
            if directory in seen:
                continue
            seen.add(directory)
            pyproject = directory / "pyproject.toml"
            try:
                text = pyproject.read_text(encoding="utf-8")
            except OSError:
                continue
            if 'name = "atlas"' in text:
                return directory
    return None


def reject_wrapping_bbox(bbox: BBox) -> None:
    """Reject a bbox that crosses the antimeridian.

    Raises:
        FldasRequestError: if ``west > east``.
    """
    if bbox.west > bbox.east:
        raise FldasRequestError(
            f"FLDAS needs a west<=east bounding box; got west={bbox.west} east={bbox.east} "
            "(a box that crosses the antimeridian cannot be sent to CMR or OPeNDAP)"
        )


def _granule_to_scene(short_name: str, version: str, item: dict[str, Any]) -> Scene | None:
    raw_meta = item.get("meta")
    raw_umm = item.get("umm")
    meta: dict[str, Any] = raw_meta if isinstance(raw_meta, dict) else {}
    umm: dict[str, Any] = raw_umm if isinstance(raw_umm, dict) else {}

    granule_ur = umm.get("GranuleUR")
    concept_id = meta.get("concept-id")
    scene_id = granule_ur if isinstance(granule_ur, str) and granule_ur else concept_id
    scene_dt = _begin_datetime(umm)
    bbox = _rectangle_bbox(umm)
    if not isinstance(scene_id, str) or not scene_id or scene_dt is None or bbox is None:
        return None

    return Scene.try_new(
        id=scene_id,
        collection=short_name,
        kind=SceneKind.landsurface,
        datetime=scene_dt,
        start_datetime=scene_dt,
        end_datetime=_parse_datetime(_temporal(umm).get("EndingDateTime")) or scene_dt,
        bbox=bbox,
        geometry_kind=GeometryKind.bbox,
        gsd_m=GSD_M,
        platform="FLDAS",
        instrument="Noah",
        assets=_opendap_assets(umm),
        properties=_properties(short_name, version, umm, concept_id),
    )


def _temporal(umm: dict[str, Any]) -> dict[str, Any]:
    temporal = umm.get("TemporalExtent")
    if not isinstance(temporal, dict):
        return {}
    range_dt = temporal.get("RangeDateTime")
    return range_dt if isinstance(range_dt, dict) else {}


def _begin_datetime(umm: dict[str, Any]) -> datetime | None:
    range_dt = _temporal(umm)
    return _parse_datetime(range_dt.get("BeginningDateTime")) or _parse_datetime(
        range_dt.get("EndingDateTime")
    )


def _parse_datetime(raw: object) -> datetime | None:
    if not isinstance(raw, str) or not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return _utc(parsed)


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _rectangle_bbox(umm: dict[str, Any]) -> BBox | None:
    spatial = umm.get("SpatialExtent")
    domain = spatial.get("HorizontalSpatialDomain") if isinstance(spatial, dict) else None
    geometry = domain.get("Geometry") if isinstance(domain, dict) else None
    rects = geometry.get("BoundingRectangles") if isinstance(geometry, dict) else None
    if not isinstance(rects, list) or not rects or not isinstance(rects[0], dict):
        return None
    rect = rects[0]
    try:
        return BBox(
            west=float(rect["WestBoundingCoordinate"]),
            south=float(rect["SouthBoundingCoordinate"]),
            east=float(rect["EastBoundingCoordinate"]),
            north=float(rect["NorthBoundingCoordinate"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _related_urls(umm: dict[str, Any]) -> list[dict[str, Any]]:
    related = umm.get("RelatedUrls")
    if not isinstance(related, list):
        return []
    return [entry for entry in related if isinstance(entry, dict)]


def _opendap_assets(umm: dict[str, Any]) -> dict[str, Asset]:
    for entry in _related_urls(umm):
        url = entry.get("URL")
        if not isinstance(url, str) or not url.startswith("https://"):
            continue
        if _is_opendap_entry(entry):
            asset = Asset.try_new(
                href=url,
                media_type="application/x-netcdf",
                title="OPeNDAP subset",
                roles=["data"],
            )
            if asset is not None:
                return {"opendap": asset}
    return {}


def _is_opendap_entry(entry: dict[str, Any]) -> bool:
    for key in ("Type", "Subtype"):
        value = entry.get(key)
        if isinstance(value, str) and "OPENDAP" in value.upper():
            return True
    return False


def _properties(
    short_name: str,
    version: str,
    umm: dict[str, Any],
    concept_id: object,
) -> dict[str, Any]:
    props: dict[str, Any] = {
        "short_name": short_name,
        "version": version,
        "full_file_url": _full_file_url(umm),
    }
    if isinstance(concept_id, str) and concept_id:
        props["concept_id"] = concept_id
    return props


def _full_file_url(umm: dict[str, Any]) -> str | None:
    """The whole-granule download URL. Kept for provenance only; never fetched."""
    for entry in _related_urls(umm):
        url = entry.get("URL")
        if entry.get("Type") == "GET DATA" and isinstance(url, str) and url.startswith("https://"):
            return url
    return None


def _opendap_href(scene: Scene) -> str:
    asset = scene.assets.get("opendap")
    if asset is None:
        raise FldasError(f"granule {scene.id} has no OPeNDAP URL")
    return _dap_base(asset.href)


def _dap_base(url: str) -> str:
    base = url.strip()
    while base.endswith(".ascii"):
        base = base[: -len(".ascii")]
    return base


def _skip_note(skipped: list[str]) -> str:
    shown = "; ".join(skipped[:8])
    extra = f" (+{len(skipped) - 8} more)" if len(skipped) > 8 else ""
    return f"skipped {len(skipped)} month(s): {shown}{extra}"


def _axis_constraint() -> str:
    """First and last center of each axis. DAP2 stop indices are inclusive."""
    return f"X[0:{_X_LAST}:{_X_LAST}],Y[0:{_Y_LAST}:{_Y_LAST}]"


def _opendap_ascii_url(base: str, constraint: str) -> str:
    """Build an ASCII URL. Brackets are encoded so the gateway accepts them."""
    return f"{base}.ascii?{quote(constraint, safe=',:')}"


def _expand_sampled_axis(axis: _Axis, last_index: int) -> _Axis:
    """Rebuild a regular axis when the response is only its two endpoints."""
    if len(axis.values) != 2:
        return axis
    first, last = axis.values
    step = (last - first) / last_index
    values = tuple(round(first + index * step, 5) for index in range(last_index + 1))
    return _Axis(name=axis.name, values=values)


def _axis(arrays: dict[str, _Array], name: str) -> _Axis:
    array = arrays.get(name)
    if array is None or len(array.dims) != 1 or len(array.values) < 2:
        raise FldasError(f"the OPeNDAP response has no usable {name} axis")
    return _Axis(name=name, values=array.values)


def _plan_cells(
    bbox: BBox,
    x_axis: _Axis,
    y_axis: _Axis,
    points: Sequence[tuple[float, float]] | None,
) -> list[_Cell]:
    """Resolve the rows to emit: requested points, or the covered cell centers."""
    if points is None:
        x_lo, x_hi = _cell_window(x_axis, bbox.west, bbox.east)
        y_lo, y_hi = _cell_window(y_axis, bbox.south, bbox.north)
        cells = [
            _Cell(lon=x_axis.values[x], lat=y_axis.values[y], x=x, y=y)
            for y in range(y_lo, y_hi + 1)
            for x in range(x_lo, x_hi + 1)
        ]
        return sorted(cells, key=lambda cell: (cell.lat, cell.lon))
    return [
        _Cell(
            lon=lon,
            lat=lat,
            x=_nearest_index(x_axis, lon, "longitude"),
            y=_nearest_index(y_axis, lat, "latitude"),
        )
        for lon, lat in points
    ]


def _cell_window(axis: _Axis, low: float, high: float) -> tuple[int, int]:
    """Inclusive index range of the axis cells whose centers lie in [low, high]."""
    order = axis.values if axis.step > 0 else tuple(reversed(axis.values))
    first = bisect_left(order, low)
    last = bisect_right(order, high) - 1
    if first > last:
        raise FldasRequestError(
            f"no {axis.name} grid-cell center falls inside the request bbox; "
            "widen the bbox or pick a different area"
        )
    if axis.step < 0:
        size = len(axis.values)
        return size - 1 - last, size - 1 - first
    return first, last


def _nearest_index(axis: _Axis, value: float, label: str) -> int:
    """Nearest axis index, rejecting values more than half a cell off-grid."""
    if axis.name == "Y" and value < GRID_SOUTH_LAT:
        raise FldasRequestError(
            f"latitude {value} is south of the FLDAS grid limit ({GRID_SOUTH_LAT})"
        )
    raw = (value - axis.values[0]) / axis.step
    index = min(max(round(raw), 0), len(axis.values) - 1)
    center = axis.values[index]
    if abs(value - center) > abs(axis.step) * _HALF_CELL + _AXIS_EPSILON:
        raise FldasRequestError(
            f"{label} {value} is more than half a cell outside the FLDAS grid "
            f"(nearest cell center {center})"
        )
    return index


def _columns(names: Sequence[str], *, with_cells: bool) -> tuple[str, ...]:
    head: tuple[str, ...] = ("longitude", "latitude")
    if with_cells:
        head = (*head, "cell_longitude", "cell_latitude")
    return (*head, "time", *names)


def _array_value(
    arrays: dict[str, _Array],
    name: str,
    cell: _Cell,
    x_lo: int,
    x_hi: int,
    y_lo: int,
    y_hi: int,
) -> float:
    array = arrays.get(name)
    if array is None:
        raise FldasError(f"the OPeNDAP response has no {name} values")
    expected = (1, y_hi - y_lo + 1, x_hi - x_lo + 1)
    if array.dims != expected or len(array.values) != expected[0] * expected[1] * expected[2]:
        raise FldasError(
            f"{name} came back as {array.dims} with {len(array.values)} values, expected {expected}"
        )
    n_x = expected[2]
    return array.values[(cell.y - y_lo) * n_x + (cell.x - x_lo)]


def _cell_text(value: float) -> str:
    return "" if value == FILL_VALUE else _number(value)


def _number(value: float) -> str:
    return repr(float(value))


_HEADER_RE = re.compile(r"^(?P<name>[A-Za-z_][A-Za-z0-9_.]*)(?P<dims>(?:\[\d+\])+)\s*;?\s*$")
_DIM_RE = re.compile(r"\[(\d+)\]")
_INDEX_PREFIX_RE = re.compile(r"^(?:\[\d+\])+\s*,?\s*")
_NUMBER_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_NUMERIC_LINE_RE = re.compile(r"^[\s\[\].,+\-0-9eE]+$")


def parse_dap2(text: str, label: str) -> dict[str, _Array]:
    """Parse an OPeNDAP ASCII response into ``name -> _Array``.

    Classic DAP2 headers look like ``name.name[d0][d1]``. The cloud gateway
    instead writes ``Dataset:`` and one CSV row per latitude.
    """
    if any(line.strip().startswith("Dataset:") for line in text.splitlines()):
        return _parse_cloud_ascii(text, label)
    arrays: dict[str, _Array] = {}
    name: str | None = None
    dims: tuple[int, ...] = ()
    numbers: list[float] = []

    def finish() -> None:
        nonlocal name, dims, numbers
        if name is not None:
            array = _Array(name=name, dims=dims, values=tuple(numbers))
            # `Var.Var` is the data array; `Var.X` is only that grid's map.
            # Keep the map under its full name so it cannot replace `Var`.
            key = _data_name(name)
            existing = arrays.get(key)
            if existing is None or len(array.values) > len(existing.values):
                arrays[key] = array
        name, dims, numbers = None, (), []

    for raw in text.splitlines():
        line = raw.strip()
        header = _HEADER_RE.match(line)
        if header is not None:
            finish()
            name = header.group("name")
            dims = tuple(int(match.group(1)) for match in _DIM_RE.finditer(header.group("dims")))
            continue
        if name is None or not line:
            continue
        if '"' in line:
            # Quoted strings are not numeric data; drop the block instead of mis-reading it.
            finish()
            continue
        if _NUMERIC_LINE_RE.match(line):
            body = _INDEX_PREFIX_RE.sub("", line)
            numbers.extend(float(match.group(0)) for match in _NUMBER_RE.finditer(body))
            continue
        finish()
    finish()
    if not arrays:
        raise FldasError(f"could not read DAP2 ASCII arrays from the {label} response")
    return arrays


_CLOUD_ROW_RE = re.compile(r"^(?P<name>[A-Za-z_][A-Za-z0-9_.]*)(?:\[[^\]]*\])*,\s*(?P<rest>.*)$")


def _parse_cloud_ascii(text: str, label: str) -> dict[str, _Array]:
    """Parse the cloud gateway's ``Dataset:`` ASCII into arrays.

    Coordinate lines are ``X, lon, lon``. A data line names the latitude in
    brackets and lists longitude samples after the comma, west to east.
    """
    axes: dict[str, tuple[float, ...]] = {}
    rows: dict[str, list[tuple[float, ...]]] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("Dataset:"):
            continue
        matched = _CLOUD_ROW_RE.match(line)
        if matched is None:
            continue
        raw_name = matched.group("name")
        numbers = tuple(float(item.group(0)) for item in _NUMBER_RE.finditer(matched.group("rest")))
        if not numbers:
            continue
        if "[" in line.split(",", 1)[0]:
            rows.setdefault(_data_name(raw_name), []).append(numbers)
            continue
        axis_name = _cloud_axis_name(raw_name)
        if axis_name is None:
            continue
        current = axes.get(axis_name)
        if current is None or len(numbers) > len(current):
            axes[axis_name] = numbers
    arrays: dict[str, _Array] = {
        name: _Array(name=name, dims=(len(values),), values=values) for name, values in axes.items()
    }
    for name, samples in rows.items():
        width = len(samples[0])
        if any(len(sample) != width for sample in samples):
            raise FldasError(f"{name} rows in the {label} response do not share a width")
        flat = tuple(value for sample in samples for value in sample)
        arrays[name] = _Array(name=name, dims=(1, len(samples), width), values=flat)
    if not arrays:
        raise FldasError(f"could not read DAP2 ASCII arrays from the {label} response")
    return arrays


def _cloud_axis_name(raw_name: str) -> str | None:
    if raw_name in {"X", "Y"}:
        return raw_name
    if raw_name.endswith(".X"):
        return "X"
    if raw_name.endswith(".Y"):
        return "Y"
    return None


def _data_name(raw_name: str) -> str:
    """Fold ``X.X`` and ``Var.Var`` onto ``X`` and ``Var``; leave ``Var.X`` alone."""
    if "." not in raw_name:
        return raw_name
    head, _, tail = raw_name.partition(".")
    if "." not in tail and head == tail:
        return head
    return raw_name


def _transient_archive(text: str) -> bool:
    """True when Hyrax itself could not read the archive, not when login failed."""
    lowered = text.lower()
    return "dmrpp" in lowered or "daac_bucket" in lowered or "bad gateway" in lowered


def _looks_like_html(text: str) -> bool:
    head = text.lstrip()[:256].lower()
    return head.startswith("<!doctype") or head.startswith("<html") or "<html" in head


def split_variables(text: str) -> list[str]:
    """Split a comma-separated variable list, dropping blanks."""
    return [item.strip() for item in text.split(",") if item.strip()]
