from __future__ import annotations

import asyncio
import contextlib
import email.utils
import enum
import hashlib
import json
import os
import re
import time
from collections import defaultdict
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

import httpx

from atlas.cli.fs import replace_with_retry
from atlas.cli.preview import (
    clear_preview,
    preview_path,
    write_file_preview,
    write_read_preview,
)
from atlas.data.base import Asset, BBox, DataPullClient, Scene
from atlas.data.cog import (
    CogError,
    CogLimitError,
    CogMissError,
    CogRangeError,
    CogRead,
    CogUnsupportedError,
    encode_cog,
    model_bounds,
    open_cog,
)
from atlas.data.profiles import AssetNotFound, calibration_for, epsg_hint, resolve_asset
from atlas.geo.chipmeta import (
    BandInfo,
    Calibration,
    ChipMeta,
    FileInfo,
    RasterInfo,
    read_chip_meta,
    sidecar_path,
    write_chip_meta,
)
from atlas.geo.errors import GeoError

_SAFE_ID = re.compile(r"[^A-Za-z0-9._-]+")
_BROWSE_KEYS = frozenset({"rendered_preview", "thumbnail", "preview", "quicklook", "browse"})
_BROWSE_ROLES = frozenset({"thumbnail", "overview", "preview", "rendered_preview"})
_PREFERRED_KEYS = ("visual",)
# Roles and key fragments of quality, mask, and geometry layers. A default pick
# skips them so an optical download is imagery, not a QA band or cloud mask.
_AUXILIARY_ROLES = frozenset(
    {"cloud", "cloud-shadow", "snow-ice", "water-mask", "data-mask", "saturation", "metadata"}
)
_AUXILIARY_PREFIXES = ("qa", "oa_", "tie_", "tie-")
_AUXILIARY_PARTS = (
    "mask",
    "cloud",
    "angle",
    "azimuth",
    "zenith",
    "meteo",
    "coordinates",
    "geometr",
    "flags",
    "quality",
    "manifest",
    "removed",
    "instrument",
)
_AUXILIARY_KEYS = frozenset({"ang", "saa", "sza", "vaa", "vza", "aot", "wvp", "scl"})
_RASTER_SUFFIXES = frozenset(
    {".tif", ".tiff", ".jp2", ".nc", ".hdf", ".h5", ".zip", ".las", ".laz"}
)
_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp"})
# Suffixes checked when a scene asset may already be on disk under a name the
# catalog does not predict (a response content type chose it).
KNOWN_SUFFIXES = (
    ".tif",
    ".tiff",
    ".bin",
    ".jp2",
    ".nc",
    ".hdf",
    ".h5",
    ".zip",
    ".png",
    ".jpg",
    ".dat.bz2",
    ".grib2.gz",
)
# A windowed chip gets a small sibling that records the AOI it was read for,
# so a later pull of the same area can skip it. The georeferenced file itself
# is not modified.
AOI_MARKER_SUFFIX = ".aoi.json"
_MAX_NAME_BYTES = 240
_WINDOWS_RESERVED = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        "CONIN$",
        "CONOUT$",
        *(f"COM{number}" for number in range(1, 10)),
        *(f"LPT{number}" for number in range(1, 10)),
    }
)
_COMPRESSED_SUFFIXES = frozenset({".bz2", ".gz"})
# A whole-file GET that receives fewer than ``STALL_MIN_BYTES`` in
# ``STALL_SECONDS`` is aborted. A trickling server would otherwise hold the
# run for hours, since the socket read timeout resets on every byte.
STALL_SECONDS = 120.0
STALL_MIN_BYTES = 64 * 1024
# Bounded retry of a whole-file GET answered 429 or 5xx.
GET_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 2.0
RETRY_AFTER_CAP_SECONDS = 60.0
# Bounded retry of an href signing call (Planetary Computer SAS) answered 429,
# 5xx, or a transport error. Same budget as the COG range reader.
SIGN_ATTEMPTS = 8
SIGN_BACKOFF_SECONDS = 0.5
SIGN_WAIT_CAP_SECONDS = 16.0
# Whole-file downloads are written under this sibling prefix and renamed on
# success, so a killed run never leaves a truncated file at the final name.
PARTIAL_PREFIX = "."
PARTIAL_SUFFIX = ".part"
# Agent-facing limits: one window per asset is capped at 36 megapixels, a whole
# file larger than 200 MB needs an explicit opt-in, and a whole-file fallback for
# a window that failed transiently is abandoned after 64 MiB.
WINDOW_MAX_PIXELS = 36_000_000
WHOLE_FILE_CAP_BYTES = 200_000_000
WINDOW_INTENDED_ABORT_BYTES = 64 * 1024 * 1024
_CHIP_TILE = 512
_SIDECAR_PROPERTIES = ("eo:cloud_cover", "s2:processing_baseline", "earthsearch:boa_offset_applied")
_SIDECAR_PROPERTY_PREFIXES = ("sat:", "sar:")
_REDACTED = "***"
_URL_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://[^\s\"'<>]+")
# FIRMS puts MAP_KEY in the path: /api/area/csv/<key>/<source>/<area>/...
_FIRMS_KEY_PATH = re.compile(r"(/api/(?:area|country)/csv/)[^/]+", re.IGNORECASE)
# A credential-shaped parameter outside a URL, e.g. echoed in a message body.
_SECRET_PARAM = re.compile(
    r"(?i)(?<![A-Za-z0-9_-])"
    r"(sig|se|st|sp|sv|sr|skoid|sktid|skt|ske|sks|skv|token|access_token|id_token|"
    r"refresh_token|api_key|apikey|key|map_key|password|x-amz-[a-z-]+)"
    r"=([^&\s\"'<>]+)"
)
_BEARER = re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/=-]+")


class BrowseAssetError(ValueError):
    """Raised when the requested asset is a preview or thumbnail."""


class DownloadStallError(OSError):
    """Raised when a whole-file GET stops making progress."""


class _WholeFileAbort(OSError):
    """A whole-file fallback for an intended window passed the abort size."""


class _WholeFileRefused(Exception):
    """A whole-file fetch was refused for size; carries the final error text."""


class _Stop(Exception):
    """Cancellation or the wall-clock cap tripped; ``status`` is the outcome status."""

    def __init__(self, status: str) -> None:
        super().__init__(status)
        self.status = status


def redact_url(url: str) -> str:
    """Return ``url`` without credentials.

    The query string and fragment are replaced by ``***`` (SAS ``sig``/``se``,
    ``X-Amz-*``, ``token``, and ``key`` all live there), user info is
    dropped, and a FIRMS ``MAP_KEY`` path segment is masked. Scheme, host,
    and path stay so the failure still names the file.
    """
    try:
        parts = urlsplit(url)
    except ValueError:
        return _REDACTED
    netloc = parts.netloc.rsplit("@", 1)[-1]
    path = _FIRMS_KEY_PATH.sub(rf"\1{_REDACTED}", parts.path)
    query = _REDACTED if parts.query else ""
    fragment = _REDACTED if parts.fragment else ""
    return urlunsplit((parts.scheme, netloc, path, query, fragment))


def redact_secrets(text: str) -> str:
    """Mask URL query strings, FIRMS path keys, and credential parameters in ``text``.

    Applied to every error that is rendered or persisted (CLI table,
    ``DownloadOutcome.error``, ``manifest.json``). httpx puts the full
    request URL in ``HTTPStatusError`` text, and a signed href carries its
    token in the query.
    """
    text = _URL_RE.sub(lambda match: redact_url(match.group(0)), text)
    text = _SECRET_PARAM.sub(rf"\1={_REDACTED}", text)
    return _BEARER.sub(rf"\1 {_REDACTED}", text)


def describe_error(exc: BaseException) -> str:
    """Readable, never-empty, redacted text for a failed download.

    Some transport errors carry no message: an ``httpx.ReadTimeout`` raised
    from a bare ``TimeoutError`` stringifies to ``""``. The class name is
    always included so a failure is never rendered blank. URL queries and
    credential parameters are masked by ``redact_secrets``.
    """
    name = type(exc).__name__
    text = str(exc).strip()
    if not text and exc.__cause__ is not None:
        text = str(exc.__cause__).strip()
    return redact_secrets(f"{name}: {text}") if text else name


@dataclass
class DownloadOutcome:
    """One (scene, asset) download attempt.

    ``error`` and ``note`` are passed through ``redact_secrets`` on construction,
    so a signed href or key in an exception message never reaches the CLI table
    or ``manifest.json``. ``status`` is one of saved, skipped, failed, no_asset,
    cancelled, timeout (the last two only when a cancel or deadline is used).
    ``windowed`` is True for a chip, False for a whole file, None when no window
    was intended. ``bytes_fetched`` counts bytes over the network for this outcome.
    """

    source: str
    scene_id: str
    status: str
    path: Path | None = None
    error: str | None = None
    bytes_read: int | None = None
    granule_size: int | None = None
    preview: Path | None = None
    preview_error: str | None = None
    asset_key: str | None = None
    alias: str | None = None
    windowed: bool | None = None
    note: str | None = None
    bytes_fetched: int = 0

    def __post_init__(self) -> None:
        if self.error is not None:
            self.error = redact_secrets(self.error)
        if self.note is not None:
            self.note = redact_secrets(self.note)


@dataclass
class DownloadReport:
    """Per-scene results grouped for the CLI summary and manifest."""

    outcomes: list[DownloadOutcome] = field(default_factory=list)

    def saved_paths(self) -> dict[str, list[Path]]:
        grouped: dict[str, list[Path]] = defaultdict(list)
        for item in self.outcomes:
            if item.status in {"saved", "skipped"} and item.path is not None:
                grouped[item.source].append(item.path)
        return dict(grouped)

    def for_source(self, name: str) -> list[DownloadOutcome]:
        return [item for item in self.outcomes if item.source == name]


class _Unwindowable(enum.Enum):
    TOKEN = "unwindowable"


UNWINDOWABLE = _Unwindowable.TOKEN
"""``_windowed_cog`` result: the asset cannot be windowed, not a failed attempt."""


def is_browse_asset(key: str, asset: Asset) -> bool:
    """True for STAC browse products (previews, thumbnails, JPEG overviews)."""
    if key.lower() in _BROWSE_KEYS:
        return True
    roles = {role.lower() for role in asset.roles}
    if roles & _BROWSE_ROLES:
        return True
    media = (asset.media_type or "").lower()
    if media.startswith("image/jpeg") or media.startswith("image/jpg"):
        return True
    return media.startswith("image/png") and "data" not in roles


def is_forbidden_asset_name(name: str) -> bool:
    """True if ``--asset`` names a browse product."""
    return name.strip().lower() in _BROWSE_KEYS


def pick_asset(scene: Scene, name: str | None = None) -> tuple[str, Asset] | None:
    """Choose one native-resolution asset. Never returns a preview or thumbnail.

    Prefers ``visual`` (full-res RGB COG) when it is not browse, then the
    ``red`` band, then a ``reflectance`` asset, then a ``data`` file that is not
    a QA, mask, angle, or geometry layer, then a whole-product archive, then
    any ``data`` asset, then the first remaining non-browse asset. ``name`` pins a band alias such as ``red``,
    otherwise a literal asset key.

    Raises:
        BrowseAssetError: if ``name`` is a preview/thumbnail key or asset.
    """
    if name:
        if is_forbidden_asset_name(name):
            raise BrowseAssetError(f"{name!r} is a preview/thumbnail and cannot be downloaded")
        # Band names are reserved and beat a literal key of the same name, so
        # `nir` is ~865 nm everywhere. Earth Search keys B08 (0.842) as `nir`.
        key = scene.band_aliases.get(name.strip().lower(), name)
        asset = scene.assets.get(key)
        if asset is None:
            folded = key.strip().casefold()
            key = next((k for k in sorted(scene.assets) if k.casefold() == folded), key)
            asset = scene.assets.get(key)
        if asset is None:
            return None
        if is_browse_asset(key, asset):
            raise BrowseAssetError(f"{key!r} is a preview/thumbnail and cannot be downloaded")
        # The native key, not the alias, so the filename records the band fetched.
        return key, asset
    preferred = [*_PREFERRED_KEYS]
    if "red" in scene.band_aliases:
        preferred.append(scene.band_aliases["red"])
    for key in preferred:
        asset = scene.assets.get(key)
        if asset is not None and not is_browse_asset(key, asset):
            return key, asset
    data_hits = [
        (key, asset)
        for key, asset in scene.assets.items()
        if not is_browse_asset(key, asset) and "data" in _roles(asset)
    ]
    reflectance = [hit for hit in data_hits if "reflectance" in _roles(hit[1])]
    # A whole-product archive (CDSE ``product``) ranks after a single file.
    imagery = [hit for hit in data_hits if not _is_auxiliary(*hit)]
    files = [hit for hit in imagery if "archive" not in _roles(hit[1])]
    for hits in (reflectance, files, imagery, data_hits):
        if hits:
            return hits[0]
    for key, asset in scene.assets.items():
        if not is_browse_asset(key, asset):
            return key, asset
    return None


def _roles(asset: Asset) -> set[str]:
    return {role.lower() for role in asset.roles}


def _is_auxiliary(key: str, asset: Asset) -> bool:
    """True for a QA, mask, or sun/view geometry layer."""
    if _roles(asset) & _AUXILIARY_ROLES:
        return True
    lower = key.lower()
    if lower in _AUXILIARY_KEYS or lower.startswith(_AUXILIARY_PREFIXES):
        return True
    return any(part in lower for part in _AUXILIARY_PARTS)


def _safe_part(text: str) -> str:
    """Sanitize one filename part and avoid a reserved Windows device name."""
    cleaned = _SAFE_ID.sub("_", text.rstrip(" .")).rstrip(" .")
    return cleaned or "_"


def _avoid_reserved(name: str) -> str:
    """Prefix ``_`` when the stem is a reserved Windows device name."""
    return f"_{name}" if name.split(".", 1)[0].upper() in _WINDOWS_RESERVED else name


def scene_filename(scene_id: str, asset_key: str, suffix: str) -> str:
    """Return a portable filename: ``<id>_<key><suffix>``.

    A scene id that already ends in ``suffix`` (GOES ids are file names such
    as ``..._c2026152.nc``) drops that ending, so the extension is not
    repeated: ``..._c2026152_data.nc``, not ``..._c2026152.nc_data.nc``.
    Any other id, and the placeholder ``.bin`` suffix, keep the id whole.

    The name is limited to 240 bytes so a long CMR asset key or scene id does
    not fail with ``ENAMETOOLONG``; the shortened stem keeps a hash of the id
    so two long ids do not collide. A Windows reserved device name is prefixed
    with ``_`` and trailing dots or spaces are dropped.
    """
    if not suffix.startswith("."):
        suffix = f".{suffix}"
    base = scene_id
    if suffix.lower() != ".bin" and base.lower().endswith(suffix.lower()):
        trimmed = base[: -len(suffix)]
        if trimmed.strip("._-"):
            base = trimmed
    stem = _safe_part(base)
    key = _safe_part(asset_key)
    tail = f"_{key}{suffix}"
    if len(tail.encode("utf-8")) > _MAX_NAME_BYTES - 24:
        digest = hashlib.sha1(asset_key.encode("utf-8")).hexdigest()[:8]
        room = max(1, _MAX_NAME_BYTES - 24 - len(suffix.encode("utf-8")) - 9)
        key = f"{key[:room]}~{digest}"
        tail = f"_{key}{suffix}"
    room = max(1, _MAX_NAME_BYTES - len(tail.encode("utf-8")))
    if len(stem.encode("utf-8")) > room:
        digest = hashlib.sha1(scene_id.encode("utf-8")).hexdigest()[:8]
        stem = f"{stem[: max(1, room - 9)]}~{digest}"
    return _avoid_reserved(f"{stem}{tail}")


def legacy_scene_filename(scene_id: str, asset_key: str, suffix: str) -> str:
    """The name ``scene_filename`` gave before ids stopped repeating the suffix.

    GOES files saved by older runs are ``<id>.nc_data.nc``. This name is only
    read to adopt such a file; nothing is written under it.
    """
    if not suffix.startswith("."):
        suffix = f".{suffix}"
    return f"{_SAFE_ID.sub('_', scene_id)}_{_SAFE_ID.sub('_', asset_key)}{suffix}"


def partial_path(final: Path) -> Path:
    """Hidden sibling a download is written to before it is renamed to ``final``.

    The name never equals a ``scene_filename`` result (those do not start
    with ``.`` for a real id and never end in ``.part``), so the skip-existing
    check cannot mistake an interrupted download for a finished one.
    """
    return final.with_name(f"{PARTIAL_PREFIX}{final.name}{PARTIAL_SUFFIX}")


def _commit_partial(partial: Path, final: Path) -> Path:
    """Atomically move a finished download to its final name and return it.

    A symlink at ``final`` is unlinked first so the rename does not follow it.
    """
    _unlink_symlink(final)
    replace_with_retry(partial, final)
    return final


def _adopt_legacy_name(planned: Path, legacy: Path) -> None:
    """Rename a file saved under a legacy name to ``planned`` when that is free.

    Only a non-empty regular file is adopted, and never over an existing
    ``planned`` entry, so the skip check then sees it as present.
    """
    if legacy == planned or planned.exists() or planned.is_symlink():
        return
    if _nonempty_file(legacy):
        replace_with_retry(legacy, planned)


def aoi_path(data_path: Path) -> Path:
    """Sibling marker that records the AOI a windowed chip was read for."""
    return data_path.with_name(data_path.name + AOI_MARKER_SUFFIX)


def write_aoi(data_path: Path, bbox: BBox) -> None:
    """Record ``bbox`` beside a windowed chip; best effort, never fails a pull."""
    target = aoi_path(data_path)
    tmp = target.with_name(target.name + ".part")
    _unlink_symlink(tmp)
    with contextlib.suppress(OSError):
        tmp.write_text(
            json.dumps(bbox.model_dump(mode="json"), sort_keys=True) + "\n",
            encoding="utf-8",
        )
        replace_with_retry(tmp, target)


def same_aoi(data_path: Path, bbox: BBox) -> bool:
    """True when the marker beside ``data_path`` is exactly this AOI."""
    marker = aoi_path(data_path)
    if marker.is_symlink() or not marker.is_file():
        return False
    try:
        stored = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    if not isinstance(stored, dict):
        return False
    current = bbox.model_dump(mode="json")
    return all(stored.get(key) == current[key] for key in ("west", "south", "east", "north"))


def clear_aoi(data_path: Path) -> None:
    """Remove an AOI marker so it cannot describe a later whole-file download."""
    _unlink_file(aoi_path(data_path))


def existing_product(
    dest_dir: Path,
    scene_id: str,
    asset_key: str,
    suffix: str,
) -> Path | None:
    """Return a non-empty file for this scene asset under any known name.

    The catalog often predicts the suffix, but a server's content type can
    choose a different one. Checking the known suffixes and the pre-rename
    GOES name keeps a later pull from downloading the same scene again.
    """
    for candidate in dict.fromkeys((suffix, *KNOWN_SUFFIXES)):
        path = dest_dir / scene_filename(scene_id, asset_key, candidate)
        if _nonempty_file(path):
            return path
    legacy = dest_dir / legacy_scene_filename(scene_id, asset_key, suffix)
    if _nonempty_file(legacy):
        return legacy
    return None


def asset_suffix(asset: Asset, content_type: str | None = None) -> str:
    """Infer a file extension from media type or the href path."""
    for media in (content_type, asset.media_type):
        if not media:
            continue
        lower = media.lower()
        if "jp2" in lower or "jpeg2000" in lower:
            return ".jp2"
        if "geotiff" in lower or "tiff" in lower or "tif" in lower:
            return ".tif"
        if "netcdf" in lower:
            return ".nc"
        if "hdf" in lower:
            return ".hdf"
        if "png" in lower:
            return ".png"
        if "jpeg" in lower or "jpg" in lower:
            return ".jpg"
    path = Path(urlsplit(asset.href).path)
    ext = path.suffix.lower()
    if ext in _IMAGE_SUFFIXES:
        return ".jpg" if ext == ".jpeg" else ext
    if ext == ".tiff":
        return ".tif"
    if ext in _RASTER_SUFFIXES:
        return ext
    if ext in _COMPRESSED_SUFFIXES:
        # A compressed file keeps its inner extension, e.g. Himawari ``.DAT.bz2``.
        inner = Path(path.stem).suffix.lower()
        return f"{inner}{ext}" if inner else ext
    return ".bin"


async def download_assets(
    scenes: list[Scene],
    out: Path,
    *,
    http: httpx.AsyncClient,
    clients: Mapping[str, DataPullClient],
    concurrency: int = 8,
    asset_name: str | None = None,
    on_progress: Callable[[str, int, int], None] | None = None,
    bbox: BBox | None = None,
    stall_seconds: float = STALL_SECONDS,
    stall_min_bytes: int = STALL_MIN_BYTES,
    attempts: int = GET_ATTEMPTS,
    retry_backoff: float = RETRY_BACKOFF_SECONDS,
    asset_names: Sequence[str] | None = None,
    whole_file: Literal["allow", "size_limited"] = "allow",
    allow_whole_file: bool = False,
    max_pixels: int | None = None,
    deadline_seconds: float | None = None,
    polygon: Sequence[Sequence[float]] | None = None,
    write_sidecar: bool | None = None,
) -> DownloadReport:
    """GET one native-resolution asset per scene, streamed to disk.

    Existing non-empty files are skipped, except a GeoTIFF read as a window.
    A file under the pre-rename GOES name (``<id>.nc_data.nc``) counts as
    present and is renamed in place to the current name.
    That read is checked again: an empty window fails the scene and moves an
    older chip aside, so it is not this request's file. A range failure, a
    window over the read limit, and a transport error leave that chip in
    place. Browse assets are not downloaded. A failed GET records ``failed``
    and does not abort the rest.

    When ``bbox`` is set, a geographic, Web Mercator, or WGS84 UTM Cloud
    Optimized GeoTIFF is stored as the overlapping window. Those requests
    use HTTP Range and do not download the rest of the granule. Any other
    asset, including a COG in an unsupported CRS, is still fetched whole.

    A saved GeoTIFF whose pixels display dark or not at all also gets a
    ``<stem>.preview.png`` beside it (see ``atlas.cli.preview``); a skipped
    whole-file GeoTIFF without one gets one too. The data file is not
    changed. A preview failure is recorded in ``preview_error`` and never
    fails the scene.

    A whole-file GET answered 429 or 5xx is tried up to ``attempts`` times in
    total, honoring ``Retry-After``; any other 4xx fails at once. Each retry
    asks the client for its download headers again, so a bearer token that
    expired during the wait is refreshed. No file is
    opened until a 2xx arrives, so a retry never leaves a second partial
    file. A body that delivers fewer than ``stall_min_bytes`` within
    ``stall_seconds`` is aborted and its partial file removed.

    Bytes are written to a hidden ``.<name>.part`` sibling and renamed to the
    final name only after the body completes, so a killed run never leaves a
    truncated file at a name the skip check treats as done. A failure removes
    the partial file.

    Every ``failed`` or ``no_asset`` outcome carries a non-empty ``error``
    with URL queries and credentials masked (see ``redact_secrets``).
    An unexpected exception fails that scene only and does not abort the
    other downloads.

    ``asset_names`` fetches several assets per scene (alias or key, any case, via
    ``atlas.data.profiles.resolve_asset``); one outcome is recorded per (scene,
    asset). Combined with ``asset_name`` it raises ``ValueError``. Each chip is
    written as tiled, deflate-compressed GeoTIFF; with ``write_sidecar`` (default:
    on when ``asset_names`` is given) a ``<chip>.meta.json`` records provenance,
    calibration, and the AOI ``polygon``. A chip is reused only when its AOI marker
    and sidecar polygon match the request.

    ``whole_file="allow"`` keeps the CLI behaviour. ``"size_limited"`` refuses a
    whole-file GET for an asset that cannot be windowed when its size is known to
    exceed ``WHOLE_FILE_CAP_BYTES`` (or turns out to, from Content-Length) unless
    ``allow_whole_file``, and abandons the whole-file fallback for a transient
    window failure after ``WINDOW_INTENDED_ABORT_BYTES``. ``max_pixels`` caps one
    window. ``deadline_seconds`` is a wall-clock cap; user cancellation
    (``atlas.geo._lazy.check_cancel``) and the cap record ``cancelled`` or ``timeout``
    for work not finished, delete partial files, and keep finished outputs. The
    caller must run this coroutine in the thread that holds the cancel scope.
    """
    from atlas.geo._lazy import check_cancel
    from atlas.geo.errors import GeoCancelled

    if asset_name and asset_names is not None:
        raise ValueError("pass asset_name or asset_names, not both")
    names = _unique_names(asset_names)
    for requested in names or ([asset_name] if asset_name else []):
        if is_forbidden_asset_name(requested):
            raise BrowseAssetError(f"{requested!r} is a preview/thumbnail and cannot be downloaded")
    pairs: list[tuple[Scene, str | None]] = (
        [(scene, name) for scene in scenes for name in names]
        if asset_names is not None
        else [(scene, asset_name) for scene in scenes]
    )
    sidecars = asset_names is not None if write_sidecar is None else write_sidecar
    ring = _ring(polygon)
    deadline = None if deadline_seconds is None else time.monotonic() + deadline_seconds

    def tripped() -> str | None:
        """Return the stop status when the cap passed or the user cancelled."""
        if deadline is not None and time.monotonic() >= deadline:
            return "timeout"
        try:
            check_cancel()
        except GeoCancelled:
            return "cancelled"
        return None

    def checkpoint() -> None:
        stop = tripped()
        if stop is not None:
            raise _Stop(stop)

    totals: dict[str, int] = {}
    for scene, _name in pairs:
        totals[scene.source] = totals.get(scene.source, 0) + 1
    done_counts: dict[str, int] = dict.fromkeys(totals, 0)
    report = DownloadReport()
    lock = asyncio.Lock()
    sem = asyncio.Semaphore(max(1, concurrency))

    async def record(outcome: DownloadOutcome) -> None:
        async with lock:
            report.outcomes.append(outcome)
            done_counts[outcome.source] = done_counts.get(outcome.source, 0) + 1
            if on_progress is not None:
                on_progress(
                    outcome.source,
                    done_counts[outcome.source],
                    totals.get(outcome.source, done_counts[outcome.source]),
                )

    async def one(scene: Scene, name: str | None) -> None:
        path: Path | None = None
        partial: Path | None = None
        stale: Path | None = None
        bytes_read: int | None = None
        granule_size: int | None = None
        bytes_fetched = 0
        chip: CogRead | None = None
        key: str | None = None
        alias: str | None = None
        windowed_flag: bool | None = None
        notes: list[str] = []

        def make(status: str, **fields: Any) -> DownloadOutcome:
            return DownloadOutcome(
                source=scene.source,
                scene_id=scene.id,
                status=status,
                asset_key=key,
                alias=alias,
                **fields,
            )

        try:
            if asset_names is not None:
                try:
                    resolved = resolve_asset(scene, name or "")
                except AssetNotFound as exc:
                    await record(make("no_asset", error=str(exc)))
                    return
                key, asset, alias = resolved.key, resolved.asset, resolved.alias
                if is_browse_asset(key, asset):
                    raise BrowseAssetError(
                        f"{key!r} is a preview/thumbnail and cannot be downloaded"
                    )
            else:
                picked = pick_asset(scene, asset_name)
                if picked is None:
                    await record(make("no_asset", error="no native-resolution asset"))
                    return
                key, asset = picked
                alias = _alias_of(scene, key)
            dest_dir = output_source_dir(out, scene.source)
            if dest_dir is None:
                await record(
                    make("failed", error="source name is not a directory inside the output")
                )
                return
            dest_dir.mkdir(parents=True, exist_ok=True)
            planned_suffix = asset_suffix(asset)
            planned = dest_dir / scene_filename(scene.id, key, planned_suffix)
            _adopt_legacy_name(
                planned, dest_dir / legacy_scene_filename(scene.id, key, planned_suffix)
            )
            window = bbox is not None and _is_geotiff_asset(asset)
            present = existing_product(dest_dir, scene.id, key, planned_suffix)
            if present is not None:
                if not window or (
                    bbox is not None
                    and same_aoi(present, bbox)
                    and (not sidecars or _sidecar_matches(present, ring))
                ):
                    # A whole-file download is AOI-independent. A chip is only
                    # reusable when its marker records this same AOI (and, with a
                    # sidecar, this same polygon).
                    preview, preview_error = await _missing_preview(present)
                    await record(
                        make(
                            "skipped",
                            path=present,
                            preview=preview,
                            preview_error=preview_error,
                            windowed=True if window else None,
                        )
                    )
                    return
                # The chip on disk is an earlier window. A miss must not report
                # that file as the result of this request.
                stale = present
            elif window and planned.is_symlink():
                stale = planned
            async with sem:
                checkpoint()
                client = clients.get(scene.source)
                href = await _signed_href(client, asset.href)
                checkpoint()
                headers = await client.download_headers(href) if client is not None else {}
                why: list[str] = []
                windowed = await _windowed_cog(
                    http,
                    href,
                    asset,
                    bbox,
                    headers,
                    epsg_hint=epsg_hint(scene),
                    max_pixels=max_pixels,
                    why=why,
                )
                checkpoint()
                if isinstance(windowed, tuple):
                    payload, chip = windowed
                    bytes_read, granule_size = chip.bytes_fetched, chip.granule_size
                    bytes_fetched = chip.bytes_fetched
                    windowed_flag = True
                    final = dest_dir / scene_filename(scene.id, key, ".tif")
                    partial = partial_path(final)
                    _unlink_symlink(partial)
                    partial.write_bytes(payload)
                    path = _commit_partial(partial, final)
                    partial = None
                    if sidecars:
                        failure = _write_sidecar(
                            path, scene, key, alias, chip, len(payload), bbox, ring
                        )
                        if failure is not None:
                            notes.append(failure)
                    if bbox is not None:
                        write_aoi(final, bbox)
                    _retire_previous(out, stale, path)
                    stale = None
                elif (
                    windowed is UNWINDOWABLE
                    and stale is not None
                    and _nonempty_file(stale)
                    and not stale.is_symlink()
                    and not aoi_path(stale).exists()
                ):
                    # This asset cannot be windowed (unsupported CRS or layout)
                    # and the file on disk is a whole-file download without an
                    # AOI marker, not an older chip. Keep it instead of fetching
                    # the granule again.
                    preview, preview_error = await _missing_preview(stale)
                    await record(
                        make("skipped", path=stale, preview=preview, preview_error=preview_error)
                    )
                    return
                else:
                    reason = "; ".join(why) or "window not read"
                    limit_check = _WholeFilePolicy.for_download(
                        whole_file, allow_whole_file, window, windowed is UNWINDOWABLE, asset
                    )
                    limit_check.pre_check(key, reason)
                    total_attempts = max(1, attempts)
                    written = 0
                    for attempt in range(1, total_attempts + 1):
                        if attempt > 1 and client is not None:
                            # A Retry-After wait can outlast a short-lived token.
                            headers = await client.download_headers(href)
                        async with http.stream(
                            "GET", href, headers=headers, follow_redirects=True
                        ) as response:
                            if attempt < total_attempts and _retryable(response.status_code):
                                delay = _retry_delay(response, attempt, retry_backoff)
                                await response.aclose()
                                await asyncio.sleep(delay)
                                continue
                            response.raise_for_status()
                            limit_check.check_headers(response, key, reason)
                            suffix = asset_suffix(asset, response.headers.get("content-type"))
                            final = dest_dir / scene_filename(scene.id, key, suffix)
                            partial = partial_path(final)
                            _unlink_symlink(partial)
                            written = 0
                            with partial.open("wb") as handle:
                                async for chunk in _guarded_chunks(
                                    response, stall_seconds, stall_min_bytes
                                ):
                                    checkpoint()
                                    written += len(chunk)
                                    limit_check.check_written(written, key, reason)
                                    handle.write(chunk)
                            path = _commit_partial(partial, final)
                            clear_aoi(path)
                            # A new whole file must not keep a sidecar that
                            # describes the chip it replaced.
                            _unlink_file(sidecar_path(path))
                            partial = None
                        break
                    bytes_fetched = written
                    if window:
                        windowed_flag = False
                        notes.append(f"whole file ({reason})")
                        if sidecars:
                            notes.append("no sidecar: whole file")
                    _retire_previous(out, stale, path)
                    stale = None
        except _Stop as stop:
            _unlink_partial(path, stale)
            _unlink_file(partial)
            await record(make(stop.status, bytes_fetched=bytes_fetched))
            return
        except CogMissError as exc:
            _retire_previous(out, stale, None)
            _unlink_partial(path, stale)
            _unlink_file(partial)
            await record(make("failed", error=str(exc) or describe_error(exc)))
            return
        except (CogRangeError, CogLimitError) as exc:
            # A refused range and a window over the read limit are not an
            # empty window. The previous chip stays. A partial file from this
            # attempt is removed.
            _unlink_partial(path, stale)
            _unlink_file(partial)
            await record(make("failed", error=_limit_error(exc, max_pixels)))
            return
        except (_WholeFileRefused, _WholeFileAbort) as exc:
            _unlink_partial(path, stale)
            _unlink_file(partial)
            await record(make("failed", error=str(exc), windowed=False))
            return
        except BrowseAssetError as exc:
            _unlink_file(partial)
            await record(make("failed", error=str(exc) or describe_error(exc)))
            return
        except Exception as exc:
            # Transport, HTTP status, stall, disk, signing, and any unexpected
            # client error fail this scene only. The partial file is removed.
            _unlink_partial(path, stale)
            _unlink_file(partial)
            await record(make("failed", error=describe_error(exc)))
            return
        if path is None:
            await record(make("failed", error="download produced no file"))
            return
        preview, preview_error = await _saved_preview(path, chip)
        await record(
            make(
                "saved",
                path=path,
                bytes_read=bytes_read,
                granule_size=granule_size,
                preview=preview,
                preview_error=preview_error,
                windowed=windowed_flag,
                note="; ".join(notes) or None,
                bytes_fetched=bytes_fetched,
            )
        )

    await asyncio.gather(*(one(scene, name) for scene, name in pairs))
    return report


async def _saved_preview(path: Path, chip: CogRead | None) -> tuple[Path | None, str | None]:
    """Write the preview for a file just saved; return its path or the error text.

    A chip is previewed from its in-memory pixels, a whole-file GeoTIFF from
    an overview of the file on disk. Any other file gets none. A failure
    removes an older preview so it does not stand in for the new file.
    """
    if chip is None and not _is_tiff_path(path):
        return None, None
    try:
        if chip is not None:
            return write_read_preview(path, chip), None
        return await write_file_preview(path), None
    except Exception as exc:
        with contextlib.suppress(OSError):
            clear_preview(path)
        return None, f"preview not written: {describe_error(exc)}"


async def _missing_preview(path: Path) -> tuple[Path | None, str | None]:
    """Preview a skipped whole-file GeoTIFF that has none yet."""
    if not _is_tiff_path(path):
        return None, None
    existing = preview_path(path)
    if existing.is_file() and not existing.is_symlink():
        return existing, None
    return await _saved_preview(path, None)


def _is_tiff_path(path: Path) -> bool:
    return path.suffix.lower() in {".tif", ".tiff"}


def _retryable(status: int) -> bool:
    """True for 429 and 5xx. Any other 4xx is the request's fault and final."""
    return status == 429 or 500 <= status < 600


def _retry_delay(response: httpx.Response, attempt: int, backoff: float) -> float:
    """Seconds to wait before the next attempt.

    ``Retry-After`` (delta-seconds or an HTTP date) wins when it parses and
    is capped at ``RETRY_AFTER_CAP_SECONDS``. Otherwise the wait is
    ``backoff * attempt``.
    """
    seconds = _retry_after(response)
    if seconds is None:
        return backoff * attempt
    return min(seconds, RETRY_AFTER_CAP_SECONDS)


def _retry_after(response: httpx.Response) -> float | None:
    """Non-negative ``Retry-After`` seconds (delta or HTTP date), or None."""
    header = (response.headers.get("retry-after") or "").strip()
    if not header:
        return None
    if header.isdigit():
        return float(header)
    try:
        when = email.utils.parsedate_to_datetime(header)
    except (TypeError, ValueError, IndexError):
        return None
    if when is None or when.tzinfo is None:
        return None
    return max(when.timestamp() - time.time(), 0.0)


async def _guarded_chunks(
    response: httpx.Response, stall_seconds: float, min_bytes: int
) -> AsyncIterator[bytes]:
    """Yield body chunks; raise ``DownloadStallError`` when progress stops.

    Progress means at least ``min_bytes`` within each ``stall_seconds``
    window. A server that sends nothing, or trickles bytes so the socket
    read timeout keeps resetting, is cut off.
    """
    loop = asyncio.get_running_loop()
    chunks = response.aiter_bytes()
    window_start = loop.time()
    window_bytes = 0
    total = 0
    while True:
        remaining = window_start + stall_seconds - loop.time()
        try:
            chunk = await asyncio.wait_for(anext(chunks), timeout=max(remaining, 0.0))
        except StopAsyncIteration:
            return
        except TimeoutError:
            raise DownloadStallError(
                f"download stalled: fewer than {min_bytes} bytes in {stall_seconds:g}s "
                f"after {total} bytes"
            ) from None
        total += len(chunk)
        window_bytes += len(chunk)
        if window_bytes >= min_bytes:
            window_start = loop.time()
            window_bytes = 0
        yield chunk


def _nonempty_file(path: Path) -> bool:
    """True when ``path`` is a file with bytes, without following a symlink."""
    if path.is_symlink() or not path.is_file():
        return False
    return path.stat().st_size > 0


def _unlink_file(path: Path | None) -> None:
    """Remove one product file. A symlink is unlinked, not followed."""
    if path is None:
        return
    if path.is_symlink() or path.is_file():
        path.unlink()


def _unlink_symlink(path: Path) -> None:
    """Unlink a destination symlink so a write does not follow it."""
    if path.is_symlink():
        path.unlink()


def plain_source_name(source: str) -> bool:
    """True when ``source`` is one path segment and not a traversal."""
    if not source or "/" in source or "\\" in source or ".." in source:
        return False
    return source not in {".", ".."} and Path(source).name == source


def inside_output(root: Path, candidate: Path) -> bool:
    """True when ``candidate`` stays inside ``root`` after resolving symlinks.

    ``root`` and its parents may be symlinks. A final symlink to a file is
    judged by where the link sits, so the link can be unlinked in place. A
    symlinked directory is resolved, and a path that lands outside ``root``
    is rejected.
    """
    try:
        resolved_root = root.resolve()
        if candidate.is_symlink() and not candidate.is_dir():
            location = candidate.parent.resolve() / candidate.name
        else:
            location = candidate.resolve()
        try:
            location.relative_to(resolved_root)
        except ValueError:
            # Windows paths are case-insensitive, so a differently-cased path
            # can still be inside. normcase is the identity on POSIX.
            location_text = os.path.normcase(str(location))
            root_text = os.path.normcase(str(resolved_root))
            if location_text != root_text and not location_text.startswith(root_text + os.sep):
                return False
    except (OSError, RuntimeError, ValueError):
        return False
    return location != resolved_root


def output_source_dir(root: Path, source: str) -> Path | None:
    """Return ``root/source`` when that directory cannot escape ``root``."""
    if not plain_source_name(source):
        return None
    dest = root / source
    if not inside_output(root, dest):
        return None
    return dest


def retire_product(path: Path) -> None:
    """Stop presenting one previous product without discarding its bytes.

    A regular file is moved to ``<name>.stale`` beside itself. A symlink is
    unlinked and not followed. Directories are left alone. A GeoTIFF's
    ``<stem>.preview.png``, AOI marker, and metadata sidecar are retired with it, so neither
    outlives its data.
    """
    if _is_tiff_path(path) and not path.name.endswith(".stale"):
        _retire_one(preview_path(path))
        _retire_one(aoi_path(path))
        _retire_one(sidecar_path(path))
    _retire_one(path)


def _retire_one(path: Path) -> None:
    if path.is_symlink():
        path.unlink()
        return
    if not path.is_file():
        return
    dest = path.with_name(path.name + ".stale")
    if dest.is_symlink() or dest.is_file():
        dest.unlink()
    elif dest.exists():
        return
    replace_with_retry(path, dest)


def _retire_previous(root: Path, previous: Path | None, keep: Path | None) -> None:
    """Move an older product aside when it is not the file just written."""
    if previous is None or previous == keep:
        return
    if not inside_output(root, previous):
        return
    retire_product(previous)


def _unlink_partial(path: Path | None, previous: Path | None) -> None:
    """Remove an incomplete file from this attempt, not the previous product."""
    if path is None or path == previous:
        return
    _unlink_file(path)


def _is_geotiff_asset(asset: Asset) -> bool:
    media = (asset.media_type or "").lower()
    if "geotiff" in media or "tiff" in media:
        return True
    return Path(urlsplit(asset.href).path).suffix.lower() in {".tif", ".tiff"}


def _unique_names(names: Sequence[str] | None) -> list[str]:
    """Return the requested asset names without case-insensitive duplicates."""
    if names is None:
        return []
    unique: dict[str, str] = {}
    for name in names:
        cleaned = name.strip()
        if cleaned:
            unique.setdefault(cleaned.casefold(), cleaned)
    if not unique:
        raise ValueError("asset_names is empty")
    return list(unique.values())


def _ring(polygon: Sequence[Sequence[float]] | None) -> list[list[float]] | None:
    """Return a lon/lat ring as plain float lists, or None."""
    if not polygon:
        return None
    return [[float(value) for value in point] for point in polygon]


def _alias_of(scene: Scene, key: str) -> str | None:
    """Return the profile alias of asset ``key``, or None."""
    try:
        return resolve_asset(scene, key).alias
    except AssetNotFound:
        return None


def _sidecar_matches(chip: Path, ring: list[list[float]] | None) -> bool:
    """True when the chip's sidecar exists and records the requested polygon."""
    try:
        meta = read_chip_meta(chip)
    except GeoError:
        return False
    if meta is None or not meta.file.windowed:
        return False
    stored = meta.aoi.get("polygon") if isinstance(meta.aoi, dict) else None
    return _ring(stored) == ring


def _float_or_none(text: str | None) -> float | None:
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _scene_properties(scene: Scene) -> dict[str, Any]:
    return {
        name: value
        for name, value in scene.properties.items()
        if name in _SIDECAR_PROPERTIES or name.startswith(_SIDECAR_PROPERTY_PREFIXES)
    }


def _chip_meta(
    scene: Scene,
    key: str,
    alias: str | None,
    read: CogRead,
    size: int,
    bbox: BBox | None,
    ring: list[list[float]] | None,
) -> ChipMeta:
    """Build the sidecar for one windowed chip from catalog facts and the read."""
    asset = scene.assets[key]
    try:
        resolved = resolve_asset(scene, key)
        band = resolved.band
    except AssetNotFound:
        band = None
    rule = calibration_for(scene, key)
    dtype = str(read.array.dtype)
    calibration = Calibration()
    if rule is not None:
        calibration = Calibration(
            scale=rule.scale,
            offset=rule.offset,
            offset_applied=rule.offset_applied,
            nodata_in=rule.nodata_in,
            dtype_in=rule.dtype_in or dtype,
            valid_range=list(rule.valid_range) if rule.valid_range else None,
            source=rule.source,
        )
    aoi = None
    if bbox is not None:
        aoi = {"bbox": [bbox.west, bbox.south, bbox.east, bbox.north], "polygon": ring}
    return ChipMeta(
        scene_id=scene.id,
        source=scene.source,
        collection=scene.collection,
        platform=scene.platform,
        datetime=scene.datetime.isoformat(),
        asset_key=key,
        alias=alias,
        band=BandInfo(
            common_name=asset.common_name or (band.common_name if band else None),
            wavelength_um=asset.center_wavelength or (band.wavelength_um if band else None),
            gsd_m=asset.spatial_resolution or (band.gsd_m if band else None),
            source_band=key,
        ),
        file=FileInfo(
            bytes=size,
            windowed=True,
            granule_bytes=read.granule_size,
            bytes_fetched=read.bytes_fetched,
        ),
        raster=RasterInfo(
            dtype=dtype,
            height=int(read.array.shape[-2]),
            width=int(read.array.shape[-1]),
            epsg=read.epsg,
            transform=[
                read.pixel_size_x,
                0.0,
                read.origin_x,
                0.0,
                -read.pixel_size_y,
                read.origin_y,
            ],
            nodata=_float_or_none(read.nodata),
            registration="area",
        ),
        calibration=calibration,
        qa_preset=band.qa_preset if band else None,
        aoi=aoi,
        scene_properties=_scene_properties(scene),
    )


def _write_sidecar(
    chip: Path,
    scene: Scene,
    key: str,
    alias: str | None,
    read: CogRead,
    size: int,
    bbox: BBox | None,
    ring: list[list[float]] | None,
) -> str | None:
    """Write the chip's sidecar; return a note on failure, never raise.

    Any older sidecar is removed first, so a chip never carries the sidecar of a
    different window. A failed write leaves none, and the chip is re-read next run.
    """
    _unlink_file(sidecar_path(chip))
    try:
        write_chip_meta(chip, _chip_meta(scene, key, alias, read, size, bbox, ring))
    except Exception as exc:
        return f"sidecar not written: {describe_error(exc)}"
    return None


def _limit_error(exc: BaseException, max_pixels: int | None) -> str:
    """Error text for a refused range or oversized window."""
    text = str(exc) or describe_error(exc)
    if max_pixels is None or "Fix:" in text or not isinstance(exc, CogLimitError):
        return text
    return f"{text}. Fix: shrink the AOI (limit {max_pixels / 1e6:g} megapixels per asset)"


@dataclass(frozen=True)
class _WholeFilePolicy:
    """Size rules for a whole-file GET when a window was intended."""

    known_size: int | None = None
    refuse_over: int | None = None
    abort_over: int | None = None

    @classmethod
    def for_download(
        cls,
        mode: str,
        allow: bool,
        window: bool,
        unwindowable: bool,
        asset: Asset,
    ) -> _WholeFilePolicy:
        if mode != "size_limited" or not window:
            return cls()
        if not unwindowable:
            return cls(abort_over=WINDOW_INTENDED_ABORT_BYTES)
        if allow:
            return cls()
        size = asset.size_bytes
        if size is None:
            return cls(refuse_over=WHOLE_FILE_CAP_BYTES)
        return cls(known_size=size if size > WHOLE_FILE_CAP_BYTES else None)

    def pre_check(self, key: str, reason: str) -> None:
        if self.known_size is not None:
            raise _WholeFileRefused(_refusal(key, reason, f"{self.known_size / 1e6:.0f} MB"))

    def check_headers(self, response: httpx.Response, key: str, reason: str) -> None:
        if self.refuse_over is None:
            return
        header = response.headers.get("content-length", "")
        if header.isdigit() and int(header) > self.refuse_over:
            raise _WholeFileRefused(_refusal(key, reason, f"{int(header) / 1e6:.0f} MB"))

    def check_written(self, written: int, key: str, reason: str) -> None:
        if self.refuse_over is not None and written > self.refuse_over:
            raise _WholeFileRefused(_refusal(key, reason, f"over {self.refuse_over / 1e6:.0f} MB"))
        if self.abort_over is not None and written > self.abort_over:
            raise _WholeFileAbort(
                f"window read failed ({reason}); whole-file fetch aborted at "
                f"{self.abort_over // (1024 * 1024)} MiB"
            )


def _refusal(key: str, reason: str, size: str) -> str:
    return (
        f"{key} cannot be windowed ({reason}) and is {size}; whole-file fetch skipped. "
        "Fix: allow_whole_file=true, or choose another asset"
    )


async def _windowed_cog(
    http: httpx.AsyncClient,
    href: str,
    asset: Asset,
    bbox: BBox | None,
    headers: Mapping[str, str] | None = None,
    *,
    epsg_hint: int | None = None,
    max_pixels: int | None = None,
    why: list[str] | None = None,
) -> tuple[bytes, CogRead] | _Unwindowable | None:
    """Return the AOI GeoTIFF bytes and the window they encode.

    ``UNWINDOWABLE`` means the asset itself cannot be windowed (no AOI, not a
    GeoTIFF, unsupported CRS or layout), so a file already on disk is a
    whole-file download. None means this attempt failed (transport error or
    undecodable tile) and the whole asset should be fetched. A miss is not a fallback:
    the AOI is outside the raster, and downloading the granule would not
    contain it. The read carries bytes fetched, the object size from
    Content-Range, and the pixels used for the preview. The chip is encoded as
    tiled, deflate-compressed GeoTIFF.

    ``epsg_hint`` resolves a user-defined projection, ``max_pixels`` caps the window,
    and each reason a window was not produced is appended to ``why``.

    A box that crosses the antimeridian is read one side at a time. A scene on
    one side is windowed normally; a global raster whose two edges both meet
    the AOI cannot be one rectangular chip, so that case stays whole-file.
    """
    reasons: list[str] = why if why is not None else []
    if bbox is None:
        reasons.append("no AOI")
        return UNWINDOWABLE
    if not _is_geotiff_asset(asset):
        reasons.append("not a GeoTIFF")
        return UNWINDOWABLE
    try:
        cog = await open_cog(http, href, headers=headers, epsg_hint=epsg_hint)
        if cog.epsg is None:
            reasons.append("raster CRS unknown and no proj:epsg hint")
            return UNWINDOWABLE
        try:
            parts = bbox.antimeridian_parts()
        except ValueError:
            reasons.append("AOI crosses the antimeridian")
            return UNWINDOWABLE
        reads: list[CogRead] = []
        unsupported = False
        for part in parts:
            bounds = model_bounds(
                cog.epsg,
                west=part.west,
                south=part.south,
                east=part.east,
                north=part.north,
            )
            if bounds is None:
                unsupported = True
                continue
            west, south, east, north = bounds
            try:
                reads.append(
                    await cog.read_window(
                        west=west, south=south, east=east, north=north, max_pixels=max_pixels
                    )
                )
            except CogMissError:
                continue
        if not reads and unsupported:
            reasons.append(f"unsupported projection EPSG:{cog.epsg}")
            return UNWINDOWABLE
        if not reads:
            raise CogMissError("AOI does not overlap the COG")
        if len(reads) > 1:
            reasons.append("AOI crosses the antimeridian and both edges meet the raster")
            return UNWINDOWABLE
        read = reads[0]
    except (CogMissError, CogRangeError, CogLimitError):
        raise
    except CogUnsupportedError as exc:
        reasons.append(exc.reason or str(exc) or "unsupported GeoTIFF layout")
        return UNWINDOWABLE
    except (CogError, httpx.HTTPError) as exc:
        reasons.append(type(exc).__name__)
        return None
    return encode_cog(read, tile=_CHIP_TILE, compress="deflate"), read


async def _signed_href(
    client: DataPullClient | None,
    href: str,
    *,
    attempts: int = SIGN_ATTEMPTS,
    backoff: float = SIGN_BACKOFF_SECONDS,
) -> str:
    """Return ``href`` signed by the client's ``sign_href``, or unchanged.

    A 429, a 5xx, or a transport error is retried up to ``attempts`` times in
    total. The wait honors ``Retry-After`` and is otherwise exponential from
    ``backoff``; every wait is capped at ``SIGN_WAIT_CAP_SECONDS``. Any other
    4xx and any non-HTTP error are raised at once.
    """
    if client is None:
        return href
    sign = getattr(client, "sign_href", None)
    if sign is None:
        return href
    total = max(1, attempts)
    for attempt in range(1, total + 1):
        try:
            signed = sign(href)
            if asyncio.iscoroutine(signed):
                signed = await signed
            if not isinstance(signed, str):
                raise TypeError(f"sign_href must return str, got {type(signed)}")
            return signed
        except httpx.HTTPStatusError as exc:
            if attempt >= total or not _retryable(exc.response.status_code):
                raise
            await asyncio.sleep(_sign_delay(exc.response, attempt, backoff))
        except httpx.TransportError:
            if attempt >= total:
                raise
            await asyncio.sleep(_sign_delay(None, attempt, backoff))
    return href


def _sign_delay(response: httpx.Response | None, attempt: int, backoff: float) -> float:
    """Wait before signing attempt ``attempt + 1``, capped at ``SIGN_WAIT_CAP_SECONDS``."""
    header = None if response is None else _retry_after(response)
    if header is not None:
        return min(header, SIGN_WAIT_CAP_SECONDS)
    return min(backoff * 2.0 ** (attempt - 1), SIGN_WAIT_CAP_SECONDS)
