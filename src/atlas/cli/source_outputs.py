"""Files for sources whose scenes carry no native-resolution asset.

GIBS scenes are WMS browse renders and FIRMS scenes are point detections.
``download_assets`` refuses both, so ``atlas data`` would otherwise save
nothing for them. This module writes:

* GIBS: the WMS GetMap image the scene already names, fetched for the
  request AOI and day, as ``<out>/<source>/<scene>_wms.png`` (or ``.jpg``).
  The name carries no AOI, so a failed fetch moves an existing image at
  that name to ``<name>.stale`` rather than leaving another AOI's render
  in place.
* FIRMS: one CSV per source, ``<out>/<source>/<source>_detections.csv``,
  with one row per detection and every column FIRMS returned.

Invariants: only clients that are ``GibsClient`` or ``FirmsClient`` route
here, so browse assets stay forbidden for every other source. Files are
written only inside ``output_source_dir(out, source)``, and each write
replaces the destination atomically without following a symlink there.
"""

from __future__ import annotations

import asyncio
import csv
import io
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx
from PIL import Image, UnidentifiedImageError

from atlas.cli.download import (
    DownloadOutcome,
    describe_error,
    inside_output,
    output_source_dir,
    retire_product,
    scene_filename,
)
from atlas.data.base import DataPullClient, Scene
from atlas.data.firms import FirmsClient
from atlas.data.gibs import GibsClient

GIBS_ASSET_KEY = "rendered_preview"
DETECTIONS_SUFFIX = "_detections.csv"
_LEAD_COLUMNS = (
    "scene_id",
    "source",
    "latitude",
    "longitude",
    "acq_datetime",
    "platform",
    "instrument",
)
_IMAGE_SUFFIXES = {"PNG": ".png", "JPEG": ".jpg"}

ProgressCallback = Callable[[str, int, int], None]


def writes_own_output(client: DataPullClient | None) -> bool:
    """True for sources this module saves instead of ``download_assets``."""
    return isinstance(client, GibsClient | FirmsClient)


def detections_path(out: Path, source: str) -> Path | None:
    """Return ``<out>/<source>/<source>_detections.csv`` or None if unsafe."""
    dest = output_source_dir(out, source)
    return None if dest is None else dest / f"{source}{DETECTIONS_SUFFIX}"


async def write_source_outputs(
    scenes: Sequence[Scene],
    out: Path,
    *,
    http: httpx.AsyncClient,
    clients: Mapping[str, DataPullClient],
    concurrency: int = 8,
    on_progress: ProgressCallback | None = None,
) -> list[DownloadOutcome]:
    """Save GIBS images and FIRMS detection tables for ``scenes``.

    Scenes whose client is neither GIBS nor FIRMS are ignored. A failed GIBS
    day records ``failed`` and does not stop other days or sources. GIBS
    images are always fetched again, because the file name does not encode
    the AOI; a failed day moves any earlier image at that name to
    ``<name>.stale``. A FIRMS table is rewritten from this pull's scenes.
    """
    gibs: list[Scene] = []
    firms: dict[str, list[Scene]] = {}
    for scene in scenes:
        client = clients.get(scene.source)
        if isinstance(client, GibsClient):
            gibs.append(scene)
        elif isinstance(client, FirmsClient):
            firms.setdefault(scene.source, []).append(scene)
    outcomes: list[DownloadOutcome] = []
    for source, rows in firms.items():
        outcomes.append(write_detections_csv(rows, out, source))
        if on_progress is not None:
            on_progress(source, len(rows), len(rows))
    outcomes.extend(
        await _save_gibs_images(
            gibs, out, http=http, concurrency=concurrency, on_progress=on_progress
        )
    )
    return outcomes


def write_detections_csv(scenes: Sequence[Scene], out: Path, source: str) -> DownloadOutcome:
    """Write one CSV row per FIRMS detection and return the outcome.

    Leading columns are the scene id, source, parsed latitude/longitude, the
    UTC acquisition time in ISO 8601, platform, and instrument. Every other
    property the client kept follows in first-seen order. Rows are ordered by
    acquisition time, then latitude and longitude.
    """
    scene_id = f"{source}-detections"
    path = detections_path(out, source)
    if path is None:
        return DownloadOutcome(
            source=source,
            scene_id=scene_id,
            status="failed",
            error="source name is not a directory inside the output",
        )
    columns = list(_LEAD_COLUMNS)
    for scene in scenes:
        for key in scene.properties:
            if key not in columns:
                columns.append(key)
    ordered = sorted(scenes, key=lambda s: (s.datetime, s.lat or 0.0, s.lon or 0.0, s.id))
    try:
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for scene in ordered:
            writer.writerow(_detection_row(scene))
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_atomic(path, buffer.getvalue().encode("utf-8"))
    except OSError as exc:
        return DownloadOutcome(source=source, scene_id=scene_id, status="failed", error=str(exc))
    except Exception as exc:
        return DownloadOutcome(
            source=source,
            scene_id=scene_id,
            status="failed",
            error=describe_error(exc),
        )
    return DownloadOutcome(source=source, scene_id=scene_id, status="saved", path=path)


def _detection_row(scene: Scene) -> dict[str, Any]:
    row: dict[str, Any] = {
        key: ("" if value is None else value) for key, value in scene.properties.items()
    }
    row.update(
        {
            "scene_id": scene.id,
            "source": scene.source or scene.collection,
            "latitude": scene.lat if scene.lat is not None else row.get("latitude", ""),
            "longitude": scene.lon if scene.lon is not None else row.get("longitude", ""),
            "acq_datetime": scene.datetime.isoformat().replace("+00:00", "Z"),
            "platform": scene.platform or "",
            "instrument": scene.instrument or "",
        }
    )
    return row


async def _save_gibs_images(
    scenes: Sequence[Scene],
    out: Path,
    *,
    http: httpx.AsyncClient,
    concurrency: int,
    on_progress: ProgressCallback | None,
) -> list[DownloadOutcome]:
    totals: dict[str, int] = {}
    for scene in scenes:
        totals[scene.source] = totals.get(scene.source, 0) + 1
    done: dict[str, int] = dict.fromkeys(totals, 0)
    sem = asyncio.Semaphore(max(1, concurrency))

    async def one(scene: Scene) -> DownloadOutcome:
        async with sem:
            try:
                outcome = await _save_gibs_image(scene, out, http=http)
            except Exception as exc:
                # An unexpected decoder or disk error fails this day only.
                outcome = DownloadOutcome(
                    source=scene.source,
                    scene_id=scene.id,
                    status="failed",
                    error=describe_error(exc),
                )
        done[scene.source] += 1
        if on_progress is not None:
            on_progress(scene.source, done[scene.source], totals[scene.source])
        return outcome

    return list(await asyncio.gather(*(one(scene) for scene in scenes)))


async def _save_gibs_image(scene: Scene, out: Path, *, http: httpx.AsyncClient) -> DownloadOutcome:
    """GET one GIBS GetMap and keep it only when it decodes and is not blank.

    The image name has no AOI, so a file already at that name may be another
    AOI's render. A failed fetch retires it (``<name>.stale``) so the
    manifest row for this request never points readers at that image.
    """
    dest_dir = output_source_dir(out, scene.source)

    def failed(error: str) -> DownloadOutcome:
        if dest_dir is not None:
            try:
                _retire_gibs_images(out, dest_dir, scene.id)
            except OSError as exc:
                # The scene already failed; a stuck old file must not hide that.
                error = f"{error}; older image not moved aside: {exc}"
        return DownloadOutcome(source=scene.source, scene_id=scene.id, status="failed", error=error)

    if dest_dir is None:
        return failed("source name is not a directory inside the output")
    asset = scene.assets.get(GIBS_ASSET_KEY)
    if asset is None:
        return failed("GIBS scene has no WMS image URL")
    try:
        response = await http.get(asset.href, follow_redirects=True)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        return failed(f"GIBS request failed: {exc}")
    payload = response.content
    try:
        suffix, blank = _inspect_image(payload)
    except ValueError as exc:
        return failed(str(exc))
    if blank:
        return failed("GIBS has no imagery for this day and area (image is fully transparent)")
    path = dest_dir / scene_filename(scene.id, "wms", suffix)
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        _write_atomic(path, payload)
    except OSError as exc:
        return failed(str(exc))
    return DownloadOutcome(
        source=scene.source,
        scene_id=scene.id,
        status="saved",
        path=path,
        bytes_read=None,
    )


def _retire_gibs_images(out: Path, dest_dir: Path, scene_id: str) -> None:
    """Move aside an earlier image saved under this scene's name, in any format."""
    for suffix in _IMAGE_SUFFIXES.values():
        previous = dest_dir / scene_filename(scene_id, "wms", suffix)
        if inside_output(out, previous):
            retire_product(previous)


def _inspect_image(payload: bytes) -> tuple[str, bool]:
    """Return the file suffix and whether every pixel is transparent.

    Raises:
        ValueError: The body is not a PNG or JPEG image, for example a WMS
            ServiceException XML document.
    """
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
            fmt = image.format or ""
            if fmt not in _IMAGE_SUFFIXES:
                raise ValueError(f"GIBS returned an unsupported image format {fmt!r}")
            blank = _fully_transparent(image)
    except (UnidentifiedImageError, OSError) as exc:
        snippet = payload[:120].decode("utf-8", errors="replace").strip()
        raise ValueError(f"GIBS did not return an image: {snippet!r}") from exc
    return _IMAGE_SUFFIXES[fmt], blank


def _fully_transparent(image: Image.Image) -> bool:
    if image.mode not in {"RGBA", "LA", "PA"} and "transparency" not in image.info:
        return False
    alpha = image.convert("RGBA").getchannel("A")
    extrema = alpha.getextrema()
    return isinstance(extrema, tuple) and extrema[1] == 0


def _write_atomic(path: Path, payload: bytes) -> None:
    """Replace ``path`` with ``payload``; a symlink at ``path`` is replaced, not followed."""
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
        # ``os.fchmod`` is Unix-only; ``os.chmod`` exists on Windows too.
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
