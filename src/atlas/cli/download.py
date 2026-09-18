"""Parallel download of native-resolution scene assets. Browse imagery is rejected."""

from __future__ import annotations

import asyncio
import re
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import httpx

from atlas.data.base import Asset, DataPullClient, Scene

_SAFE_ID = re.compile(r"[^A-Za-z0-9._-]+")
_BROWSE_KEYS = frozenset({"rendered_preview", "thumbnail", "preview"})
_BROWSE_ROLES = frozenset({"thumbnail", "overview", "preview", "rendered_preview"})
_PREFERRED_KEYS = ("visual",)
_RASTER_SUFFIXES = frozenset(
    {".tif", ".tiff", ".jp2", ".nc", ".hdf", ".h5", ".zip", ".las", ".laz"}
)


class BrowseAssetError(ValueError):
    """Raised when the requested asset is a preview or thumbnail."""


@dataclass
class DownloadOutcome:
    """One scene's download attempt."""

    source: str
    scene_id: str
    status: str
    path: Path | None = None
    error: str | None = None


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

    Prefers ``visual`` (full-res RGB COG) when it is not browse, then any ``data``
    role, then the first remaining non-browse asset. ``name`` pins a key.

    Raises:
        BrowseAssetError: if ``name`` is a preview/thumbnail key or asset.
    """
    if name:
        if is_forbidden_asset_name(name):
            raise BrowseAssetError(f"{name!r} is a preview/thumbnail and cannot be downloaded")
        asset = scene.assets.get(name)
        if asset is None:
            return None
        if is_browse_asset(name, asset):
            raise BrowseAssetError(f"{name!r} is a preview/thumbnail and cannot be downloaded")
        return name, asset
    for key in _PREFERRED_KEYS:
        asset = scene.assets.get(key)
        if asset is not None and not is_browse_asset(key, asset):
            return key, asset
    data_hits = [
        (key, asset)
        for key, asset in scene.assets.items()
        if not is_browse_asset(key, asset) and "data" in {role.lower() for role in asset.roles}
    ]
    if data_hits:
        return data_hits[0]
    for key, asset in scene.assets.items():
        if not is_browse_asset(key, asset):
            return key, asset
    return None


def scene_filename(scene_id: str, asset_key: str, suffix: str) -> str:
    """Filesystem-safe name ``<id>_<asset><suffix>``."""
    stem = _SAFE_ID.sub("_", scene_id)
    key = _SAFE_ID.sub("_", asset_key)
    if not suffix.startswith("."):
        suffix = f".{suffix}"
    return f"{stem}_{key}{suffix}"


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
    ext = Path(urlparse(asset.href).path).suffix.lower()
    if ext == ".tiff":
        return ".tif"
    if ext in _RASTER_SUFFIXES:
        return ext
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
) -> DownloadReport:
    """GET one native-resolution asset per scene, streamed to disk.

    Existing non-empty files are skipped. Browse assets are not downloaded.
    A failed GET records ``failed`` and does not abort the rest.
    """
    if asset_name and is_forbidden_asset_name(asset_name):
        raise BrowseAssetError(f"{asset_name!r} is a preview/thumbnail and cannot be downloaded")

    totals: dict[str, int] = {}
    for scene in scenes:
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

    async def one(scene: Scene) -> None:
        path: Path | None = None
        try:
            picked = pick_asset(scene, asset_name)
            if picked is None:
                await record(
                    DownloadOutcome(
                        source=scene.source,
                        scene_id=scene.id,
                        status="no_asset",
                        error="no native-resolution asset",
                    )
                )
                return
            key, asset = picked
            dest_dir = out / scene.source
            dest_dir.mkdir(parents=True, exist_ok=True)
            planned = dest_dir / scene_filename(scene.id, key, asset_suffix(asset))
            if planned.is_file() and planned.stat().st_size > 0:
                await record(
                    DownloadOutcome(
                        source=scene.source,
                        scene_id=scene.id,
                        status="skipped",
                        path=planned,
                    )
                )
                return
            async with sem:
                href = await _signed_href(clients.get(scene.source), asset.href)
                async with http.stream("GET", href, follow_redirects=True) as response:
                    response.raise_for_status()
                    suffix = asset_suffix(asset, response.headers.get("content-type"))
                    path = dest_dir / scene_filename(scene.id, key, suffix)
                    with path.open("wb") as handle:
                        async for chunk in response.aiter_bytes():
                            handle.write(chunk)
        except BrowseAssetError as exc:
            await record(
                DownloadOutcome(
                    source=scene.source, scene_id=scene.id, status="failed", error=str(exc)
                )
            )
            return
        except (httpx.HTTPError, OSError, TypeError) as exc:
            if path is not None:
                path.unlink(missing_ok=True)
            await record(
                DownloadOutcome(
                    source=scene.source, scene_id=scene.id, status="failed", error=str(exc)
                )
            )
            return
        if path is None:
            await record(
                DownloadOutcome(
                    source=scene.source,
                    scene_id=scene.id,
                    status="failed",
                    error="download produced no file",
                )
            )
            return
        await record(
            DownloadOutcome(source=scene.source, scene_id=scene.id, status="saved", path=path)
        )

    await asyncio.gather(*(one(scene) for scene in scenes))
    return report


async def _signed_href(client: DataPullClient | None, href: str) -> str:
    if client is None:
        return href
    sign = getattr(client, "sign_href", None)
    if sign is None:
        return href
    last_error: httpx.HTTPError | None = None
    for attempt in range(3):
        try:
            signed = sign(href)
            if asyncio.iscoroutine(signed):
                signed = await signed
            if not isinstance(signed, str):
                raise TypeError(f"sign_href must return str, got {type(signed)}")
            return signed
        except httpx.HTTPError as exc:
            last_error = exc
            await asyncio.sleep(0.4 * (attempt + 1))
    if last_error is not None:
        raise last_error
    return href
