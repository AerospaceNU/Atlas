"""Parallel preview download for scenes returned by search."""

from __future__ import annotations

import asyncio
import io
import re
from collections.abc import Callable, Mapping
from pathlib import Path

import httpx
from PIL import Image

from atlas.data.base import Asset, DataPullClient, Scene

_DEFAULT_ASSET_KEYS = ("rendered_preview", "visual", "thumbnail")
_SAFE_ID = re.compile(r"[^A-Za-z0-9._-]+")


def pick_asset(scene: Scene, name: str | None = None) -> tuple[str, Asset] | None:
    """Choose a downloadable preview asset from a scene.

    Prefers ``rendered_preview``, then ``visual``, then ``thumbnail``, then any
    asset whose roles look like an overview. ``name`` pins a specific key.
    """
    if name:
        asset = scene.assets.get(name)
        return (name, asset) if asset is not None else None
    for key in _DEFAULT_ASSET_KEYS:
        asset = scene.assets.get(key)
        if asset is not None:
            return key, asset
    for key, asset in scene.assets.items():
        roles = {role.lower() for role in asset.roles}
        if roles & {"thumbnail", "overview", "visual", "rendered_preview"}:
            return key, asset
    return None


def scene_filename(scene_id: str) -> str:
    """Filesystem-safe PNG name for a scene id."""
    return _SAFE_ID.sub("_", scene_id) + ".png"


def save_preview(content: bytes, destination: Path) -> Path:
    """Write GET bytes as an RGB PNG, or raw bytes if they are not an image."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        image = Image.open(io.BytesIO(content))
        image.convert("RGB").save(destination, format="PNG")
        return destination
    except OSError:
        raw = destination.with_suffix(".bin")
        raw.write_bytes(content)
        return raw


async def download_previews(
    scenes: list[Scene],
    out: Path,
    *,
    http: httpx.AsyncClient,
    clients: Mapping[str, DataPullClient],
    concurrency: int = 8,
    asset_name: str | None = None,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> dict[str, list[Path]]:
    """GET one preview per scene, concurrently, grouped by source.

    A missing asset or a failed GET skips that scene and does not abort the rest.

    Args:
        scenes: Scenes to download (already stamped with ``source``).
        out: Root output directory; files land in ``out/<source>/<id>.png``.
        http: Shared HTTP client.
        clients: Search clients, used for optional ``sign_href``.
        concurrency: Maximum in-flight downloads.
        asset_name: Optional asset key to prefer.
        on_progress: ``(source, saved, total)`` after each successful save.

    Returns:
        Source name to list of written paths.
    """
    totals: dict[str, int] = {}
    for scene in scenes:
        totals[scene.source] = totals.get(scene.source, 0) + 1
    saved_counts: dict[str, int] = dict.fromkeys(totals, 0)
    saved: dict[str, list[Path]] = {name: [] for name in totals}
    lock = asyncio.Lock()
    sem = asyncio.Semaphore(max(1, concurrency))

    async def one(scene: Scene) -> None:
        picked = pick_asset(scene, asset_name)
        if picked is None:
            return
        _key, asset = picked
        href = await _signed_href(clients.get(scene.source), asset.href)
        dest = out / scene.source / scene_filename(scene.id)
        try:
            async with sem:
                response = await http.get(href)
                response.raise_for_status()
                path = save_preview(response.content, dest)
        except (httpx.HTTPError, OSError, TypeError):
            return
        async with lock:
            saved[scene.source].append(path)
            saved_counts[scene.source] += 1
            if on_progress is not None:
                on_progress(scene.source, saved_counts[scene.source], totals[scene.source])

    await asyncio.gather(*(one(scene) for scene in scenes))
    return saved


async def _signed_href(client: DataPullClient | None, href: str) -> str:
    if client is None:
        return href
    sign = getattr(client, "sign_href", None)
    if sign is None:
        return href
    signed = sign(href)
    if asyncio.iscoroutine(signed):
        signed = await signed
    if not isinstance(signed, str):
        raise TypeError(f"sign_href must return str, got {type(signed)}")
    return signed
