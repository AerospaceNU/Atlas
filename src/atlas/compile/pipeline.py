from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path

from atlas.compile.aggregate import aggregate
from atlas.compile.mosaic import build_mosaic
from atlas.compile.product import CompiledProduct, MosaicResult, SourceScenes
from atlas.compile.select import coverage_fraction, representative_cloud, select
from atlas.data.base import DataPullClient, PullRequest
from atlas.data.planetary_computer import PlanetaryComputerClient


async def consolidate(
    request: PullRequest,
    clients: Mapping[str, DataPullClient],
    *,
    out_dir: Path | None = None,
    render: bool = True,
) -> CompiledProduct:
    """Search all sources, select scenes, and (optionally) render mosaics.

    `out_dir` is required when `render=True` (mosaics are written there as
    `<source>.png`). Set `render=False` for a metadata-only consolidation.
    """
    if render and out_dir is None:
        raise ValueError("out_dir is required when render=True")

    catalog = await aggregate(request, clients)

    scene_ids: dict[str, list[str]] = {}
    n_scenes: dict[str, int] = {}
    cloud_cover: dict[str, float | None] = {}
    coverage: dict[str, float] = {}
    render_targets: list[tuple[str, SourceScenes]] = []

    for src in catalog.ok_sources:
        chosen = select(src.scenes, max_cloud=request.max_cloud_cover)
        if not chosen:
            continue
        scene_ids[src.source] = [s.id for s in chosen]
        n_scenes[src.source] = len(chosen)
        cloud_cover[src.source] = representative_cloud(chosen)
        coverage[src.source] = coverage_fraction(chosen, request.bbox)
        render_targets.append((src.source, src))

    mosaics: list[MosaicResult] = []
    if render and out_dir is not None:
        mosaics = await _render_all(request, clients, render_targets, out_dir)

    return CompiledProduct(
        request=request,
        sources=list(scene_ids),
        scene_ids=scene_ids,
        n_scenes=n_scenes,
        cloud_cover=cloud_cover,
        coverage=coverage,
        mosaics=mosaics,
    )


async def _render_all(
    request: PullRequest,
    clients: Mapping[str, DataPullClient],
    targets: list[tuple[str, SourceScenes]],
    out_dir: Path,
) -> list[MosaicResult]:
    """Render each source's mosaic concurrently (off the event loop thread)."""

    def render_one(source: str, src: SourceScenes) -> MosaicResult | None:
        client = clients[source]
        if not isinstance(client, PlanetaryComputerClient):
            return None  # mosaic path only supports the PC API for now
        # The clearest scene also carries the render recipe we want.
        sample = min(
            src.scenes,
            key=lambda s: s.cloud_cover if s.cloud_cover is not None else float("inf"),
        )
        return build_mosaic(
            client, request, sample, source=source, out_path=out_dir / f"{source}.png"
        )

    results = await asyncio.gather(
        *(asyncio.to_thread(render_one, source, src) for source, src in targets)
    )
    return [r for r in results if r is not None]
