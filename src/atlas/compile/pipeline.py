from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path

import httpx

from atlas.compile.aggregate import aggregate
from atlas.compile.mosaic import build_mosaic
from atlas.compile.product import CompiledProduct, MosaicResult, MosaicSkip, SourceScenes
from atlas.compile.select import coverage_fraction, filter_cloud, representative_cloud, select
from atlas.data.base import DataPullClient, PullRequest, SceneKind
from atlas.data.planetary_computer import PlanetaryComputerClient
from atlas.data.sentinel1 import Sentinel1Client

# Optical counts as clear at or under this cloud cover when the request sets no cap.
_CLEAR_CLOUD_COVER = 20.0
# Below this AOI fraction under clear optical, SAR becomes the analysis input.
MIN_CLEAR_OPTICAL_COVERAGE = 0.5
# Registry name searched for SAR when the request brought no SAR source.
_SAR_FALLBACK_SOURCE = "sentinel1"


async def consolidate(
    request: PullRequest,
    clients: Mapping[str, DataPullClient],
    *,
    out_dir: Path | None = None,
    render: bool = True,
    min_clear_optical_coverage: float = MIN_CLEAR_OPTICAL_COVERAGE,
    sar_fallback: DataPullClient | None = None,
) -> CompiledProduct:
    """Search all sources, select scenes, and (optionally) render mosaics.

    When optical scenes at or under the cloud cap (the request's
    ``max_cloud_cover``, else 20%) cover less than
    ``min_clear_optical_coverage`` of the AOI, SAR becomes the analysis input
    and optical mosaics are skipped for cloud. Optical with no reported cloud
    counts as clear. SAR sources in ``clients`` are used first; if there are
    none, Sentinel-1 is searched. With no SAR scenes, optical stays the input.
    A request with ``min_cloud_cover`` set is seeking cloud and never falls back.

    Args:
        request: The AOI and time window to consolidate.
        clients: Search clients by source name.
        out_dir: Directory for rendered mosaics, written as ``<source>.png``.
            Required when ``render`` is true.
        render: Whether to render mosaics. False gives a metadata-only
            consolidation.
        min_clear_optical_coverage: AOI fraction, 0 to 1, that clear optical
            must cover before SAR is used instead.
        sar_fallback: Client searched as ``sentinel1`` for the SAR fallback.
            None builds a ``Sentinel1Client`` and closes it afterwards.

    Returns:
        The consolidated product. Sources selected for render that produced no
        mosaic appear in ``skipped_mosaics`` with a reason.

    Raises:
        ValueError: If ``render`` is true and ``out_dir`` is None.
    """
    if render and out_dir is None:
        raise ValueError("out_dir is required when render=True")

    catalog = await aggregate(request, clients)
    sources = catalog.ok_sources
    render_clients: Mapping[str, DataPullClient] = clients
    owned_fallback: Sentinel1Client | None = None
    analysis_input: SceneKind | None = None
    note: str | None = None

    optical = [s for src in sources for s in src.scenes if s.kind is SceneKind.optical]
    if optical:
        analysis_input = SceneKind.optical
        max_cloud = (
            request.max_cloud_cover if request.max_cloud_cover is not None else _CLEAR_CLOUD_COVER
        )
        clear = coverage_fraction(filter_cloud(optical, max_cloud), request.bbox)
        if request.min_cloud_cover is None and clear < min_clear_optical_coverage:
            if _SAR_FALLBACK_SOURCE not in clients and not _sources_of(sources, SceneKind.sar):
                fallback = sar_fallback
                if fallback is None:
                    fallback = owned_fallback = Sentinel1Client()
                render_clients = {**clients, _SAR_FALLBACK_SOURCE: fallback}
                extra = await aggregate(request, {_SAR_FALLBACK_SOURCE: fallback})
                sources = sources + extra.ok_sources
            sar = _sources_of(sources, SceneKind.sar)
            if sar:
                analysis_input = SceneKind.sar
                note = (
                    f"optical skipped for cloud: {clear:.0%} of the AOI has optical at or "
                    f"under {max_cloud:g}% cloud, below the "
                    f"{min_clear_optical_coverage:.0%} threshold; using SAR from {', '.join(sar)}"
                )

    scene_ids: dict[str, list[str]] = {}
    n_scenes: dict[str, int] = {}
    cloud_cover: dict[str, float | None] = {}
    coverage: dict[str, float] = {}
    render_targets: list[tuple[str, SourceScenes]] = []
    cloud_skips: list[MosaicSkip] = []

    for src in sources:
        chosen = select(
            src.scenes,
            min_cloud=request.min_cloud_cover,
            max_cloud=request.max_cloud_cover,
        )
        if not chosen:
            continue
        scene_ids[src.source] = [s.id for s in chosen]
        n_scenes[src.source] = len(chosen)
        cloud_cover[src.source] = representative_cloud(chosen)
        coverage[src.source] = coverage_fraction(chosen, request.bbox)
        if note is not None and any(s.kind is SceneKind.optical for s in src.scenes):
            cloud_skips.append(
                MosaicSkip(source=src.source, collection=src.collection, reason=note)
            )
            continue
        render_targets.append((src.source, src))

    mosaics: list[MosaicResult] = []
    skipped: list[MosaicSkip] = []
    try:
        if render and out_dir is not None:
            mosaics, skipped = await _render_all(request, render_clients, render_targets, out_dir)
            skipped = cloud_skips + skipped
    finally:
        if owned_fallback is not None:
            await owned_fallback.aclose()

    return CompiledProduct(
        request=request,
        sources=list(scene_ids),
        scene_ids=scene_ids,
        n_scenes=n_scenes,
        cloud_cover=cloud_cover,
        coverage=coverage,
        mosaics=mosaics,
        skipped_mosaics=skipped,
        analysis_input=analysis_input,
        analysis_note=note,
    )


def _sources_of(sources: list[SourceScenes], kind: SceneKind) -> list[str]:
    """Names of sources with at least one scene of ``kind``, in order."""
    return [src.source for src in sources if any(s.kind is kind for s in src.scenes)]


async def _render_all(
    request: PullRequest,
    clients: Mapping[str, DataPullClient],
    targets: list[tuple[str, SourceScenes]],
    out_dir: Path,
) -> tuple[list[MosaicResult], list[MosaicSkip]]:
    """Render each source's mosaic concurrently (off the event loop thread).

    Args:
        request: The AOI and time window being rendered.
        clients: Search clients by source name; only
            ``PlanetaryComputerClient`` instances can be rendered.
        targets: Source name and scenes for each source that survived
            selection.
        out_dir: Directory that receives one ``<source>.png`` per render.

    Returns:
        The rendered mosaics, and a ``MosaicSkip`` carrying a reason for every
        target that produced none.
    """

    def render_one(source: str, src: SourceScenes) -> MosaicResult | MosaicSkip:
        client = clients[source]
        if not isinstance(client, PlanetaryComputerClient):
            return MosaicSkip(
                source=source,
                collection=src.collection,
                reason=(
                    f"mosaic rendering supports the Planetary Computer API only; "
                    f"{type(client).__name__} is not a PlanetaryComputerClient"
                ),
            )
        # The clearest scene also carries the render recipe we want.
        sample = min(
            src.scenes,
            key=lambda s: s.cloud_cover if s.cloud_cover is not None else float("inf"),
        )
        try:
            return build_mosaic(
                client, request, sample, source=source, out_path=out_dir / f"{source}.png"
            )
        # KeyError and TypeError come from _register_search when Planetary
        # Computer answers 200 with a missing or non-string searchid.
        except (httpx.HTTPError, KeyError, OSError, TypeError, ValueError) as exc:
            return MosaicSkip(
                source=source,
                collection=src.collection,
                reason=f"{type(exc).__name__}: {exc}",
            )

    results = await asyncio.gather(
        *(asyncio.to_thread(render_one, source, src) for source, src in targets)
    )
    mosaics = [r for r in results if isinstance(r, MosaicResult)]
    skips = [r for r in results if isinstance(r, MosaicSkip)]
    return mosaics, skips
