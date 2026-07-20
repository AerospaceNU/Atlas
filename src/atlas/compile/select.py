"""Scene selection: filter, dedup, prioritize, and estimate AOI coverage.

This is the decision layer between raw search results and compositing. It works
purely on `Scene` metadata (no pixels), so it is cheap and provider-agnostic.
Cloud *masking* (per-pixel, from SCL/qa bands) is a separate, heavier concern
that belongs to the local-COG path and is intentionally not done here.
"""

from __future__ import annotations

from atlas.data.base import BBox, Scene


def filter_cloud(scenes: list[Scene], max_cloud: float | None) -> list[Scene]:
    """Drop scenes above the cloud threshold. Scenes without a cloud value
    (e.g. SAR) are always kept."""
    if max_cloud is None:
        return list(scenes)
    return [s for s in scenes if s.cloud_cover is None or s.cloud_cover <= max_cloud]


def dedup(scenes: list[Scene]) -> list[Scene]:
    """Remove duplicate scenes by id, preserving order."""
    seen: set[str] = set()
    out: list[Scene] = []
    for s in scenes:
        if s.id not in seen:
            seen.add(s.id)
            out.append(s)
    return out


def sort_clearest(scenes: list[Scene]) -> list[Scene]:
    """Order so the best pixels come first: lowest cloud, then most recent.
    Scenes without cloud cover (SAR) sort by recency alone."""
    return sorted(
        scenes,
        key=lambda s: (
            s.cloud_cover if s.cloud_cover is not None else -1.0,
            -s.datetime.timestamp(),
        ),
    )


def select(
    scenes: list[Scene], *, max_cloud: float | None = None, limit: int | None = None
) -> list[Scene]:
    """Filter by cloud, dedup, order clearest-first, and optionally cap count."""
    chosen = sort_clearest(dedup(filter_cloud(scenes, max_cloud)))
    return chosen[:limit] if limit is not None else chosen


def representative_cloud(scenes: list[Scene]) -> float | None:
    """Lowest cloud cover among scenes (None if no scene reports cloud)."""
    values = [s.cloud_cover for s in scenes if s.cloud_cover is not None]
    return min(values) if values else None


def coverage_fraction(scenes: list[Scene], aoi: BBox, *, grid: int = 16) -> float:
    """Approximate fraction of the AOI covered by the union of scene footprints.

    Samples a grid x grid lattice of points across the AOI and reports the
    fraction that fall inside at least one scene's bbox. Cheap and coarse, but
    enough to tell the difference between full coverage and a clipped sliver.
    """
    if not scenes:
        return 0.0
    boxes = [s.bbox for s in scenes]
    hits = 0
    total = grid * grid
    for i in range(grid):
        lon = aoi.west + (aoi.east - aoi.west) * (i + 0.5) / grid
        for j in range(grid):
            lat = aoi.south + (aoi.north - aoi.south) * (j + 0.5) / grid
            if any(b.west <= lon <= b.east and b.south <= lat <= b.north for b in boxes):
                hits += 1
    return hits / total
