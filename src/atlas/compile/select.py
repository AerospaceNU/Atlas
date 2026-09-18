from __future__ import annotations

from atlas.data.base import BBox, GeometryKind, Scene, SceneKind

_CLOUD_KINDS = frozenset({SceneKind.optical, SceneKind.browse})
_COVERAGE_SKIP_KINDS = frozenset({SceneKind.browse, SceneKind.lidar, SceneKind.altimetry})


def _within_cloud_bounds(scene: Scene, min_cloud: float | None, max_cloud: float | None) -> bool:
    if scene.kind not in _CLOUD_KINDS or scene.cloud_cover is None:
        return True
    if min_cloud is not None and scene.cloud_cover < min_cloud:
        return False
    return max_cloud is None or scene.cloud_cover <= max_cloud


def filter_cloud(
    scenes: list[Scene], max_cloud: float | None, *, min_cloud: float | None = None
) -> list[Scene]:
    """Drop optical/browse scenes outside the cloud range.

    ``min_cloud`` keeps only scenes at or above that cover, which is how a
    cloud-seeking request (training a cloud detector) asks for cloudy pixels.

    Other kinds (SAR, thermal, detections, …) are kept regardless of cloud.
    Optical/browse with ``cloud_cover is None`` are kept at either bound.
    """
    if max_cloud is None and min_cloud is None:
        return list(scenes)
    return [s for s in scenes if _within_cloud_bounds(s, min_cloud, max_cloud)]


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

    Cloud is only considered for optical/browse. Other kinds sort by recency.
    """
    return sorted(
        scenes,
        key=lambda s: (
            s.cloud_cover if s.kind in _CLOUD_KINDS and s.cloud_cover is not None else -1.0,
            -s.datetime.timestamp(),
        ),
    )


def select(
    scenes: list[Scene],
    *,
    min_cloud: float | None = None,
    max_cloud: float | None = None,
    limit: int | None = None,
) -> list[Scene]:
    """Filter by cloud range, dedup, order clearest-first, and optionally cap count.

    Ordering stays clearest-first even when ``min_cloud`` is set, so a capped
    cloud-seeking request returns the least cloudy scenes that still clear the
    floor. Pass a larger ``limit`` if you want the cloudiest end.
    """
    chosen = sort_clearest(dedup(filter_cloud(scenes, max_cloud, min_cloud=min_cloud)))
    return chosen[:limit] if limit is not None else chosen


def representative_cloud(scenes: list[Scene]) -> float | None:
    """Lowest cloud cover among optical/browse scenes (None if none report cloud)."""
    values = [s.cloud_cover for s in scenes if s.kind in _CLOUD_KINDS and s.cloud_cover is not None]
    return min(values) if values else None


def _counts_for_coverage(scene: Scene) -> bool:
    """Area-filling footprints only. Points, AOI snapshots, browse, lidar, altimetry skip."""
    if scene.geometry_kind is GeometryKind.point or scene.footprint_is_request:
        return False
    return scene.kind not in _COVERAGE_SKIP_KINDS


def coverage_fraction(scenes: list[Scene], aoi: BBox, *, grid: int = 16) -> float:
    """Approximate fraction of the AOI covered by area-filling scene footprints.

    Samples a grid x grid lattice of points across the AOI and reports the
    fraction that fall inside at least one counted scene's bbox.
    """
    boxes = [s.bbox for s in scenes if _counts_for_coverage(s)]
    if not boxes:
        return 0.0
    hits = 0
    total = grid * grid
    for i in range(grid):
        lon = aoi.west + (aoi.east - aoi.west) * (i + 0.5) / grid
        for j in range(grid):
            lat = aoi.south + (aoi.north - aoi.south) * (j + 0.5) / grid
            if any(b.west <= lon <= b.east and b.south <= lat <= b.north for b in boxes):
                hits += 1
    return hits / total
