from __future__ import annotations

import re
from datetime import date, datetime
from typing import Literal

from atlas.data.base import BBox, GeometryKind, LimitMode, Scene, SceneKind

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
    """Remove duplicate scenes by id, preserving order.

    Matching is id-only, not source-plus-id. Catalogs that serve the same
    observation under their own ids (Landsat via Planetary Computer and
    HLS; MODIS and GOES via two backends each) survive as separate entries.

    Args:
        scenes: Scenes to filter, in the order they should be kept.

    Returns:
        The scenes with later repeats of an id dropped. Group on
        ``(platform, datetime)`` downstream for one row per acquisition.
    """
    seen: set[str] = set()
    out: list[Scene] = []
    for s in scenes:
        if s.id not in seen:
            seen.add(s.id)
            out.append(s)
    return out


def pick_my_mode(
    scene: list[Scene], limit: int | None, limit_mode: LimitMode | None
) -> list[Scene]:
    if limit is None:
        return scene
    if len(scene) <= limit:
        return scene
    if limit_mode is None:
        limit_mode = LimitMode.first
    if limit_mode is LimitMode.first:
        return scene[:limit]
    elif limit_mode is LimitMode.last:
        return scene[-limit:]
    elif limit_mode is LimitMode.even:
        if limit == 1:
            return [scene[0]]
        scene_count = len(scene)
        result = []
        for i in range(limit):
            pick_index = round(i * (scene_count - 1) / (limit - 1))
            result.append(scene[pick_index])
        return result


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
    limit_mode: LimitMode | None = None,
) -> list[Scene]:
    """Filter by cloud range, dedup, order clearest-first, and optionally cap count.

    Ordering stays clearest-first even when ``min_cloud`` is set, so a capped
    cloud-seeking request returns the least cloudy scenes that still clear the
    floor. Pass a larger ``limit`` if you want the cloudiest end.
    """
    chosen = dedup(filter_cloud(scenes, max_cloud, min_cloud=min_cloud))
    if limit is None:
        return sort_clearest(chosen)
    return pick_my_mode(sorted(chosen, key=lambda s: s.datetime), limit, limit_mode)


Period = Literal["month", "season", "year"]
SelectMode = Literal["clearest", "newest", "oldest", "even", "per_month", "per_period"]

_STATE_WORDS = {
    "asc": "ascending",
    "ascending": "ascending",
    "desc": "descending",
    "descending": "descending",
}
_ORBIT_RE = re.compile(r"^(?:(\d+)(?::([a-z]+))?|([a-z]+))$")
_NO_CLOUD_SCORE = 101.0


def acquisition_key(scene: Scene) -> tuple[str, date]:
    """Return the key shared by the tiles of one acquisition (platform and UTC day)."""
    return (scene.platform, scene.datetime.date())


def period_key(moment: datetime, period: Period) -> tuple[int, int]:
    """Return the bucket of ``moment`` for ``period``.

    Seasons are meteorological: December belongs to the winter that ends the next
    February, so the key year is the year of that February (0 DJF, 1 MAM, 2 JJA, 3 SON).
    """
    if period == "month":
        return (moment.year, moment.month)
    if period == "year":
        return (moment.year, 0)
    year = moment.year + 1 if moment.month == 12 else moment.year
    return (year, (moment.month % 12) // 3)


def _acquisition_score(group: list[Scene]) -> float:
    """Lower is better: mean known cloud, 101 when unknown, 0 for kinds without cloud."""
    cloudy = [s for s in group if s.kind in _CLOUD_KINDS]
    if not cloudy:
        return 0.0
    known = [s.cloud_cover for s in cloudy if s.cloud_cover is not None]
    return sum(known) / len(known) if known else _NO_CLOUD_SCORE


def select_per_period(scenes: list[Scene], period: Period = "month") -> list[Scene]:
    """Keep the best acquisition of each period, with every tile of that acquisition.

    Best is the lowest mean cloud cover, then the newest. SAR and other kinds
    without cloud score 0 and are kept. Scenes of one acquisition are never split.

    Args:
        scenes: Candidate scenes, already deduplicated.
        period: Bucket size: month, meteorological season, or year.

    Returns:
        The chosen scenes sorted by time.
    """
    acquisitions: dict[tuple[str, date], list[Scene]] = {}
    for scene in scenes:
        acquisitions.setdefault(acquisition_key(scene), []).append(scene)
    best: dict[tuple[int, int], list[Scene]] = {}
    best_rank: dict[tuple[int, int], tuple[float, float]] = {}
    for group in acquisitions.values():
        bucket = period_key(group[0].datetime, period)
        rank = (_acquisition_score(group), -max(s.datetime.timestamp() for s in group))
        if bucket not in best or rank < best_rank[bucket]:
            best[bucket], best_rank[bucket] = group, rank
    return sorted((s for g in best.values() for s in g), key=lambda s: (s.datetime, s.id))


def select_scenes(
    scenes: list[Scene],
    mode: SelectMode,
    *,
    limit: int,
    period: Period = "month",
    max_cloud: float | None = None,
) -> list[Scene]:
    """Filter by cloud, deduplicate, then pick scenes by ``mode``.

    ``per_month`` and ``per_period`` ignore ``limit``; the caller caps their total.
    """
    chosen = dedup(filter_cloud(scenes, max_cloud))
    by_time = sorted(chosen, key=lambda s: (s.datetime, s.id))
    if mode == "clearest":
        return sort_clearest(chosen)[:limit]
    if mode == "newest":
        return by_time[::-1][:limit]
    if mode == "oldest":
        return by_time[:limit]
    if mode == "even":
        return pick_my_mode(by_time, limit, LimitMode.even)
    return select_per_period(chosen, "month" if mode == "per_month" else period)


def parse_orbit(orbit: str) -> tuple[str | None, str | None]:
    """Parse ``8``, ``8:ascending`` or ``ascending`` into (relative orbit, state).

    Raises:
        ValueError: The text is none of those forms.
    """
    match = _ORBIT_RE.match(orbit.strip().lower())
    word = (match.group(2) or match.group(3)) if match else None
    if match is None or (word is not None and word not in _STATE_WORDS):
        raise ValueError(
            f"orbit {orbit!r} is not understood. Fix: use a relative orbit number, a direction "
            "(ascending or descending), or both as 8:descending"
        )
    return match.group(1), _STATE_WORDS[word] if word else None


def orbit_of(scene: Scene) -> tuple[str | None, str | None]:
    """Return the scene's (relative orbit, orbit state) properties, None when absent."""
    rel = scene.properties.get("sat:relative_orbit")
    state = scene.properties.get("sat:orbit_state")
    return (None if rel is None else str(rel), None if state is None else str(state).lower())


def filter_orbit(scenes: list[Scene], orbit: str) -> tuple[list[Scene], list[Scene]]:
    """Split scenes into (kept, dropped) by relative orbit and/or orbit state.

    A scene missing a property the filter needs is dropped.
    """
    want_rel, want_state = parse_orbit(orbit)
    kept: list[Scene] = []
    dropped: list[Scene] = []
    for scene in scenes:
        rel, state = orbit_of(scene)
        ok = (want_rel is None or rel == want_rel) and (want_state is None or state == want_state)
        (kept if ok else dropped).append(scene)
    return kept, dropped


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
        lon = aoi.west + aoi.lon_span * (i + 0.5) / grid
        if lon > 180:
            lon -= 360
        for j in range(grid):
            lat = aoi.south + (aoi.north - aoi.south) * (j + 0.5) / grid
            if any(b.contains(lon, lat) for b in boxes):
                hits += 1
    return hits / total
