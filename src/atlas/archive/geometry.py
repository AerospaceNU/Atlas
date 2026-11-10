"""Footprint tests for scene queries, in plain Python (no geometry dependency).

Boxes are ``(west, south, east, north)`` tuples that never cross the
antimeridian; :func:`split_box` turns a wrapping :class:`BBox` into one or two
of them. Boundaries are inclusive: a scene whose edge touches the AOI matches,
the same rule as :meth:`BBox.contains`.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

from atlas.data.base import BBox

Box = tuple[float, float, float, float]
Ring = Sequence[Sequence[float]]


def split_box(bbox: BBox) -> list[Box]:
    """Split a box that crosses the antimeridian into its two halves."""
    if bbox.west > bbox.east:
        return [
            (bbox.west, bbox.south, 180.0, bbox.north),
            (-180.0, bbox.south, bbox.east, bbox.north),
        ]
    return [(bbox.west, bbox.south, bbox.east, bbox.north)]


def boxes_intersect(a: Box, b: Box) -> bool:
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def footprint_intersects(
    footprint: Sequence[Ring] | None,
    bbox: BBox,
    aoi: BBox,
) -> bool:
    """True if a scene's footprint (or its bbox when there is none) meets the AOI."""
    aoi_boxes = split_box(aoi)
    if not footprint:
        return any(boxes_intersect(s, a) for s in split_box(bbox) for a in aoi_boxes)
    return any(ring_intersects_box(ring, box) for ring in footprint for box in aoi_boxes)


def ring_intersects_box(ring: Ring, box: Box) -> bool:
    """True if a closed polygon ring and a box share any point."""
    west, south, east, north = box
    if any(west <= lon <= east and south <= lat <= north for lon, lat, *_ in ring):
        return True
    corners = [(west, south), (east, south), (east, north), (west, north)]
    if any(_point_in_ring(lon, lat, ring) for lon, lat in corners):
        return True
    edges = list(pairwise([*corners, corners[0]]))
    return any(
        _segments_intersect(p1, p2, q1, q2)
        for p1, p2 in pairwise((point[0], point[1]) for point in ring)
        for q1, q2 in edges
    )


def _point_in_ring(lon: float, lat: float, ring: Ring) -> bool:
    inside = False
    for (x1, y1, *_), (x2, y2, *_) in pairwise(ring):
        if (y1 > lat) != (y2 > lat):
            cross = x1 + (lat - y1) * (x2 - x1) / (y2 - y1)
            if lon < cross:
                inside = not inside
    return inside


def _segments_intersect(
    p1: tuple[float, float],
    p2: tuple[float, float],
    q1: tuple[float, float],
    q2: tuple[float, float],
) -> bool:
    d1 = _orient(q1, q2, p1)
    d2 = _orient(q1, q2, p2)
    d3 = _orient(p1, p2, q1)
    d4 = _orient(p1, p2, q2)
    if ((d1 > 0 > d2) or (d1 < 0 < d2)) and ((d3 > 0 > d4) or (d3 < 0 < d4)):
        return True
    return (
        (d1 == 0 and _on_segment(q1, q2, p1))
        or (d2 == 0 and _on_segment(q1, q2, p2))
        or (d3 == 0 and _on_segment(p1, p2, q1))
        or (d4 == 0 and _on_segment(p1, p2, q2))
    )


def _orient(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: tuple[float, float], b: tuple[float, float], p: tuple[float, float]) -> bool:
    return min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= p[1] <= max(a[1], b[1])
