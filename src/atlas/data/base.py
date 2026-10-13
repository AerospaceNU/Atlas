from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from datetime import date
from datetime import datetime as DateTime
from enum import StrEnum
from itertools import pairwise
from typing import Any, ClassVar, Self
from urllib.parse import urlparse

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator


class SceneKind(StrEnum):
    optical = "optical"
    sar = "sar"
    thermal = "thermal"
    lidar = "lidar"
    dem = "dem"
    atmosphere = "atmosphere"
    browse = "browse"
    detection = "detection"
    altimetry = "altimetry"
    precipitation = "precipitation"
    landcover = "landcover"
    landsurface = "landsurface"


class GeometryKind(StrEnum):
    point = "point"
    bbox = "bbox"
    polygon = "polygon"


class LimitMode(StrEnum):
    first = "first"
    last = "last"
    even = "even"


class BBox(BaseModel):
    """West/south/east/north box in degrees.

    Boxes may cross the antimeridian. ``west > east`` means the box runs east
    from ``west`` through 180° to ``east``, as in RFC 7946 and STAC. Geometry
    helpers keep that wrapping box. :meth:`antimeridian_parts` is the cut for
    an API that requires ``west <= east``.
    """

    west: float = Field(..., ge=-180, le=180)
    south: float = Field(..., ge=-90, le=90)
    east: float = Field(..., ge=-180, le=180)
    north: float = Field(..., ge=-90, le=90)

    @classmethod
    def from_corners(cls, lon1: float, lat1: float, lon2: float, lat2: float) -> Self:
        return cls.enclosing([lon1, lon2], [lat1, lat2])

    @classmethod
    def enclosing(cls, lons: Sequence[float], lats: Sequence[float]) -> Self:
        ordered = sorted(lons)
        west, east = ordered[0], ordered[-1]
        # The box is the circle minus its widest empty gap; start with the gap across 180°.
        widest_gap = 360.0 - (east - west)
        for left, right in pairwise(ordered):
            if right - left > widest_gap:
                widest_gap = right - left
                west, east = right, left
        return cls(west=west, south=min(lats), east=east, north=max(lats))

    @classmethod
    def from_point(cls, lon: float, lat: float, *, pad: float = 0.01) -> Self:
        return cls(
            west=max(-180.0, lon - pad),
            south=max(-90.0, lat - pad),
            east=min(180.0, lon + pad),
            north=min(90.0, lat + pad),
        )

    @property
    def lon_span(self) -> float:
        span = self.east - self.west
        return span + 360.0 if span < 0 else span

    def contains(self, lon: float, lat: float) -> bool:
        if not self.south <= lat <= self.north:
            return False
        if self.west > self.east:
            return lon >= self.west or lon <= self.east
        return self.west <= lon <= self.east

    def intersects(self, other: BBox) -> bool:
        """True if the boxes share any point, including boxes that cross 180°.

        Args:
            other: Box to test against this one.

        Returns:
            Whether the two boxes overlap or touch.
        """
        if self.north < other.south or other.north < self.south:
            return False
        return any(
            west <= other_east and other_west <= east
            for west, east in self._lon_ranges()
            for other_west, other_east in other._lon_ranges()
        )

    def iou(self, other: BBox) -> float:
        """Return the intersection-over-union of the two boxes.

        The score is the shared area divided by the combined area, in square
        degrees. Boxes that only touch score 0. A box that crosses 180° is
        split there before the longitude overlap is measured, so a dateline
        footprint is not compared as the long way around the globe.

        Args:
            other: Box to compare with this one.

        Returns:
            A score from 0 to 1. Identical boxes score 1.
        """
        intersection = self._overlap_area(other)
        if intersection <= 0.0:
            return 0.0
        union = self._area() + other._area() - intersection
        if union <= intersection:
            return 1.0
        return intersection / union

    def _area(self) -> float:
        height = self.north - self.south
        if height <= 0.0:
            return 0.0
        return self.lon_span * height

    def _overlap_area(self, other: BBox) -> float:
        height = min(self.north, other.north) - max(self.south, other.south)
        if height <= 0.0:
            return 0.0
        width = 0.0
        for west, east in self._lon_ranges():
            for other_west, other_east in other._lon_ranges():
                overlap = min(east, other_east) - max(west, other_west)
                if overlap > 0.0:
                    width += overlap
        return width * height

    def _lon_ranges(self) -> list[tuple[float, float]]:
        if self.west > self.east:
            return [(self.west, 180.0), (-180.0, self.east)]
        return [(self.west, self.east)]

    def antimeridian_parts(self) -> list[Self]:
        """Non-wrapping boxes that cover this one.

        WMS, the FIRMS area API, and GOES disk selection all require the west
        edge to be less than or equal to the east edge. A box that does not
        cross 180° is returned unchanged. A box that crosses is cut at ±180°.
        A cut with no width is omitted, because +180° and -180° are the same
        meridian and are already the edge of the other part.

        Returns:
            One or two boxes. Each has ``west <= east``.

        Raises:
            ValueError: The box crosses the antimeridian but neither side has width.
        """
        if self.west <= self.east:
            return [self]
        parts: list[Self] = []
        for west, east in self._lon_ranges():
            if east <= west:
                continue
            parts.append(type(self)(west=west, south=self.south, east=east, north=self.north))
        if not parts:
            raise ValueError(
                "bbox crosses the antimeridian but has no width on either side of 180 "
                f"(west={self.west}, east={self.east})"
            )
        return parts

    def as_list(self) -> list[float]:
        return [self.west, self.south, self.east, self.north]

    @classmethod
    def try_new(cls, **data: Any) -> Self | None:
        try:
            return cls.model_validate(data)
        except ValidationError:
            return None


_GEOM_EPS = 1e-9
# Past this, callers should simplify. The intersection test stays fast up to the cap.
MAX_POLYGON_VERTICES = 4096
_POLE_RING = (
    "ring wraps a pole; the boundary winds around a pole and does not close "
    "inside one 360-degree window, so the interior is not a single polygon a "
    "bounding box or a STAC intersects query can describe"
)


def _as_position(value: object) -> tuple[float, float]:
    if isinstance(value, str) or not isinstance(value, Sequence) or len(value) != 2:
        raise ValueError("each position must be lon, lat")
    lon, lat = value
    if isinstance(lon, bool) or isinstance(lat, bool):
        raise ValueError("each position must be lon, lat")
    if not isinstance(lon, int | float) or not isinstance(lat, int | float):
        raise ValueError("each position must be lon, lat")
    lon_f = float(lon)
    lat_f = float(lat)
    if not math.isfinite(lon_f) or not math.isfinite(lat_f):
        raise ValueError("positions must be finite")
    if not -180.0 <= lon_f <= 180.0:
        raise ValueError("longitude must be between -180 and 180")
    if not -90.0 <= lat_f <= 90.0:
        raise ValueError("latitude must be between -90 and 90")
    return lon_f, lat_f


def _same_position(left: tuple[float, float], right: tuple[float, float]) -> bool:
    return abs(left[0] - right[0]) <= _GEOM_EPS and abs(left[1] - right[1]) <= _GEOM_EPS


def _cross(
    start: tuple[float, float],
    end: tuple[float, float],
    point: tuple[float, float],
) -> float:
    return (end[0] - start[0]) * (point[1] - start[1]) - (end[1] - start[1]) * (point[0] - start[0])


def _within(left: float, right: float, value: float) -> bool:
    low, high = (left, right) if left <= right else (right, left)
    return low - _GEOM_EPS <= value <= high + _GEOM_EPS


def _point_on_segment(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> bool:
    if abs(_cross(start, end, point)) > _GEOM_EPS:
        return False
    return _within(start[0], end[0], point[0]) and _within(start[1], end[1], point[1])


def _segments_cross(
    a_start: tuple[float, float],
    a_end: tuple[float, float],
    b_start: tuple[float, float],
    b_end: tuple[float, float],
) -> bool:
    orientations = (
        _cross(a_start, a_end, b_start),
        _cross(a_start, a_end, b_end),
        _cross(b_start, b_end, a_start),
        _cross(b_start, b_end, a_end),
    )

    def _sign(value: float) -> int:
        if abs(value) <= _GEOM_EPS:
            return 0
        return 1 if value > 0 else -1

    signs = tuple(_sign(value) for value in orientations)
    if signs[0] == 0 and _point_on_segment(b_start, a_start, a_end):
        return True
    if signs[1] == 0 and _point_on_segment(b_end, a_start, a_end):
        return True
    if signs[2] == 0 and _point_on_segment(a_start, b_start, b_end):
        return True
    if signs[3] == 0 and _point_on_segment(a_end, b_start, b_end):
        return True
    return signs[0] * signs[1] < 0 and signs[2] * signs[3] < 0


def _unwrap_ring(ring: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    unwrapped: list[tuple[float, float]] = [ring[0]]
    for lon, lat in ring[1:]:
        shifted = lon
        previous = unwrapped[-1][0]
        while shifted - previous > 180.0:
            shifted -= 360.0
        while previous - shifted > 180.0:
            shifted += 360.0
        unwrapped.append((shifted, lat))
    closed = (
        abs(unwrapped[-1][0] - unwrapped[0][0]) <= 1e-6
        and abs(unwrapped[-1][1] - unwrapped[0][1]) <= 1e-6
    )
    span = max(point[0] for point in unwrapped) - min(point[0] for point in unwrapped)
    if not closed or span >= 360.0:
        raise ValueError(_POLE_RING)
    return unwrapped


def _ring_area(points: Sequence[tuple[float, float]]) -> float:
    total = 0.0
    for (x1, y1), (x2, y2) in pairwise(points):
        total += x1 * y2 - x2 * y1
    return total / 2.0


def _reverse_closed(ring: list[tuple[float, float]]) -> list[tuple[float, float]]:
    body = [ring[0], *reversed(ring[1:-1])]
    return [*body, body[0]]


def _planar_contains(planar: Sequence[tuple[float, float]], point: tuple[float, float]) -> bool:
    x, y = point
    edges = len(planar) - 1
    for index in range(edges):
        if _point_on_segment(point, planar[index], planar[index + 1]):
            return True
    inside = False
    for index in range(edges):
        x1, y1 = planar[index]
        x2, y2 = planar[index + 1]
        if (y1 > y) == (y2 > y):
            continue
        x_cross = (x2 - x1) * (y - y1) / (y2 - y1) + x1
        if x < x_cross:
            inside = not inside
    return inside


def _shift_near(lon: float, center: float) -> float:
    shifted = lon
    while shifted - center > 180.0:
        shifted -= 360.0
    while center - shifted > 180.0:
        shifted += 360.0
    return shifted


def _crossed_meridian(planar: Sequence[tuple[float, float]]) -> float | None:
    for (x1, _y1), (x2, _y2) in pairwise(planar):
        lo, hi = (x1, x2) if x1 <= x2 else (x2, x1)
        k = math.floor((lo - 180.0) / 360.0) + 1
        meridian = 180.0 + 360.0 * k
        if lo + 1e-9 < meridian < hi - 1e-9:
            return meridian
    return None


def _pkey(point: tuple[float, float]) -> tuple[float, float]:
    return (round(point[0], 8), round(point[1], 8))


def _dedupe_open(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    if not points:
        return []
    output = [points[0]]
    for point in points[1:]:
        if not _same_position(output[-1], point):
            output.append(point)
    return output


def _side_of_meridian(x: float, *, keep_left: bool, meridian: float) -> int:
    """1 on the kept side, 0 on the cut, -1 outside."""
    if abs(x - meridian) <= 1e-8:
        return 0
    if keep_left:
        return 1 if x < meridian else -1
    return 1 if x > meridian else -1


def _meridian_hit(
    start: tuple[float, float],
    end: tuple[float, float],
    meridian: float,
) -> tuple[float, float]:
    x1, y1 = start
    x2, y2 = end
    denom = x2 - x1
    if abs(denom) <= _GEOM_EPS:
        return meridian, y1
    t = min(1.0, max(0.0, (meridian - x1) / denom))
    return meridian, y1 + t * (y2 - y1)


def _inside_chains(
    planar: Sequence[tuple[float, float]],
    *,
    keep_left: bool,
    meridian: float,
) -> list[list[tuple[float, float]]]:
    """Pieces of the ring that stay on the kept side of ``meridian``.

    Each open chain runs from where the ring enters that side to where it
    leaves. A ring that never leaves is returned as one closed chain.
    """
    pts = list(planar[:-1])
    if not pts:
        return []

    def inside(point: tuple[float, float]) -> bool:
        return _side_of_meridian(point[0], keep_left=keep_left, meridian=meridian) >= 0

    outside_at = next(
        (
            index
            for index, point in enumerate(pts)
            if _side_of_meridian(point[0], keep_left=keep_left, meridian=meridian) < 0
        ),
        None,
    )
    if outside_at is None:
        closed = _dedupe_closed(list(planar))
        return [closed] if len(closed) >= 4 else []

    pts = pts[outside_at:] + pts[:outside_at]
    chains: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] | None = None
    for start, end in pairwise([*pts, pts[0]]):
        start_in = inside(start)
        end_in = inside(end)
        if start_in and end_in:
            if current is None:
                current = [start]
            current.append(end)
            continue
        if start_in and not end_in:
            if current is None:
                current = [start]
            current.append(_meridian_hit(start, end, meridian))
            chain = _dedupe_open(current)
            if len(chain) >= 2:
                chains.append(chain)
            current = None
            continue
        if not start_in and end_in:
            current = [_meridian_hit(start, end, meridian), end]
    return chains


def _cut_edges(
    chains: Sequence[Sequence[tuple[float, float]]],
    planar: Sequence[tuple[float, float]],
    *,
    keep_left: bool,
    meridian: float,
) -> list[list[tuple[float, float]]]:
    """Meridian edges that close chains, with the interior on the left.

    A segment between two crossings is emitted only when the ring covers the
    kept side there. Lobes that only meet along the cut stay separate polygons
    instead of one ring joined by a zero-width bridge.
    """
    points: dict[tuple[float, float], tuple[float, float]] = {}
    for chain in chains:
        if _same_position(chain[0], chain[-1]):
            continue
        points.setdefault(_pkey(chain[0]), chain[0])
        points.setdefault(_pkey(chain[-1]), chain[-1])
    ordered = sorted(points.values(), key=lambda point: (point[1], point[0]))
    edges: list[list[tuple[float, float]]] = []
    probe_x = meridian - 1e-7 if keep_left else meridian + 1e-7
    for lower, upper in pairwise(ordered):
        if abs(upper[1] - lower[1]) <= 1e-8:
            continue
        mid_y = (lower[1] + upper[1]) / 2.0
        if not _planar_contains(planar, (probe_x, mid_y)):
            continue
        if keep_left:
            edges.append([lower, upper])
        else:
            edges.append([upper, lower])
    return edges


def _link_rings(
    chains: Sequence[list[tuple[float, float]]],
    cuts: Sequence[list[tuple[float, float]]],
) -> list[list[tuple[float, float]]]:
    """Walk chain and cut edges into closed rings."""
    pieces: list[list[tuple[float, float]]] = [list(chain) for chain in chains]
    pieces.extend(list(cut) for cut in cuts)
    outgoing: dict[tuple[float, float], list[list[tuple[float, float]]]] = {}
    for piece in pieces:
        outgoing.setdefault(_pkey(piece[0]), []).append(piece)

    used: set[int] = set()
    rings: list[list[tuple[float, float]]] = []
    for piece in pieces:
        if id(piece) in used:
            continue
        used.add(id(piece))
        ring = list(piece)
        for _ in pieces:
            if _same_position(ring[0], ring[-1]):
                break
            nxt = next(
                (
                    candidate
                    for candidate in outgoing.get(_pkey(ring[-1]), [])
                    if id(candidate) not in used
                ),
                None,
            )
            if nxt is None:
                break
            used.add(id(nxt))
            ring.extend(nxt[1:])
        if len(ring) >= 4 and _same_position(ring[0], ring[-1]):
            rings.append(ring)
    return rings


def _halfplane_rings(
    planar: Sequence[tuple[float, float]],
    *,
    keep_left: bool,
    meridian: float,
) -> list[list[tuple[float, float]]]:
    """Polygons on one side of ``meridian``, one ring per disjoint lobe."""
    chains = _inside_chains(planar, keep_left=keep_left, meridian=meridian)
    if not chains:
        return []
    if len(chains) == 1 and _same_position(chains[0][0], chains[0][-1]):
        return chains
    return _link_rings(chains, _cut_edges(chains, planar, keep_left=keep_left, meridian=meridian))


def _dedupe_closed(ring: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    if not ring:
        return []
    body = [ring[0]]
    for point in ring[1:]:
        if not _same_position(body[-1], point):
            body.append(point)
    if len(body) == 1 or not _same_position(body[0], body[-1]):
        body.append(body[0])
    else:
        body[-1] = body[0]
    return body


def _wrap_lon(lon: float) -> float:
    while lon > 180.0:
        lon -= 360.0
    while lon < -180.0:
        lon += 360.0
    return lon


def _piece_coordinates(
    planar: Sequence[tuple[float, float]],
    *,
    eastern: bool,
    meridian: float,
) -> list[list[float]]:
    coords: list[tuple[float, float]] = []
    for x, y in planar:
        on_cut = abs(x - meridian) <= 1e-8
        lon = (meridian - 360.0 if eastern else meridian) if on_cut else x
        lon = _wrap_lon(lon)
        if not eastern and abs(lon + 180.0) <= 1e-8 and abs(x - meridian) <= 1e-8:
            lon = 180.0
        if eastern and abs(lon - 180.0) <= 1e-8 and abs(x - meridian) <= 1e-8:
            lon = -180.0
        coords.append((lon, y))
    coords = _dedupe_closed(coords)
    if len(coords) < 4 or abs(_ring_area(coords)) <= 1e-12:
        return []
    if _ring_area(coords) < 0:
        coords = _reverse_closed(coords)
    body = coords[:-1]
    start = min(range(len(body)), key=lambda index: (body[index][0], body[index][1]))
    rotated = body[start:] + body[:start]
    rotated.append(rotated[0])
    return [[lon, lat] for lon, lat in rotated]


def _batch_crosses(
    a_start: NDArray[np.float64],
    a_end: NDArray[np.float64],
    b_start: NDArray[np.float64],
    b_end: NDArray[np.float64],
) -> bool:
    """Whether segment ``a`` meets any segment in the ``b`` arrays."""
    ax = float(a_start[0])
    ay = float(a_start[1])
    bx = float(a_end[0])
    by = float(a_end[1])
    cx = b_start[:, 0]
    cy = b_start[:, 1]
    dx = b_end[:, 0]
    dy = b_end[:, 1]
    o1 = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    o2 = (bx - ax) * (dy - ay) - (by - ay) * (dx - ax)
    o3 = (dx - cx) * (ay - cy) - (dy - cy) * (ax - cx)
    o4 = (dx - cx) * (by - cy) - (dy - cy) * (bx - cx)

    def _sign(values: NDArray[np.float64]) -> NDArray[np.float64]:
        signed = np.sign(values).astype(np.float64, copy=False)
        signed[np.abs(values) <= _GEOM_EPS] = 0.0
        return signed

    s1 = _sign(o1)
    s2 = _sign(o2)
    s3 = _sign(o3)
    s4 = _sign(o4)
    if np.any((s1 * s2 < 0) & (s3 * s4 < 0)):
        return True
    touch = (s1 == 0) | (s2 == 0) | (s3 == 0) | (s4 == 0)
    if not bool(np.any(touch)):
        return False
    for index in np.flatnonzero(touch):
        if _segments_cross(
            (ax, ay),
            (bx, by),
            (float(cx[index]), float(cy[index])),
            (float(dx[index]), float(dy[index])),
        ):
            return True
    return False


def _ring_crosses_itself(planar: Sequence[tuple[float, float]]) -> bool:
    """Whether a non-adjacent pair of edges meets.

    Edge bounding boxes drop pairs that cannot meet, so a ring at
    ``MAX_POLYGON_VERTICES`` does not do a Python-level quadratic scan.
    """
    count = len(planar) - 1
    if count < 4:
        return False
    coords = np.asarray(planar[:-1], dtype=np.float64)
    start = coords
    end = np.roll(coords, -1, axis=0)
    lo = np.minimum(start, end)
    hi = np.maximum(start, end)
    for index in range(count - 2):
        j0 = index + 2
        j1 = count - 1 if index == 0 else count
        if j0 >= j1:
            continue
        overlap = np.nonzero(
            (hi[j0:j1, 0] >= lo[index, 0] - _GEOM_EPS)
            & (lo[j0:j1, 0] <= hi[index, 0] + _GEOM_EPS)
            & (hi[j0:j1, 1] >= lo[index, 1] - _GEOM_EPS)
            & (lo[j0:j1, 1] <= hi[index, 1] + _GEOM_EPS)
        )[0]
        if overlap.size == 0:
            continue
        chosen = overlap + j0
        if _batch_crosses(start[index], end[index], start[chosen], end[chosen]):
            return True
    return False


def _polygon_hits_rect(
    planar: Sequence[tuple[float, float]],
    rect: tuple[float, float, float, float],
) -> bool:
    xmin, ymin, xmax, ymax = rect
    if xmax < xmin or ymax < ymin:
        return False
    if any(
        xmin - _GEOM_EPS <= x <= xmax + _GEOM_EPS and ymin - _GEOM_EPS <= y <= ymax + _GEOM_EPS
        for x, y in planar[:-1]
    ):
        return True
    corners = ((xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax))
    if any(_planar_contains(planar, corner) for corner in corners):
        return True
    edges = (
        (corners[0], corners[1]),
        (corners[1], corners[2]),
        (corners[2], corners[3]),
        (corners[3], corners[0]),
    )
    for start, end in pairwise(planar):
        for edge_start, edge_end in edges:
            if _segments_cross(start, end, edge_start, edge_end):
                return True
    return False


class Polygon(BaseModel):
    """One exterior ring in degrees, longitude then latitude.

    The ring follows RFC 7946: at least four positions, and the first and last
    positions are the same. Holes are not an AOI. At most
    ``MAX_POLYGON_VERTICES`` vertices are accepted; a larger ring should be
    simplified before it is used as an area of interest. A ring may cross the
    antimeridian. The envelope is the narrow box from ``BBox.enclosing``.
    Clockwise rings are stored counter-clockwise.

    A ring that wraps a pole is rejected. Walking each edge the short way
    accumulates a full turn of longitude, so the ring does not close inside one
    360-degree window. Its interior contains the pole. A bounding box cannot
    describe that interior, and splitting the ring at plus or minus 180 (RFC
    7946 section 3.1.9) cannot either: that split assumes the ring lives in a
    single 360-degree window. Catalogs that only accept a bbox would be searched
    with a box that is not the area of interest.
    """

    ring: list[tuple[float, float]]

    @field_validator("ring", mode="before")
    @classmethod
    def _positions(cls, value: object) -> list[tuple[float, float]]:
        if isinstance(value, str) or not isinstance(value, Sequence):
            raise ValueError("polygon ring needs at least four positions")
        # Reject before reading coordinates so a huge ring never reaches the
        # self-intersection test. Callers should simplify past the cap.
        if len(value) > MAX_POLYGON_VERTICES + 1:
            raise ValueError(
                f"polygon ring has more than {MAX_POLYGON_VERTICES} vertices; simplify the polygon"
            )
        if len(value) < 4:
            raise ValueError("polygon ring needs at least four positions")
        return [_as_position(item) for item in value]

    @model_validator(mode="after")
    def _simple_ring(self) -> Self:
        ring = list(self.ring)
        if ring[0] != ring[-1]:
            if _same_position(ring[0], ring[-1]):
                ring[-1] = ring[0]
            else:
                raise ValueError("ring must start and end at the same position")
        for left, right in pairwise(ring):
            if left == right:
                raise ValueError("ring has a zero-length edge")
        planar = _unwrap_ring(ring)
        if abs(_ring_area(planar)) <= 1e-12:
            raise ValueError("polygon has no area")
        if _ring_crosses_itself(planar):
            raise ValueError("polygon ring crosses itself")
        if _ring_area(planar) < 0:
            ring = _reverse_closed(ring)
        self.ring = ring
        return self

    @classmethod
    def from_geojson(cls, geometry: dict[str, Any]) -> Self:
        """Build a ring from a GeoJSON Polygon.

        Args:
            geometry: A GeoJSON object with ``type`` ``Polygon``.

        Returns:
            The exterior ring.

        Raises:
            ValueError: If the object is not a single exterior ring.
        """
        if geometry.get("type") != "Polygon":
            raise ValueError("AOI geometry must be a Polygon")
        coordinates = geometry.get("coordinates")
        if not isinstance(coordinates, list) or not coordinates:
            raise ValueError("polygon coordinates are required")
        if len(coordinates) != 1:
            raise ValueError("polygon holes are not accepted")
        return cls(ring=coordinates[0])

    def envelope(self) -> BBox:
        """Narrow bounding box of the ring, including antimeridian rings.

        Returns:
            The west/south/east/north box that contains every vertex.
        """
        lons = [lon for lon, _lat in self.ring[:-1]]
        lats = [lat for _lon, lat in self.ring[:-1]]
        return BBox.enclosing(lons, lats)

    def contains(self, lon: float, lat: float) -> bool:
        """Whether a longitude/latitude is inside the ring or on its boundary.

        Args:
            lon: Longitude in degrees.
            lat: Latitude in degrees.

        Returns:
            True when the point is inside the AOI or on an edge.
        """
        if not math.isfinite(lon) or not math.isfinite(lat):
            return False
        planar = _unwrap_ring(self.ring)
        xs = [point[0] for point in planar]
        ys = [point[1] for point in planar]
        placed = _shift_near(lon, planar[0][0])
        if placed < min(xs) - _GEOM_EPS or placed > max(xs) + _GEOM_EPS:
            return False
        if lat < min(ys) - _GEOM_EPS or lat > max(ys) + _GEOM_EPS:
            return False
        return _planar_contains(planar, (placed, lat))

    def intersects_bbox(self, bbox: BBox) -> bool:
        """Whether this ring meets a box, including a box that crosses 180°.

        Args:
            bbox: Box to test. A point on the shared boundary counts.

        Returns:
            True when the ring and the box have any point in common.
        """
        planar = _unwrap_ring(self.ring)
        center = (min(point[0] for point in planar) + max(point[0] for point in planar)) / 2.0
        west = _shift_near(bbox.west, center)
        rect = (west, bbox.south, west + bbox.lon_span, bbox.north)
        return _polygon_hits_rect(planar, rect)

    def overlaps(self, scene: Scene) -> bool:
        """Whether a scene meets this area of interest.

        Point scenes must fall inside the ring. Every other scene is tested by
        footprint bounding-box overlap. ``Scene`` stores that box, not the
        catalog footprint ring, so a box that only clips the polygon still
        counts. A scene whose real outline misses the ring can be kept.

        Args:
            scene: Scene returned by any catalog.

        Returns:
            True when the scene should survive a polygon post-filter.
        """
        if scene.geometry_kind is GeometryKind.point:
            if scene.lon is None or scene.lat is None:
                return False
            return self.contains(scene.lon, scene.lat)
        return self.intersects_bbox(scene.bbox)

    def as_geojson(self) -> dict[str, Any]:
        """GeoJSON Polygon geometry for the ring.

        Returns:
            A ``Polygon`` geometry whose only ring is this exterior.
        """
        return {
            "type": "Polygon",
            "coordinates": [[[lon, lat] for lon, lat in self.ring]],
        }

    def as_stac_geometry(self) -> dict[str, Any]:
        """GeoJSON geometry safe to send as a STAC ``intersects`` value.

        A ring that crosses the antimeridian is cut at plus or minus 180. Each
        side becomes one or more polygons, so a concave ring that crosses the
        meridian twice does not keep a zero-width bridge along the cut (RFC
        7946 section 3.1.9). A ring that does not cross stays one Polygon.

        Returns:
            A Polygon or MultiPolygon geometry in longitude, latitude order.
        """
        planar = _unwrap_ring(self.ring)
        meridian = _crossed_meridian(planar)
        if meridian is None:
            return self.as_geojson()
        pieces: list[list[list[float]]] = []
        for keep_left, eastern in ((True, False), (False, True)):
            for ring in _halfplane_rings(planar, keep_left=keep_left, meridian=meridian):
                piece = _piece_coordinates(ring, eastern=eastern, meridian=meridian)
                if piece:
                    pieces.append(piece)
        pieces.sort(key=lambda ring: (ring[0][0], ring[0][1], ring))
        if len(pieces) <= 1:
            if len(pieces) == 1:
                return {"type": "Polygon", "coordinates": [pieces[0]]}
            return self.as_geojson()
        return {"type": "MultiPolygon", "coordinates": [[piece] for piece in pieces]}


class PullRequest(BaseModel):
    start_date: date
    end_date: date
    bbox: BBox
    polygon: Polygon | None = None
    min_cloud_cover: float | None = Field(default=None, ge=0, le=100)
    max_cloud_cover: float | None = Field(default=None, ge=0, le=100)
    limit: int | None = Field(default=None, ge=1)
    limit_mode: LimitMode | None = Field(default=None)

    @model_validator(mode="before")
    @classmethod
    def _derive_bbox(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if data.get("bbox") is not None or data.get("polygon") is None:
            return data
        polygon = data["polygon"]
        parsed = polygon if isinstance(polygon, Polygon) else Polygon.model_validate(polygon)
        return {**data, "bbox": parsed.envelope()}

    @model_validator(mode="after")
    def _validate_dates(self) -> Self:
        if self.end_date < self.start_date:
            raise ValueError("end_date must be on or after start_date")
        return self

    @model_validator(mode="after")
    def _polygon_inside_bbox(self) -> Self:
        if self.polygon is None:
            return self
        outside = [
            (lon, lat) for lon, lat in self.polygon.ring[:-1] if not self.bbox.contains(lon, lat)
        ]
        if outside:
            raise ValueError("polygon lies outside bbox")
        return self

    @model_validator(mode="after")
    def _validate_cloud_bounds(self) -> Self:
        if (
            self.min_cloud_cover is not None
            and self.max_cloud_cover is not None
            and self.max_cloud_cover < self.min_cloud_cover
        ):
            raise ValueError("max_cloud_cover must be greater than or equal to min_cloud_cover")
        return self


class Asset(BaseModel):
    href: str
    media_type: str | None = None
    title: str | None = None
    roles: list[str] = Field(default_factory=list)
    # First ``raster:bands`` entry. None when the item does not publish one.
    scale: float | None = None
    offset: float | None = None

    @field_validator("href")
    @classmethod
    def href_must_be_absolute_http(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("href must be an absolute http(s) URL")
        return value

    @field_validator("scale", "offset")
    @classmethod
    def scale_and_offset_must_be_finite(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if not math.isfinite(value):
            raise ValueError("scale and offset must be finite")
        return value

    @classmethod
    def try_new(cls, **data: Any) -> Self | None:
        try:
            return cls.model_validate(data)
        except ValidationError:
            return None


class Scene(BaseModel):
    id: str
    source: str = ""
    collection: str = ""
    kind: SceneKind = SceneKind.optical
    datetime: DateTime
    start_datetime: DateTime | None = None
    end_datetime: DateTime | None = None
    bbox: BBox
    geometry_kind: GeometryKind = GeometryKind.bbox
    lon: float | None = Field(default=None, ge=-180, le=180)
    lat: float | None = Field(default=None, ge=-90, le=90)
    gsd_m: float | None = Field(default=None, gt=0)
    platform: str
    instrument: str | None = None
    cloud_cover: float | None = Field(default=None, ge=0, le=100)
    footprint_is_request: bool = False
    assets: dict[str, Asset] = Field(default_factory=dict)
    # Common band name to a key in `assets`. Catalogs name the same band
    # differently: S30 NIR is B8A, L30 NIR is B05.
    band_aliases: dict[str, str] = Field(default_factory=dict)
    properties: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _consistency(self) -> Self:
        if (
            self.start_datetime is not None
            and self.end_datetime is not None
            and self.end_datetime < self.start_datetime
        ):
            raise ValueError("end_datetime must be on or after start_datetime")
        if self.geometry_kind is GeometryKind.point and (self.lon is None or self.lat is None):
            raise ValueError("point geometry requires lon and lat")
        unknown = set(self.band_aliases.values()) - set(self.assets)
        if unknown:
            raise ValueError(f"band_aliases point at missing assets: {sorted(unknown)}")
        return self

    @classmethod
    def try_new(cls, **data: Any) -> Self | None:
        try:
            return cls.model_validate(data)
        except ValidationError:
            return None


class PullResult(BaseModel):
    request: PullRequest
    scenes: list[Scene]
    # Opaque STAC next-page cursor. None when the response has no next link.
    next_cursor: str | None = None
    # Why the next page could not be recorded. None when the cursor is present
    # or the response has no next link. "signed" means the only continuation
    # was a signed href, which is not stored.
    next_cursor_unavailable: str | None = None

    @model_validator(mode="after")
    def _clip_to_polygon(self) -> Self:
        polygon = self.request.polygon
        if polygon is None:
            return self
        # Every catalog, STAC or not, is clipped here. The test is footprint
        # bounding-box overlap (a point-in-ring test for point scenes). Scene
        # does not carry the catalog footprint geometry, so a box that merely
        # clips the ring can keep a scene whose outline misses it. Bbox-only
        # catalogs search the envelope, so scenes outside the ring still arrive.
        self.scenes = [scene for scene in self.scenes if polygon.overlaps(scene)]
        return self


class DataPullClient(ABC):
    # The CLI catalog is derived from these ClassVars on registered clients.
    # Subclasses that set satellite + scene_kind are picked up automatically.
    satellite: ClassVar[str] = ""
    scene_kind: ClassVar[SceneKind] = SceneKind.optical

    @abstractmethod
    async def search(self, request: PullRequest) -> PullResult:
        raise NotImplementedError

    async def search_from_cursor(self, request: PullRequest, page_cursor: str) -> PullResult:
        """Continue a search from a stored page cursor.

        Sources that do not page return no new scenes and keep the cursor.
        Resume then does not send the page-one request again.

        Args:
            request: Pull whose filters produced ``page_cursor``.
            page_cursor: Cursor recorded for this source.

        Returns:
            An empty page that still carries ``page_cursor``.
        """
        return PullResult(request=request, scenes=[], next_cursor=page_cursor)

    async def resolve_scene_ids(self, scene_ids: Sequence[str]) -> Mapping[str, Scene] | None:
        """Return stored scenes whose ids still exist.

        Only STAC sources resolve replacements. ``StacApiClient`` confirms
        which ids the collection still serves. Every other source keeps this
        default and returns None, and the pull leaves that source's ids
        unchecked: they are not reported missing and they are not replaced.

        Args:
            scene_ids: Ids recorded in the pull manifest.

        Returns:
            A map of ids the catalog still serves. Missing keys are gone.
            None means this client cannot check ids, so the caller leaves
            the stored ids as they are.
        """
        del scene_ids
        return None

    async def find_scenes_by_place_time(
        self,
        *,
        collection: str,
        when: DateTime,
        footprint: BBox,
    ) -> list[Scene] | None:
        """Return scenes near one acquisition and footprint.

        Only STAC sources search this way. Other sources return None, and
        the pull does not invent a replacement for them.

        Args:
            collection: Collection the stored scene came from.
            when: Acquisition time of the stored scene.
            footprint: Footprint box of the stored scene.

        Returns:
            Catalog hits to filter by collection, time tolerance, footprint
            IoU, and grid key. None means this client cannot search that way,
            so the caller leaves the stored ids unchecked.
        """
        del collection, when, footprint
        return None
