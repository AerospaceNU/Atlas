from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import date
from datetime import datetime as DateTime
from enum import StrEnum
from itertools import pairwise
from typing import Any, ClassVar, Self
from urllib.parse import urlparse

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
    helpers must handle wrapping boxes instead of splitting or rejecting them.
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

    def _lon_ranges(self) -> list[tuple[float, float]]:
        if self.west > self.east:
            return [(self.west, 180.0), (-180.0, self.east)]
        return [(self.west, self.east)]

    def as_list(self) -> list[float]:
        return [self.west, self.south, self.east, self.north]

    @classmethod
    def try_new(cls, **data: Any) -> Self | None:
        try:
            return cls.model_validate(data)
        except ValidationError:
            return None


_GEOM_EPS = 1e-9


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
        raise ValueError("ring wraps the globe")
    return unwrapped


def _ring_area(points: Sequence[tuple[float, float]]) -> float:
    total = 0.0
    for (x1, y1), (x2, y2) in pairwise(points):
        total += x1 * y2 - x2 * y1
    return total / 2.0


def _reverse_closed(ring: list[tuple[float, float]]) -> list[tuple[float, float]]:
    body = [ring[0], *reversed(ring[1:-1])]
    return [*body, body[0]]


class Polygon(BaseModel):
    """One exterior ring in degrees, longitude then latitude.

    The ring follows RFC 7946: at least four positions, and the first and last
    positions are the same. Holes are not an AOI. A ring may cross the
    antimeridian. The envelope is the narrow box from ``BBox.enclosing``.
    Clockwise rings are stored counter-clockwise.
    """

    ring: list[tuple[float, float]]

    @field_validator("ring", mode="before")
    @classmethod
    def _positions(cls, value: object) -> list[tuple[float, float]]:
        if isinstance(value, str) or not isinstance(value, Sequence) or len(value) < 4:
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
        edges = len(planar) - 1
        for i in range(edges):
            for j in range(i + 1, edges):
                adjacent = j == i + 1 or (i == 0 and j == edges - 1)
                if adjacent:
                    continue
                if _segments_cross(planar[i], planar[i + 1], planar[j], planar[j + 1]):
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
        placed = lon
        origin = planar[0][0]
        while placed - origin > 180.0:
            placed -= 360.0
        while origin - placed > 180.0:
            placed += 360.0
        if placed < min(xs) - _GEOM_EPS or placed > max(xs) + _GEOM_EPS:
            return False
        if lat < min(ys) - _GEOM_EPS or lat > max(ys) + _GEOM_EPS:
            return False
        point = (placed, lat)
        edges = len(planar) - 1
        for index in range(edges):
            if _point_on_segment(point, planar[index], planar[index + 1]):
                return True
        inside = False
        for index in range(edges):
            x1, y1 = planar[index]
            x2, y2 = planar[index + 1]
            if (y1 > lat) == (y2 > lat):
                continue
            x_cross = (x2 - x1) * (lat - y1) / (y2 - y1) + x1
            if placed < x_cross:
                inside = not inside
        return inside

    def as_geojson(self) -> dict[str, Any]:
        """GeoJSON Polygon geometry for the ring.

        Returns:
            A ``Polygon`` geometry whose only ring is this exterior.
        """
        return {
            "type": "Polygon",
            "coordinates": [[[lon, lat] for lon, lat in self.ring]],
        }


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

    @field_validator("href")
    @classmethod
    def href_must_be_absolute_http(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("href must be an absolute http(s) URL")
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


class DataPullClient(ABC):
    # The CLI catalog is derived from these ClassVars on registered clients.
    # Subclasses that set satellite + scene_kind are picked up automatically.
    satellite: ClassVar[str] = ""
    scene_kind: ClassVar[SceneKind] = SceneKind.optical

    @abstractmethod
    async def search(self, request: PullRequest) -> PullResult:
        raise NotImplementedError
