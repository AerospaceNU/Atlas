from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from datetime import datetime as DateTime
from enum import StrEnum
from typing import Any, Self
from urllib.parse import urlparse

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator


class SceneKind(StrEnum):
    """What a scene *is*, for routing — not a vendor collection id."""

    optical = "optical"
    sar = "sar"
    thermal = "thermal"
    lidar = "lidar"
    dem = "dem"
    atmosphere = "atmosphere"  # retrieval rasters, including microwave (e.g. SMAP)
    browse = "browse"
    detection = "detection"
    altimetry = "altimetry"
    precipitation = "precipitation"
    landcover = "landcover"


class GeometryKind(StrEnum):
    point = "point"
    bbox = "bbox"
    polygon = "polygon"


class BBox(BaseModel):
    west: float = Field(..., ge=-180, le=180)
    south: float = Field(..., ge=-90, le=90)
    east: float = Field(..., ge=-180, le=180)
    north: float = Field(..., ge=-90, le=90)

    @classmethod
    def from_corners(cls, lon1: float, lat1: float, lon2: float, lat2: float) -> Self:
        return cls(
            west=min(lon1, lon2),
            south=min(lat1, lat2),
            east=max(lon1, lon2),
            north=max(lat1, lat2),
        )

    @classmethod
    def from_point(cls, lon: float, lat: float, *, pad: float = 0.01) -> Self:
        return cls(
            west=max(-180.0, lon - pad),
            south=max(-90.0, lat - pad),
            east=min(180.0, lon + pad),
            north=min(90.0, lat + pad),
        )

    def as_list(self) -> list[float]:
        return [self.west, self.south, self.east, self.north]

    @classmethod
    def try_new(cls, **data: Any) -> Self | None:
        try:
            return cls.model_validate(data)
        except ValidationError:
            return None


class PullRequest(BaseModel):
    start_date: date
    end_date: date
    bbox: BBox
    max_cloud_cover: float | None = Field(default=None, ge=0, le=100)
    limit: int = Field(default=100, ge=1, le=1000)

    @model_validator(mode="after")
    def _validate_dates(self) -> Self:
        if self.end_date < self.start_date:
            raise ValueError("end_date must be on or after start_date")
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
    @abstractmethod
    async def search(self, request: PullRequest) -> PullResult:
        raise NotImplementedError
