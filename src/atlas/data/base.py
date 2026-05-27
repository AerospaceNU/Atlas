from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import Any, Self

from pydantic import BaseModel, Field, model_validator


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

    def as_list(self) -> list[float]:
        return [self.west, self.south, self.east, self.north]


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


class Scene(BaseModel):
    id: str
    datetime: datetime
    bbox: BBox
    platform: str
    instrument: str | None = None
    cloud_cover: float | None = None
    assets: dict[str, Asset] = Field(default_factory=dict)
    properties: dict[str, Any] = Field(default_factory=dict)


class PullResult(BaseModel):
    request: PullRequest
    scenes: list[Scene]


class DataPullClient(ABC):
    @abstractmethod
    async def search(self, request: PullRequest) -> PullResult:
        raise NotImplementedError
