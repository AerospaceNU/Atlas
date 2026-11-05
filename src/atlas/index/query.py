"""The scene query contract shared by the CLI and agent tools.

Results carry only scene cards and counts, never raw STAC metadata or asset
URLs. ``QueryResult.to_json()`` is the payload a tool hands to the model.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, Field, computed_field, model_validator

from atlas.data.base import BBox
from atlas.index.card import SceneCard


class SceneOrder(StrEnum):
    time = "time"  # oldest first
    cloud = "cloud"  # clearest first, then oldest; scenes without cloud data last


class SceneQuery(BaseModel):
    """Find indexed scenes over an AOI, for a date range or for specific dates.

    Dates are UTC calendar days and inclusive, like ``PullRequest``. With
    ``on``, each date is answered separately so a date with no imagery is
    reported as a gap instead of being filled from a neighbouring day.
    ``limit`` applies after ``order``, so ``order="cloud"`` keeps the
    clearest scenes when the window holds more than ``limit``.
    """

    bbox: BBox
    start: date | None = None
    end: date | None = None
    on: list[date] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    platforms: list[str] = Field(default_factory=list)
    max_cloud_cover: float | None = Field(default=None, ge=0, le=100)
    order: SceneOrder = SceneOrder.time
    limit: int = Field(default=50, ge=1, le=1000)

    @model_validator(mode="after")
    def _one_time_filter(self) -> Self:
        has_range = self.start is not None or self.end is not None
        if has_range == bool(self.on):
            raise ValueError("give either start and end, or one or more `on` dates")
        if has_range:
            if self.start is None or self.end is None:
                raise ValueError("a date range needs both start and end")
            if self.end < self.start:
                raise ValueError("end must be on or after start")
        return self

    def windows(self) -> list[tuple[date, date]]:
        """Inclusive (first, last) day pairs, one per result group."""
        if self.on:
            return [(day, day) for day in dict.fromkeys(self.on)]
        assert self.start is not None and self.end is not None
        return [(self.start, self.end)]


class QueryGroup(BaseModel):
    start: date
    end: date
    cards: list[SceneCard]
    matched: int  # scenes matching before `limit`; > len(cards) means truncated
    excluded_unknown_cloud: int = 0  # dropped by max_cloud_cover for lacking cloud data

    @computed_field  # type: ignore[prop-decorator]
    @property
    def gap(self) -> bool:
        """True when no scene matched this window; reported, never back-filled."""
        return not self.cards


class QueryResult(BaseModel):
    query: SceneQuery
    groups: list[QueryGroup]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def empty(self) -> bool:
        return all(group.gap for group in self.groups)

    @property
    def index_ids(self) -> list[str]:
        return [card.index_id for group in self.groups for card in group.cards]

    def to_json(self, *, indent: int | None = None) -> str:
        """Compact by default: this is the payload a tool hands to the model."""
        return self.model_dump_json(indent=indent)
