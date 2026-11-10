"""The scene query contract shared by the CLI and agent tools.

Results carry only scene cards and counts, never raw STAC metadata or asset
URLs. ``QueryResult.to_json()`` is the payload a tool hands to the model.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import date
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, Field, computed_field, model_validator

from atlas.archive.card import CloudStatus, SceneCard
from atlas.data.base import BBox


class SceneOrder(StrEnum):
    time = "time"  # oldest first
    cloud = "cloud"  # clearest first, then cloud not applicable, then cloud unknown


_CLOUD_RANK = {CloudStatus.value: 0, CloudStatus.not_applicable: 1, CloudStatus.unknown: 2}


def sort_cards(cards: Iterable[SceneCard], order: SceneOrder) -> list[SceneCard]:
    """Cards in ``order``; ties fall back to oldest first, then index id."""

    def key(card: SceneCard) -> tuple[int, float, float, str]:
        when = card.datetime.timestamp()
        if order is SceneOrder.cloud:
            return (_CLOUD_RANK[card.cloud_status], card.cloud_cover or 0.0, when, card.index_id)
        return (0, 0.0, when, card.index_id)

    return sorted(cards, key=key)


class SceneQuery(BaseModel):
    """Find indexed scenes over an AOI, for a date range or for specific dates.

    Dates are UTC calendar days and inclusive, like ``PullRequest``. With
    ``on``, each date is answered separately so a date with no imagery is
    reported as a gap instead of being filled from a neighbouring day.

    ``max_cloud_cover`` keeps scenes at or under the threshold and scenes whose
    sensor does not measure cloud (SAR, lidar, elevation). Scenes whose cloud
    is unknown are dropped and counted. ``limit`` applies after ``order``, so
    ``order="cloud"`` keeps the clearest scenes when a window holds more.
    """

    bbox: BBox
    start: date | None = None
    end: date | None = None
    on: list[date] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    satellites: list[str] = Field(default_factory=list)
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
    """Cards per window, from one index or several merged.

    ``index_dirs`` are the index folders read. ``index``-scheme locators, used
    only when a folder is outside the workspace, resolve against the first.
    """

    query: SceneQuery
    index_dirs: list[str]
    groups: list[QueryGroup]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def empty(self) -> bool:
        return all(group.gap for group in self.groups)

    @property
    def index_ids(self) -> list[str]:
        return [card.index_id for group in self.groups for card in group.cards]

    def to_json(self, *, indent: int | None = None) -> str:
        """Compact by default, without null fields: this is the payload a tool hands to the model."""
        return self.model_dump_json(indent=indent, exclude_none=True)


def merge_results(query: SceneQuery, results: Sequence[QueryResult]) -> QueryResult:
    """Answer ``query`` from several indexes, as if they were one.

    Each result must come from ``query``, so each already holds its own top
    ``limit`` cards and their union holds the merged top. A scene found in more
    than one index is kept once, preferring the copy with local data, and
    counted once in ``matched``.
    """
    groups: list[QueryGroup] = []
    for position, (first, last) in enumerate(query.windows()):
        cards: dict[str, SceneCard] = {}
        matched = 0
        excluded = 0
        for result in results:
            group = result.groups[position]
            matched += group.matched
            excluded += group.excluded_unknown_cloud
            for card in group.cards:
                held = cards.get(card.index_id)
                if held is None:
                    cards[card.index_id] = card
                    continue
                matched -= 1
                if card.has_local_data and not held.has_local_data:
                    cards[card.index_id] = card
        groups.append(
            QueryGroup(
                start=first,
                end=last,
                cards=sort_cards(cards.values(), query.order)[: query.limit],
                matched=matched,
                excluded_unknown_cloud=excluded,
            )
        )
    dirs = [folder for result in results for folder in result.index_dirs]
    return QueryResult(query=query, index_dirs=dirs, groups=groups)
