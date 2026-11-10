"""Agent tool: look up already-indexed scenes without searching a catalog."""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, Field, model_validator

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import Tool, ToolResult
from atlas.agent.layout import project_data_dir
from atlas.agent.tools.retrieve_data_tool import area_to_bbox
from atlas.archive import (
    INDEX_FILENAME,
    IndexVersionError,
    QueryResult,
    SceneIndex,
    SceneOrder,
    SceneQuery,
    merge_results,
)
from atlas.data.base import BBox


class FindScenesInput(BaseModel):
    """Place, time, and filters for a scene lookup in the local index."""

    latitude: float | None = Field(
        default=None,
        ge=-90,
        le=90,
        description="Latitude of the point of interest, north positive. Give it with longitude.",
    )
    longitude: float | None = Field(
        default=None,
        ge=-180,
        le=180,
        description="Longitude of the point of interest, east positive. Give it with latitude.",
    )
    radius_km: float = Field(
        default=2.0,
        gt=0.0,
        le=100.0,
        description="Half-width in km of the square around latitude/longitude. Defaults to 2.",
    )
    bbox: BBox | None = Field(
        default=None,
        description="Area as west, south, east, north in decimal degrees, instead of a point.",
    )
    start_date: date | None = Field(
        default=None,
        description="First day of the window, YYYY-MM-DD in UTC, inclusive. Give with end_date.",
    )
    end_date: date | None = Field(
        default=None,
        description="Last day of the window, YYYY-MM-DD in UTC, inclusive.",
    )
    on_dates: list[date] = Field(
        default_factory=list,
        description=(
            "Specific days, answered one by one, instead of start_date/end_date. Use it for "
            "before/after: a day with no scene comes back as a gap, never filled from another day."
        ),
    )
    sources: list[str] = Field(
        default_factory=list,
        description="Registry sources that found the scene, such as sentinel2 or landsat9.",
    )
    satellites: list[str] = Field(
        default_factory=list,
        description="Satellite families, such as sentinel2, landsat8, or sentinel1.",
    )
    max_cloud_cover: float | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description=(
            "Keep scenes at or under this cloud percentage. Radar, lidar, and elevation pass "
            "because they do not see cloud. Optical scenes with unknown cloud are dropped and "
            "counted in excluded_unknown_cloud."
        ),
    )
    order: SceneOrder = Field(
        default=SceneOrder.time,
        description="time: oldest first. cloud: clearest first; use it with limit=1 for 'clearest'.",
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Most scenes per window, after ordering. Defaults to 5, at most 20.",
    )

    @model_validator(mode="after")
    def _one_place(self) -> Self:
        point = self.latitude is not None and self.longitude is not None
        if point == (self.bbox is not None):
            raise ValueError("give latitude and longitude, or a bbox, but not both")
        return self

    def to_query(self) -> SceneQuery:
        return SceneQuery(
            bbox=area_to_bbox(self.latitude, self.longitude, self.radius_km, self.bbox),
            start=self.start_date,
            end=self.end_date,
            on=self.on_dates,
            sources=self.sources,
            satellites=self.satellites,
            max_cloud_cover=self.max_cloud_cover,
            order=self.order,
            limit=self.limit,
        )


class FindScenesTool(Tool):
    name = "find_scenes"
    description = (
        "Look up satellite scenes Atlas has already found or downloaded, from the local scene "
        "index. It never searches a catalog, so it is instant and costs nothing. Use it before "
        "get_satellite_data, and to answer questions like 'the clearest Sentinel-2 here this "
        "week' (satellites=['sentinel2'], order='cloud', limit=1). Each scene is a short card "
        "with an index_id: cite the index_id of every scene you use. Each layer says whether "
        "it is absent, windowed (a chip covering its window), full, or missing (deleted); a "
        "local layer's locator key is its workspace path. A window with gap=true has no "
        "indexed scene: say so and do not substitute another date. The index holds only what "
        "earlier searches found, so to look further call get_satellite_data, which adds what "
        "it finds."
    )
    trust = "default"

    @property
    def input_model(self) -> type[BaseModel]:
        return FindScenesInput

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        query = FindScenesInput.model_validate(arguments).to_query()
        results: list[QueryResult] = []
        problems: list[str] = []
        for folder in index_folders(store):
            path = folder / INDEX_FILENAME
            try:
                with SceneIndex(path) as index:
                    results.append(index.query(query, workspace=store.root))
            except (IndexVersionError, sqlite3.Error, OSError) as exc:
                problems.append(f"cannot read {store.relative(path)}: {exc}")
        if not results:
            lines = problems or [
                "no scene index in this workspace yet; get_satellite_data builds one as it searches"
            ]
            return ToolResult(text="\n".join(lines))
        merged = merge_results(query, results)
        return ToolResult(text="\n".join([merged.to_json(), *problems]))


def index_folders(store: LocalArtifactStore) -> list[Path]:
    """Folders holding a scene index: where get_satellite_data saves, then ``atlas data``.

    Only folders that already have an index are returned, so a lookup never
    creates one.
    """
    folders = [store.resolve("data"), project_data_dir(store.root)]
    unique = list(dict.fromkeys(folder.resolve() for folder in folders))
    return [folder for folder in unique if (folder / INDEX_FILENAME).is_file()]
