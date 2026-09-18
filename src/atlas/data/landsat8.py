from __future__ import annotations

from typing import Any, ClassVar

from atlas.data.base import PullRequest, SceneKind
from atlas.data.planetary_computer import PlanetaryComputerClient


class Landsat8Client(PlanetaryComputerClient):
    default_collection: ClassVar[str] = "landsat-c2-l2"
    satellite: ClassVar[str] = "landsat8"
    scene_kind: ClassVar[SceneKind] = SceneKind.optical
    nominal_gsd_m: ClassVar[float | None] = 30.0

    def _build_query_filter(self, request: PullRequest) -> dict[str, Any] | None:
        query: dict[str, Any] = {"platform": {"eq": "landsat-8"}}
        cloud = self._cloud_query(request)
        if cloud:
            query.update(cloud)
        return query
