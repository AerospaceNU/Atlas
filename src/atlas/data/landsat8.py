from __future__ import annotations

from typing import Any, ClassVar

from atlas.data.base import PullRequest
from atlas.data.planetary_computer import PlanetaryComputerClient


class Landsat8Client(PlanetaryComputerClient):
    default_collection: ClassVar[str] = "landsat-c2-l2"

    def _build_query_filter(self, request: PullRequest) -> dict[str, Any] | None:
        query: dict = {"platform": {"eq": "landsat-8"}}
        if request.max_cloud_cover is None:
            return query
        query["eo:cloud_cover"] = {"lte": request.max_cloud_cover}
        return query
