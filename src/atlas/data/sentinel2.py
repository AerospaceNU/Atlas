from __future__ import annotations

from typing import Any, ClassVar

from atlas.data.base import PullRequest
from atlas.data.planetary_computer import PlanetaryComputerClient


class Sentinel2Client(PlanetaryComputerClient):
    default_collection: ClassVar[str] = "sentinel-2-l2a"

    def _build_query_filter(self, request: PullRequest) -> dict[str, Any] | None:
        if request.max_cloud_cover is None:
            return None
        return {"eo:cloud_cover": {"lte": request.max_cloud_cover}}
