from __future__ import annotations

from typing import Any, ClassVar

from atlas.data.base import PullRequest
from atlas.data.planetary_computer import PlanetaryComputerClient


class HLSSentinelClient(PlanetaryComputerClient):
    """Harmonized Landsat Sentinel-2 v2, Sentinel-2 at 30 m (PC ``hls2-s30``)."""

    default_collection: ClassVar[str] = "hls2-s30"

    def _build_query_filter(self, request: PullRequest) -> dict[str, Any] | None:
        if request.max_cloud_cover is None:
            return None
        return {"eo:cloud_cover": {"lte": request.max_cloud_cover}}
