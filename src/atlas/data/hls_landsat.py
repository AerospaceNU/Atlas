from __future__ import annotations

from typing import Any, ClassVar

from atlas.data.base import PullRequest
from atlas.data.planetary_computer import PlanetaryComputerClient


class HLSLandsatClient(PlanetaryComputerClient):
    """Harmonized Landsat Sentinel-2 v2, Landsat 8/9 at 30 m (PC ``hls2-l30``)."""

    default_collection: ClassVar[str] = "hls2-l30"

    def _build_query_filter(self, request: PullRequest) -> dict[str, Any] | None:
        if request.max_cloud_cover is None:
            return None
        return {"eo:cloud_cover": {"lte": request.max_cloud_cover}}
