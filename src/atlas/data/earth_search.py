from __future__ import annotations

from typing import ClassVar

from atlas.data.base import SceneKind
from atlas.data.stac import StacApiClient

EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1/search"


class EarthSearchClient(StacApiClient):
    search_url: ClassVar[str] = EARTH_SEARCH_URL


class EarthSearchSentinel1GrdClient(EarthSearchClient):
    """Sentinel-1 GRD on AWS Open Data (not the PC RTC collection)."""

    default_collection: ClassVar[str] = "sentinel-1-grd"
    scene_kind: ClassVar[SceneKind] = SceneKind.sar
    nominal_gsd_m: ClassVar[float | None] = 10.0
