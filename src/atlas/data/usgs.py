from __future__ import annotations

from typing import ClassVar

from atlas.data.base import SceneKind
from atlas.data.stac import StacApiClient

USGS_SEARCH_URL = "https://landsatlook.usgs.gov/stac-server/search"


class UsgsStacClient(StacApiClient):
    search_url: ClassVar[str] = USGS_SEARCH_URL


class UsgsLandsatC2L1Client(UsgsStacClient):
    default_collection: ClassVar[str] = "landsat-c2l1"
    satellite: ClassVar[str] = "landsat"
    scene_kind: ClassVar[SceneKind] = SceneKind.optical
    nominal_gsd_m: ClassVar[float | None] = 30.0


class UsgsLandsatC2L2SrClient(UsgsStacClient):
    default_collection: ClassVar[str] = "landsat-c2l2-sr"
    satellite: ClassVar[str] = "landsat"
    scene_kind: ClassVar[SceneKind] = SceneKind.optical
    nominal_gsd_m: ClassVar[float | None] = 30.0
