from __future__ import annotations

from typing import ClassVar

from atlas.data.base import SceneKind
from atlas.data.stac import StacApiClient

DEA_SEARCH = "https://explorer.dea.ga.gov.au/stac/search"
DEAfrica_SEARCH = "https://explorer.digitalearth.africa/stac/search"


class DeaS2ArdClient(StacApiClient):
    """Digital Earth Australia Sentinel-2 analysis-ready data."""

    search_url: ClassVar[str] = DEA_SEARCH
    default_collection: ClassVar[str] = "ga_s2am_ard_3"
    scene_kind: ClassVar[SceneKind] = SceneKind.optical
    nominal_gsd_m: ClassVar[float | None] = 10.0


class DeAfricaS2Client(StacApiClient):
    """Digital Earth Africa Sentinel-2 L2A."""

    search_url: ClassVar[str] = DEAfrica_SEARCH
    default_collection: ClassVar[str] = "s2_l2a"
    scene_kind: ClassVar[SceneKind] = SceneKind.optical
    nominal_gsd_m: ClassVar[float | None] = 10.0
