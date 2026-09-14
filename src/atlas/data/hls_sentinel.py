from __future__ import annotations

from typing import ClassVar

from atlas.data.base import SceneKind
from atlas.data.planetary_computer import PlanetaryComputerClient


class HLSSentinelClient(PlanetaryComputerClient):
    """Harmonized Landsat Sentinel-2 v2, Sentinel-2 at 30 m (PC ``hls2-s30``)."""

    default_collection: ClassVar[str] = "hls2-s30"
    scene_kind: ClassVar[SceneKind] = SceneKind.optical
    nominal_gsd_m: ClassVar[float | None] = 30.0
