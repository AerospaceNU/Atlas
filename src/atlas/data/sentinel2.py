from __future__ import annotations

from typing import ClassVar

from atlas.data.base import SceneKind
from atlas.data.planetary_computer import PlanetaryComputerClient


class Sentinel2Client(PlanetaryComputerClient):
    default_collection: ClassVar[str] = "sentinel-2-l2a"
    scene_kind: ClassVar[SceneKind] = SceneKind.optical
    nominal_gsd_m: ClassVar[float | None] = 10.0
