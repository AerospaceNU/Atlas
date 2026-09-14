from __future__ import annotations

from typing import ClassVar

from atlas.data.base import SceneKind
from atlas.data.planetary_computer import PlanetaryComputerClient


class Sentinel1Client(PlanetaryComputerClient):
    default_collection: ClassVar[str] = "sentinel-1-rtc"
    scene_kind: ClassVar[SceneKind] = SceneKind.sar
    nominal_gsd_m: ClassVar[float | None] = 10.0
