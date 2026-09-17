from __future__ import annotations

from typing import ClassVar

from atlas.data.base import SceneKind
from atlas.data.planetary_computer import PlanetaryComputerClient


class NaipClient(PlanetaryComputerClient):
    """USDA NAIP ~1 m aerial imagery over the United States (PC ``naip``)."""

    default_collection: ClassVar[str] = "naip"
    satellite: ClassVar[str] = "naip"
    scene_kind: ClassVar[SceneKind] = SceneKind.optical
    nominal_gsd_m: ClassVar[float | None] = 1.0
