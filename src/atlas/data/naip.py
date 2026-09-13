from __future__ import annotations

from typing import ClassVar

from atlas.data.planetary_computer import PlanetaryComputerClient


class NaipClient(PlanetaryComputerClient):
    """USDA NAIP ~1 m aerial imagery over the United States (PC ``naip``)."""

    default_collection: ClassVar[str] = "naip"
