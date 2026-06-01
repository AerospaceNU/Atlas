from __future__ import annotations

from typing import ClassVar

from atlas.data.planetary_computer import PlanetaryComputerClient


class Sentinel1Client(PlanetaryComputerClient):
    default_collection: ClassVar[str] = "sentinel-1-rtc"
