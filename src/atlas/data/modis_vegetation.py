from __future__ import annotations

from typing import ClassVar

from atlas.data.modis_base import MODISClient


class MODISVegetationClient(MODISClient):
    """MOD13A2 - MODIS/Terra Vegetation Indices (NDVI, EVI). 16-day, 1 km."""

    short_name: ClassVar[str] = "MOD13A2"
