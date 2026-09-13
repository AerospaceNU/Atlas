from __future__ import annotations

from typing import ClassVar

from atlas.data.planetary_computer import PlanetaryComputerClient


class CopDemGlo30Client(PlanetaryComputerClient):
    """Copernicus DEM GLO-30. Static elevation; datetime is omitted from search."""

    default_collection: ClassVar[str] = "cop-dem-glo-30"
    include_datetime: ClassVar[bool] = False
