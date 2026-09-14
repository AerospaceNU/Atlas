from __future__ import annotations

from typing import ClassVar

from atlas.data.base import SceneKind
from atlas.data.planetary_computer import PlanetaryComputerClient


class CopDemGlo30Client(PlanetaryComputerClient):
    """Copernicus DEM GLO-30. Static elevation; datetime is omitted from search."""

    default_collection: ClassVar[str] = "cop-dem-glo-30"
    include_datetime: ClassVar[bool] = False
    scene_kind: ClassVar[SceneKind] = SceneKind.dem
    nominal_gsd_m: ClassVar[float | None] = 30.0


class CopDemGlo90Client(PlanetaryComputerClient):
    """Copernicus DEM GLO-90. Static elevation; datetime is omitted from search."""

    default_collection: ClassVar[str] = "cop-dem-glo-90"
    include_datetime: ClassVar[bool] = False
    scene_kind: ClassVar[SceneKind] = SceneKind.dem
    nominal_gsd_m: ClassVar[float | None] = 90.0
