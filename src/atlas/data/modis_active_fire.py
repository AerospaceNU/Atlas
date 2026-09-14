from __future__ import annotations

from typing import ClassVar

from atlas.data.base import SceneKind
from atlas.data.modis_base import MODISClient


class MODISActiveFireClient(MODISClient):
    """MOD14A1 - MODIS/Terra Thermal Anomalies and Fire. Daily, 1 km."""

    short_name: ClassVar[str] = "MOD14A1"
    scene_kind: ClassVar[SceneKind] = SceneKind.thermal
    nominal_gsd_m: ClassVar[float | None] = 1000.0
