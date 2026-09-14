from __future__ import annotations

from typing import ClassVar

from atlas.data.base import SceneKind
from atlas.data.modis_base import MODISClient


class MODISLandCoverClient(MODISClient):
    """MCD12Q1 - MODIS Land Cover Type (Terra + Aqua). Annual, 500 m."""

    short_name: ClassVar[str] = "MCD12Q1"
    scene_kind: ClassVar[SceneKind] = SceneKind.landcover
    nominal_gsd_m: ClassVar[float | None] = 500.0
