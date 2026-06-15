from __future__ import annotations

from typing import ClassVar

from atlas.data.modis_base import MODISClient


class MODISSurfaceReflClient(MODISClient):
    """MOD09GA - MODIS/Terra Surface Reflectance (bands 1-7). Daily, 500 m."""

    short_name: ClassVar[str] = "MOD09GA"
