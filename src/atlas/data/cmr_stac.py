"""NASA CMR-STAC / CloudSTAC collections (search is public; some assets need Earthdata)."""

from __future__ import annotations

from typing import ClassVar

from atlas.data.base import SceneKind
from atlas.data.stac import StacApiClient

LPCLOUD_SEARCH = "https://cmr.earthdata.nasa.gov/stac/LPCLOUD/search"
NSIDC_SEARCH = "https://cmr.earthdata.nasa.gov/stac/NSIDC_CPRD/search"


class CmrLpcloudClient(StacApiClient):
    search_url: ClassVar[str] = LPCLOUD_SEARCH


class CmrNsidcClient(StacApiClient):
    search_url: ClassVar[str] = NSIDC_SEARCH


class GediL2AClient(CmrLpcloudClient):
    default_collection: ClassVar[str] = "GEDI02_A_002"
    satellite: ClassVar[str] = "gedi"
    scene_kind: ClassVar[SceneKind] = SceneKind.lidar
    nominal_gsd_m: ClassVar[float | None] = 25.0


class ViirsVnp09gaClient(CmrLpcloudClient):
    """VIIRS/NPP surface reflectance daily (VNP09GA)."""

    default_collection: ClassVar[str] = "VNP09GA.v002"
    satellite: ClassVar[str] = "viirs"
    scene_kind: ClassVar[SceneKind] = SceneKind.optical
    nominal_gsd_m: ClassVar[float | None] = 500.0


class IceSat2Atl03Client(CmrNsidcClient):
    default_collection: ClassVar[str] = "ATL03_007"
    satellite: ClassVar[str] = "icesat2"
    scene_kind: ClassVar[SceneKind] = SceneKind.lidar


class SmapL3PassiveClient(CmrNsidcClient):
    default_collection: ClassVar[str] = "SPL3SMP_E_006"
    satellite: ClassVar[str] = "smap"
    scene_kind: ClassVar[SceneKind] = SceneKind.atmosphere
    nominal_gsd_m: ClassVar[float | None] = 9000.0
