"""NASA CMR-STAC / CloudSTAC collections (search is public; some assets need Earthdata)."""

from __future__ import annotations

from typing import ClassVar

from atlas.data.stac import StacApiClient

LPCLOUD_SEARCH = "https://cmr.earthdata.nasa.gov/stac/LPCLOUD/search"
NSIDC_SEARCH = "https://cmr.earthdata.nasa.gov/stac/NSIDC_CPRD/search"


class CmrLpcloudClient(StacApiClient):
    search_url: ClassVar[str] = LPCLOUD_SEARCH


class CmrNsidcClient(StacApiClient):
    search_url: ClassVar[str] = NSIDC_SEARCH


class GediL2AClient(CmrLpcloudClient):
    default_collection: ClassVar[str] = "GEDI02_A_002"


class ViirsVnp09gaClient(CmrLpcloudClient):
    """VIIRS/NPP surface reflectance daily (VNP09GA)."""

    default_collection: ClassVar[str] = "VNP09GA.v002"


class IceSat2Atl03Client(CmrNsidcClient):
    default_collection: ClassVar[str] = "ATL03_007"


class SmapL3PassiveClient(CmrNsidcClient):
    default_collection: ClassVar[str] = "SPL3SMP_E_006"
