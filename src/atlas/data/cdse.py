from __future__ import annotations

from typing import ClassVar

import httpx

from atlas.data.stac import StacApiClient

CDSE_SEARCH_URL = "https://stac.dataspace.copernicus.eu/v1/search"


class CdseStacClient(StacApiClient):
    """Copernicus Data Space Ecosystem STAC. Search is public; S3 download needs a CDSE account."""

    search_url: ClassVar[str] = CDSE_SEARCH_URL

    def __init__(
        self,
        *,
        collection: str | None = None,
        client: httpx.AsyncClient | None = None,
        timeout: float = 60.0,
    ) -> None:
        super().__init__(collection=collection, client=client, timeout=timeout)


class CdseSentinel3OlciClient(CdseStacClient):
    """Sentinel-3 OLCI L1 EFR, not-time-critical."""

    default_collection: ClassVar[str] = "sentinel-3-olci-1-efr-ntc"


class CdseSentinel5PNo2Client(CdseStacClient):
    """Sentinel-5P TROPOMI L2 NO2, offline processing."""

    default_collection: ClassVar[str] = "sentinel-5p-l2-no2-offl"
