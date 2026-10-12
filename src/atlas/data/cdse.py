from __future__ import annotations

from typing import ClassVar

import httpx

from atlas.data.base import SceneKind
from atlas.data.credentials import CDSE_CREDENTIALS
from atlas.data.stac import StacApiClient

CDSE_SEARCH_URL = "https://stac.dataspace.copernicus.eu/v1/search"


class CdseStacClient(StacApiClient):
    """Copernicus Data Space Ecosystem STAC.

    Search is public when the catalog allows it. A 401 or 403 while
    ``CDSE_USERNAME`` or ``CDSE_PASSWORD`` is unset is a source error that
    names each missing variable and says to set the env var. The same status
    with both values present names the rejection and the status. The value
    is not included. CDSE has no ``/key`` command. The check runs after the
    response, not before the request.
    """

    search_url: ClassVar[str] = CDSE_SEARCH_URL
    required_credentials: ClassVar[tuple[str, ...]] = CDSE_CREDENTIALS

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
    satellite: ClassVar[str] = "sentinel3"
    scene_kind: ClassVar[SceneKind] = SceneKind.optical
    nominal_gsd_m: ClassVar[float | None] = 300.0


class CdseSentinel5PNo2Client(CdseStacClient):
    """Sentinel-5P TROPOMI L2 NO2, offline processing."""

    default_collection: ClassVar[str] = "sentinel-5p-l2-no2-offl"
    satellite: ClassVar[str] = "sentinel5p"
    scene_kind: ClassVar[SceneKind] = SceneKind.atmosphere
    nominal_gsd_m: ClassVar[float | None] = 5500.0


class CdseSentinel3SlstrLstClient(CdseStacClient):
    default_collection: ClassVar[str] = "sentinel-3-sl-2-lst-ntc"
    satellite: ClassVar[str] = "sentinel3"
    scene_kind: ClassVar[SceneKind] = SceneKind.thermal
    nominal_gsd_m: ClassVar[float | None] = 1000.0


class CdseSentinel3SlstrFrpClient(CdseStacClient):
    default_collection: ClassVar[str] = "sentinel-3-sl-2-frp-ntc"
    satellite: ClassVar[str] = "sentinel3"
    scene_kind: ClassVar[SceneKind] = SceneKind.thermal
    nominal_gsd_m: ClassVar[float | None] = 1000.0


class CdseSentinel3SralWatClient(CdseStacClient):
    default_collection: ClassVar[str] = "sentinel-3-sr-2-wat-ntc"
    satellite: ClassVar[str] = "sentinel3"
    scene_kind: ClassVar[SceneKind] = SceneKind.altimetry


class CdseSentinel5PCh4Client(CdseStacClient):
    default_collection: ClassVar[str] = "sentinel-5p-l2-ch4-offl"
    satellite: ClassVar[str] = "sentinel5p"
    scene_kind: ClassVar[SceneKind] = SceneKind.atmosphere
    nominal_gsd_m: ClassVar[float | None] = 5500.0


class CdseSentinel5PCoClient(CdseStacClient):
    default_collection: ClassVar[str] = "sentinel-5p-l2-co-offl"
    satellite: ClassVar[str] = "sentinel5p"
    scene_kind: ClassVar[SceneKind] = SceneKind.atmosphere
    nominal_gsd_m: ClassVar[float | None] = 5500.0


class CdseClmsBurntAreaClient(CdseStacClient):
    default_collection: ClassVar[str] = "clms_ba_global_300m_daily_v4_cog"
    satellite: ClassVar[str] = "clms"
    scene_kind: ClassVar[SceneKind] = SceneKind.detection
    nominal_gsd_m: ClassVar[float | None] = 300.0
