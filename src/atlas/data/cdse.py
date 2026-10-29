from __future__ import annotations

import asyncio
import os
import time
from typing import ClassVar
from urllib.parse import urlparse

import httpx

from atlas.data.base import SceneKind
from atlas.data.credentials import CDSE_CREDENTIALS, CDSE_PASSWORD, CDSE_USERNAME
from atlas.data.stac import StacApiClient

CDSE_SEARCH_URL = "https://stac.dataspace.copernicus.eu/v1/search"
CDSE_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
)
CDSE_DOWNLOAD_HOST = "download.dataspace.copernicus.eu"
# Refresh this long before the token's stated expiry.
_TOKEN_MARGIN_SECONDS = 60.0


class CdseStacClient(StacApiClient):
    """Copernicus Data Space Ecosystem STAC.

    Search is public when the catalog allows it. A 401 or 403 while
    ``CDSE_USERNAME`` or ``CDSE_PASSWORD`` is unset is a source error that
    names each missing variable and says to set the env var. The same status
    with both values present names the rejection and the status. The value
    is not included. CDSE has no ``/key`` command. The check runs after the
    response, not before the request.

    Downloads go to the OData https alternate of each asset. The username and
    password are exchanged for an access token, which is sent only to
    ``download.dataspace.copernicus.eu`` and reused until shortly before it
    expires. A rejected exchange is an HTTP error for that scene. The values
    are never logged.
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
        self._access_token: str | None = None
        self._token_expires_at = 0.0
        self._token_lock = asyncio.Lock()

    async def download_headers(self, href: str) -> dict[str, str]:
        parsed = urlparse(href)
        if parsed.scheme != "https" or (parsed.hostname or "").lower() != CDSE_DOWNLOAD_HOST:
            return {}
        token = await self._token()
        return {"Authorization": f"Bearer {token}"} if token else {}

    async def _token(self) -> str | None:
        """Access token for downloads, or None when the credentials are unset.

        Raises:
            httpx.HTTPStatusError: CDSE rejected the username and password.
        """
        username = os.environ.get(CDSE_USERNAME, "").strip()
        password = os.environ.get(CDSE_PASSWORD, "")
        if not username or not password.strip():
            return None
        async with self._token_lock:
            if self._access_token and time.monotonic() < self._token_expires_at:
                return self._access_token
            response = await self._client.post(
                CDSE_TOKEN_URL,
                data={
                    "grant_type": "password",
                    "client_id": "cdse-public",
                    "username": username,
                    "password": password,
                },
            )
            response.raise_for_status()
            payload = response.json()
            self._access_token = str(payload["access_token"])
            lifetime = float(payload.get("expires_in", 600))
            self._token_expires_at = time.monotonic() + max(0.0, lifetime - _TOKEN_MARGIN_SECONDS)
            return self._access_token


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
