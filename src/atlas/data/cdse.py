from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Mapping
from typing import Any, ClassVar
from urllib.parse import urlparse

import httpx

from atlas.data.base import PullRequest, PullResult, Scene, SceneKind
from atlas.data.credentials import (
    CDSE_CREDENTIALS,
    CDSE_PASSWORD,
    CDSE_USERNAME,
    CredentialRejectedError,
    MissingCredentialError,
)
from atlas.data.stac import StacApiClient

CDSE_SEARCH_URL = "https://stac.dataspace.copernicus.eu/v1/search"
CDSE_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
)
# A full product is served from ``download``; a per-node asset (a single band
# or metadata file) is served from ``zipper``. Both need the same bearer token.
CDSE_DOWNLOAD_HOST = "download.dataspace.copernicus.eu"
CDSE_ASSET_HOST = "zipper.dataspace.copernicus.eu"
CDSE_DOWNLOAD_HOSTS = frozenset({CDSE_DOWNLOAD_HOST, CDSE_ASSET_HOST})
# Refresh this long before the token's stated expiry. A whole-file GET reuses
# its headers across up to three attempts with Retry-After waits of up to 60 s
# each, plus the transfer itself, so the margin covers several minutes.
_TOKEN_MARGIN_SECONDS = 300.0
# Keycloak answers a wrong username or password with 401, or 400 invalid_grant.
_REJECTED_GRANT_STATUSES = frozenset({400, 401})


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
    ``download.dataspace.copernicus.eu`` and ``zipper.dataspace.copernicus.eu``
    and reused until a few minutes
    before it expires. A download from that host with either variable unset
    raises ``MissingCredentialError`` naming the unset variables, and sends no
    token request. A rejected exchange (401, or 400 ``invalid_grant``) raises
    ``CredentialRejectedError`` and is remembered for the client's lifetime:
    later scenes fail with the same error without posting the password again,
    so a wrong password cannot lock the account. The values are never logged.
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
        # Status of a rejected password grant; a fresh error is raised each time
        # so one shared exception does not collect every scene's traceback.
        self._rejected_status: int | None = None

    async def download_headers(self, href: str) -> dict[str, str]:
        parsed = urlparse(href)
        if parsed.scheme != "https" or (parsed.hostname or "").lower() not in CDSE_DOWNLOAD_HOSTS:
            return {}
        return {"Authorization": f"Bearer {await self._token()}"}

    async def _token(self) -> str:
        """Access token for downloads.

        Raises:
            MissingCredentialError: ``CDSE_USERNAME`` or ``CDSE_PASSWORD`` is
                unset. No request is sent.
            CredentialRejectedError: CDSE rejected the username and password,
                now or earlier in this client's lifetime.
            httpx.HTTPStatusError: The token endpoint failed for another
                reason. That failure is not remembered.
        """
        username = os.environ.get(CDSE_USERNAME, "").strip()
        password = os.environ.get(CDSE_PASSWORD, "")
        missing = tuple(
            name
            for name, value in ((CDSE_USERNAME, username), (CDSE_PASSWORD, password.strip()))
            if not value
        )
        if missing:
            raise MissingCredentialError(*missing)
        async with self._token_lock:
            if self._rejected_status is not None:
                raise CredentialRejectedError(self._rejected_status, *CDSE_CREDENTIALS)
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
            if _is_rejected_grant(response):
                self._rejected_status = response.status_code
                raise CredentialRejectedError(response.status_code, *CDSE_CREDENTIALS)
            response.raise_for_status()
            payload = response.json()
            self._access_token = str(payload["access_token"])
            lifetime = float(payload.get("expires_in", 600))
            self._token_expires_at = time.monotonic() + max(0.0, lifetime - _TOKEN_MARGIN_SECONDS)
            return self._access_token


def _is_rejected_grant(response: httpx.Response) -> bool:
    """True when the token endpoint refused the username and password.

    401 always counts. 400 counts only with ``invalid_grant``; another 400
    says nothing about the password, so it is not remembered.
    """
    if response.status_code not in _REJECTED_GRANT_STATUSES:
        return False
    if response.status_code == 401:
        return True
    try:
        payload = response.json()
    except ValueError:
        return False
    return isinstance(payload, dict) and payload.get("error") == "invalid_grant"


# OLCI items publish no ``eo:bands``, so wavelength matching finds nothing and
# the default pick would fall to the first radiance file, Oa01 at 400 nm.
# Centres from the OLCI band table: Oa06 560 nm, Oa08 665 nm, Oa17 865 nm.
_OLCI_BAND_ALIASES: Mapping[str, str] = {
    "green": "Oa06_radianceData",
    "red": "Oa08_radianceData",
    "nir": "Oa17_radianceData",
}


class CdseSentinel3OlciClient(CdseStacClient):
    """Sentinel-3 OLCI L1 EFR, not-time-critical.

    ``red``, ``green``, and ``nir`` name the Oa08, Oa06, and Oa17 radiance
    files, so the default download is the 665 nm band. An alias is set only
    when the item has that asset. Other aliases from the item are kept.
    """

    default_collection: ClassVar[str] = "sentinel-3-olci-1-efr-ntc"
    satellite: ClassVar[str] = "sentinel3"
    scene_kind: ClassVar[SceneKind] = SceneKind.optical
    nominal_gsd_m: ClassVar[float | None] = 300.0

    def _scenes_in(self, payload: Mapping[str, Any]) -> list[Scene]:
        return [_with_olci_aliases(scene) for scene in super()._scenes_in(payload)]

    def _result_from_payload(self, request: PullRequest, payload: Any) -> PullResult:
        result = super()._result_from_payload(request, payload)
        scenes = [_with_olci_aliases(scene) for scene in result.scenes]
        return result.model_copy(update={"scenes": scenes})


def _with_olci_aliases(scene: Scene) -> Scene:
    pinned = {alias: key for alias, key in _OLCI_BAND_ALIASES.items() if key in scene.assets}
    if not pinned:
        return scene
    return scene.model_copy(update={"band_aliases": {**scene.band_aliases, **pinned}})


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
