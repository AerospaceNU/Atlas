from __future__ import annotations

from typing import ClassVar
from urllib.parse import urlencode

from atlas.data.stac import StacApiClient

STAC_SEARCH_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SAS_SIGN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/sign"


class PlanetaryComputerClient(StacApiClient):
    search_url: ClassVar[str] = STAC_SEARCH_URL

    async def sign_href(self, href: str) -> str:
        resp = await self._client.get(f"{SAS_SIGN_URL}?{urlencode({'href': href})}")
        resp.raise_for_status()
        signed = resp.json()["href"]
        if not isinstance(signed, str):
            raise TypeError(f"Expected string href from sign endpoint, got {type(signed)}")
        return signed
