from __future__ import annotations

from datetime import datetime
from typing import Any, ClassVar, Self
from urllib.parse import urlencode

import httpx

from atlas.data.base import (
    Asset,
    BBox,
    DataPullClient,
    PullRequest,
    PullResult,
    Scene,
)

STAC_SEARCH_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SAS_SIGN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/sign"


class PlanetaryComputerClient(DataPullClient):
    default_collection: ClassVar[str] = ""

    def __init__(
        self,
        *,
        collection: str | None = None,
        client: httpx.AsyncClient | None = None,
        timeout: float = 30.0,
    ) -> None:
        if collection is None and not self.default_collection:
            raise ValueError(
                f"{type(self).__name__} has no default_collection; pass `collection=...`"
            )
        self.collection = collection or self.default_collection
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = client is None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.aclose()

    async def search(self, request: PullRequest) -> PullResult:
        body: dict[str, Any] = {
            "collections": [self.collection],
            "bbox": request.bbox.as_list(),
            "datetime": (
                f"{request.start_date.isoformat()}T00:00:00Z/"
                f"{request.end_date.isoformat()}T23:59:59Z"
            ),
            "limit": request.limit,
        }
        query = self._build_query_filter(request)
        if query:
            body["query"] = query

        resp = await self._client.post(STAC_SEARCH_URL, json=body)
        resp.raise_for_status()
        payload = resp.json()

        scenes = [_feature_to_scene(f) for f in payload.get("features", [])]
        return PullResult(request=request, scenes=scenes)

    async def sign_href(self, href: str) -> str:
        resp = await self._client.get(f"{SAS_SIGN_URL}?{urlencode({'href': href})}")
        resp.raise_for_status()
        signed = resp.json()["href"]
        if not isinstance(signed, str):
            raise TypeError(f"Expected string href from sign endpoint, got {type(signed)}")
        return signed

    def _build_query_filter(self, request: PullRequest) -> dict[str, Any] | None:
        # Hook: subclasses return a STAC `query` filter (e.g. cloud cover), or None.
        return None


def _feature_to_scene(feature: dict[str, Any]) -> Scene:
    props: dict[str, Any] = feature.get("properties", {})
    bbox_list = feature.get("bbox") or [0.0, 0.0, 0.0, 0.0]
    instruments = props.get("instruments") or []
    raw_assets = feature.get("assets", {})

    assets = {
        name: Asset(
            href=a["href"],
            media_type=a.get("type"),
            title=a.get("title"),
            roles=a.get("roles", []),
        )
        for name, a in raw_assets.items()
    }

    return Scene(
        id=feature["id"],
        datetime=datetime.fromisoformat(props["datetime"].replace("Z", "+00:00")),
        bbox=BBox(
            west=bbox_list[0],
            south=bbox_list[1],
            east=bbox_list[2],
            north=bbox_list[3],
        ),
        platform=props["platform"],
        instrument=instruments[0] if instruments else None,
        cloud_cover=props.get("eo:cloud_cover"),
        assets=assets,
        properties=props,
    )
