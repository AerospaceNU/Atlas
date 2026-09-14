"""Fan-out search: run one request across every source concurrently."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping

from atlas.compile.product import SceneCatalog, SourceScenes
from atlas.data.base import DataPullClient, PullRequest, Scene


async def _search_one(name: str, client: DataPullClient, request: PullRequest) -> SourceScenes:
    collection = getattr(client, "collection", "")
    try:
        result = await client.search(request)
    except Exception as exc:
        return SourceScenes(
            source=name, collection=collection, scenes=[], error=f"{type(exc).__name__}: {exc}"
        )
    stamped: list[Scene] = []
    for scene in result.scenes:
        updates: dict[str, str] = {}
        if not scene.source:
            updates["source"] = name
        if not scene.collection and collection:
            updates["collection"] = collection
        stamped.append(scene.model_copy(update=updates) if updates else scene)
    return SourceScenes(source=name, collection=collection, scenes=stamped)


async def aggregate(request: PullRequest, clients: Mapping[str, DataPullClient]) -> SceneCatalog:
    """Search every source for `request` and collect results into one catalog.

    A failure in one source is captured on its `SourceScenes.error` rather than
    aborting the whole fan-out.
    """
    sources = await asyncio.gather(
        *(_search_one(name, client, request) for name, client in clients.items())
    )
    return SceneCatalog(request=request, sources=list(sources))
