"""Fan-out search: run one request across every source concurrently."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping

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


async def aggregate(
    request: PullRequest,
    clients: Mapping[str, DataPullClient],
    *,
    on_source: Callable[[SourceScenes], None] | None = None,
) -> SceneCatalog:
    """Search every source for `request` and collect results into one catalog.

    A failure in one source is captured on its `SourceScenes.error` rather than
    aborting the whole fan-out. ``on_source`` is called as each source finishes.
    """

    async def run(name: str, client: DataPullClient) -> SourceScenes:
        result = await _search_one(name, client, request)
        if on_source is not None:
            on_source(result)
        return result

    sources = await asyncio.gather(*(run(name, client) for name, client in clients.items()))
    return SceneCatalog(request=request, sources=list(sources))
