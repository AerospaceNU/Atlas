from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime
from typing import Any

from atlas.archive import (
    CloudStatus,
    LookupStatus,
    TTLCardStore,
    cards_from_catalog,
    make_index_id,
    record_catalog,
)
from atlas.compile.aggregate import aggregate
from atlas.compile.product import SceneCatalog, SourceScenes
from atlas.data.base import (
    Asset,
    BBox,
    DataPullClient,
    PullRequest,
    PullResult,
    Scene,
    SceneKind,
)

AOI = BBox(west=-71.12, south=42.32, east=-71.02, north=42.40)
WHEN = datetime(2024, 7, 1, 15, 30, tzinfo=UTC)


def pull_request() -> PullRequest:
    return PullRequest(start_date=date(2024, 7, 1), end_date=date(2024, 7, 3), bbox=AOI, limit=5)


def scene(scene_id: str = "S2_A", **overrides: Any) -> Scene:
    data: dict[str, Any] = {
        "id": scene_id,
        "source": "sentinel2",
        "collection": "sentinel-2-l2a",
        "platform": "sentinel-2a",
        "datetime": WHEN,
        "bbox": AOI,
        "cloud_cover": 8.0,
        "assets": {"B04": Asset(href="https://example.com/b04.tif", size_bytes=10)},
    }
    data.update(overrides)
    return Scene(**data)


def source(name: str, scenes: list[Scene], **kwargs: Any) -> SourceScenes:
    return SourceScenes(source=name, collection="sentinel-2-l2a", scenes=scenes, **kwargs)


def catalog(*sources: SourceScenes) -> SceneCatalog:
    return SceneCatalog(request=pull_request(), sources=list(sources))


def test_every_scene_becomes_a_card_with_its_index_id() -> None:
    batch = cards_from_catalog(catalog(source("sentinel2", [scene("A"), scene("B")])))
    assert [c.provider_id for c in batch.cards] == ["A", "B"]
    assert batch.cards[0].index_id == make_index_id("sentinel2", "sentinel-2-l2a", "A")
    assert batch.skipped == []


def test_empty_source_and_collection_come_from_the_returning_source() -> None:
    bare = scene("A", source="", collection="")
    batch = cards_from_catalog(catalog(source("sentinel2", [bare])))
    (card,) = batch.cards
    assert (card.source, card.collection, card.satellite) == (
        "sentinel2",
        "sentinel-2-l2a",
        "sentinel2",
    )
    # The scene itself is not changed.
    assert bare.source == "" and bare.collection == ""


def test_a_scene_that_cannot_form_a_card_is_skipped_not_fatal() -> None:
    bad = scene("BAD", source="not_a_source")
    batch = cards_from_catalog(catalog(source("not_a_source", [bad, scene("OK")])))
    assert [s.scene_id for s in batch.skipped] == ["BAD"]
    assert "satellite" in batch.skipped[0].reason
    assert batch.skipped[0].source == "not_a_source"
    # The bad scene does not stop the good one after it.
    assert [c.provider_id for c in batch.cards] == ["OK"]


def test_failed_sources_add_no_cards_and_the_rest_still_do() -> None:
    failed = SourceScenes(source="landsat8", collection="c", error="TimeoutError")
    batch = cards_from_catalog(catalog(failed, source("sentinel2", [scene("A")])))
    assert [c.provider_id for c in batch.cards] == ["A"]
    assert batch.skipped == []


def test_the_same_scene_twice_gives_one_card_and_the_later_wins() -> None:
    first, again = scene("A", cloud_cover=5.0), scene("A", cloud_cover=9.0)
    batch = cards_from_catalog(catalog(source("sentinel2", [first]), source("sentinel2", [again])))
    (card,) = batch.cards
    assert card.cloud_cover == 9.0 and card.cloud_status is CloudStatus.value


def test_same_scene_from_two_sources_stays_two_cards() -> None:
    other = scene("A", source="earthsearch_sentinel2_l2a")
    batch = cards_from_catalog(
        catalog(source("sentinel2", [scene("A")]), source("earthsearch_sentinel2_l2a", [other]))
    )
    assert len({c.index_id for c in batch.cards}) == 2


def test_record_catalog_puts_cards_in_the_store() -> None:
    store = TTLCardStore()
    batch = record_catalog(store, catalog(source("sentinel2", [scene("A"), scene("B")])))
    assert len(batch.cards) == 2
    for card in batch.cards:
        assert store.lookup(card.index_id).card == card
    assert store.lookup("sc_unknown").status is LookupStatus.missing


class _Client(DataPullClient):
    collection = "sentinel-2-l2a"
    satellite = "sentinel2"
    scene_kind = SceneKind.optical

    def __init__(self, scenes: list[Scene], error: Exception | None = None) -> None:
        self._scenes = scenes
        self._error = error

    async def search(self, request: PullRequest) -> PullResult:
        if self._error is not None:
            raise self._error
        return PullResult(request=request, scenes=self._scenes)

    async def download(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


def test_aggregate_fills_the_store_and_returns_the_same_catalog() -> None:
    store = TTLCardStore()
    clients: dict[str, DataPullClient] = {
        "sentinel2": _Client([scene("A", source="", collection="")]),
        "landsat8": _Client([], error=RuntimeError("boom")),
    }
    result = asyncio.run(aggregate(pull_request(), clients, card_store=store))
    assert [s.source for s in result.sources] == ["sentinel2", "landsat8"]
    assert result.sources[1].error is not None
    assert [c.provider_id for c in store.list()] == ["A"]


def test_aggregate_without_a_store_keeps_no_cards() -> None:
    clients: dict[str, DataPullClient] = {"sentinel2": _Client([scene("A")])}
    result = asyncio.run(aggregate(pull_request(), clients))
    assert result.total == 1


def test_a_bad_scene_never_fails_the_search() -> None:
    store = TTLCardStore()
    clients: dict[str, DataPullClient] = {
        "not_a_source": _Client([scene("A", source="not_a_source")]),
    }
    result = asyncio.run(aggregate(pull_request(), clients, card_store=store))
    assert result.total == 1 and store.list() == []
