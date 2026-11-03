from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from atlas.archive import (
    CardLookup,
    CardStore,
    LookupStatus,
    SceneCard,
    TTLCardStore,
    make_index_id,
)
from atlas.data.base import BBox, SceneKind

BOX = BBox(west=-71.2, south=42.2, east=-70.9, north=42.5)


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, minutes: float) -> None:
        self.now += minutes * 60


def card(scene_id: str = "LC08_X", **overrides: Any) -> SceneCard:
    data: dict[str, Any] = {
        "index_id": make_index_id("landsat8", "landsat-c2-l2", scene_id),
        "source": "landsat8",
        "satellite": "landsat8",
        "platform": "landsat-8",
        "collection": "landsat-c2-l2",
        "provider_id": scene_id,
        "kind": SceneKind.optical,
        "datetime": datetime(2024, 7, 1, 15, 30, tzinfo=UTC),
        "bbox": BOX,
    }
    data.update(overrides)
    return SceneCard(**data)


def make_store(clock: Clock, **kwargs: Any) -> TTLCardStore:
    return TTLCardStore(clock=clock, **kwargs)


def test_store_satisfies_the_card_store_interface() -> None:
    store: CardStore = TTLCardStore()
    store.put(card())
    assert store.lookup(card().index_id).status is LookupStatus.found


def test_found_lookup_returns_the_card() -> None:
    clock = Clock()
    store = make_store(clock)
    c = card()
    store.put(c)
    result = store.lookup(c.index_id)
    assert result == CardLookup(c.index_id, LookupStatus.found, c)


def test_unknown_id_is_missing_not_expired() -> None:
    result = make_store(Clock()).lookup("sc_unknown")
    assert result.status is LookupStatus.missing and result.card is None


def test_card_expires_after_ttl_and_says_so() -> None:
    clock = Clock()
    store = make_store(clock)
    c = card()
    store.put(c)
    clock.advance(4.9)
    assert store.lookup(c.index_id).status is LookupStatus.found
    # The read at 4.9 min renewed the card; 5 min later it is gone.
    clock.advance(5)
    result = store.lookup(c.index_id)
    assert result.status is LookupStatus.expired and result.card is None
    assert store.lookup(c.index_id).status is LookupStatus.expired


def test_expiry_boundary_is_exclusive() -> None:
    clock = Clock()
    store = make_store(clock)
    c = card()
    store.put(c)
    clock.advance(5)
    assert store.lookup(c.index_id).status is LookupStatus.expired


def test_read_renews_but_never_past_max_age() -> None:
    clock = Clock()
    store = make_store(clock, max_age=timedelta(minutes=12))
    c = card()
    store.put(c)
    for _ in range(2):
        clock.advance(4)
        assert store.lookup(c.index_id).status is LookupStatus.found
    # 8 min in; the read just now renewed to min(8 + 5, 12) = 12.
    clock.advance(3.9)
    assert store.lookup(c.index_id).status is LookupStatus.found
    clock.advance(0.2)
    assert store.lookup(c.index_id).status is LookupStatus.expired


def test_putting_again_starts_afresh_and_replaces_the_card() -> None:
    clock = Clock()
    store = make_store(clock)
    first = card()
    store.put(first)
    clock.advance(4)
    newer = card(cloud_cover=3.0, cloud_status="value")
    store.put(newer)
    clock.advance(4)
    assert store.lookup(first.index_id).card == newer


def test_putting_an_expired_id_makes_it_found_again() -> None:
    clock = Clock()
    store = make_store(clock)
    c = card()
    store.put(c)
    clock.advance(6)
    assert store.lookup(c.index_id).status is LookupStatus.expired
    store.put(c)
    assert store.lookup(c.index_id).status is LookupStatus.found


def test_over_capacity_drops_least_recently_used_as_expired() -> None:
    clock = Clock()
    store = make_store(clock, max_cards=2)
    a, b, c = card("A"), card("B"), card("C")
    store.put_many([a, b])
    store.lookup(a.index_id)  # A is now more recent than B
    store.put(c)
    assert store.lookup(b.index_id).status is LookupStatus.expired
    assert store.lookup(a.index_id).status is LookupStatus.found
    assert store.lookup(c.index_id).status is LookupStatus.found


def test_list_returns_live_cards_in_time_then_id_order() -> None:
    clock = Clock()
    store = make_store(clock)
    late = card("LATE", datetime=datetime(2024, 8, 1, tzinfo=UTC))
    early_b, early_a = card("B"), card("A")
    store.put_many([late, early_b, early_a])
    expected = [*sorted([early_a, early_b], key=lambda x: x.index_id), late]
    assert store.list() == expected
    clock.advance(6)
    assert store.list() == []


def test_list_does_not_renew_cards() -> None:
    clock = Clock()
    store = make_store(clock)
    c = card()
    store.put(c)
    clock.advance(4)
    assert store.list() == [c]
    clock.advance(1.1)
    assert store.lookup(c.index_id).status is LookupStatus.expired


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ttl": timedelta(0)},
        {"ttl": timedelta(minutes=-1)},
        {"ttl": timedelta(minutes=10), "max_age": timedelta(minutes=5)},
        {"max_cards": 0},
    ],
)
def test_invalid_limits_are_rejected(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        TTLCardStore(**kwargs)
