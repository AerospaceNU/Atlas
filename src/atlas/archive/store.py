"""Where scene cards are kept, behind an interface.

``CardStore`` is the contract callers use. ``TTLCardStore`` is the first
implementation: an in-memory store that holds cards for a few minutes so a
follow-up question can read more of a card without searching again. It is a
stopgap until the persistent store is chosen. Cards are not written to disk,
and an expired index id can only be rebuilt by searching again.

A lookup says why a card is missing. ``expired`` means the store once held it
and no longer does; ``missing`` means it never did. Callers must not present
either as "no such scene".
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum
from typing import Protocol

from atlas.archive.card import SceneCard

DEFAULT_TTL = timedelta(minutes=5)
DEFAULT_MAX_AGE = timedelta(minutes=30)
DEFAULT_MAX_CARDS = 1000


class LookupStatus(StrEnum):
    found = "found"
    expired = "expired"
    missing = "missing"


@dataclass(frozen=True)
class CardLookup:
    """Result of asking a store for one index id.

    Attributes:
        index_id: The id that was asked for.
        status: ``found``, ``expired``, or ``missing``.
        card: The card, set exactly when ``status`` is ``found``.
    """

    index_id: str
    status: LookupStatus
    card: SceneCard | None = None


class CardStore(Protocol):
    def put(self, card: SceneCard) -> None:
        """Store a card, replacing any card with the same index id."""

    def put_many(self, cards: Iterable[SceneCard]) -> None:
        """Store several cards."""

    def lookup(self, index_id: str) -> CardLookup:
        """Return the card for ``index_id``, or why it is not available."""

    def list(self) -> list[SceneCard]:
        """Return every card currently held, ordered by time then index id."""


@dataclass
class _Entry:
    card: SceneCard
    stored_at: float
    expires_at: float


class TTLCardStore:
    """In-memory card store whose cards expire.

    A card lives ``ttl`` after it is stored. Each lookup that finds it renews
    that time, but never past ``max_age`` from when it was stored, so a card
    that is read often still ages out. Storing a card again starts it afresh,
    because a new search is newer information. When more than ``max_cards`` are
    held, the least recently used are dropped and reported as ``expired``.

    Args:
        ttl: How long a card lives without being read.
        max_age: Hard limit on a card's life, however often it is read.
        max_cards: Most cards held at once.
        clock: Seconds from a monotonic source. Tests pass a fake.

    Raises:
        ValueError: If a limit is not positive, or ``max_age`` is below ``ttl``.
    """

    def __init__(
        self,
        ttl: timedelta = DEFAULT_TTL,
        *,
        max_age: timedelta = DEFAULT_MAX_AGE,
        max_cards: int = DEFAULT_MAX_CARDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if ttl <= timedelta(0):
            raise ValueError("ttl must be positive")
        if max_age < ttl:
            raise ValueError("max_age must be at least ttl")
        if max_cards < 1:
            raise ValueError("max_cards must be at least 1")
        self._ttl = ttl.total_seconds()
        self._max_age = max_age.total_seconds()
        self._max_cards = max_cards
        self._clock = clock
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        # Ids once held, so a lookup can say "expired" instead of "missing".
        self._gone: OrderedDict[str, None] = OrderedDict()
        self._lock = threading.Lock()

    def put(self, card: SceneCard) -> None:
        with self._lock:
            self._put(card)

    def put_many(self, cards: Iterable[SceneCard]) -> None:
        with self._lock:
            for card in cards:
                self._put(card)

    def lookup(self, index_id: str) -> CardLookup:
        with self._lock:
            now = self._clock()
            entry = self._entries.get(index_id)
            if entry is None:
                status = LookupStatus.expired if index_id in self._gone else LookupStatus.missing
                return CardLookup(index_id, status)
            if self._is_expired(entry, now):
                self._drop(index_id)
                return CardLookup(index_id, LookupStatus.expired)
            entry.expires_at = min(now + self._ttl, entry.stored_at + self._max_age)
            self._entries.move_to_end(index_id)
            return CardLookup(index_id, LookupStatus.found, entry.card)

    def list(self) -> list[SceneCard]:
        with self._lock:
            self._purge(self._clock())
            cards = [entry.card for entry in self._entries.values()]
        return sorted(cards, key=lambda card: (card.datetime, card.index_id))

    def _put(self, card: SceneCard) -> None:
        now = self._clock()
        self._purge(now)
        self._entries[card.index_id] = _Entry(card, now, now + self._ttl)
        self._entries.move_to_end(card.index_id)
        self._gone.pop(card.index_id, None)
        while len(self._entries) > self._max_cards:
            self._drop(next(iter(self._entries)))

    def _is_expired(self, entry: _Entry, now: float) -> bool:
        return now >= entry.expires_at or now >= entry.stored_at + self._max_age

    def _purge(self, now: float) -> None:
        for index_id in [i for i, e in self._entries.items() if self._is_expired(e, now)]:
            self._drop(index_id)

    def _drop(self, index_id: str) -> None:
        del self._entries[index_id]
        self._gone[index_id] = None
        while len(self._gone) > self._max_cards:
            self._gone.popitem(last=False)
