from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from atlas.archive.card import SceneCard
from atlas.archive.store import CardStore

if TYPE_CHECKING:
    from atlas.compile.product import SceneCatalog


@dataclass(frozen=True)
class SkippedScene:
    """A scene that could not become a card, and why.

    Attributes:
        source: Registry key of the source that returned the scene.
        scene_id: Provider scene id.
        reason: First line of the validation error. Never a URL or a query.
    """

    source: str
    scene_id: str
    reason: str


@dataclass(frozen=True)
class CardBatch:
    """Cards built from one search, and the scenes that were left out.

    Attributes:
        cards: One card per distinct index id, in the order first seen. When a
            search returns the same scene twice, the later copy wins.
        skipped: Scenes that could not become a card. A failed source has no
            scenes and so adds nothing here.
    """

    cards: list[SceneCard] = field(default_factory=list)
    skipped: list[SkippedScene] = field(default_factory=list)


def cards_from_catalog(catalog: SceneCatalog) -> CardBatch:
    """Build a card for every scene in a search result.

    A scene with no ``source`` or ``collection`` takes them from the source
    that returned it, because some clients leave them empty. A scene that still
    cannot form a card is reported in ``skipped`` and does not stop the others.
    """
    built: dict[str, SceneCard] = {}
    skipped: list[SkippedScene] = []
    for src in catalog.sources:
        if src.error is not None:
            continue
        for scene in src.scenes:
            updates: dict[str, str] = {}
            if not scene.source and src.source:
                updates["source"] = src.source
            if not scene.collection and src.collection:
                updates["collection"] = src.collection
            filled = scene.model_copy(update=updates) if updates else scene
            try:
                card = SceneCard.from_scene(filled)
            except ValueError as exc:
                reason = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
                skipped.append(SkippedScene(src.source, scene.id, reason))
                continue
            built.pop(card.index_id, None)
            built[card.index_id] = card
    return CardBatch(cards=list(built.values()), skipped=skipped)


def record_catalog(store: CardStore, catalog: SceneCatalog) -> CardBatch:
    """Build cards from a search result and put them in ``store``."""
    batch = cards_from_catalog(catalog)
    store.put_many(batch.cards)
    return batch
