"""Match a reprocessed scene by collection, time, and footprint.

A dynamic reprocess keeps the acquisition and the footprint but issues a new
scene id. The pull manifest stores those three facts beside the id so a later
refresh or resume can find the replacement when the stored id is gone.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from atlas.data.base import BBox, Scene

# Reprocessing keeps the acquisition time. A minute absorbs timestamp rounding
# without reaching the next distinct pass.
SCENE_DATETIME_TOLERANCE = timedelta(seconds=60)

UnresolvedReason = Literal["no match", "ambiguous"]


@dataclass(frozen=True)
class StoredScene:
    """One scene row read from a pull manifest.

    Attributes:
        id: Scene id stored beside the place and time.
        collection: Catalog collection, when the manifest recorded one.
        observed_at: Acquisition time, when the manifest recorded one.
        footprint: Footprint box, when the manifest recorded one.
    """

    id: str
    collection: str | None = None
    observed_at: datetime | None = None
    footprint: BBox | None = None


@dataclass(frozen=True)
class SceneRecord:
    """Scene identity written back to the manifest."""

    id: str
    collection: str | None = None
    observed_at: datetime | None = None
    footprint: BBox | None = None

    def as_manifest(self) -> dict[str, object]:
        """Return the JSON object stored beside this id.

        Returns:
            ``id`` plus collection, datetime, and footprint when they are known.
        """
        body: dict[str, object] = {"id": self.id}
        if self.collection:
            body["collection"] = self.collection
        if self.observed_at is not None:
            body["datetime"] = format_datetime(self.observed_at)
        if self.footprint is not None:
            body["footprint"] = self.footprint.as_list()
        return body


@dataclass(frozen=True)
class Replacement:
    """A stored id that was replaced by exactly one place-and-time match."""

    old_id: str
    new_id: str
    scene: Scene


@dataclass(frozen=True)
class Unresolved:
    """A stored id that is gone and has no unique replacement."""

    scene_id: str
    reason: UnresolvedReason
    candidates: tuple[str, ...] = ()


@dataclass(frozen=True)
class SourceResolution:
    """Identity check for one source.

    Attributes:
        checked: True when the catalog confirmed which stored ids still exist.
        replacements: Stored ids replaced by exactly one match.
        unresolved: Stored ids that are gone, with no match or several matches.
        confirmed: Stored ids the catalog still serves.
    """

    checked: bool
    replacements: tuple[Replacement, ...] = ()
    unresolved: tuple[Unresolved, ...] = ()
    confirmed: tuple[Scene, ...] = ()

    @staticmethod
    def unchecked() -> SourceResolution:
        """Return a resolution that leaves stored ids as they are."""
        return SourceResolution(checked=False)


def format_datetime(value: datetime) -> str:
    """Return ``value`` as a UTC timestamp ending in ``Z``.

    Args:
        value: Acquisition time. A naive value is treated as UTC.

    Returns:
        An ISO-8601 timestamp in UTC.
    """
    aware = value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    text = aware.isoformat()
    if text.endswith("+00:00"):
        return text[:-6] + "Z"
    return text


def parse_datetime(value: object) -> datetime | None:
    """Parse a manifest datetime, or return None when it is not one.

    Args:
        value: JSON value from a scene record.

    Returns:
        An aware UTC datetime, or None.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00").replace(" ", "T", 1)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def stored_scenes(row: Mapping[str, Any] | None) -> list[StoredScene]:
    """Read scene identity from one manifest source.

    ``scenes`` carries collection, datetime, and footprint beside the id.
    ``scene_ids`` from an older manifest are ids with no place or time.
    A missing ``scenes`` field still loads.

    Args:
        row: One ``sources`` entry, or None.

    Returns:
        Stored scenes in manifest order, richer rows winning over a bare id.
    """
    if row is None:
        return []
    found: dict[str, StoredScene] = {}
    order: list[str] = []
    scenes = row.get("scenes")
    if isinstance(scenes, list):
        for item in scenes:
            parsed = _parse_stored(item)
            if parsed is None or parsed.id in found:
                continue
            found[parsed.id] = parsed
            order.append(parsed.id)
    scene_ids = row.get("scene_ids")
    if isinstance(scene_ids, list):
        for item in scene_ids:
            if isinstance(item, str) and item and item not in found:
                found[item] = StoredScene(id=item)
                order.append(item)
    return [found[scene_id] for scene_id in order]


def reconcile(
    stored: Sequence[StoredScene],
    present: Mapping[str, Scene],
    candidates: Mapping[str, Sequence[Scene]],
    *,
    collection: str,
) -> SourceResolution:
    """Replace stored ids that no longer resolve.

    A missing id is replaced only when exactly one candidate shares the
    collection, falls within :data:`SCENE_DATETIME_TOLERANCE`, and overlaps
    the stored footprint. Zero matches or several matches are reported and
    the stored id is kept.

    Args:
        stored: Scene rows from the manifest.
        present: Stored ids the catalog still serves, keyed by id.
        candidates: Place-and-time hits keyed by the stored id that was missing.
        collection: Collection this source searched, used when a row omits one.

    Returns:
        The replacements, unresolved ids, and ids that still resolve.
    """
    claimed = {scene_id for scene_id in present if scene_id}
    replacements: list[Replacement] = []
    unresolved: list[Unresolved] = []
    confirmed: list[Scene] = []
    for item in stored:
        current = present.get(item.id)
        if current is not None and current.id == item.id:
            confirmed.append(current)
            continue
        if item.observed_at is None or item.footprint is None:
            unresolved.append(Unresolved(item.id, "no match"))
            continue
        wanted = item.collection or collection
        matches = _good_matches(
            item,
            candidates.get(item.id, ()),
            collection=wanted,
            claimed=claimed,
        )
        if len(matches) == 1:
            scene = matches[0]
            claimed.add(scene.id)
            replacements.append(Replacement(old_id=item.id, new_id=scene.id, scene=scene))
        elif len(matches) == 0:
            unresolved.append(Unresolved(item.id, "no match"))
        else:
            names = tuple(sorted({scene.id for scene in matches}))
            unresolved.append(Unresolved(item.id, "ambiguous", names))
    return SourceResolution(
        checked=True,
        replacements=tuple(replacements),
        unresolved=tuple(unresolved),
        confirmed=tuple(confirmed),
    )


def project_records(
    current: Sequence[Scene],
    stored: Sequence[StoredScene],
    resolution: SourceResolution,
    *,
    collection: str,
    keep_stored: bool,
) -> list[SceneRecord]:
    """Build the scene rows to write for one source.

    A resume keeps every stored id, with replacements rewritten in place.
    A refresh writes the new search. It also rewrites a replaced id and keeps
    an unresolved id so the region is not reported as empty.

    Args:
        current: Scenes this pull found, including an injected replacement.
        stored: Rows from the previous manifest.
        resolution: Identity check for this source.
        collection: Collection to store when a scene omits one.
        keep_stored: True when this pull resumes and must keep earlier ids.

    Returns:
        Scene records in write order.
    """
    replaced = {item.old_id: item for item in resolution.replacements}
    confirmed = {scene.id: scene for scene in resolution.confirmed}
    unresolved = {item.scene_id for item in resolution.unresolved}
    records: list[SceneRecord] = []
    seen: set[str] = set()

    def add(record: SceneRecord) -> None:
        if not record.id or record.id in seen:
            return
        seen.add(record.id)
        records.append(record)

    walk_stored = keep_stored or bool(resolution.replacements) or bool(resolution.unresolved)
    if walk_stored:
        for item in stored:
            replacement = replaced.get(item.id)
            if replacement is not None:
                add(_record_from_scene(replacement.scene, collection=collection))
                continue
            if not keep_stored and item.id not in unresolved:
                continue
            confirmed_scene = confirmed.get(item.id)
            if confirmed_scene is not None:
                add(_record_from_scene(confirmed_scene, collection=collection))
                continue
            add(_record_from_stored(item))
    for scene in current:
        add(_record_from_scene(scene, collection=collection))
    return records


def resolution_lines(source: str, resolution: SourceResolution) -> list[str]:
    """Return the user-facing lines for one source's identity check.

    Args:
        source: Registry name.
        resolution: Check result. An unchecked result has nothing to say.

    Returns:
        One line per replacement, missing match, or ambiguous match.
    """
    if not resolution.checked:
        return []
    lines: list[str] = []
    for replacement in resolution.replacements:
        lines.append(f"{source}: replaced scene {replacement.old_id} with {replacement.new_id}")
    for pending in resolution.unresolved:
        if pending.reason == "ambiguous":
            listed = ", ".join(pending.candidates)
            lines.append(f"{source}: scene {pending.scene_id} is ambiguous ({listed})")
        else:
            lines.append(f"{source}: scene {pending.scene_id} has no match")
    return lines


def _good_matches(
    stored: StoredScene,
    candidates: Sequence[Scene],
    *,
    collection: str,
    claimed: set[str],
) -> list[Scene]:
    if stored.observed_at is None or stored.footprint is None:
        return []
    found: list[Scene] = []
    seen: set[str] = set()
    for scene in candidates:
        if not scene.id or scene.id == stored.id or scene.id in claimed or scene.id in seen:
            continue
        scene_collection = scene.collection or collection
        if collection and scene_collection != collection:
            continue
        if not _within_tolerance(stored.observed_at, scene.datetime):
            continue
        if not stored.footprint.intersects(scene.bbox):
            continue
        seen.add(scene.id)
        found.append(scene)
    return found


def _within_tolerance(stored_at: datetime, observed_at: datetime) -> bool:
    left = stored_at if stored_at.tzinfo is not None else stored_at.replace(tzinfo=UTC)
    right = observed_at if observed_at.tzinfo is not None else observed_at.replace(tzinfo=UTC)
    return abs(left - right) <= SCENE_DATETIME_TOLERANCE


def _parse_stored(value: object) -> StoredScene | None:
    if isinstance(value, str) and value:
        return StoredScene(id=value)
    if not isinstance(value, dict):
        return None
    scene_id = value.get("id")
    if not isinstance(scene_id, str) or not scene_id:
        return None
    collection = value.get("collection")
    return StoredScene(
        id=scene_id,
        collection=collection if isinstance(collection, str) and collection else None,
        observed_at=parse_datetime(value.get("datetime")),
        footprint=_parse_footprint(value.get("footprint")),
    )


def _parse_footprint(value: object) -> BBox | None:
    if isinstance(value, list) and len(value) == 4:
        west, south, east, north = value
        if any(isinstance(item, bool) or not isinstance(item, int | float) for item in value):
            return None
        return BBox.try_new(
            west=float(west), south=float(south), east=float(east), north=float(north)
        )
    if isinstance(value, dict):
        return BBox.try_new(
            west=value.get("west"),
            south=value.get("south"),
            east=value.get("east"),
            north=value.get("north"),
        )
    return None


def _record_from_scene(scene: Scene, *, collection: str) -> SceneRecord:
    return SceneRecord(
        id=scene.id,
        collection=scene.collection or collection or None,
        observed_at=scene.datetime,
        footprint=scene.bbox,
    )


def _record_from_stored(item: StoredScene) -> SceneRecord:
    return SceneRecord(
        id=item.id,
        collection=item.collection,
        observed_at=item.observed_at,
        footprint=item.footprint,
    )
