"""Match a reprocessed scene by collection, time, footprint, and grid key.

A dynamic reprocess keeps the acquisition and the footprint but issues a new
scene id. The pull manifest stores those facts beside the id so a later
refresh or resume can find the replacement when the stored id is gone.

Only STAC sources resolve replacements. ``StacApiClient`` confirms which
stored ids still exist and searches by place and time. Every other source
returns None from those hooks, and the pull leaves that source's ids
unchecked.

An old manifest row has no grid key. Matching then uses the time window and
footprint IoU, and the replacement line says so. ``sat:relative_orbit`` is a
weak key: many scenes on one pass share it, so a match on that key still
depends on the time window and the footprint IoU.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from itertools import combinations
from typing import Any, Literal

from atlas.data.base import BBox, Scene

# Reprocessing keeps the acquisition time. A minute absorbs timestamp rounding
# without reaching the next distinct pass.
SCENE_DATETIME_TOLERANCE = timedelta(seconds=60)

# A neighbour that merely touches, or overlaps on a sliver at the AOI edge,
# is a different tile. Reprocessing keeps the footprint, so the boxes agree.
FOOTPRINT_IOU_MIN = 0.8

# Float noise on a ratio that is exactly the threshold still counts as enough.
_IOU_ABS = 1e-9

_MGRS = "s2:mgrs_tile"
_GRID_CODE = "grid:code"
_WRS_PATH = "landsat:wrs_path"
_WRS_ROW = "landsat:wrs_row"
_ORBIT = "sat:relative_orbit"
_BASELINE = "s2:processing_baseline"
_BASELINE_WIDTH = 4

UnresolvedReason = Literal["no match", "ambiguous"]
GridValue = str | int


@dataclass(frozen=True)
class GridKey:
    """Stable tile or orbit identity taken from STAC properties.

    Attributes:
        fields: Property names and values, in canonical order.
    """

    fields: tuple[tuple[str, GridValue], ...]

    def as_manifest(self) -> dict[str, GridValue]:
        """Return the JSON object stored on a scene row.

        Returns:
            The property names and values that identify the tile or orbit.
        """
        return dict(self.fields)


@dataclass(frozen=True)
class StoredScene:
    """One scene row read from a pull manifest.

    Attributes:
        id: Scene id stored beside the place and time.
        collection: Catalog collection, when the manifest recorded one.
        observed_at: Acquisition time, when the manifest recorded one.
        footprint: Footprint box, when the manifest recorded one.
        grid_key: Tile or orbit key, when the manifest recorded one.
    """

    id: str
    collection: str | None = None
    observed_at: datetime | None = None
    footprint: BBox | None = None
    grid_key: GridKey | None = None


@dataclass(frozen=True)
class SceneRecord:
    """Scene identity written back to the manifest."""

    id: str
    collection: str | None = None
    observed_at: datetime | None = None
    footprint: BBox | None = None
    grid_key: GridKey | None = None

    def as_manifest(self) -> dict[str, object]:
        """Return the JSON object stored beside this id.

        Returns:
            ``id`` plus collection, datetime, footprint, and grid key when
            they are known.
        """
        body: dict[str, object] = {"id": self.id}
        if self.collection:
            body["collection"] = self.collection
        if self.observed_at is not None:
            body["datetime"] = format_datetime(self.observed_at)
        if self.footprint is not None:
            body["footprint"] = self.footprint.as_list()
        if self.grid_key is not None:
            body["grid_key"] = self.grid_key.as_manifest()
        return body


@dataclass(frozen=True)
class Replacement:
    """A stored id that was replaced by a place-and-time match.

    Attributes:
        old_id: Id that no longer resolves.
        new_id: Id that replaced it.
        scene: Catalog scene for ``new_id``.
        missing_grid_key: True when the stored row had no grid key, so the
            match used the time window and footprint IoU.
    """

    old_id: str
    new_id: str
    scene: Scene
    missing_grid_key: bool = False


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
        """Return a resolution that leaves stored ids as they are.

        Non-STAC sources use this. Their ids are not checked and are not
        replaced.
        """
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

    ``scenes`` carries collection, datetime, footprint, and grid key beside
    the id. ``scene_ids`` from an older manifest are ids with no place or time.
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

    A missing id is replaced when one candidate shares the collection, falls
    within :data:`SCENE_DATETIME_TOLERANCE`, has footprint IoU of at least
    :data:`FOOTPRINT_IOU_MIN`, and carries the stored grid key when one was
    recorded. Several matches collapse to the newest processing baseline
    only when they are otherwise the same scene. Zero matches, or several
    matches that are different scenes, are reported and the stored id is kept.

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
        chosen = _choose_match(matches)
        if chosen is not None:
            claimed.add(chosen.id)
            replacements.append(
                Replacement(
                    old_id=item.id,
                    new_id=chosen.id,
                    scene=chosen,
                    missing_grid_key=item.grid_key is None,
                )
            )
        elif not matches:
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

    A replacement of a row that stored no grid key says it matched by time
    and footprint. A row that stored a grid key does not add that note.

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
        note = ""
        if replacement.missing_grid_key:
            note = " (no stored grid key; matched by time and footprint)"
        lines.append(
            f"{source}: replaced scene {replacement.old_id} with {replacement.new_id}{note}"
        )
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
        if not _same_grid(stored.grid_key, scene):
            continue
        if not _footprint_matches(stored.footprint, scene.bbox):
            continue
        seen.add(scene.id)
        found.append(scene)
    return found


def _choose_match(matches: Sequence[Scene]) -> Scene | None:
    """Pick the replacement, or None when the candidates stay ambiguous.

    One match is that scene. Several matches are a replacement only when
    they are the same scene apart from processing, in which case the newest
    baseline wins. A tie, or two different scenes, stays unresolved.
    """
    if len(matches) == 1:
        return matches[0]
    if len(matches) < 2 or not _same_scene(matches):
        return None
    newest = max(matches, key=_recency)
    top = _recency(newest)
    if sum(1 for scene in matches if _recency(scene) == top) != 1:
        return None
    return newest


def _same_scene(matches: Sequence[Scene]) -> bool:
    """True when every candidate is one acquisition with a new id."""
    keys = [grid_key_from_properties(scene.properties) for scene in matches]
    if any(key != keys[0] for key in keys):
        return False
    for left, right in combinations(matches, 2):
        if not _within_tolerance(left.datetime, right.datetime):
            return False
        if not _footprint_matches(left.bbox, right.bbox):
            return False
    return True


def _same_grid(stored: GridKey | None, scene: Scene) -> bool:
    """True when the candidate may stand in for this stored row.

    An old manifest has no grid key. The candidate is then kept or dropped
    by the time window and footprint IoU alone, and the replacement line
    says that. ``sat:relative_orbit`` is a weak key: it only names the
    pass, so those same checks still have to agree.
    """
    if stored is None:
        return True
    return grid_key_from_properties(scene.properties) == stored


def _footprint_matches(stored: BBox, candidate: BBox) -> bool:
    return stored.iou(candidate) + _IOU_ABS >= FOOTPRINT_IOU_MIN


def _recency(scene: Scene) -> tuple[int, tuple[int, ...], float]:
    """Rank a candidate by processing baseline, then created or updated."""
    baseline = _processing_baseline(scene.properties)
    stamp = _processing_stamp(scene.properties)
    stamp_key = float("-inf") if stamp is None else stamp.timestamp()
    if baseline is None:
        return (0, (), stamp_key)
    return (1, baseline, stamp_key)


def grid_key_from_properties(properties: Mapping[str, Any]) -> GridKey | None:
    """Return the stable tile or orbit key carried by these properties.

    The first family that is present wins: ``s2:mgrs_tile``, then
    ``grid:code``, then ``landsat:wrs_path`` with ``landsat:wrs_row``,
    then ``sat:relative_orbit``. The manifest stores that key, and a
    later lookup requires the same key.

    ``sat:relative_orbit`` is a weak key. Many scenes on one pass share
    it, so storing it does not by itself identify the tile. The time
    window and the footprint IoU still have to agree.

    Args:
        properties: STAC item properties, or a manifest ``grid_key`` object.

    Returns:
        The key to store and compare, or None when none of those fields
        is usable.
    """
    mgrs = _token(properties.get(_MGRS))
    if mgrs is not None:
        return GridKey(((_MGRS, mgrs),))
    code = _token(properties.get(_GRID_CODE))
    if code is not None:
        return GridKey(((_GRID_CODE, code),))
    path = _whole_number(properties.get(_WRS_PATH))
    row = _whole_number(properties.get(_WRS_ROW))
    if path is not None and row is not None:
        return GridKey(((_WRS_PATH, path), (_WRS_ROW, row)))
    orbit = _whole_number(properties.get(_ORBIT))
    if orbit is not None:
        return GridKey(((_ORBIT, orbit),))
    return None


def _processing_baseline(properties: Mapping[str, Any]) -> tuple[int, ...] | None:
    raw = properties.get(_BASELINE)
    if isinstance(raw, bool) or raw is None:
        return None
    if isinstance(raw, int):
        text = str(raw)
    elif isinstance(raw, str):
        text = raw.strip()
    else:
        return None
    if not text:
        return None
    pieces = text.split(".")
    if any(not piece.isdigit() for piece in pieces):
        return None
    numbers = [int(piece) for piece in pieces]
    if len(numbers) < _BASELINE_WIDTH:
        numbers.extend([0] * (_BASELINE_WIDTH - len(numbers)))
    return tuple(numbers)


def _processing_stamp(properties: Mapping[str, Any]) -> datetime | None:
    created = parse_datetime(properties.get("created"))
    updated = parse_datetime(properties.get("updated"))
    if created is None:
        return updated
    if updated is None:
        return created
    return max(created, updated)


def _token(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _whole_number(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        number = value
    elif isinstance(value, float):
        if not value.is_integer():
            return None
        number = int(value)
    elif isinstance(value, str) and value.strip().isdigit():
        number = int(value.strip())
    else:
        return None
    if number < 0:
        return None
    return number


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
        grid_key=_parse_grid_key(value.get("grid_key")),
    )


def _parse_grid_key(value: object) -> GridKey | None:
    if not isinstance(value, dict) or not value:
        return None
    parsed = grid_key_from_properties(value)
    if parsed is None:
        return None
    if set(parsed.as_manifest()) != set(value):
        return None
    return parsed


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
        grid_key=grid_key_from_properties(scene.properties),
    )


def _record_from_stored(item: StoredScene) -> SceneRecord:
    return SceneRecord(
        id=item.id,
        collection=item.collection,
        observed_at=item.observed_at,
        footprint=item.footprint,
        grid_key=item.grid_key,
    )
