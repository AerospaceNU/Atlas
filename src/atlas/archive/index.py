"""SQLite scene index: one :class:`SceneCard` per observation, an R*Tree on bboxes.

The index lives at ``<out>/index.sqlite`` next to the files it describes. A
local layer is stored with an ``index`` locator, a path relative to that
folder, so the whole folder can move. Queries show it as a ``workspace``
locator when the folder is inside the workspace.

Each row keeps a cleaned snapshot of the source ``Scene`` (``scene_json``), so
cards are re-derived offline when :data:`CARD_VERSION` changes. What is local
is not in the scene, so a rebuild carries each layer's local state over.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, date, time
from datetime import datetime as DateTime
from pathlib import Path
from types import TracebackType
from typing import Any, Self
from urllib.parse import urlsplit, urlunsplit

from atlas.archive.card import (
    CARD_VERSION,
    Availability,
    CloudStatus,
    LayerRef,
    Locator,
    SceneCard,
    make_index_id,
)
from atlas.archive.geometry import footprint_intersects, split_box
from atlas.archive.query import QueryGroup, QueryResult, SceneQuery, sort_cards
from atlas.archive.store import CardLookup, LookupStatus
from atlas.data.base import BBox, Scene

INDEX_FILENAME = "index.sqlite"
SCHEMA_VERSION = 1
INDEX_SCHEME = "index"
WORKSPACE_SCHEME = "workspace"

_LOCAL = frozenset({Availability.windowed.value, Availability.full.value})

_META_TABLE = "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
_SCHEMA = """
CREATE TABLE IF NOT EXISTS scenes (
    rowid INTEGER PRIMARY KEY,
    index_id TEXT NOT NULL UNIQUE,
    card_version INTEGER NOT NULL,
    source TEXT NOT NULL,
    satellite TEXT NOT NULL,
    platform TEXT NOT NULL,
    datetime_utc INTEGER NOT NULL,
    west REAL NOT NULL,
    south REAL NOT NULL,
    east REAL NOT NULL,
    north REAL NOT NULL,
    footprint TEXT,
    cloud_cover REAL,
    cloud_status TEXT NOT NULL,
    card_json TEXT NOT NULL,
    scene_json TEXT NOT NULL,
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS scenes_datetime ON scenes (datetime_utc);
CREATE VIRTUAL TABLE IF NOT EXISTS scene_bbox USING rtree (id, min_lon, max_lon, min_lat, max_lat);
"""


class IndexVersionError(RuntimeError):
    """The index was written by a newer Atlas than this one."""


@dataclass(frozen=True)
class SavedLayer:
    """One layer a download wrote: its asset key, its file, and the chip's window.

    ``window`` is the AOI of a bbox chip, and None for a whole-asset download.
    """

    key: str
    path: Path
    window: BBox | None = None


@dataclass(frozen=True)
class IndexEntry:
    """A scene to record and, when a download saved one, its local layer.

    ``satellite`` defaults to the ``satellite`` of the client registered under
    ``scene.source``, as in :meth:`SceneCard.from_scene`.
    """

    scene: Scene
    satellite: str | None = None
    saved: SavedLayer | None = None


@dataclass
class UpsertStats:
    added: int = 0
    updated: int = 0
    skipped_request_footprint: int = 0  # gridded/browse products whose footprint is the request
    skipped_invalid: int = 0  # no source, collection, or registered satellite
    index_ids: dict[tuple[str, str], str] = field(default_factory=dict)  # (source, scene id)


class SceneIndex:
    """Read and write the scene index at ``path``.

    Opening an index written with an older :data:`CARD_VERSION` rebuilds its
    cards in place; opening one from a newer Atlas raises ``IndexVersionError``
    and leaves the file unchanged.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.root = path.parent
        self.root.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, timeout=10.0)
        try:
            self._db.row_factory = sqlite3.Row
            with self._db:
                self._db.execute(_META_TABLE)
            # Refuse a newer index before adding tables or changing its journal mode.
            versions = self._versions()
            self._db.execute("PRAGMA journal_mode=WAL")
            with self._db:
                self._db.executescript(_SCHEMA)
            self._migrate(versions)
        except BaseException:
            # The caller never gets an object to close, so close it here.
            self._db.close()
            raise

    @classmethod
    def at(cls, out_dir: Path) -> Self:
        return cls(out_dir / INDEX_FILENAME)

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def upsert(self, entries: Iterable[IndexEntry], *, now: DateTime | None = None) -> UpsertStats:
        """Record scenes from a pull.

        A scene seen again refreshes its card and ``last_seen`` and keeps its
        local layers; a later search-only pull never forgets a download. A
        saved layer replaces that layer's earlier state, chip or full.
        """
        stamp = _stamp(now)
        stats = UpsertStats()
        with self._db:
            for entry in entries:
                if entry.scene.footprint_is_request:
                    stats.skipped_request_footprint += 1
                    continue
                self._upsert_one(entry, stamp, stats)
        return stats

    def query(self, query: SceneQuery, *, workspace: Path | None = None) -> QueryResult:
        """Answer ``query`` from the index alone. Never contacts a provider."""
        satellites = {item.lower() for item in query.satellites}
        platforms = {item.lower() for item in query.platforms}
        groups: list[QueryGroup] = []
        for first, last in query.windows():
            rows: list[sqlite3.Row] = []
            excluded_unknown_cloud = 0
            for row in self._candidates(query.bbox, _day_start(first), _day_start(last) + 86400):
                footprint = json.loads(row["footprint"]) if row["footprint"] else None
                if not footprint_intersects(footprint, _row_bbox(row), query.bbox):
                    continue
                if query.sources and row["source"] not in query.sources:
                    continue
                if satellites and row["satellite"] not in satellites:
                    continue
                if platforms and row["platform"].lower() not in platforms:
                    continue
                status = CloudStatus(row["cloud_status"])
                if query.max_cloud_cover is not None:
                    if status is CloudStatus.unknown:
                        excluded_unknown_cloud += 1
                        continue
                    if status is CloudStatus.value and row["cloud_cover"] > query.max_cloud_cover:
                        continue
                rows.append(row)
            cards = sort_cards((self._present(row, workspace) for row in rows), query.order)
            groups.append(
                QueryGroup(
                    start=first,
                    end=last,
                    cards=cards[: query.limit],
                    matched=len(rows),
                    excluded_unknown_cloud=excluded_unknown_cloud,
                )
            )
        return QueryResult(query=query, index_dirs=[self._shown_root(workspace)], groups=groups)

    def lookup(self, index_id: str, *, workspace: Path | None = None) -> CardLookup:
        """Resolve a cited index id. The index keeps cards, so none expire."""
        row = self._db.execute("SELECT * FROM scenes WHERE index_id = ?", (index_id,)).fetchone()
        if row is None:
            return CardLookup(index_id, LookupStatus.missing)
        return CardLookup(index_id, LookupStatus.found, self._present(row, workspace))

    def rebuild(self) -> int:
        """Re-derive every card from its stored scene snapshot. Returns the count."""
        rows = self._db.execute("SELECT rowid, card_json, scene_json FROM scenes").fetchall()
        with self._db:
            for row in rows:
                scene = Scene.model_validate_json(row["scene_json"])
                stored = json.loads(row["card_json"])
                local = _local_layers(stored)
                card = _with_local(
                    SceneCard.from_scene(scene, satellite=stored.get("satellite"), keep=local),
                    local,
                )
                columns = _columns(card, scene)
                assignments = ", ".join(f"{name} = :{name}" for name in columns)
                self._db.execute(
                    f"UPDATE scenes SET {assignments} WHERE rowid = :rowid",
                    {**columns, "rowid": row["rowid"]},
                )
                self._write_bbox(row["rowid"], card.bbox)
            self._set_meta("card_version", str(CARD_VERSION))
        return len(rows)

    def _versions(self) -> dict[str, int]:
        """Stored versions, empty for a new index. Raises if a newer Atlas wrote it."""
        meta = {
            row["key"]: int(row["value"])
            for row in self._db.execute("SELECT key, value FROM meta").fetchall()
        }
        schema = meta.get("schema_version", SCHEMA_VERSION)
        cards = meta.get("card_version", CARD_VERSION)
        if schema > SCHEMA_VERSION or cards > CARD_VERSION:
            raise IndexVersionError(
                f"{self.path} was written by a newer Atlas "
                f"(schema {schema}, cards {cards}); upgrade Atlas to read it"
            )
        return meta

    def _migrate(self, versions: dict[str, int]) -> None:
        if not versions:
            with self._db:
                self._set_meta("schema_version", str(SCHEMA_VERSION))
                self._set_meta("card_version", str(CARD_VERSION))
            return
        if versions.get("card_version", CARD_VERSION) < CARD_VERSION:
            self.rebuild()

    def _set_meta(self, key: str, value: str) -> None:
        self._db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))

    def _upsert_one(self, entry: IndexEntry, stamp: str, stats: UpsertStats) -> None:
        scene = entry.scene
        index_id = make_index_id(scene.source, scene.collection, scene.id)
        row = self._db.execute(
            "SELECT rowid, card_json FROM scenes WHERE index_id = ?", (index_id,)
        ).fetchone()
        local = _local_layers(json.loads(row["card_json"])) if row is not None else {}
        if entry.saved is not None:
            key = self._locator_key(entry.saved.path)
            if key is not None:
                local[entry.saved.key] = _layer_state(key, entry.saved.window)
        try:
            card = SceneCard.from_scene(scene, satellite=entry.satellite, keep=local)
        except ValueError:
            stats.skipped_invalid += 1
            return
        card = _with_local(card, local)
        columns = _columns(card, scene)
        if row is None:
            columns.update(index_id=index_id, first_seen=stamp, last_seen=stamp)
            names = ", ".join(columns)
            params = ", ".join(f":{name}" for name in columns)
            cursor = self._db.execute(f"INSERT INTO scenes ({names}) VALUES ({params})", columns)
            assert cursor.lastrowid is not None
            self._write_bbox(cursor.lastrowid, card.bbox)
            stats.added += 1
        else:
            columns["last_seen"] = stamp
            assignments = ", ".join(f"{name} = :{name}" for name in columns)
            self._db.execute(
                f"UPDATE scenes SET {assignments} WHERE rowid = :rowid",
                {**columns, "rowid": row["rowid"]},
            )
            self._write_bbox(row["rowid"], card.bbox)
            stats.updated += 1
        stats.index_ids[(scene.source, scene.id)] = index_id

    def _write_bbox(self, rowid: int, bbox: BBox) -> None:
        # R*Tree ids must be unique; a wrapping bbox needs two entries per scene,
        # so scene ``rowid`` owns R*Tree ids ``2*rowid`` and ``2*rowid + 1``.
        self._db.execute("DELETE FROM scene_bbox WHERE id IN (?, ?)", (2 * rowid, 2 * rowid + 1))
        for part, (west, south, east, north) in enumerate(split_box(bbox)):
            self._db.execute(
                "INSERT INTO scene_bbox (id, min_lon, max_lon, min_lat, max_lat) "
                "VALUES (?, ?, ?, ?, ?)",
                (2 * rowid + part, west, east, south, north),
            )

    def _candidates(self, aoi: BBox, start: int, stop: int) -> list[sqlite3.Row]:
        """Scenes whose bbox may meet the AOI in ``[start, stop)``, oldest first."""
        found: dict[int, sqlite3.Row] = {}
        for west, south, east, north in split_box(aoi):
            for row in self._db.execute(
                "SELECT s.rowid AS rid, s.* FROM scene_bbox r JOIN scenes s ON s.rowid = r.id / 2 "
                "WHERE r.max_lon >= ? AND r.min_lon <= ? AND r.max_lat >= ? AND r.min_lat <= ? "
                "AND s.datetime_utc >= ? AND s.datetime_utc < ?",
                (west, east, south, north, start, stop),
            ):
                found[row["rid"]] = row
        return sorted(found.values(), key=lambda row: (row["datetime_utc"], row["index_id"]))

    def _locator_key(self, path: Path) -> str | None:
        """``path`` relative to the index folder, or None when it lies outside it."""
        try:
            return path.resolve().relative_to(self.root.resolve()).as_posix()
        except ValueError:
            return None

    def _workspace_prefix(self, workspace: Path | None) -> Path | None:
        if workspace is None:
            return None
        try:
            return self.root.resolve().relative_to(workspace.resolve())
        except ValueError:
            return None

    def _shown_root(self, workspace: Path | None) -> str:
        prefix = self._workspace_prefix(workspace)
        return prefix.as_posix() if prefix is not None else os.fspath(self.root.resolve())

    def _present(self, row: sqlite3.Row, workspace: Path | None) -> SceneCard:
        """The stored card, with gone files marked ``missing`` and workspace locators."""
        card = SceneCard.model_validate_json(row["card_json"])
        prefix = self._workspace_prefix(workspace)
        layers = dict(card.layers)
        for key, layer in layers.items():
            if layer.locator is None:
                continue
            update: dict[str, Any] = {}
            if not (self.root / layer.locator.key).exists():
                update["availability"] = Availability.missing
            if prefix is not None:
                update["locator"] = Locator(
                    scheme=WORKSPACE_SCHEME, key=(prefix / layer.locator.key).as_posix()
                )
            if update:
                layers[key] = layer.model_copy(update=update)
        return card.model_copy(update={"layers": layers})


def _layer_state(key: str, window: BBox | None) -> dict[str, Any]:
    return {
        "availability": Availability.windowed if window is not None else Availability.full,
        "locator": {"scheme": INDEX_SCHEME, "key": key},
        "window": window.model_dump(mode="json") if window is not None else None,
    }


def _local_layers(card: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Local state of each chip or full layer in a stored card's JSON.

    Read from raw JSON so an older card version still hands its state on.
    """
    found: dict[str, dict[str, Any]] = {}
    layers = card.get("layers")
    if not isinstance(layers, dict):
        return found
    for key, layer in layers.items():
        if isinstance(layer, dict) and layer.get("availability") in _LOCAL:
            found[key] = {
                "availability": layer["availability"],
                "locator": layer.get("locator"),
                "window": layer.get("window"),
            }
    return found


def _with_local(card: SceneCard, local: dict[str, dict[str, Any]]) -> SceneCard:
    """``card`` with each still-listed layer's local state applied."""
    layers = dict(card.layers)
    for key, state in local.items():
        if key in layers:
            layers[key] = LayerRef.model_validate({**layers[key].model_dump(), **state})
    return card.model_copy(update={"layers": layers})


def _columns(card: SceneCard, scene: Scene) -> dict[str, Any]:
    return {
        "card_version": card.card_version,
        "source": card.source,
        "satellite": card.satellite,
        "platform": card.platform,
        "datetime_utc": math.floor(card.datetime.timestamp()),
        "west": card.bbox.west,
        "south": card.bbox.south,
        "east": card.bbox.east,
        "north": card.bbox.north,
        "footprint": json.dumps(scene.footprint) if scene.footprint else None,
        "cloud_cover": card.cloud_cover,
        "cloud_status": card.cloud_status.value,
        "card_json": card.model_dump_json(),
        "scene_json": json.dumps(_scene_snapshot(scene), sort_keys=True),
    }


def _scene_snapshot(scene: Scene) -> dict[str, Any]:
    """Scene as JSON for rebuilds, with query strings (signatures, tokens) cut from hrefs."""
    data = scene.model_dump(mode="json")
    for asset in data.get("assets", {}).values():
        parts = urlsplit(asset["href"])
        asset["href"] = urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
    return data


def _row_bbox(row: sqlite3.Row) -> BBox:
    return BBox(west=row["west"], south=row["south"], east=row["east"], north=row["north"])


def _day_start(day: date) -> int:
    return math.floor(DateTime.combine(day, time(), tzinfo=UTC).timestamp())


def _stamp(now: DateTime | None) -> str:
    moment = now or DateTime.now(UTC)
    return moment.astimezone(UTC).isoformat(timespec="seconds")
