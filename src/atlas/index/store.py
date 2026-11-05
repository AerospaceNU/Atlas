"""SQLite scene index: one row per observation, an R*Tree on scene bboxes.

The index lives at ``<out>/index.sqlite`` next to the files it describes, and
stores local paths relative to that folder so the whole folder can move.
Each row keeps a cleaned snapshot of the source ``Scene`` (``scene_json``) so
cards can be re-derived offline when :data:`CARD_VERSION` changes.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, time
from datetime import datetime as DateTime
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from atlas.data.base import BBox, Scene
from atlas.index.card import (
    CARD_VERSION,
    MAX_CARD_BANDS,
    Availability,
    SceneCard,
    scene_band_aliases,
    scene_bands,
    scene_index_id,
    scene_snapshot,
    utc,
)
from atlas.index.geometry import footprint_intersects, split_box
from atlas.index.query import QueryGroup, QueryResult, SceneOrder, SceneQuery

INDEX_FILENAME = "index.sqlite"
SCHEMA_VERSION = 1

_META_TABLE = "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
_SCHEMA = """
CREATE TABLE IF NOT EXISTS scenes (
    rowid INTEGER PRIMARY KEY,
    index_id TEXT NOT NULL UNIQUE,
    card_version INTEGER NOT NULL,
    catalog TEXT NOT NULL,
    collection TEXT NOT NULL,
    item_id TEXT NOT NULL,
    datetime_utc INTEGER NOT NULL,
    platform TEXT NOT NULL,
    instrument TEXT,
    kind TEXT NOT NULL,
    gsd_m REAL,
    west REAL NOT NULL,
    south REAL NOT NULL,
    east REAL NOT NULL,
    north REAL NOT NULL,
    footprint TEXT,
    cloud_cover REAL,
    bands TEXT NOT NULL,
    band_aliases TEXT NOT NULL,
    availability TEXT NOT NULL,
    local_path TEXT,
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    scene_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS scenes_datetime ON scenes (datetime_utc);
CREATE TABLE IF NOT EXISTS scene_sources (
    index_id TEXT NOT NULL,
    source TEXT NOT NULL,
    PRIMARY KEY (index_id, source)
);
CREATE TABLE IF NOT EXISTS id_aliases (old_id TEXT PRIMARY KEY, index_id TEXT NOT NULL);
CREATE VIRTUAL TABLE IF NOT EXISTS scene_bbox USING rtree (id, min_lon, max_lon, min_lat, max_lat);
"""

# Columns re-derived from a Scene on every upsert and rebuild.
_DERIVED = (
    "card_version",
    "catalog",
    "collection",
    "item_id",
    "datetime_utc",
    "platform",
    "instrument",
    "kind",
    "gsd_m",
    "west",
    "south",
    "east",
    "north",
    "footprint",
    "cloud_cover",
    "bands",
    "band_aliases",
    "scene_json",
)


class IndexVersionError(RuntimeError):
    """The index was written by a newer Atlas than this one."""


@dataclass(frozen=True)
class IndexEntry:
    """A scene to record, the registry source that found it, and its local file if saved."""

    scene: Scene
    source: str
    local_path: Path | None = None


@dataclass
class UpsertStats:
    added: int = 0
    updated: int = 0
    skipped_request_footprint: int = 0  # gridded/browse products whose footprint is the request


class SceneIndex:
    """Read and write the scene index at ``path``.

    Opening an index written with an older :data:`CARD_VERSION` rebuilds its
    cards in place; opening one from a newer Atlas raises ``IndexVersionError``.
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

        A scene seen again refreshes its metadata and ``last_seen``. A saved file
        marks it ``downloaded``; a later search-only pull never undoes that.
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

    def query(self, query: SceneQuery, *, cwd: Path | None = None) -> QueryResult:
        """Answer ``query`` from the index alone. Never contacts a provider."""
        base = cwd or Path.cwd()
        platforms = {item.lower() for item in query.platforms}
        groups: list[QueryGroup] = []
        for first, last in query.windows():
            cards: list[SceneCard] = []
            excluded_unknown_cloud = 0
            for row in self._candidates(query.bbox, _day_start(first), _day_start(last) + 86400):
                footprint = json.loads(row["footprint"]) if row["footprint"] else None
                if not footprint_intersects(footprint, _row_bbox(row), query.bbox):
                    continue
                if platforms and row["platform"].lower() not in platforms:
                    continue
                sources = self._sources(row["index_id"])
                if query.sources and not set(query.sources) & set(sources):
                    continue
                if query.max_cloud_cover is not None:
                    if row["cloud_cover"] is None:
                        excluded_unknown_cloud += 1
                        continue
                    if row["cloud_cover"] > query.max_cloud_cover:
                        continue
                cards.append(self._card(row, sources, base))
            if query.order is SceneOrder.cloud:
                # Stable sort: equal cloud cover keeps the oldest-first order.
                cards.sort(key=lambda card: (card.cloud_cover is None, card.cloud_cover or 0.0))
            groups.append(
                QueryGroup(
                    start=first,
                    end=last,
                    cards=cards[: query.limit],
                    matched=len(cards),
                    excluded_unknown_cloud=excluded_unknown_cloud,
                )
            )
        return QueryResult(query=query, groups=groups)

    def get(self, index_id: str, *, cwd: Path | None = None) -> SceneCard | None:
        """Resolve a cited index id, following aliases left by merges."""
        alias = self._db.execute(
            "SELECT index_id FROM id_aliases WHERE old_id = ?", (index_id,)
        ).fetchone()
        target = alias["index_id"] if alias else index_id
        row = self._db.execute("SELECT * FROM scenes WHERE index_id = ?", (target,)).fetchone()
        if row is None:
            return None
        return self._card(row, self._sources(target), cwd or Path.cwd())

    def rebuild(self) -> int:
        """Re-derive every card from its stored scene snapshot. Returns the count."""
        rows = self._db.execute("SELECT rowid, index_id, scene_json FROM scenes").fetchall()
        with self._db:
            for row in rows:
                scene = Scene.model_validate_json(row["scene_json"])
                new_id = scene_index_id(scene)
                if new_id != row["index_id"]:
                    self._db.execute(
                        "INSERT OR REPLACE INTO id_aliases (old_id, index_id) VALUES (?, ?)",
                        (row["index_id"], new_id),
                    )
                    self._db.execute(
                        "UPDATE scene_sources SET index_id = ? WHERE index_id = ?",
                        (new_id, row["index_id"]),
                    )
                columns = {**_derived(scene), "index_id": new_id}
                assignments = ", ".join(f"{name} = :{name}" for name in columns)
                self._db.execute(
                    f"UPDATE scenes SET {assignments} WHERE rowid = :rowid",
                    {**columns, "rowid": row["rowid"]},
                )
                self._write_bbox(row["rowid"], scene.bbox)
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
        index_id = scene_index_id(scene)
        columns = _derived(scene)
        local = self._stored_path(entry.local_path) if entry.local_path is not None else None
        row = self._db.execute(
            "SELECT rowid, west, south, east, north FROM scenes WHERE index_id = ?", (index_id,)
        ).fetchone()
        if row is None:
            columns.update(
                index_id=index_id,
                availability=(Availability.downloaded if local else Availability.found).value,
                local_path=local,
                first_seen=stamp,
                last_seen=stamp,
            )
            names = ", ".join(columns)
            params = ", ".join(f":{name}" for name in columns)
            cursor = self._db.execute(f"INSERT INTO scenes ({names}) VALUES ({params})", columns)
            assert cursor.lastrowid is not None
            self._write_bbox(cursor.lastrowid, scene.bbox)
            stats.added += 1
        else:
            columns["last_seen"] = stamp
            if local is not None:
                columns.update(availability=Availability.downloaded.value, local_path=local)
            assignments = ", ".join(f"{name} = :{name}" for name in columns)
            self._db.execute(
                f"UPDATE scenes SET {assignments} WHERE rowid = :rowid",
                {**columns, "rowid": row["rowid"]},
            )
            if _row_bbox(row) != scene.bbox:
                self._write_bbox(row["rowid"], scene.bbox)
            stats.updated += 1
        if entry.source:
            self._db.execute(
                "INSERT OR IGNORE INTO scene_sources (index_id, source) VALUES (?, ?)",
                (index_id, entry.source),
            )

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

    def _sources(self, index_id: str) -> list[str]:
        rows = self._db.execute(
            "SELECT source FROM scene_sources WHERE index_id = ? ORDER BY source", (index_id,)
        ).fetchall()
        return [row["source"] for row in rows]

    def _stored_path(self, path: Path) -> str:
        resolved = path.resolve()
        try:
            return resolved.relative_to(self.root.resolve()).as_posix()
        except ValueError:
            return str(resolved)

    def _card(self, row: sqlite3.Row, sources: list[str], cwd: Path) -> SceneCard:
        availability = Availability(row["availability"])
        shown: str | None = None
        if row["local_path"]:
            full = self.root / row["local_path"]
            if availability is Availability.downloaded and not full.exists():
                availability = Availability.missing
            shown = _display_path(full, cwd)
        bands: list[str] = json.loads(row["bands"])
        return SceneCard(
            index_id=row["index_id"],
            catalog=row["catalog"],
            collection=row["collection"],
            item_id=row["item_id"],
            found_via=sources,
            datetime=DateTime.fromtimestamp(row["datetime_utc"], UTC),
            platform=row["platform"],
            instrument=row["instrument"],
            kind=row["kind"],
            gsd_m=row["gsd_m"],
            bbox=[row["west"], row["south"], row["east"], row["north"]],
            footprint_exact=row["footprint"] is not None,
            cloud_cover=row["cloud_cover"],
            bands=bands[:MAX_CARD_BANDS],
            band_count=len(bands),
            band_aliases=json.loads(row["band_aliases"]),
            availability=availability,
            local_path=shown,
        )


def _derived(scene: Scene) -> dict[str, Any]:
    return {
        "card_version": CARD_VERSION,
        "catalog": scene.catalog or scene.source,
        "collection": scene.collection,
        "item_id": scene.id,
        "datetime_utc": math.floor(utc(scene.datetime).timestamp()),
        "platform": scene.platform,
        "instrument": scene.instrument,
        "kind": scene.kind.value,
        "gsd_m": scene.gsd_m,
        "west": scene.bbox.west,
        "south": scene.bbox.south,
        "east": scene.bbox.east,
        "north": scene.bbox.north,
        "footprint": json.dumps(scene.footprint) if scene.footprint else None,
        "cloud_cover": scene.cloud_cover,
        "bands": json.dumps(scene_bands(scene)),
        "band_aliases": json.dumps(scene_band_aliases(scene), sort_keys=True),
        "scene_json": json.dumps(scene_snapshot(scene), sort_keys=True),
    }


def _row_bbox(row: sqlite3.Row) -> BBox:
    return BBox(west=row["west"], south=row["south"], east=row["east"], north=row["north"])


def _day_start(day: date) -> int:
    return math.floor(DateTime.combine(day, time(), tzinfo=UTC).timestamp())


def _stamp(now: DateTime | None) -> str:
    return utc(now or DateTime.now(UTC)).isoformat(timespec="seconds")


def _display_path(path: Path, cwd: Path) -> str:
    """Show a path relative to ``cwd`` when it lies inside it, else absolute."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(cwd.resolve()).as_posix()
    except ValueError:
        return os.fspath(resolved)


__all__ = [
    "INDEX_FILENAME",
    "IndexEntry",
    "IndexVersionError",
    "SceneIndex",
    "UpsertStats",
]
