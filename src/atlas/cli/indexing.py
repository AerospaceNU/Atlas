"""Record a pull's scenes in the scene index; shared by ``atlas data`` and the agent."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from atlas.archive import (
    INDEX_FILENAME,
    IndexEntry,
    IndexVersionError,
    SavedLayer,
    SceneIndex,
    UpsertStats,
)
from atlas.cli.download import DownloadReport, read_aoi
from atlas.compile.product import SceneCatalog
from atlas.data.base import DataPullClient
from atlas.data.sources.fldas import FldasClient


@dataclass(frozen=True)
class PullIndexSummary:
    """What :func:`index_pull` recorded, or why it recorded nothing."""

    path: Path
    stats: UpsertStats | None = None
    error: str | None = None

    def index_id(self, source: str, scene_id: str) -> str | None:
        """The index id a scene was recorded under, or None if it was not indexed."""
        if self.stats is None:
            return None
        return self.stats.index_ids.get((source, scene_id))

    def describe(self, *, path: str | None = None) -> str:
        """One plain-text line for a CLI summary or a tool result.

        ``path`` replaces the index path in the line, for a caller that shows
        paths relative to its workspace.
        """
        if self.stats is None:
            return f"scene index not updated: {self.error}"
        shown = path if path is not None else self.path
        line = f"scene index {shown}: {self.stats.added} new, {self.stats.updated} refreshed"
        if self.stats.skipped_request_footprint:
            line += (
                f", {self.stats.skipped_request_footprint} request-footprint products not indexed"
            )
        if self.stats.skipped_invalid:
            line += f", {self.stats.skipped_invalid} without source or collection not indexed"
        return line


def index_pull(
    out_dir: Path,
    catalog: SceneCatalog,
    report: DownloadReport,
    clients: Mapping[str, DataPullClient],
) -> PullIndexSummary | None:
    """Add a pull's scenes, and the layers it saved, to ``<out>/index.sqlite``.

    Scenes from failed searches and FLDAS tables are left out. Returns None when
    there was nothing to record. An index failure is returned, never raised: the
    pull's own record is the manifest, so indexing must not fail a pull.
    """
    saved = {
        (item.source, item.scene_id): SavedLayer(item.asset, item.path, read_aoi(item.path))
        for item in report.outcomes
        if item.status in {"saved", "skipped"} and item.path is not None and item.asset
    }
    entries = [
        IndexEntry(
            scene=scene,
            satellite=_satellite(clients.get(src.source)),
            saved=saved.get((src.source, scene.id)),
        )
        for src in catalog.sources
        if src.error is None and not isinstance(clients.get(src.source), FldasClient)
        for scene in src.scenes
    ]
    if not entries:
        return None
    path = out_dir / INDEX_FILENAME
    try:
        with SceneIndex(path) as index:
            return PullIndexSummary(path, index.upsert(entries))
    except (sqlite3.Error, OSError, ValueError, IndexVersionError) as exc:
        return PullIndexSummary(path, error=f"{type(exc).__name__}: {exc}")


def _satellite(client: DataPullClient | None) -> str | None:
    family = getattr(client, "satellite", None)
    return family if isinstance(family, str) and family.strip() else None
