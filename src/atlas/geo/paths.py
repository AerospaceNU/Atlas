"""Where analysis products live: the persistent project ``.atlas/analysis`` directory.

Stacks, figures, tables, exports, scripts and ``runs.jsonl`` outlive a session. They sit
under the workspace's ``.atlas`` root, not under the per-session artifact root, so a new
session on the same workspace sees them by id. ``ATLAS_ANALYSIS_DIR`` overrides the
directory for scripts; the bash tool sets it so scripts and tools share one location.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from atlas.agent.artifacts import LocalArtifactStore

__all__ = [
    "SUBDIRS",
    "analysis_dir",
    "latest_stack_dir",
    "legacy_analysis_dir",
    "list_stack_ids",
    "runs_path",
    "stacks_dir",
    "workspace",
]

SUBDIRS = ("stacks", "figures", "tables", "exports", "scripts")
_SCRATCH_MARKERS = (".tmp-", ".old-")


def workspace() -> Path:
    """Return the directory that holds ``analysis/`` (the project ``.atlas`` root).

    ``ATLAS_ANALYSIS_DIR`` gives its parent; else ``ATLAS_ARTIFACT_ROOT`` when set, else
    ``./.atlas``.
    """
    return analysis_dir().parent


def analysis_dir(store: LocalArtifactStore | None = None, *, ensure: bool = False) -> Path:
    """Return the persistent ``<workspace>/.atlas/analysis`` directory.

    Args:
        store: Session store; its project root (``read_root`` or ``root``) picks the
            workspace. None reads ``ATLAS_ANALYSIS_DIR``, then ``ATLAS_ARTIFACT_ROOT``,
            then uses ``./.atlas``.
        ensure: Create the standard subdirectories.

    Returns:
        The analysis directory.
    """
    if store is not None:
        adir = (store.read_root or store.root) / ".atlas" / "analysis"
    elif os.environ.get("ATLAS_ANALYSIS_DIR"):
        adir = Path(os.environ["ATLAS_ANALYSIS_DIR"]).expanduser()
    elif os.environ.get("ATLAS_ARTIFACT_ROOT"):
        adir = Path(os.environ["ATLAS_ARTIFACT_ROOT"]).expanduser() / "analysis"
    else:
        adir = Path.cwd() / ".atlas" / "analysis"
    if ensure:
        for sub in SUBDIRS:
            (adir / sub).mkdir(parents=True, exist_ok=True)
    return adir


def legacy_analysis_dir(store: LocalArtifactStore) -> Path:
    """Return the old per-session ``<artifact_root>/analysis`` directory (read-only fallback)."""
    return store.artifact_root / "analysis"


def stacks_dir(adir: Path | None = None) -> Path:
    """Return the stacks directory of an analysis directory."""
    return (adir if adir is not None else analysis_dir()) / "stacks"


def runs_path(adir: Path | None = None) -> Path:
    """Return the ``runs.jsonl`` path of an analysis directory."""
    return (adir if adir is not None else analysis_dir()) / "runs.jsonl"


def _committed(adir: Path | None) -> list[tuple[float, str, Path]]:
    root = stacks_dir(adir)
    found: list[tuple[float, str, Path]] = []
    if not root.is_dir():
        return found
    for child in root.iterdir():
        marker = child / "stack.json"
        if any(m in child.name for m in _SCRATCH_MARKERS) or not marker.is_file():
            continue
        found.append((marker.stat().st_mtime, child.name, child))
    found.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return found


def list_stack_ids(adir: Path | None = None) -> list[str]:
    """Return committed stack ids, newest first."""
    return [name for _, name, _ in _committed(adir)]


def latest_stack_dir(adir: Path | None = None) -> Path | None:
    """Return the newest committed stack directory, or None."""
    found = _committed(adir)
    return found[0][2] if found else None
