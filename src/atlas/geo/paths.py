"""Where analysis products live under an artifact root."""

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
    "list_stack_ids",
    "runs_path",
    "stacks_dir",
    "workspace",
]

SUBDIRS = ("stacks", "figures", "tables", "exports", "scripts")
_SCRATCH_MARKERS = (".tmp-", ".old-")


def workspace() -> Path:
    """Return ``ATLAS_ARTIFACT_ROOT`` when set, else ``./.atlas``."""
    env = os.environ.get("ATLAS_ARTIFACT_ROOT")
    return Path(env).expanduser() if env else Path.cwd() / ".atlas"


def analysis_dir(store: LocalArtifactStore | None = None, *, ensure: bool = False) -> Path:
    """Return ``<artifact_root>/analysis``.

    Args:
        store: Session store; None uses ``workspace()``.
        ensure: Create the standard subdirectories.

    Returns:
        The analysis directory.
    """
    root = store.artifact_root if store is not None else workspace()
    adir = root / "analysis"
    if ensure:
        for sub in SUBDIRS:
            (adir / sub).mkdir(parents=True, exist_ok=True)
    return adir


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
