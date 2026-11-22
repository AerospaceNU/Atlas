"""Per-turn file checkpoints so bad edits or multi-file patches can be undone."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Sequence
from pathlib import Path

from atlas.agent.artifacts import LocalArtifactStore

_MANIFEST = "manifest.json"


def _session_name(store: LocalArtifactStore) -> str:
    """Name the undo chain; a subagent shares its parent's chain, however deep."""
    current = store.artifact_root
    while current != store.root and current.parent.name == "subagents":
        current = current.parent.parent
    return "session" if current == store.root else current.name


def checkpoints_root(store: LocalArtifactStore) -> Path:
    """Return the checkpoint directory for this session under ``.atlas/checkpoints``."""
    return store.root / ".atlas" / "checkpoints" / _session_name(store)


def _next_sequence(root: Path) -> int:
    if not root.is_dir():
        return 1
    numbers = [int(p.name) for p in root.iterdir() if p.is_dir() and p.name.isdigit()]
    return max(numbers, default=0) + 1


def _saved_path(directory: Path, path: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute() and ".." not in candidate.parts:
        return directory / candidate
    digest = hashlib.sha256(str(candidate).encode("utf-8")).hexdigest()[:16]
    return directory / "external" / f"{digest}.bin"


def _record_file(
    directory: Path, store: LocalArtifactStore, relative_path: str
) -> dict[str, object] | None:
    source = store.resolve_read(relative_path)
    if source.is_symlink() or (source.exists() and not source.is_file()):
        return None
    entry: dict[str, object] = {"path": Path(relative_path).as_posix(), "existed": False}
    if source.is_file():
        destination = _saved_path(directory, relative_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        entry["existed"] = True
    return entry


def snapshot(store: LocalArtifactStore, relative_path: str) -> None:
    """Copy one file before a mutating tool changes it."""
    base = checkpoints_root(store)
    root = base / str(_next_sequence(base))
    entry = _record_file(root, store, relative_path)
    if entry is None:
        return
    root.mkdir(parents=True, exist_ok=True)
    (root / _MANIFEST).write_text(json.dumps(entry) + "\n", encoding="utf-8")


def snapshot_many(store: LocalArtifactStore, paths: Sequence[str]) -> None:
    """Snapshot one or more target paths as a single undoable checkpoint."""
    unique = [p for p in dict.fromkeys(paths) if p]
    if not unique:
        return
    if len(unique) == 1:
        snapshot(store, unique[0])
        return
    base = checkpoints_root(store)
    root = base / str(_next_sequence(base))
    entries = [entry for rel in unique if (entry := _record_file(root, store, rel)) is not None]
    if not entries:
        return
    root.mkdir(parents=True, exist_ok=True)
    (root / _MANIFEST).write_text(json.dumps({"files": entries}) + "\n", encoding="utf-8")


def _restore_entry(
    directory: Path, store: LocalArtifactStore, entry: dict[str, object]
) -> str | None:
    relative = str(entry.get("path") or "")
    if not relative:
        return None
    target = store.resolve(relative)
    saved = _saved_path(directory, relative)
    if entry.get("existed") and saved.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(saved, target)
    elif not entry.get("existed"):
        target.unlink(missing_ok=True)
    return relative


def restore_latest(store: LocalArtifactStore) -> str | None:
    """Restore the newest checkpoint (single or multi-file) and remove it."""
    root = checkpoints_root(store)
    if not root.is_dir():
        return None
    sequences = sorted(
        (p for p in root.iterdir() if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name),
    )
    for directory in reversed(sequences):
        manifest = directory / _MANIFEST
        if not manifest.is_file():
            shutil.rmtree(directory, ignore_errors=True)
            continue
        data = json.loads(manifest.read_text(encoding="utf-8"))
        entries = data["files"] if isinstance(data.get("files"), list) else [data]
        restored = [
            rel
            for item in entries
            if isinstance(item, dict) and (rel := _restore_entry(directory, store, item))
        ]
        shutil.rmtree(directory, ignore_errors=True)
        if restored:
            return restored[0] if len(restored) == 1 else ", ".join(restored)
    return None
