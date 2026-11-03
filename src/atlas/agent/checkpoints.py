"""Per-turn file checkpoints so one bad edit can be undone.

A mutating file tool asks this module to copy the target before it writes.
The copy lives beside the session artifacts, under
``<workspace>/.atlas/checkpoints/<session>/<sequence>/<relative path>``, and
the highest sequence is the one ``restore_latest`` puts back.

Only paths that exist before the write are copied, so a newly created file is
undone by removing it rather than by restoring bytes. The shell tool is not
checkpointed: its targets are not known before it runs.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from atlas.agent.artifacts import LocalArtifactStore

_MANIFEST = "manifest.json"


def checkpoints_root(store: LocalArtifactStore) -> Path:
    """Return the checkpoint directory for this session.

    Checkpoints live in the workspace's ``.atlas`` tree, beside artifacts but
    outside them, so removing a session does not throw away its undo history.
    """
    name = store.artifact_root.name if store.artifact_root != store.root else "session"
    return store.root / ".atlas" / "checkpoints" / name


def _next_sequence(root: Path) -> int:
    if not root.is_dir():
        return 1
    numbers = [int(path.name) for path in root.iterdir() if path.is_dir() and path.name.isdigit()]
    return max(numbers, default=0) + 1


def _saved_path(directory: Path, path: str) -> Path:
    """Return where a checkpoint copy of ``path`` lives under ``directory``.

    A plain relative path keeps its shape so the checkpoint is easy to
    inspect. An absolute path or a ``..`` traversal gets a digest name, since
    neither can be joined beneath the checkpoint directory safely.
    """
    candidate = Path(path).expanduser()
    if not candidate.is_absolute() and ".." not in candidate.parts:
        return directory / candidate
    digest = hashlib.sha256(str(candidate).encode("utf-8")).hexdigest()[:16]
    return directory / "external" / f"{digest}.bin"


def snapshot(store: LocalArtifactStore, relative_path: str) -> None:
    """Copy one file before a mutating tool changes it.

    A path that does not exist yet is recorded without bytes so a restore can
    delete it. A symlink or a non-regular file is not copied and does not
    raise.

    Args:
        store: Session store whose working tree and artifacts bound the copy.
        relative_path: Path the tool is about to change.
    """
    source = store.resolve_read(relative_path)
    if source.is_symlink() or (source.exists() and not source.is_file()):
        return
    root = checkpoints_root(store) / str(_next_sequence(checkpoints_root(store)))
    entry: dict[str, object] = {"path": Path(relative_path).as_posix()}
    if source.is_file():
        destination = _saved_path(root, relative_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        entry["existed"] = True
    else:
        entry["existed"] = False
    root.mkdir(parents=True, exist_ok=True)
    (root / _MANIFEST).write_text(json.dumps(entry) + "\n", encoding="utf-8")


def restore_latest(store: LocalArtifactStore) -> str | None:
    """Restore the newest checkpoint and remove it.

    Args:
        store: Session store to restore into.

    Returns:
        The restored workspace-relative path, or None when no checkpoint
        exists.
    """
    root = checkpoints_root(store)
    if not root.is_dir():
        return None
    sequences = sorted(
        (path for path in root.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    )
    for directory in reversed(sequences):
        manifest = directory / _MANIFEST
        if not manifest.is_file():
            shutil.rmtree(directory, ignore_errors=True)
            continue
        entry = json.loads(manifest.read_text(encoding="utf-8"))
        relative = str(entry.get("path") or "")
        if not relative:
            shutil.rmtree(directory, ignore_errors=True)
            continue
        target = store.resolve(relative)
        saved = _saved_path(directory, relative)
        if entry.get("existed") and saved.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(saved, target)
        elif not entry.get("existed"):
            target.unlink(missing_ok=True)
        shutil.rmtree(directory, ignore_errors=True)
        return relative
    return None
