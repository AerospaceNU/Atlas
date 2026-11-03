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

import json
import shutil
from pathlib import Path

from atlas.agent.artifacts import LocalArtifactStore, SandboxDenied

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


def snapshot(store: LocalArtifactStore, relative_path: str) -> None:
    """Copy one workspace file before a mutating tool changes it.

    A path that does not exist yet is recorded without bytes so a restore can
    delete it. A path that escapes the sandbox, is a symlink, or is a
    non-regular file is not copied and does not raise: the write itself will
    fail closed.

    Args:
        store: Session store whose workspace and artifacts bound the copy.
        relative_path: Workspace-relative path the tool is about to change.
    """
    try:
        source = store.resolve_read(relative_path)
    except SandboxDenied:
        return
    if source.is_symlink() or (source.exists() and not source.is_file()):
        return
    root = checkpoints_root(store) / str(_next_sequence(checkpoints_root(store)))
    destination = root / Path(relative_path)
    entry: dict[str, object] = {"path": Path(relative_path).as_posix()}
    if source.is_file():
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
        saved = directory / Path(relative)
        if entry.get("existed") and saved.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(saved, target)
        elif not entry.get("existed"):
            target.unlink(missing_ok=True)
        shutil.rmtree(directory, ignore_errors=True)
        return relative
    return None
