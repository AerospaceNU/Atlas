"""Project ``.atlas/`` and user ``~/.atlas/`` locations.

Project root (the workspace):

- ``sessions/`` — one directory per session, no keys
- ``artifacts/`` — files that belong to a session id
- ``data/`` — pulls from the data CLI (existing ``.atlas/data``)

User root (the home directory):

- ``keys/`` — API keys, never written into the project tree. The OpenRouter
  key file is mode ``0600`` and is not written or read through a symlink.
- ``weights/`` — user-level weights (existing ``~/.atlas/weights``)
- ``defaults/`` — user defaults

Removing a session removes its artifacts. Removing entries from the project
root does not touch the user root. Absolute paths and ``..`` fail closed.
Keys and weights are deleted only when that kind is named.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Literal

_PROJECT_DIRS = ("sessions", "artifacts", "data")
_USER_DIRS = ("keys", "weights", "defaults")
_LIST_KINDS = frozenset({"sessions", "artifacts"})
_REMOVE_KINDS = frozenset({"sessions", "artifacts", "keys", "weights"})
_KEY_FILE = "openrouter_api_key"

Kind = Literal["sessions", "artifacts", "keys", "weights"]


class AtlasPathError(ValueError):
    """A layout path left the chosen ``.atlas`` root or skipped confirmation."""


def project_atlas_root(workspace: Path) -> Path:
    """Return the project ``.atlas`` directory (it may not exist yet)."""
    return workspace / ".atlas"


def user_atlas_root(home: Path | None = None) -> Path:
    """Return the user ``.atlas`` directory for ``home`` or the real home."""
    base = Path.home() if home is None else home
    return base / ".atlas"


def ensure_project_layout(workspace: Path) -> Path:
    """Create project session, artifact, and data directories if missing.

    Existing files, including ``.atlas/data``, are left in place.
    """
    root = project_atlas_root(workspace)
    for name in _PROJECT_DIRS:
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


def ensure_user_layout(home: Path | None = None) -> Path:
    """Create user key, weight, and default directories if missing.

    Existing ``~/.atlas/weights`` contents are left in place.
    """
    root = user_atlas_root(home)
    for name in _USER_DIRS:
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


def list_entries(atlas_root: Path, kind: str) -> list[str]:
    """List session or artifact names inside ``atlas_root``.

    Names are the directory entries only. Host paths are not returned.
    """
    if kind not in _LIST_KINDS:
        raise AtlasPathError("only sessions and artifacts can be listed")
    directory = atlas_root / kind
    if not directory.is_dir():
        return []
    return sorted(path.name for path in directory.iterdir())


def remove_entry(atlas_root: Path, kind: str, name: str, *, confirm: bool) -> None:
    """Remove one named entry inside ``atlas_root``.

    ``confirm`` must be true. A session removal also removes the artifact
    directory of the same name. ``keys`` and ``weights`` are removed only
    when ``kind`` names them.
    """
    if not confirm:
        raise AtlasPathError("confirmation required")
    if kind not in _REMOVE_KINDS:
        raise AtlasPathError("unknown atlas entry kind")
    if kind in {"keys", "weights"} and not name.strip():
        raise AtlasPathError("keys and weights must be named explicitly")
    target = _resolve_inside(atlas_root / kind, name)
    if kind == "sessions":
        if not target.exists():
            raise FileNotFoundError(name)
        _delete_matching_artifact(atlas_root, name)
    if not target.exists():
        raise FileNotFoundError(name)
    _delete(target)


def read_user_key(home: Path | None = None) -> str | None:
    """Return the user OpenRouter key, or None when the file is absent.

    A symlink for the key file or the keys directory is ignored, as is a
    keys directory that resolves outside the user ``.atlas`` root.
    """
    path = user_key_path(home)
    keys = path.parent
    if keys.is_symlink() or path.is_symlink() or not path.is_file():
        return None
    if not _is_inside(keys.parent, keys):
        return None
    text = path.read_text(encoding="utf-8").strip()
    return text or None


def write_user_key(key: str, home: Path | None = None) -> Path:
    """Persist a user OpenRouter key under ``~/.atlas/keys`` and return the path.

    The file is created mode ``0o600``. ``home`` is the directory that
    contains ``.atlas``, not the project workspace. A symlink at the key
    path or the keys directory is refused, and so is a keys directory that
    resolves outside the user root.
    """
    cleaned = key.strip()
    if not cleaned:
        raise ValueError("key is empty")
    root = ensure_user_layout(home)
    keys = root / "keys"
    if keys.is_symlink():
        raise AtlasPathError("refusing to follow a key symlink")
    if not _is_inside(root, keys):
        raise AtlasPathError("key path escapes the user atlas root")
    path = keys / _KEY_FILE
    if path.is_symlink():
        raise AtlasPathError("refusing to follow a key symlink")
    _write_private_text(path, cleaned + "\n")
    return path


def user_key_path(home: Path | None = None) -> Path:
    """Path of the user OpenRouter key file."""
    return user_atlas_root(home) / "keys" / _KEY_FILE


def resolve_openrouter_key(
    *,
    session_key: str | None,
    env_key: str | None,
    home: Path | None = None,
) -> str | None:
    """Choose the OpenRouter key.

    A non-empty session key wins over the environment and over the user file.
    The environment wins over the user file. Missing everywhere returns None.
    """
    if session_key is not None and session_key.strip():
        return session_key.strip()
    if env_key is not None and env_key.strip():
        return env_key.strip()
    return read_user_key(home)


def _is_inside(root: Path, candidate: Path) -> bool:
    """Return whether ``candidate`` resolves inside ``root``."""
    try:
        root_resolved = root.resolve()
        candidate_resolved = candidate.resolve()
    except OSError:
        return False
    return candidate_resolved == root_resolved or root_resolved in candidate_resolved.parents


def _write_private_text(path: Path, text: str) -> None:
    """Create or replace ``path`` as user-read/write text, without following a symlink."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    if isinstance(no_follow, int):
        flags |= no_follow
    fd = os.open(path, flags, 0o600)
    try:
        payload = memoryview(text.encode("utf-8"))
        while payload:
            written = os.write(fd, payload)
            if written <= 0:
                raise OSError("could not write the key file")
            payload = payload[written:]
    finally:
        os.close(fd)
    os.chmod(path, 0o600)


def _delete_matching_artifact(atlas_root: Path, name: str) -> None:
    artifact = _resolve_inside(atlas_root / "artifacts", name)
    if artifact.exists():
        _delete(artifact)


def _resolve_inside(parent: Path, name: str) -> Path:
    relative = Path(name)
    if (
        not name.strip()
        or relative.is_absolute()
        or ".." in relative.parts
        or relative.parts == (".",)
    ):
        raise AtlasPathError("path escapes the atlas root")
    parent_resolved = parent.resolve()
    candidate = (parent / relative).resolve()
    if candidate == parent_resolved or parent_resolved not in candidate.parents:
        raise AtlasPathError("path escapes the atlas root")
    return candidate


def _delete(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
        return
    path.unlink()
