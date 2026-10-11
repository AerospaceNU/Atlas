"""Project ``.atlas/`` and user ``~/.atlas/`` locations.

Project root (the workspace):

- ``sessions/`` — one directory per session, no keys
- ``artifacts/`` — files that belong to a session id
- ``data/`` — pulls from the data CLI (existing ``.atlas/data``)

User root (the home directory):

- ``keys/`` — API keys, never written into the project tree. The directory
  is mode ``0700`` and the OpenRouter key file is mode ``0600``. Neither is
  read or written through a symlink.
- ``weights/`` — user-level weights (existing ``~/.atlas/weights``)
- ``defaults/`` — user defaults

Removing a session removes its artifacts. Removing entries from the project
root does not touch the user root. Absolute paths and ``..`` fail closed.
Keys and weights are deleted only when that kind is named.
"""

from __future__ import annotations

import os
import shutil
import warnings
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


class OpenRouterKeyWarning(UserWarning):
    """The OpenRouter key was kept, but its shape or file mode looks wrong."""


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

    Existing ``~/.atlas/weights`` contents are left in place. An existing
    ``keys`` directory is tightened to mode ``0700``. On Windows that mode
    is requested but not enforced; the directory ACL is the real control.
    """
    root = user_atlas_root(home)
    for name in _USER_DIRS:
        (root / name).mkdir(parents=True, exist_ok=True)
    _tighten_keys_dir(root)
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

    The file is opened with ``O_NOFOLLOW`` where the OS provides it. A
    symlink, or a keys directory that resolves outside the user ``.atlas``
    root, is ignored. On POSIX, a file that is readable by group or others
    produces :class:`OpenRouterKeyWarning`. Windows does not report those
    Unix bits, so the warning is skipped there.
    """
    path = user_key_path(home)
    keys = path.parent
    if keys.is_symlink() or path.is_symlink() or not _is_inside(keys.parent, keys):
        return None
    if not path.is_file():
        return None
    flags = os.O_RDONLY | _no_follow_flag()
    try:
        fd = os.open(path, flags)
    except OSError:
        return None
    try:
        info = os.fstat(fd)
        if os.name == "posix" and info.st_mode & 0o077:
            warnings.warn(
                "OpenRouter key file permissions are too loose",
                OpenRouterKeyWarning,
                stacklevel=2,
            )
        chunks: list[bytes] = []
        while True:
            block = os.read(fd, 4096)
            if not block:
                break
            chunks.append(block)
    finally:
        os.close(fd)
    text = b"".join(chunks).decode("utf-8").strip()
    if not text or "\n" in text or "\r" in text:
        return None
    return text


def prepare_openrouter_key(key: str, *, warn: bool = True) -> tuple[str, str]:
    """Strip ``key`` and return ``(cleaned, warning)``.

    A value that is empty after stripping, or that still contains a newline
    or carriage return, raises ``ValueError``. A key that does not start
    with ``sk-or-`` is kept. When ``warn`` is true that shape also emits
    :class:`OpenRouterKeyWarning`.
    """
    cleaned = key.strip()
    if not cleaned or "\n" in cleaned or "\r" in cleaned:
        raise ValueError("key is empty or contains a newline")
    warning = ""
    if not cleaned.startswith("sk-or-"):
        warning = "OpenRouter key does not start with sk-or-"
        if warn:
            warnings.warn(warning, OpenRouterKeyWarning, stacklevel=2)
    return cleaned, warning


def write_user_key(key: str, home: Path | None = None) -> Path:
    """Persist a user OpenRouter key under ``~/.atlas/keys`` and return the path.

    ``home`` is the directory that contains ``.atlas``, not the project
    workspace. The keys directory is mode ``0700``. The file is replaced
    atomically: a temporary file is created with ``O_EXCL`` and mode
    ``0600``, fsynced, then renamed into place. An existing file is chmodded
    to ``0600`` before that rename and is never opened with ``O_TRUNC``.

    A symlink at the key path or the keys directory is refused, and so is a
    keys directory that resolves outside the user root.

    On Windows the Unix modes are not enforced. ``chmod`` is still called,
    but the file ACL decides who can read the key, and symlinks are rejected
    with ``is_symlink`` because ``O_NOFOLLOW`` is not available.
    """
    cleaned, _warning = prepare_openrouter_key(key)
    root = ensure_user_layout(home)
    keys = _require_keys_dir(root)
    path = keys / _KEY_FILE
    _atomic_write_private(keys, path, cleaned + "\n")
    return path


def delete_user_key(home: Path | None = None) -> bool:
    """Remove the user OpenRouter key file.

    Returns true when a file was removed. A missing file returns false. A
    symlink is refused and left in place.
    """
    path = user_key_path(home)
    keys = path.parent
    if keys.is_symlink() or path.is_symlink():
        raise AtlasPathError("refusing to follow a key symlink")
    if keys.exists() and not _is_inside(keys.parent, keys):
        raise AtlasPathError("key path escapes the user atlas root")
    if not path.is_file():
        return False
    path.unlink()
    return True


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


def _tighten_keys_dir(root: Path) -> None:
    """Set an existing real keys directory to mode ``0700``."""
    keys = root / "keys"
    if keys.is_symlink() or not keys.is_dir() or not _is_inside(root, keys):
        return
    _chmod(keys, 0o700)


def _require_keys_dir(root: Path) -> Path:
    """Return the user keys directory, refusing a symlink or an escaped path."""
    keys = root / "keys"
    if keys.is_symlink():
        raise AtlasPathError("refusing to follow a key symlink")
    if not keys.is_dir() or not _is_inside(root, keys):
        raise AtlasPathError("key path escapes the user atlas root")
    _chmod(keys, 0o700)
    return keys


def _no_follow_flag() -> int:
    """Return ``O_NOFOLLOW`` when this OS has it, otherwise zero.

    Windows has no ``O_NOFOLLOW``. Callers still reject a symlink with
    ``Path.is_symlink`` before they open.
    """
    flag = getattr(os, "O_NOFOLLOW", 0)
    if isinstance(flag, int):
        return flag
    return 0


def _chmod(path: Path, mode: int) -> None:
    """Request ``mode``. On Windows a failure is ignored; POSIX re-raises.

    Windows does not apply Unix permission bits. The call is still made so
    a POSIX host, including a Windows drive mounted that way, is tightened.
    """
    try:
        os.chmod(path, mode)
    except OSError:
        if os.name == "posix":
            raise


def _atomic_write_private(directory: Path, path: Path, text: str) -> None:
    """Replace ``path`` via a mode-``0600`` temporary file in ``directory``.

    An existing regular file is chmodded to ``0600`` before the rename so a
    loose file is never truncated in place. The temporary file is opened
    with ``O_EXCL`` and fsynced before ``os.replace``.
    """
    if path.is_symlink():
        raise AtlasPathError("refusing to follow a key symlink")
    if path.exists() and not path.is_file():
        raise AtlasPathError("key path is not a file")
    if path.is_file():
        _chmod(path, 0o600)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _no_follow_flag()
    temp = directory / f".{path.name}.{os.getpid()}.{os.urandom(4).hex()}.tmp"
    fd = os.open(temp, flags, 0o600)
    try:
        payload = memoryview(text.encode("utf-8"))
        while payload:
            written = os.write(fd, payload)
            if written <= 0:
                raise OSError("could not write the key file")
            payload = payload[written:]
        os.fsync(fd)
    except Exception:
        os.close(fd)
        temp.unlink(missing_ok=True)
        raise
    os.close(fd)
    try:
        os.replace(temp, path)
    except Exception:
        temp.unlink(missing_ok=True)
        raise
    _chmod(path, 0o600)


def _is_inside(root: Path, candidate: Path) -> bool:
    """Return whether ``candidate`` resolves inside ``root``."""
    try:
        root_resolved = root.resolve()
        candidate_resolved = candidate.resolve()
    except OSError:
        return False
    return candidate_resolved == root_resolved or root_resolved in candidate_resolved.parents


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
