from __future__ import annotations

import datetime
import logging
import os
import stat
from pathlib import Path

from PIL import Image
from pydantic import BaseModel

_IMAGE_SUFFIXES = frozenset({".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})

_AUDIT_LOGGER = logging.getLogger("atlas.sandbox.audit")
_AUDIT_LOG_PATH = ".atlas/sandbox_audit.log"


class SandboxDenied(ValueError):
    """A sandbox invariant rejected a path or a size (see docs/adr/0001)."""


def record_denial(store: LocalArtifactStore, operation: str, detail: str) -> None:
    """Log one denied access and append it as a single line to the workspace audit log.

    The log path goes through ``store.resolve`` like any tool path, and the file
    write is skipped if resolve denies it. Other write failures raise.
    """
    timestamp = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
    # Escape control characters so a newline in a model-chosen path can't split
    # or forge records.
    record = f"{operation}: {detail}".encode("unicode_escape").decode("ascii")
    line = f"{timestamp} DENIED {record}"
    _AUDIT_LOGGER.warning(line)
    try:
        log_path = store.resolve(_AUDIT_LOG_PATH)
    except SandboxDenied as exc:
        _AUDIT_LOGGER.error("Skipped the audit file write to %s: %s", _AUDIT_LOG_PATH, exc)
        return
    log_path.parent.mkdir(exist_ok=True)
    # A symlink or hardlink planted after resolve() returned is refused here.
    fd = os.open(log_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise OSError(f"{_AUDIT_LOG_PATH} is not a single-link regular file")
        handle.write(line + "\n")


class ImageInfo(BaseModel):
    """Metadata for an image stored under an artifact root."""

    path: str
    width: int
    height: int
    mode: str
    format: str | None


def resolve_within(base: Path, relative_path: str) -> Path:
    """Resolve ``relative_path`` under ``base``, rejecting escapes.

    This is the public containment check. The artifact store and the skill
    loader both use it, so a skill path is held to the same sandbox rules as
    a tool path.

    Absolute paths and ``..`` escapes raise ``SandboxDenied``. Resolution
    follows symlinks before the containment check, so the result is always
    inside ``base`` and callers never receive a host path outside the sandbox.
    An existing regular file with more than one hardlink is refused too, since
    its other name may be outside ``base``.

    Args:
        base: Directory that bounds the resolved path.
        relative_path: Caller-supplied path, relative to ``base``.

    Returns:
        The resolved absolute path inside ``base``.

    Raises:
        SandboxDenied: If ``relative_path`` is absolute, escapes ``base``, or
            names a hardlinked file.
    """
    path = Path(relative_path)
    if path.is_absolute():
        raise SandboxDenied("Artifact paths must be relative to the artifact root")
    candidate = (base / path).resolve()
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise SandboxDenied("Artifact path escapes the artifact root") from exc
    try:
        info = candidate.stat()
    except OSError:
        return candidate
    if stat.S_ISREG(info.st_mode) and info.st_nlink > 1:
        raise SandboxDenied("Artifact path is a hardlinked file that may lead outside the root")
    return candidate


class LocalArtifactStore:
    """Constrain an agent's work to one local directory.

    Paths passed through the tool interface are always relative to ``root``
    (the write sandbox). An optional ``read_root`` widens *reads* to a second
    tree -- the project workspace -- so an agent can inspect source files and
    inputs without gaining write access to them. Writes always stay in
    ``root``, and both roots reject absolute paths and ``..`` escapes.
    """

    def __init__(self, root: Path, *, read_root: Path | None = None) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.read_root = read_root.resolve() if read_root is not None else None

    def resolve(self, relative_path: str) -> Path:
        """Resolve a relative path for writing, following symlinks, and reject escapes."""
        return resolve_within(self.root, relative_path)

    def resolve_read(self, relative_path: str) -> Path:
        """Resolve a relative path for reading, preferring the write sandbox.

        A path that exists under ``root`` wins. Otherwise, when ``read_root``
        is set, the same relative path is tried there. A path that exists in
        neither returns its ``root``-relative form, so the caller's error stays
        root-relative and never leaks a host path.
        """
        primary = resolve_within(self.root, relative_path)
        if primary.exists() or self.read_root is None:
            return primary
        fallback = resolve_within(self.read_root, relative_path)
        return fallback if fallback.exists() else primary

    def relative(self, path: Path) -> str:
        """Return a root-relative, portable artifact path."""
        try:
            return path.resolve().relative_to(self.root).as_posix()
        except ValueError as exc:
            raise SandboxDenied("Path is outside the artifact root") from exc

    def list_images(self) -> list[ImageInfo]:
        """List readable image files recursively, skipping symlinks and hardlinks that may escape."""
        images: list[ImageInfo] = []
        for path in sorted(self.root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in _IMAGE_SUFFIXES:
                continue
            try:
                images.append(self.inspect(self.relative(path)))
            except SandboxDenied:
                continue
        return images

    def inspect(self, relative_path: str) -> ImageInfo:
        """Read image dimensions without exposing an absolute host path."""
        path = self.resolve(relative_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image does not exist: {relative_path}")
        with Image.open(path) as image:
            return ImageInfo(
                path=self.relative(path),
                width=image.width,
                height=image.height,
                mode=image.mode,
                format=image.format,
            )
