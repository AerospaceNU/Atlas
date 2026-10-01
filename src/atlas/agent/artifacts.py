from __future__ import annotations

import datetime
import logging
from pathlib import Path

from PIL import Image
from pydantic import BaseModel

_IMAGE_SUFFIXES = frozenset({".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})

_AUDIT_LOGGER = logging.getLogger("atlas.sandbox.audit")
_AUDIT_LOG_RELATIVE_PATH = Path(".atlas") / "sandbox_audit.log"


class SandboxDenied(ValueError):
    """A sandbox invariant rejected a path or a size (see docs/adr/0001)."""


def record_denial(root: Path, operation: str, detail: str) -> None:
    """Log one denied access and append it to ``root/.atlas/sandbox_audit.log``."""
    timestamp = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
    line = f"{timestamp} DENIED {operation}: {detail}"
    _AUDIT_LOGGER.warning(line)
    log_path = root / _AUDIT_LOG_RELATIVE_PATH
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


class ImageInfo(BaseModel):
    """Metadata for an image stored under an artifact root."""

    path: str
    width: int
    height: int
    mode: str
    format: str | None


class LocalArtifactStore:
    """Constrain an agent's image work to one local directory.

    Paths passed through the tool interface are always relative to ``root``.
    This prevents tools from reading or writing arbitrary locations on the host.
    """

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve(self, relative_path: str) -> Path:
        """Resolve a relative path, following symlinks, and reject escapes from ``root``."""
        path = Path(relative_path)
        if path.is_absolute():
            raise SandboxDenied("Artifact paths must be relative to the artifact root")
        candidate = (self.root / path).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise SandboxDenied("Artifact path escapes the artifact root") from exc
        return candidate

    def relative(self, path: Path) -> str:
        """Return a root-relative, portable artifact path."""
        try:
            return path.resolve().relative_to(self.root).as_posix()
        except ValueError as exc:
            raise SandboxDenied("Path is outside the artifact root") from exc

    def list_images(self) -> list[ImageInfo]:
        """List readable image files recursively, skipping symlinks that escape ``root``."""
        images: list[ImageInfo] = []
        for path in sorted(self.root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in _IMAGE_SUFFIXES:
                continue
            try:
                relative_path = self.relative(path)
            except SandboxDenied:
                continue
            images.append(self.inspect(relative_path))
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
