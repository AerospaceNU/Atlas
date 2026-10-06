from __future__ import annotations

from pathlib import Path

from PIL import Image
from pydantic import BaseModel

_IMAGE_SUFFIXES = frozenset({".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})


class ImageInfo(BaseModel):
    """Metadata for an image stored under an artifact root."""

    path: str
    width: int
    height: int
    mode: str
    format: str | None


def _within(base: Path, relative_path: str) -> Path:
    """Resolve ``relative_path`` under ``base``, rejecting escapes.

    Absolute paths and ``..`` escapes raise ``ValueError``. The result is
    always inside ``base``, so callers never receive a host path outside the
    sandbox.

    Args:
        base: Directory that bounds the resolved path.
        relative_path: Caller-supplied path, relative to ``base``.

    Returns:
        The resolved absolute path inside ``base``.

    Raises:
        ValueError: If ``relative_path`` is absolute or escapes ``base``.
    """
    path = Path(relative_path)
    if path.is_absolute():
        raise ValueError("Artifact paths must be relative to the artifact root")
    candidate = (base / path).resolve()
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise ValueError("Artifact path escapes the artifact root") from exc
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
        """Resolve a relative path for writing and reject attempts to escape ``root``."""
        return _within(self.root, relative_path)

    def resolve_read(self, relative_path: str) -> Path:
        """Resolve a relative path for reading, preferring the write sandbox.

        A path that exists under ``root`` wins. Otherwise, when ``read_root``
        is set, the same relative path is tried there. A path that exists in
        neither returns its ``root``-relative form, so the caller's error stays
        root-relative and never leaks a host path.
        """
        primary = _within(self.root, relative_path)
        if primary.exists() or self.read_root is None:
            return primary
        fallback = _within(self.read_root, relative_path)
        return fallback if fallback.exists() else primary

    def relative(self, path: Path) -> str:
        """Return a root-relative, portable artifact path."""
        try:
            return path.resolve().relative_to(self.root).as_posix()
        except ValueError as exc:
            raise ValueError("Path is outside the artifact root") from exc

    def list_images(self) -> list[ImageInfo]:
        """List readable image files recursively, ordered by path."""
        images: list[ImageInfo] = []
        for path in sorted(self.root.rglob("*")):
            if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES:
                images.append(self.inspect(self.relative(path)))
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
