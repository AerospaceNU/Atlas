"""A filesystem boundary for local image inputs and generated artifacts."""

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


class LocalArtifactStore:
    """Constrain an agent's image work to one local directory.

    Paths passed through the tool interface are always relative to ``root``.
    This prevents tools from reading or writing arbitrary locations on the host.
    """

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve(self, relative_path: str) -> Path:
        """Resolve a relative path and reject attempts to escape ``root``."""
        path = Path(relative_path)
        if path.is_absolute():
            raise ValueError("Artifact paths must be relative to the artifact root")
        candidate = (self.root / path).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("Artifact path escapes the artifact root") from exc
        return candidate

    def relative(self, path: Path) -> str:
        """Return a root-relative, portable artifact path."""
        try:
            return str(path.resolve().relative_to(self.root))
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
