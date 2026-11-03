from __future__ import annotations

from pathlib import Path

from PIL import Image
from pydantic import BaseModel

_IMAGE_SUFFIXES = frozenset({".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})


class ArtifactError(ValueError):
    """A tool path or size request was rejected."""


class ImageInfo(BaseModel):
    """Metadata for an image stored under an artifact root."""

    path: str
    width: int
    height: int
    mode: str
    format: str | None


class LocalArtifactStore:
    """An agent's working tree and its tool-output directory.

    ``root`` is the working tree, usually the directory the person pointed the
    session at. A relative tool path resolves against it. An absolute path is
    used as given, ``~`` is expanded, and ``..`` is allowed.

    ``artifact_root`` is where tool-produced outputs are written. It defaults
    to ``root`` so removing a session removes its artifacts.

    ``read_root`` is an optional second tree that reads may fall back to. It
    lets a caller keep artifacts outside the working tree.
    """

    def __init__(
        self,
        root: Path,
        *,
        read_root: Path | None = None,
        artifact_root: Path | None = None,
    ) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.artifact_root = (artifact_root or root).resolve()
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.read_root = read_root.resolve() if read_root is not None else None

    def _path(self, base: Path, path: str) -> Path:
        """Resolve ``path`` against ``base``, honoring an absolute path or ``~``."""
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = base / candidate
        return candidate.resolve()

    def resolve(self, path: str) -> Path:
        """Resolve a path for writing in the working tree."""
        return self._path(self.root, path)

    def resolve_artifact(self, path: str) -> Path:
        """Resolve a path for writing a tool-produced artifact."""
        return self._path(self.artifact_root, path)

    def resolve_read(self, path: str) -> Path:
        """Resolve a path for reading.

        A path that exists in the working tree wins, then the artifact root,
        then ``read_root`` when one is mounted. A path that exists in none of
        them returns its ``root``-relative form.
        """
        primary = self._path(self.root, path)
        if primary.exists():
            return primary
        if self.artifact_root != self.root:
            artifact = self._path(self.artifact_root, path)
            if artifact.exists():
                return artifact
        if self.read_root is None:
            return primary
        fallback = self._path(self.read_root, path)
        return fallback if fallback.exists() else primary

    def read_project_bytes(self, path: str, *, max_bytes: int | None = None) -> bytes:
        """Read one file from the project workspace.

        The project root is ``read_root`` when this store has one, and ``root``
        otherwise. Any path the host can read is accepted, including an
        absolute path and ``..``.

        Args:
            path: Path relative to the project root, or an absolute path.
            max_bytes: When set, a longer file is refused instead of truncated.

        Returns:
            The file bytes, at most ``max_bytes`` when that limit is set.

        Raises:
            ArtifactError: The path is not a regular file, or exceeds
                ``max_bytes``.
            FileNotFoundError: The file is missing.
            OSError: The file could not be read.
        """
        base = self.read_root if self.read_root is not None else self.root
        resolved = self._path(base, path)
        if not resolved.exists():
            raise FileNotFoundError(path)
        if not resolved.is_file():
            raise ArtifactError("Project path is not a regular file")
        if max_bytes is not None and resolved.stat().st_size > max_bytes:
            raise ArtifactError(f"{path} exceeds the {max_bytes} byte read limit")
        return resolved.read_bytes()

    def relative(self, path: Path) -> str:
        """Return a portable path, relative to root when it sits inside root."""
        resolved = path.resolve()
        try:
            return resolved.relative_to(self.root).as_posix()
        except ValueError:
            return resolved.as_posix()

    def list_images(self) -> list[ImageInfo]:
        """List readable image files recursively under the artifact root."""
        images: list[ImageInfo] = []
        for path in sorted(self.artifact_root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in _IMAGE_SUFFIXES:
                continue
            try:
                images.append(self.inspect(self.relative(path)))
            except (ArtifactError, OSError, ValueError):
                continue
        return images

    def inspect(self, path: str) -> ImageInfo:
        """Read image dimensions for ``path``."""
        resolved = self.resolve(path)
        if not resolved.is_file():
            raise FileNotFoundError(f"Image does not exist: {path}")
        with Image.open(resolved) as image:
            return ImageInfo(
                path=self.relative(resolved),
                width=image.width,
                height=image.height,
                mode=image.mode,
                format=image.format,
            )
