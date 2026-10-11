"""Read sandbox image artifacts into vision-model content."""

from __future__ import annotations

import base64
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from atlas.agent.artifacts import LocalArtifactStore, SandboxDenied
from atlas.agent.contracts import ModelImage

# Formats OpenRouter vision models accept as inline images.
_MEDIA_TYPES = {
    ".gif": "image/gif",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}

# Inline vision requests cap the payload. Refuse anything larger before reading it.
MAX_MODEL_IMAGE_BYTES = 10_000_000


def is_model_image(relative_path: str) -> bool:
    """Return whether ``relative_path`` has a suffix the model can see."""
    return Path(relative_path).suffix.lower() in _MEDIA_TYPES


def ingest_image(store: LocalArtifactStore, relative_path: str) -> ModelImage:
    """Read one image artifact into a model content part.

    The path is resolved only through ``store``, so it cannot leave the
    artifact root. The size is checked on disk before the file is read.

    Args:
        store: Sandbox that owns the artifact.
        relative_path: Image path relative to ``store``.

    Returns:
        The file bytes as base64, with the workspace-relative path preserved.

    Raises:
        ValueError: The suffix is unsupported, or Pillow cannot read the file.
        FileNotFoundError: The path is not a file in the artifact root.
        SandboxDenied: The path escapes the root, is a hardlink, or exceeds
            ``MAX_MODEL_IMAGE_BYTES``.
    """
    suffix = Path(relative_path).suffix.lower()
    media_type = _MEDIA_TYPES.get(suffix)
    if media_type is None:
        supported = ", ".join(sorted(_MEDIA_TYPES))
        label = suffix or "(none)"
        raise ValueError(f"Unsupported image type {label}. Supported types: {supported}")
    path = store.resolve(relative_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image does not exist: {relative_path}")
    if path.stat().st_size > MAX_MODEL_IMAGE_BYTES:
        raise SandboxDenied(
            f"Image {relative_path} exceeds the {MAX_MODEL_IMAGE_BYTES} byte read limit"
        )
    data = path.read_bytes()
    if len(data) > MAX_MODEL_IMAGE_BYTES:
        raise SandboxDenied(
            f"Image {relative_path} exceeds the {MAX_MODEL_IMAGE_BYTES} byte read limit"
        )
    try:
        with Image.open(path) as image:
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise ValueError(f"Image {relative_path} is not a readable image") from exc
    return ModelImage(
        path=store.relative(path),
        media_type=media_type,
        data=base64.b64encode(data).decode("ascii"),
    )


def ingest_images(
    store: LocalArtifactStore, relative_paths: list[str] | tuple[str, ...]
) -> list[ModelImage]:
    """Read each path in order. The first failure aborts the rest."""
    return [ingest_image(store, path) for path in relative_paths]
