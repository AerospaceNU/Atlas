"""Read image artifacts into vision-model content."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import TYPE_CHECKING

from atlas.agent.artifacts import ArtifactError, LocalArtifactStore
from atlas.agent.contracts import ChatMessage, ModelImage, Role

if TYPE_CHECKING:
    from PIL.Image import Image as PilImage

_FORMAT_MEDIA_TYPES = {
    "GIF": "image/gif",
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}
_IMAGE_SUFFIXES = frozenset({".gif", ".jpeg", ".jpg", ".png", ".webp"})
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_REQUEST_IMAGE_BYTES = 15 * 1024 * 1024
MAX_SOURCE_BYTES = 32 * 1024 * 1024


def _b64_size(n: int) -> int:
    return ((n + 2) // 3) * 4


def is_model_image(relative_path: str) -> bool:
    """Return whether a tool artifact should be ingested as an image."""
    return Path(relative_path).suffix.lower() in _IMAGE_SUFFIXES


def ingest_image(store: LocalArtifactStore, relative_path: str) -> ModelImage:
    """Read and downscale one image into a base64 ``ModelImage`` under ``MAX_IMAGE_BYTES``."""
    from PIL import Image, UnidentifiedImageError

    path = store.resolve_read(relative_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image does not exist: {relative_path}")
    size = path.stat().st_size
    if size > MAX_SOURCE_BYTES:
        raise ArtifactError(f"Image {relative_path} exceeds the {MAX_SOURCE_BYTES} byte read limit")
    try:
        with Image.open(path) as image:
            try:
                image.load()
            except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
                raise ValueError(f"Image {relative_path} is not a readable image") from exc
            fmt = image.format or ""
            media_type = _FORMAT_MEDIA_TYPES.get(fmt)
            if media_type is None:
                supported = ", ".join(sorted(set(_FORMAT_MEDIA_TYPES.values())))
                raise ValueError(
                    f"Unsupported image format {fmt or 'unknown'} for {relative_path}. "
                    f"Supported formats: {supported}"
                )
            encoded = (
                path.read_bytes()
                if _b64_size(size) <= MAX_IMAGE_BYTES
                else _shrink(image, fmt, relative_path)
            )
    except Image.DecompressionBombError as exc:
        raise ValueError(f"Image {relative_path} was rejected as a decompression bomb") from exc
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ValueError(f"Image {relative_path} is not a readable image") from exc
    return ModelImage(
        path=_portable_path(store, path),
        media_type=media_type,
        data=base64.b64encode(encoded).decode("ascii"),
    )


def ingest_images(
    store: LocalArtifactStore, relative_paths: list[str] | tuple[str, ...]
) -> list[ModelImage]:
    """Read each path in order. The first failure aborts the rest."""
    return [ingest_image(store, path) for path in relative_paths]


def apply_image_budget(messages: list[ChatMessage]) -> None:
    """Drop older images in place until the newest ones fit in ``MAX_REQUEST_IMAGE_BYTES``."""
    remaining = MAX_REQUEST_IMAGE_BYTES
    chosen: dict[int, list[ModelImage]] = {}
    stop = False
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.role is Role.TOOL:
            continue
        picked: list[ModelImage] = []
        for image in reversed(message.images):
            size = len(image.data)
            if size > remaining:
                stop = True
                break
            picked.append(image)
            remaining -= size
        chosen[index] = list(reversed(picked))
        if stop:
            break
    for index, message in enumerate(messages):
        if message.role is Role.TOOL:
            if message.images:
                message.images = []
            continue
        kept = chosen.get(index)
        if kept is None:
            omitted = list(message.images)
            kept = []
        else:
            kept_ids = {id(image) for image in kept}
            omitted = [image for image in message.images if id(image) not in kept_ids]
        if omitted:
            note = "\n".join(f"[image {image.path} omitted]" for image in omitted)
            message.content = f"{message.content}\n{note}" if message.content else note
        message.images = kept


def _shrink(image: PilImage, fmt: str, relative_path: str) -> bytes:
    from PIL import Image

    current = image
    while True:
        saving = (
            current.convert("RGB")
            if fmt == "JPEG" and current.mode not in ("L", "RGB")
            else current
        )
        buf = io.BytesIO()
        saving.save(buf, format=fmt)
        encoded = buf.getvalue()
        if _b64_size(len(encoded)) <= MAX_IMAGE_BYTES:
            return encoded
        width, height = current.size
        if width <= 1 or height <= 1:
            raise ArtifactError(f"Image {relative_path} exceeds the {MAX_IMAGE_BYTES} byte limit")
        resized = (max(1, width // 2), max(1, height // 2))
        if resized == (width, height):
            raise ArtifactError(f"Image {relative_path} exceeds the {MAX_IMAGE_BYTES} byte limit")
        current = current.resize(resized, Image.Resampling.LANCZOS)


def _portable_path(store: LocalArtifactStore, resolved: Path) -> str:
    candidate = resolved.resolve()
    for base in (store.root, store.read_root):
        if base is None:
            continue
        try:
            return candidate.relative_to(base.resolve()).as_posix()
        except ValueError:
            continue
    return candidate.as_posix()
