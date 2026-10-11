"""Read sandbox image artifacts into vision-model content."""

from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from atlas.agent.artifacts import LocalArtifactStore, SandboxDenied
from atlas.agent.contracts import ChatMessage, ModelImage, Role

# Pillow format names the vision model can accept. The suffix is not consulted.
_FORMAT_MEDIA_TYPES = {
    "GIF": "image/gif",
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}

# Tool results with these suffixes are image artifacts. The media type still
# comes from Pillow, so a mislabeled file is not sent as the wrong type.
_IMAGE_SUFFIXES = frozenset({".gif", ".jpeg", ".jpg", ".png", ".webp"})

# Each encoded image stays at or under this size. Larger images are downscaled.
MAX_IMAGE_BYTES = 5 * 1024 * 1024

# Newest images are sent until this request budget is full. Older images are
# dropped instead of being attached again on the next request.
MAX_REQUEST_IMAGE_BYTES = 15 * 1024 * 1024

# Files above this are refused before they are read. Images between
# ``MAX_IMAGE_BYTES`` and this limit are opened and downscaled.
MAX_SOURCE_BYTES = 32 * 1024 * 1024


def is_model_image(relative_path: str) -> bool:
    """Return whether a tool artifact should be ingested as an image."""
    return Path(relative_path).suffix.lower() in _IMAGE_SUFFIXES


def ingest_image(store: LocalArtifactStore, relative_path: str) -> ModelImage:
    """Read one image into a model content part.

    The path is resolved with :meth:`~atlas.agent.artifacts.LocalArtifactStore.resolve_read`,
    so a file in the write sandbox wins and a file in the project workspace is
    still visible. The stored path is relative to whichever root held the file.

    Args:
        store: Sandbox that owns the artifact and the optional read root.
        relative_path: Image path relative to the write sandbox or the workspace.

    Returns:
        Base64 image bytes, at most ``MAX_IMAGE_BYTES``, with a portable path.

    Raises:
        ValueError: Pillow cannot read the file, or its format is unsupported.
        FileNotFoundError: The path is not a file in either root.
        SandboxDenied: The path escapes a root, is a hardlink, exceeds
            ``MAX_SOURCE_BYTES``, or cannot be downscaled to ``MAX_IMAGE_BYTES``.
    """
    path = store.resolve_read(relative_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image does not exist: {relative_path}")
    if path.stat().st_size > MAX_SOURCE_BYTES:
        raise SandboxDenied(f"Image {relative_path} exceeds the {MAX_SOURCE_BYTES} byte read limit")
    try:
        with Image.open(path) as image:
            try:
                image.load()
            except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
                raise ValueError(f"Image {relative_path} is not a readable image") from exc
            media_type = _media_type(image.format, relative_path)
            encoded = _within_cap(path, image, image.format or "", relative_path)
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
    """Drop older images until the newest ones fit in one request.

    Images are considered from the newest message backward. An image that
    does not fit replaces nothing newer: it and every older image are removed.
    Tool messages never carry image parts. The messages are updated in place
    so a later request does not send the evicted bytes again.
    """
    remaining = MAX_REQUEST_IMAGE_BYTES
    chosen: dict[int, list[ModelImage]] = {}
    stop = False
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.role is Role.TOOL:
            continue
        picked: list[ModelImage] = []
        for image in reversed(message.images):
            size = _encoded_size(image.data)
            if size > remaining:
                stop = True
                break
            picked.append(image)
            remaining -= size
        chosen[index] = list(reversed(picked))
        if stop:
            break
    for index, message in enumerate(messages):
        if index in chosen:
            message.images = chosen[index]
        elif message.images:
            message.images = []


def _media_type(fmt: str | None, relative_path: str) -> str:
    media_type = _FORMAT_MEDIA_TYPES.get(fmt or "")
    if media_type is not None:
        return media_type
    supported = ", ".join(sorted(set(_FORMAT_MEDIA_TYPES.values())))
    label = fmt or "unknown"
    raise ValueError(
        f"Unsupported image format {label} for {relative_path}. Supported formats: {supported}"
    )


def _within_cap(path: Path, image: Image.Image, fmt: str, relative_path: str) -> bytes:
    """Return the original bytes when they already fit, otherwise a smaller encoding."""
    if path.stat().st_size <= MAX_IMAGE_BYTES:
        data = path.read_bytes()
        if len(data) <= MAX_IMAGE_BYTES:
            return data
    return _shrink(image, fmt, relative_path)


def _shrink(image: Image.Image, fmt: str, relative_path: str) -> bytes:
    current = image
    while True:
        encoded = _save_bytes(current, fmt)
        if len(encoded) <= MAX_IMAGE_BYTES:
            return encoded
        width, height = current.size
        if width <= 1 or height <= 1:
            raise SandboxDenied(f"Image {relative_path} exceeds the {MAX_IMAGE_BYTES} byte limit")
        resized = (max(1, width // 2), max(1, height // 2))
        if resized == (width, height):
            raise SandboxDenied(f"Image {relative_path} exceeds the {MAX_IMAGE_BYTES} byte limit")
        current = current.resize(resized, Image.Resampling.LANCZOS)


def _save_bytes(image: Image.Image, fmt: str) -> bytes:
    saving = image
    if fmt == "JPEG" and image.mode not in ("L", "RGB"):
        saving = image.convert("RGB")
    buffer = io.BytesIO()
    saving.save(buffer, format=fmt)
    return buffer.getvalue()


def _encoded_size(data: str) -> int:
    return len(base64.b64decode(data))


def _portable_path(store: LocalArtifactStore, resolved: Path) -> str:
    """Return a root-relative path, never an absolute host path."""
    candidate = resolved.resolve()
    bases = [store.root]
    if store.read_root is not None:
        bases.append(store.read_root)
    for base in bases:
        try:
            return candidate.relative_to(base.resolve()).as_posix()
        except ValueError:
            continue
    raise SandboxDenied("Path is outside the artifact root")
