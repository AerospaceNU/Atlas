from __future__ import annotations

import hashlib
import json
import os
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from pydantic import BaseModel, Field

_MAX_PIXELS = 8_000_000

_WEIGHTS_ENV = "ATLAS_WEIGHTS_DIR"


class PluginInput(BaseModel):
    """Declared tensor / image contract for one plugin."""

    kind: str
    shape: list[int]
    dtype: str
    sizes: list[int] = Field(
        default_factory=list,
        description="Allowed square tile sizes. Empty means any spatial size.",
    )


class PluginOutput(BaseModel):
    """Declared output contract for one plugin."""

    kind: str
    classes: list[str] = Field(default_factory=list)


class PluginSpec(BaseModel):
    """Manifest loaded from a model package ``plugin.toml``."""

    name: str
    description: str
    runtime: str
    weight: str = ""
    sha256: str = ""
    rule: str = ""
    input: PluginInput
    output: PluginOutput


def repo_root() -> Path | None:
    """Return the checkout root if a ``pyproject.toml`` is found, else None."""
    seen: set[Path] = set()
    for start in (Path(__file__).resolve().parent, Path.cwd()):
        for candidate in (start, *start.parents):
            if candidate in seen:
                continue
            seen.add(candidate)
            if (candidate / "pyproject.toml").is_file():
                return candidate
    return None


def weights_root() -> Path:
    """Directory that holds exported weights (not the artifact sandbox).

    ``ATLAS_WEIGHTS_DIR`` wins. Otherwise ``<repo>/weights``, then
    ``~/.atlas/weights`` when no checkout is visible.
    """
    env = os.environ.get(_WEIGHTS_ENV)
    if env:
        return Path(env).expanduser().resolve()
    root = repo_root()
    if root is not None:
        return (root / "weights").resolve()
    return (Path.home() / ".atlas" / "weights").resolve()


def weight_path(spec: PluginSpec) -> Path:
    """Resolve a plugin's relative weight key under ``weights_root()``."""
    return (weights_root() / spec.weight).resolve()


def parse_plugin_toml(text: str) -> PluginSpec:
    """Parse a ``plugin.toml`` body into a spec."""
    return PluginSpec.model_validate(tomllib.loads(text))


def read_weight_bytes(spec: PluginSpec) -> bytes:
    """Read weight bytes; fail closed on missing file or sha256 mismatch.

    Error messages use the repo-relative weight key, never a host path.
    """
    path = weight_path(spec)
    if not path.is_file():
        raise FileNotFoundError(
            f"Weight {spec.weight} is missing. Run train.py and export.py or set {_WEIGHTS_ENV}."
        )
    data = path.read_bytes()
    expected = spec.sha256.strip().lower()
    if expected:
        digest = hashlib.sha256(data).hexdigest()
        if digest != expected:
            raise ValueError(f"Weight {spec.weight} sha256 mismatch")
    return data


def load_centroids(spec: PluginSpec) -> dict[str, tuple[float, float, float]]:
    """Load per-class RGB centroids from the exported weight JSON."""
    payload: dict[str, Any] = json.loads(read_weight_bytes(spec).decode("utf-8"))
    classes = payload.get("classes")
    if classes != spec.output.classes:
        raise ValueError("Weight classes do not match plugin.toml")
    raw_centroids = payload.get("centroids")
    if not isinstance(raw_centroids, dict):
        raise ValueError("Weight JSON must contain a centroids object")
    centroids: dict[str, tuple[float, float, float]] = {}
    for name in spec.output.classes:
        rgb = raw_centroids[name]
        if not isinstance(rgb, list) or len(rgb) != 3:
            raise ValueError(f"Centroid {name} must be an RGB triple")
        centroids[name] = (float(rgb[0]), float(rgb[1]), float(rgb[2]))
    return centroids


def require_max_pixels(height: int, width: int) -> None:
    """Reject rasters above ``_MAX_PIXELS``.

    Args:
        height: Raster height.
        width: Raster width.

    Raises:
        ValueError: If ``height * width`` exceeds ``_MAX_PIXELS``.
    """
    if height * width > _MAX_PIXELS:
        raise ValueError(f"Image exceeds {_MAX_PIXELS} pixels ({width}x{height})")


def load_rgb_array(path: Path) -> NDArray[np.uint8]:
    """Read ``path`` as an ``HxWx3`` uint8 RGB array.

    Args:
        path: Image file to open. Any mode Pillow can convert is accepted.

    Returns:
        A copy of the image in RGB, at its native spatial size.

    Raises:
        ValueError: If the converted image is not three-channel or exceeds
            ``_MAX_PIXELS``.
    """
    with Image.open(path) as image:
        width, height = image.size
        require_max_pixels(height, width)
        rgb = image.convert("RGB")
        array = np.array(rgb, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Expected an RGB image")
    return array


def load_weight_payload(spec: PluginSpec) -> dict[str, Any]:
    """Load a JSON weight file and check its runtime and class names.

    Args:
        spec: Plugin whose relative ``weight`` key and ``output.classes`` form
            the contract. ``sha256`` is enforced by :func:`read_weight_bytes`.

    Returns:
        The JSON object stored at that key.

    Raises:
        FileNotFoundError: If the weight file is missing. The message names the
            relative key, not a host path.
        ValueError: If the hash, runtime, or class list does not match.
    """
    payload: Any = json.loads(read_weight_bytes(spec).decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Weight {spec.weight} must be a JSON object")
    if payload.get("runtime") != spec.runtime:
        raise ValueError(f"Weight runtime does not match plugin.toml for {spec.name}")
    if payload.get("classes") != spec.output.classes:
        raise ValueError("Weight classes do not match plugin.toml")
    return payload


def weight_tile(payload: dict[str, Any], spec: PluginSpec) -> int:
    """Return the mosaic tile edge declared by a weight file.

    Args:
        payload: Object returned by :func:`load_weight_payload`.
        spec: Plugin whose ``input.sizes`` lists allowed tile edges.

    Returns:
        The tile edge in pixels.

    Raises:
        ValueError: If ``tile`` is missing, not an int, or not allowed.
    """
    tile = payload.get("tile")
    if isinstance(tile, bool) or not isinstance(tile, int):
        raise ValueError(f"Weight {spec.weight} tile must be an int")
    if spec.input.sizes and tile not in spec.input.sizes:
        raise ValueError(f"Weight tile {tile} is not in plugin input.sizes")
    if tile < 2:
        raise ValueError(f"Weight tile {tile} must be >= 2")
    return tile


def mean_rgb(image: Image.Image) -> tuple[float, float, float]:
    """Mean RGB of the image at native resolution (no resize)."""
    arr = np.asarray(image.convert("RGB"), dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("Expected an RGB image")
    channel_means = arr.reshape(-1, 3).mean(axis=0)
    return (float(channel_means[0]), float(channel_means[1]), float(channel_means[2]))


def label_pixels(
    image: Image.Image,
    centroids: dict[str, tuple[float, float, float]],
    classes: list[str],
) -> NDArray[np.uint8]:
    """Per-pixel nearest RGB centroid at native HxW. Never resizes the scene.

    Large rasters are processed in row strips so memory stays bounded; the
    result is still one label per original pixel.
    """
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("Expected an RGB image")
    cents = np.asarray([centroids[name] for name in classes], dtype=np.float32)
    height, width, _ = arr.shape
    if height * width <= _MAX_PIXELS:
        return _nearest_centroid(arr, cents)
    labels = np.empty((height, width), dtype=np.uint8)
    step = max(1, _MAX_PIXELS // max(width, 1))
    for row in range(0, height, step):
        labels[row : row + step] = _nearest_centroid(arr[row : row + step], cents)
    return labels


def _nearest_centroid(arr: NDArray[np.float32], cents: NDArray[np.float32]) -> NDArray[np.uint8]:
    delta = arr[:, :, None, :] - cents[None, None, :, :]
    return np.square(delta).sum(axis=-1).argmin(axis=-1).astype(np.uint8)


def colorize_labels(
    labels: NDArray[np.uint8],
    colors: list[tuple[int, int, int]],
) -> Image.Image:
    """RGB visualization of a label map using a per-class palette."""
    palette = np.asarray(colors, dtype=np.uint8)
    return Image.fromarray(palette[labels], mode="RGB")


def labels_png(
    labels: NDArray[np.uint8],
    colors: list[tuple[int, int, int]],
) -> Image.Image:
    """Palette PNG, one byte per pixel, same HxW as the input."""
    image = Image.fromarray(labels, mode="P")
    table: list[int] = []
    for color in colors:
        table.extend(color)
    table.extend([0, 0, 0] * (256 - len(colors)))
    image.putpalette(table)
    return image


def class_fractions(
    labels: NDArray[np.uint8],
    classes: list[str],
) -> dict[str, float]:
    """Fraction of pixels in each class, in class-list order."""
    total = labels.size
    counts = np.bincount(labels.ravel(), minlength=len(classes))
    return {name: float(counts[i]) / float(total) for i, name in enumerate(classes)}
