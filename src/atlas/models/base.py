"""Shared plugin spec, weight resolution, and centroid inference helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tomllib
from pathlib import Path
from typing import Any

from PIL import Image
from pydantic import BaseModel, Field

_WEIGHTS_ENV = "ATLAS_WEIGHTS_DIR"


class PluginInput(BaseModel):
    """Declared tensor / image contract for one plugin."""

    kind: str
    shape: list[int]
    dtype: str


class PluginOutput(BaseModel):
    """Declared output contract for one plugin."""

    kind: str
    classes: list[str] = Field(default_factory=list)


class PluginSpec(BaseModel):
    """Manifest loaded from a model package ``plugin.toml``."""

    name: str
    description: str
    runtime: str
    weight: str
    sha256: str = ""
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
            f"Weight {spec.weight} is missing. Run acquire.sh, train.py, and "
            f"export.py or set {_WEIGHTS_ENV}."
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


def mean_rgb(image: Image.Image, size: int = 224) -> tuple[float, float, float]:
    """Resize to ``size`` then reduce to a single mean RGB in 0-255."""
    rgb = image.convert("RGB").resize((size, size), Image.Resampling.BOX)
    pixel = rgb.resize((1, 1), Image.Resampling.BOX).getpixel((0, 0))
    if not isinstance(pixel, tuple) or len(pixel) < 3:
        raise ValueError("Expected an RGB pixel from the reduced image")
    return (float(pixel[0]), float(pixel[1]), float(pixel[2]))


def softmax_neg_l2(
    mean: tuple[float, float, float],
    centroids: dict[str, tuple[float, float, float]],
    classes: list[str],
) -> dict[str, float]:
    """Class scores from softmax of negative L2 distance to each centroid."""
    logits: list[float] = []
    for name in classes:
        centroid = centroids[name]
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(mean, centroid, strict=True)))
        logits.append(-dist)
    peak = max(logits)
    exps = [math.exp(value - peak) for value in logits]
    total = sum(exps)
    return {name: exp / total for name, exp in zip(classes, exps, strict=True)}
