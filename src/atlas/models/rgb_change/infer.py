from __future__ import annotations

import json
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field
from scipy import ndimage

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import Tool, ToolResult
from atlas.models import base as model_base
from atlas.models.base import PluginSpec, load_rgb_array

CHANGE_THRESHOLD = 0.2
MIN_BLOB_PIXELS = 16


def _reject_oversized(height: int, width: int) -> None:
    """Refuse a pair whose raster exceeds ``_MAX_PIXELS``."""
    if height * width > model_base._MAX_PIXELS:
        raise ValueError(f"Image exceeds {model_base._MAX_PIXELS} pixels ({width}x{height})")


class RgbChangeInput(BaseModel):
    """Two workspace-relative RGB PNGs of the same size."""

    before: str = Field(description="Earlier RGB PNG, relative to the local artifact workspace.")
    after: str = Field(description="Later RGB PNG, relative to the local artifact workspace.")
    heatmap_path: str | None = Field(
        default=None,
        description="Float32 heatmap .npy path relative to the workspace.",
    )
    json_path: str | None = Field(
        default=None,
        description="JSON polygon summary path relative to the workspace.",
    )


def rgb_heatmap(before: NDArray[np.uint8], after: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Mean absolute channel delta, scaled to ``[0, 1]``.

    Args:
        before: Earlier ``HxWx3`` uint8 image.
        after: Later image. Must match ``before.shape``.

    Returns:
        ``HxW`` float32 heatmap.

    Raises:
        ValueError: If the two images differ in shape.
    """
    if before.shape != after.shape:
        raise ValueError(
            "RGB pair must share HxW, got "
            f"{before.shape[1]}x{before.shape[0]} and {after.shape[1]}x{after.shape[0]}"
        )
    _reject_oversized(int(before.shape[0]), int(before.shape[1]))
    delta = np.abs(after.astype(np.float32) - before.astype(np.float32))
    return (delta.mean(axis=2) / np.float32(255.0)).astype(np.float32)


def _convex_hull(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    unique = sorted(set(points))
    if len(unique) < 3:
        return []

    def cross(origin: tuple[int, int], left: tuple[int, int], right: tuple[int, int]) -> int:
        return (left[0] - origin[0]) * (right[1] - origin[1]) - (left[1] - origin[1]) * (
            right[0] - origin[0]
        )

    lower: list[tuple[int, int]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[int, int]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return []
    return hull


def blob_polygons(
    heatmap: NDArray[np.float32],
    *,
    threshold: float = CHANGE_THRESHOLD,
    min_pixels: int = MIN_BLOB_PIXELS,
) -> list[dict[str, Any]]:
    """Convex hulls of 4-connected blobs at or above ``threshold``.

    Args:
        heatmap: ``HxW`` float array in ``[0, 1]``.
        threshold: Minimum heatmap value that counts as change.
        min_pixels: Blobs smaller than this are dropped.

    Returns:
        Polygons sorted by area, each with ``area`` and ``points`` as ``[x, y]``.
    """
    _reject_oversized(int(heatmap.shape[0]), int(heatmap.shape[1]))
    mask = heatmap >= threshold
    structure = ndimage.generate_binary_structure(2, 1)
    labeled, _count = ndimage.label(np.asarray(mask, dtype=bool), structure=structure)
    polygons: list[dict[str, Any]] = []
    for index, window in enumerate(ndimage.find_objects(labeled), start=1):
        if window is None:
            continue
        rows, cols = np.nonzero(labeled[window] == index)
        if int(rows.size) < min_pixels:
            continue
        y0 = int(window[0].start or 0)
        x0 = int(window[1].start or 0)
        pixels = [
            (x0 + int(x), y0 + int(y)) for y, x in zip(rows.tolist(), cols.tolist(), strict=True)
        ]
        hull = _convex_hull(pixels)
        if len(hull) < 3:
            continue
        polygons.append({"area": len(pixels), "points": [[px, py] for px, py in hull]})
    polygons.sort(
        key=lambda item: (
            -int(item["area"]),
            int(item["points"][0][0]),
            int(item["points"][0][1]),
        )
    )
    return polygons


class RgbChangeTool(Tool):
    """Absdiff heatmap and blob polygons for an RGB pair. Does not resize."""

    def __init__(self, spec: PluginSpec) -> None:
        if spec.runtime != "rgb_delta":
            raise ValueError(f"Unsupported model runtime {spec.runtime!r} for {spec.name}")
        self.spec = spec
        self.name = spec.name
        self.description = spec.description

    @property
    def input_model(self) -> type[BaseModel]:
        return RgbChangeInput

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        request = RgbChangeInput.model_validate(arguments)
        before_path = store.resolve_read(request.before)
        after_path = store.resolve_read(request.after)
        heatmap_rel = request.heatmap_path or f"artifacts/{self.name}_heatmap.npy"
        json_rel = request.json_path or f"artifacts/{self.name}.json"
        heatmap_path = store.resolve(heatmap_rel)
        json_path = store.resolve(json_rel)
        if heatmap_path.suffix.lower() != ".npy":
            raise ValueError("heatmap output must end in .npy")
        if json_path.suffix.lower() != ".json":
            raise ValueError("json output must end in .json")

        before = load_rgb_array(before_path)
        after = load_rgb_array(after_path)
        heatmap = rgb_heatmap(before, after)
        height, width = int(heatmap.shape[0]), int(heatmap.shape[1])
        polygons = blob_polygons(heatmap)
        changed = float((heatmap >= CHANGE_THRESHOLD).mean())

        heatmap_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(heatmap_path, heatmap)
        payload = {
            "before": request.before,
            "after": request.after,
            "width": width,
            "height": height,
            "threshold": CHANGE_THRESHOLD,
            "min_pixels": MIN_BLOB_PIXELS,
            "changed_fraction": round(changed, 6),
            "heatmap": store.relative(heatmap_path),
            "polygons": polygons,
        }
        json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return ToolResult(
            text=(
                f"{self.name}: {width}x{height} (changed={changed:.2f}, polygons={len(polygons)})"
            ),
            artifacts=[store.relative(heatmap_path), store.relative(json_path)],
        )
