from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from pydantic import BaseModel, Field

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import Tool, ToolResult
from atlas.models.base import PluginSpec, load_rgb_array


class MaskInput(BaseModel):
    """Workspace-relative RGB tile in; same-size binary mask out."""

    path: str = Field(description="PNG tile path relative to the local artifact workspace.")
    mask_path: str | None = Field(
        default=None,
        description="Mask PNG path relative to the workspace (0 background, 1 positive).",
    )
    json_path: str | None = Field(
        default=None,
        description="JSON summary path relative to the workspace.",
    )


def cloud_mask(rgb: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Label bright, low-chroma pixels as cloud.

    Args:
        rgb: ``HxWx3`` uint8 image.

    Returns:
        ``HxW`` uint8 mask, ``1`` where the pixel is cloud.
    """
    channels = rgb.astype(np.int16)
    luma = channels.sum(axis=2) // 3
    chroma = channels.max(axis=2) - channels.min(axis=2)
    return ((luma >= 200) & (chroma <= 25)).astype(np.uint8)


def burn_mask(rgb: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Label dark red-brown pixels as burn scar.

    Red must be strictly greater than green and blue, and luma must be at least
    25 and below 80. Black and dark grey fail that test.

    Args:
        rgb: ``HxWx3`` uint8 image.

    Returns:
        ``HxW`` uint8 mask, ``1`` where the pixel is burned.
    """
    red = rgb[..., 0].astype(np.int32)
    green = rgb[..., 1].astype(np.int32)
    blue = rgb[..., 2].astype(np.int32)
    luma = (299 * red + 587 * green + 114 * blue) // 1000
    # Red must strictly dominate, and luma has a floor, so black and dark grey
    # (equal channels, or too dark to be a scar) stay unburned.
    burned = (red > green) & (red > blue) & (luma >= 25) & (luma < 80) & (green < 90)
    return burned.astype(np.uint8)


def water_mask(rgb: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Label blue-dominant pixels as open water.

    Args:
        rgb: ``HxWx3`` uint8 image.

    Returns:
        ``HxW`` uint8 mask, ``1`` where the pixel is water.
    """
    red = rgb[..., 0].astype(np.int16)
    green = rgb[..., 1].astype(np.int16)
    blue = rgb[..., 2].astype(np.int16)
    water = (blue > red + 30) & (blue > green + 20) & (blue >= 80)
    return water.astype(np.uint8)


_RULES: dict[str, Callable[[NDArray[np.uint8]], NDArray[np.uint8]]] = {
    "cloud": cloud_mask,
    "burn": burn_mask,
    "water": water_mask,
}


def apply_tiled_mask(
    rgb: NDArray[np.uint8],
    rule: Callable[[NDArray[np.uint8]], NDArray[np.uint8]],
    tile: int,
) -> NDArray[np.uint8]:
    """Apply ``rule`` on ``tile`` windows and stitch a native-size mask.

    Args:
        rgb: ``HxWx3`` uint8 mosaic.
        rule: Maps one tile to a 0/1 mask of the same spatial size.
        tile: Window edge in pixels.

    Returns:
        ``HxW`` uint8 mask. The output is not resized to the tile.
    """
    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    mask = np.zeros((height, width), dtype=np.uint8)
    for y0 in range(0, height, tile):
        for x0 in range(0, width, tile):
            y1 = min(y0 + tile, height)
            x1 = min(x0 + tile, width)
            mask[y0:y1, x0:x1] = rule(rgb[y0:y1, x0:x1])
    return mask


class BinaryMaskTool(Tool):
    """Heuristic 0/1 mask. Large mosaics are tiled and stitched, not rejected."""

    def __init__(self, spec: PluginSpec) -> None:
        if spec.runtime != "binary_mask":
            raise ValueError(f"Unsupported model runtime {spec.runtime!r} for {spec.name}")
        rule = _RULES.get(spec.rule)
        if rule is None:
            raise ValueError(f"Unknown binary mask rule {spec.rule!r} for {spec.name}")
        if len(spec.output.classes) != 2:
            raise ValueError(f"{spec.name} must declare exactly two output classes")
        self.spec = spec
        self.name = spec.name
        self.description = spec.description
        self._rule = rule
        self._tile = max(spec.input.sizes) if spec.input.sizes else 512

    @property
    def input_model(self) -> type[BaseModel]:
        return MaskInput

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        request = MaskInput.model_validate(arguments)
        image_path = store.resolve_read(request.path)
        mask_rel = request.mask_path or f"artifacts/{self.name}_mask.png"
        json_rel = request.json_path or f"artifacts/{self.name}.json"
        mask_path = store.resolve(mask_rel)
        json_path = store.resolve(json_rel)
        if mask_path.suffix.lower() != ".png":
            raise ValueError("mask output must end in .png")
        if json_path.suffix.lower() != ".json":
            raise ValueError("json output must end in .json")

        rgb = load_rgb_array(image_path)
        height, width = int(rgb.shape[0]), int(rgb.shape[1])
        mask = apply_tiled_mask(rgb, self._rule, self._tile)
        if mask.shape != (height, width):
            raise RuntimeError("Mask spatial size must match the input")

        mask_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask, mode="L").save(mask_path)
        positive = self.spec.output.classes[1]
        fraction = float(mask.sum()) / float(mask.size)
        payload = {
            "path": request.path,
            "width": width,
            "height": height,
            "positive": positive,
            "positive_fraction": round(fraction, 6),
            "mask": store.relative(mask_path),
        }
        json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return ToolResult(
            text=f"{self.name}: {width}x{height} ({positive}={fraction:.2f})",
            artifacts=[store.relative(mask_path), store.relative(json_path)],
        )
