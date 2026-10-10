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

    Args:
        rgb: ``HxWx3`` uint8 image.

    Returns:
        ``HxW`` uint8 mask, ``1`` where the pixel is burned.
    """
    red = rgb[..., 0].astype(np.int32)
    green = rgb[..., 1].astype(np.int32)
    blue = rgb[..., 2].astype(np.int32)
    luma = (299 * red + 587 * green + 114 * blue) // 1000
    burned = (luma < 80) & (red >= green) & (red >= blue) & (green < 90)
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


def _require_tile(width: int, height: int, sizes: list[int]) -> None:
    if width == height and width in sizes:
        return
    allowed = ", ".join(f"{size}x{size}" for size in sizes)
    raise ValueError(f"Expected a tile of {allowed}, got {width}x{height}")


class BinaryMaskTool(Tool):
    """Per-pixel 0/1 mask on a 256 or 512 RGB tile. Does not resize the input."""

    def __init__(self, spec: PluginSpec) -> None:
        if spec.runtime != "binary_mask":
            raise ValueError(f"Unsupported model runtime {spec.runtime!r} for {spec.name}")
        rule = _RULES.get(spec.rule)
        if rule is None:
            raise ValueError(f"Unknown binary mask rule {spec.rule!r} for {spec.name}")
        if len(spec.output.classes) != 2:
            raise ValueError(f"{spec.name} must declare exactly two output classes")
        if not spec.input.sizes:
            raise ValueError(f"{spec.name} must declare input.sizes")
        self.spec = spec
        self.name = spec.name
        self.description = spec.description
        self._rule = rule

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
        _require_tile(width, height, self.spec.input.sizes)
        mask = self._rule(rgb)
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
