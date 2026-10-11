from __future__ import annotations

import json
from typing import Any

import lightgbm as lgb
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from pydantic import BaseModel, Field

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import Tool, ToolResult
from atlas.models.base import PluginSpec, load_rgb_array, load_weight_payload, weight_tile

_FEATURES = ["r", "g", "b"]


class SegmentInput(BaseModel):
    """Workspace-relative RGB PNG in; same-size class mask out."""

    path: str = Field(description="PNG path relative to the local artifact workspace.")
    mask_path: str | None = Field(
        default=None,
        description="Class-index PNG path relative to the workspace.",
    )
    json_path: str | None = Field(
        default=None,
        description="JSON summary path relative to the workspace.",
    )


class LightGBMWeights:
    """A LightGBM booster plus the tile edge from the weight contract."""

    def __init__(self, spec: PluginSpec, payload: dict[str, Any]) -> None:
        if payload.get("features") != _FEATURES:
            raise ValueError(f"Weight {spec.weight} features must be {_FEATURES}")
        model = payload.get("model")
        if not isinstance(model, str) or not model.strip():
            raise ValueError(f"Weight {spec.weight} must contain a LightGBM model string")
        self.tile = weight_tile(payload, spec)
        self.booster = lgb.Booster(model_str=model)


def predict_lightgbm(rgb: NDArray[np.uint8], weights: LightGBMWeights) -> NDArray[np.uint8]:
    """Tile ``rgb``, score RGB pixels, and stitch a 0/1 mask.

    Args:
        rgb: ``HxWx3`` uint8 mosaic.
        weights: Booster loaded from the plugin weight file.

    Returns:
        ``HxW`` uint8 mask. ``1`` is the positive class (probability >= 0.5).
    """
    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    labels = np.zeros((height, width), dtype=np.uint8)
    tile = weights.tile
    for y0 in range(0, height, tile):
        for x0 in range(0, width, tile):
            patch = rgb[y0 : y0 + tile, x0 : x0 + tile]
            features = patch.reshape(-1, 3).astype(np.float64)
            scores = np.asarray(weights.booster.predict(features), dtype=np.float64)
            patch_h, patch_w = int(patch.shape[0]), int(patch.shape[1])
            labels[y0 : y0 + patch_h, x0 : x0 + patch_w] = (scores >= 0.5).reshape(patch_h, patch_w)
    return labels


class LightGBMTool(Tool):
    """LightGBM pixel classifier. Large mosaics are tiled, then stitched."""

    def __init__(self, spec: PluginSpec) -> None:
        if spec.runtime != "lightgbm":
            raise ValueError(f"Unsupported model runtime {spec.runtime!r} for {spec.name}")
        if len(spec.output.classes) != 2:
            raise ValueError(f"{spec.name} must declare exactly two output classes")
        if not spec.weight:
            raise ValueError(f"{spec.name} must declare a weight key")
        self.spec = spec
        self.name = spec.name
        self.description = spec.description

    @property
    def input_model(self) -> type[BaseModel]:
        return SegmentInput

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        request = SegmentInput.model_validate(arguments)
        mask_rel = request.mask_path or f"artifacts/{self.name}_mask.png"
        json_rel = request.json_path or f"artifacts/{self.name}.json"
        mask_path = store.resolve(mask_rel)
        json_path = store.resolve(json_rel)
        if mask_path.suffix.lower() != ".png":
            raise ValueError("mask output must end in .png")
        if json_path.suffix.lower() != ".json":
            raise ValueError("json output must end in .json")

        weights = LightGBMWeights(self.spec, load_weight_payload(self.spec))
        rgb = load_rgb_array(store.resolve_read(request.path))
        labels = predict_lightgbm(rgb, weights)
        height, width = int(labels.shape[0]), int(labels.shape[1])
        positive = self.spec.output.classes[1]
        fraction = float(labels.sum()) / float(labels.size)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(labels, mode="L").save(mask_path)
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
