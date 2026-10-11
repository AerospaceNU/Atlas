from __future__ import annotations

import json
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from pydantic import BaseModel, Field

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import Tool, ToolResult
from atlas.models.base import PluginSpec, load_rgb_array, load_weight_payload, weight_tile


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


class UnetWeights:
    """Conv kernels for the CPU U-Net. Shapes are part of the weight contract."""

    def __init__(self, spec: PluginSpec, payload: dict[str, Any]) -> None:
        classes = len(spec.output.classes)
        self.tile = weight_tile(payload, spec)
        if self.tile % 2:
            raise ValueError(f"Weight tile {self.tile} must be even")
        self.enc_w = _kernel(payload, "enc_w", cin=3)
        hidden = int(self.enc_w.shape[0])
        self.enc_b = _bias(payload, "enc_b", hidden)
        self.bn_w = _kernel(payload, "bn_w", cin=hidden, cout=hidden)
        self.bn_b = _bias(payload, "bn_b", hidden)
        self.dec_w = _kernel(payload, "dec_w", cin=hidden * 2, cout=hidden)
        self.dec_b = _bias(payload, "dec_b", hidden)
        self.head_w = _kernel(payload, "head_w", cin=hidden, cout=classes)
        self.head_b = _bias(payload, "head_b", classes)


def _kernel(
    payload: dict[str, Any],
    key: str,
    *,
    cin: int,
    cout: int | None = None,
) -> NDArray[np.float32]:
    array = np.asarray(payload.get(key), dtype=np.float32)
    if array.ndim != 4 or array.shape[1] != cin or array.shape[2:] != (3, 3):
        raise ValueError(f"Weight {key} must have shape (Cout, {cin}, 3, 3)")
    if cout is not None and array.shape[0] != cout:
        raise ValueError(f"Weight {key} must have {cout} output channels")
    return array


def _bias(payload: dict[str, Any], key: str, width: int) -> NDArray[np.float32]:
    array = np.asarray(payload.get(key), dtype=np.float32)
    if array.shape != (width,):
        raise ValueError(f"Weight {key} must have shape ({width},)")
    return array


def _conv2d(
    image: NDArray[np.float32],
    weight: NDArray[np.float32],
    bias: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Same-padded 3x3 conv. ``image`` is HxWxCin, ``weight`` is CoutxCinx3x3."""
    cout = int(weight.shape[0])
    cin = int(weight.shape[1])
    height, width, _ = image.shape
    padded = np.pad(image, ((1, 1), (1, 1), (0, 0)), mode="edge")
    acc = np.zeros((height, width, cout), dtype=np.float32)
    for out_c in range(cout):
        total = np.zeros((height, width), dtype=np.float32)
        for in_c in range(cin):
            kernel = weight[out_c, in_c]
            for ky in range(3):
                for kx in range(3):
                    total += padded[ky : ky + height, kx : kx + width, in_c] * kernel[ky, kx]
        acc[:, :, out_c] = total + bias[out_c]
    return acc


def _max_pool2(image: NDArray[np.float32]) -> NDArray[np.float32]:
    height, width, channels = image.shape
    pooled_h = height // 2
    pooled_w = width // 2
    cropped = image[: pooled_h * 2, : pooled_w * 2]
    folded = cropped.reshape(pooled_h, 2, pooled_w, 2, channels)
    return folded.max(axis=(1, 3))


def _upsample2(image: NDArray[np.float32], height: int, width: int) -> NDArray[np.float32]:
    up = np.repeat(np.repeat(image, 2, axis=0), 2, axis=1)
    return up[:height, :width]


def unet_forward(rgb: NDArray[np.uint8], weights: UnetWeights) -> NDArray[np.uint8]:
    """Run one even-sized tile and return a class-index map.

    Args:
        rgb: ``HxWx3`` uint8 tile. Both spatial sizes must be even.
        weights: Kernels loaded from the plugin weight file.

    Returns:
        ``HxW`` uint8 class indexes.
    """
    features = rgb.astype(np.float32)
    encoded = np.maximum(_conv2d(features, weights.enc_w, weights.enc_b), 0)
    pooled = _max_pool2(encoded)
    bottleneck = np.maximum(_conv2d(pooled, weights.bn_w, weights.bn_b), 0)
    restored = _upsample2(bottleneck, int(encoded.shape[0]), int(encoded.shape[1]))
    skipped = np.concatenate([restored, encoded], axis=2)
    decoded = np.maximum(_conv2d(skipped, weights.dec_w, weights.dec_b), 0)
    logits = _conv2d(decoded, weights.head_w, weights.head_b)
    return np.argmax(logits, axis=2).astype(np.uint8)


def predict_unet(rgb: NDArray[np.uint8], weights: UnetWeights) -> NDArray[np.uint8]:
    """Tile ``rgb`` at ``weights.tile``, run the U-Net, and stitch to native size.

    Args:
        rgb: ``HxWx3`` uint8 mosaic.
        weights: Loaded U-Net weights. ``tile`` is the window edge.

    Returns:
        Class-index mask with the same height and width as ``rgb``.
    """
    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    labels = np.zeros((height, width), dtype=np.uint8)
    tile = weights.tile
    for y0 in range(0, height, tile):
        for x0 in range(0, width, tile):
            patch = rgb[y0 : y0 + tile, x0 : x0 + tile]
            patch_h, patch_w = int(patch.shape[0]), int(patch.shape[1])
            pad_h = patch_h % 2
            pad_w = patch_w % 2
            padded = patch
            if pad_h or pad_w:
                padded = np.pad(patch, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
            pred = unet_forward(padded, weights)
            labels[y0 : y0 + patch_h, x0 : x0 + patch_w] = pred[:patch_h, :patch_w]
    return labels


class UnetCpuTool(Tool):
    """CPU U-Net over RGB mosaics. Tiles large inputs and stitches the labels."""

    def __init__(self, spec: PluginSpec) -> None:
        if spec.runtime != "unet_cpu":
            raise ValueError(f"Unsupported model runtime {spec.runtime!r} for {spec.name}")
        if len(spec.output.classes) < 2:
            raise ValueError(f"{spec.name} must declare at least two output classes")
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

        weights = UnetWeights(self.spec, load_weight_payload(self.spec))
        rgb = load_rgb_array(store.resolve_read(request.path))
        labels = predict_unet(rgb, weights)
        height, width = int(labels.shape[0]), int(labels.shape[1])
        if (height, width) != (int(rgb.shape[0]), int(rgb.shape[1])):
            raise RuntimeError("Label map spatial size must match the input")

        mask_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(labels, mode="L").save(mask_path)
        positive = self.spec.output.classes[1]
        fraction = float(np.count_nonzero(labels == 1)) / float(labels.size)
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
