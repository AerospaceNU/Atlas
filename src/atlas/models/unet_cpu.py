from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from pydantic import BaseModel

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import Tool, ToolResult
from atlas.models.base import (
    PluginSpec,
    load_rgb_array,
    parse_weight_payload,
    read_weight_bytes,
    weight_tile,
)
from atlas.models.inputs import SegmentInput

_MAX_HIDDEN = 32
_TILE_MARGIN = 8
_WEIGHT_CACHE: dict[tuple[str, str, tuple[str, ...]], UnetWeights] = {}


class UnetWeights:
    """Conv kernels for the CPU U-Net. Shapes are part of the weight contract."""

    def __init__(self, spec: PluginSpec, payload: dict[str, Any]) -> None:
        classes = len(spec.output.classes)
        self.tile = weight_tile(payload, spec)
        if self.tile % 2:
            raise ValueError(f"Weight tile {self.tile} must be even")
        self.scale = _scalar(payload, "scale")
        self.mean = _rgb_triple(payload, "mean")
        self.std = _rgb_triple(payload, "std")
        if not bool(np.all(np.abs(self.std) > 0)):
            raise ValueError("Weight std must be a non-zero RGB triple")
        self.enc_w = _kernel(payload, "enc_w", cin=3)
        hidden = int(self.enc_w.shape[0])
        if hidden > _MAX_HIDDEN:
            raise ValueError(f"Weight hidden size {hidden} exceeds {_MAX_HIDDEN}")
        self.enc_b = _bias(payload, "enc_b", hidden)
        self.bn_w = _kernel(payload, "bn_w", cin=hidden, cout=hidden)
        self.bn_b = _bias(payload, "bn_b", hidden)
        self.dec_w = _kernel(payload, "dec_w", cin=hidden * 2, cout=hidden)
        self.dec_b = _bias(payload, "dec_b", hidden)
        self.head_w = _kernel(payload, "head_w", cin=hidden, cout=classes)
        self.head_b = _bias(payload, "head_b", classes)


def load_unet_weights(spec: PluginSpec) -> UnetWeights:
    """Load U-Net weights, reusing a previous parse of the same bytes.

    Args:
        spec: Plugin whose relative weight key points at the JSON contract.

    Returns:
        Parsed kernels. A second call with the same name, class list, and
        sha256 returns the same object.
    """
    data = read_weight_bytes(spec)
    digest = hashlib.sha256(data).hexdigest()
    key = (spec.name, digest, tuple(spec.output.classes))
    cached = _WEIGHT_CACHE.get(key)
    if cached is not None:
        return cached
    weights = UnetWeights(spec, parse_weight_payload(spec, data))
    _WEIGHT_CACHE[key] = weights
    return weights


def _scalar(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Weight {key} must be a number")
    return float(value)


def _rgb_triple(payload: dict[str, Any], key: str) -> NDArray[np.float32]:
    array = np.asarray(payload.get(key), dtype=np.float32)
    if array.shape != (3,):
        raise ValueError(f"Weight {key} must be an RGB triple")
    return array


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


def _normalize(rgb: NDArray[np.uint8], weights: UnetWeights) -> NDArray[np.float32]:
    features = rgb.astype(np.float32) * np.float32(weights.scale)
    return (features - weights.mean.reshape(1, 1, 3)) / weights.std.reshape(1, 1, 3)


def unet_forward(rgb: NDArray[np.uint8], weights: UnetWeights) -> NDArray[np.uint8]:
    """Run one even-sized tile and return a class-index map.

    Args:
        rgb: ``HxWx3`` uint8 tile. Both spatial sizes must be even.
        weights: Kernels loaded from the plugin weight file.

    Returns:
        ``HxW`` uint8 class indexes. Inputs are scaled by ``weights.scale``,
        then shifted by ``mean`` and ``std``, before the convolutions.
    """
    features = _normalize(rgb, weights)
    encoded = np.maximum(_conv2d(features, weights.enc_w, weights.enc_b), 0)
    pooled = _max_pool2(encoded)
    bottleneck = np.maximum(_conv2d(pooled, weights.bn_w, weights.bn_b), 0)
    restored = _upsample2(bottleneck, int(encoded.shape[0]), int(encoded.shape[1]))
    skipped = np.concatenate([restored, encoded], axis=2)
    decoded = np.maximum(_conv2d(skipped, weights.dec_w, weights.dec_b), 0)
    logits = _conv2d(decoded, weights.head_w, weights.head_b)
    return np.argmax(logits, axis=2).astype(np.uint8)


def _labels_for_patch(patch: NDArray[np.uint8], weights: UnetWeights) -> NDArray[np.uint8]:
    patch_h, patch_w = int(patch.shape[0]), int(patch.shape[1])
    pad_h = patch_h % 2
    pad_w = patch_w % 2
    padded = patch
    if pad_h or pad_w:
        padded = np.pad(patch, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    return unet_forward(padded, weights)[:patch_h, :patch_w]


def predict_unet(
    rgb: NDArray[np.uint8],
    weights: UnetWeights,
    *,
    margin: int = _TILE_MARGIN,
) -> NDArray[np.uint8]:
    """Tile ``rgb`` with an overlap of ``margin`` and stitch the core of each tile.

    Args:
        rgb: ``HxWx3`` uint8 mosaic.
        weights: Loaded U-Net weights. ``tile`` is the core window edge.
        margin: Context kept around each core, then cropped away. The default
            is 8 pixels, enough for this network's 3x3 stack. ``tile`` and the
            default margin are even so 2x2 pooling stays aligned with a
            single pass over the whole image.

    Returns:
        Class-index mask with the same height and width as ``rgb``.
    """
    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    labels = np.zeros((height, width), dtype=np.uint8)
    tile = weights.tile
    for y0 in range(0, height, tile):
        for x0 in range(0, width, tile):
            y1 = min(y0 + tile, height)
            x1 = min(x0 + tile, width)
            cy0 = max(0, y0 - margin)
            cx0 = max(0, x0 - margin)
            cy1 = min(height, y1 + margin)
            cx1 = min(width, x1 + margin)
            pred = _labels_for_patch(rgb[cy0:cy1, cx0:cx1], weights)
            top = y0 - cy0
            left = x0 - cx0
            labels[y0:y1, x0:x1] = pred[top : top + (y1 - y0), left : left + (x1 - x0)]
    return labels


class UnetCpuTool(Tool):
    """CPU U-Net over RGB mosaics. Overlapping tiles are cropped and stitched."""

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

        weights = load_unet_weights(self.spec)
        rgb = load_rgb_array(store.resolve_read(request.path))
        labels = predict_unet(rgb, weights)
        height, width = int(labels.shape[0]), int(labels.shape[1])
        if (height, width) != (int(rgb.shape[0]), int(rgb.shape[1])):
            raise RuntimeError("Label map spatial size must match the input")

        mask_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(labels, mode="L").save(mask_path)
        total = float(labels.size)
        fractions = {
            name: round(float(np.count_nonzero(labels == index)) / total, 6)
            for index, name in enumerate(self.spec.output.classes)
        }
        positive = self.spec.output.classes[1]
        fraction = fractions[positive]
        payload = {
            "path": request.path,
            "width": width,
            "height": height,
            "positive": positive,
            "positive_fraction": fraction,
            "fractions": fractions,
            "mask": store.relative(mask_path),
        }
        json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return ToolResult(
            text=f"{self.name}: {width}x{height} ({positive}={fraction:.2f})",
            artifacts=[store.relative(mask_path), store.relative(json_path)],
        )
