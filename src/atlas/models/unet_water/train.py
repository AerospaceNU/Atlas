from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from atlas.models.base import load_rgb_array, parse_plugin_toml, repo_root

_PLUGIN = Path(__file__).with_name("plugin.toml")
_SPEC = parse_plugin_toml(_PLUGIN.read_text(encoding="utf-8"))
_CLASSES = _SPEC.output.classes
_TILE = 256
_MAX_PIXELS_PER_CLASS = 4096


def _default_data_dir() -> Path:
    root = repo_root()
    if root is None:
        raise SystemExit("No checkout found; pass --data-dir")
    return root / "local" / "models" / "unet_water" / "train"


def _default_out() -> Path:
    root = repo_root()
    if root is None:
        raise SystemExit("No checkout found; pass --out")
    return root / "local" / "models" / "unet_water" / "model.json"


def _center(cout: int, cin: int, src: int) -> list[list[list[list[float]]]]:
    weight = np.zeros((cout, cin, 3, 3), dtype=np.float64)
    for channel in range(cout):
        weight[channel, src + channel, 1, 1] = 1.0
    return cast(list[list[list[list[float]]]], weight.tolist())


def _class_pixels(class_dir: Path) -> NDArray[np.float64]:
    paths = sorted(class_dir.glob("*.png")) + sorted(class_dir.glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"No PNG chips for class {class_dir.name}")
    samples: list[NDArray[np.float64]] = []
    for path in paths:
        flat = load_rgb_array(path).reshape(-1, 3).astype(np.float64)
        samples.append(flat)
    pixels = np.concatenate(samples, axis=0)
    if len(pixels) > _MAX_PIXELS_PER_CLASS:
        step = int(np.ceil(len(pixels) / _MAX_PIXELS_PER_CLASS))
        pixels = pixels[::step][:_MAX_PIXELS_PER_CLASS]
    return pixels


def fit_unet(data_dir: Path) -> dict[str, object]:
    """Fit a nearest-mean head and return a ``unet_cpu`` weight document.

    Chips live in ``data_dir/<class>/*.png``. RGB is scaled by ``1/255`` with
    mean 0 and std 1, so the convolutions see ``[0, 1]``. The encoder and
    decoder are identity 3x3 kernels. The head is a nearest class-mean
    classifier in that space: logit ``c`` is ``2 * dot(x, mean_c) - ||mean_c||^2``.

    Args:
        data_dir: Directory of per-class chip folders named in ``plugin.toml``.

    Returns:
        JSON-ready weight object with ``tile`` 256.
    """
    means: list[NDArray[np.float64]] = []
    for name in _CLASSES:
        class_dir = data_dir / name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {name}")
        means.append(_class_pixels(class_dir).mean(axis=0) / 255.0)
    hidden = 3
    class_count = len(_CLASSES)
    head_w = np.zeros((class_count, hidden, 3, 3), dtype=np.float64)
    head_b = np.zeros(class_count, dtype=np.float64)
    for index, mean in enumerate(means):
        head_w[index, :, 1, 1] = 2.0 * mean
        head_b[index] = -float(np.dot(mean, mean))
    return {
        "runtime": "unet_cpu",
        "classes": list(_CLASSES),
        "tile": _TILE,
        "scale": 1.0 / 255.0,
        "mean": [0.0, 0.0, 0.0],
        "std": [1.0, 1.0, 1.0],
        "enc_w": _center(hidden, 3, 0),
        "enc_b": [0.0, 0.0, 0.0],
        "bn_w": _center(hidden, hidden, 0),
        "bn_b": [0.0, 0.0, 0.0],
        "dec_w": _center(hidden, hidden * 2, hidden),
        "dec_b": [0.0, 0.0, 0.0],
        "head_w": head_w.tolist(),
        "head_b": head_b.tolist(),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fit a CPU U-Net head from RGB chips.")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    data_dir = args.data_dir or _default_data_dir()
    out = args.out or _default_out()
    payload = fit_unet(data_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(out.as_posix())


if __name__ == "__main__":
    main(sys.argv[1:])
