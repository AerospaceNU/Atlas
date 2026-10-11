from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

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
    return root / "local" / "models" / "lgbm_clouds" / "train"


def _default_out() -> Path:
    root = repo_root()
    if root is None:
        raise SystemExit("No checkout found; pass --out")
    return root / "local" / "models" / "lgbm_clouds" / "model.json"


def _class_pixels(class_dir: Path) -> NDArray[np.float64]:
    paths = sorted(class_dir.glob("*.png")) + sorted(class_dir.glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"No PNG chips for class {class_dir.name}")
    samples = [load_rgb_array(path).reshape(-1, 3).astype(np.float64) for path in paths]
    pixels = np.concatenate(samples, axis=0)
    if len(pixels) > _MAX_PIXELS_PER_CLASS:
        step = int(np.ceil(len(pixels) / _MAX_PIXELS_PER_CLASS))
        pixels = pixels[::step][:_MAX_PIXELS_PER_CLASS]
    return pixels


def fit_lightgbm(data_dir: Path) -> dict[str, object]:
    """Fit a binary LightGBM model on raw 0-255 RGB pixels.

    The positive class is the second name in ``plugin.toml``. Features are
    ``r``, ``g``, and ``b`` on the 0-255 scale, matching ``predict_lightgbm``.

    Args:
        data_dir: Directory of per-class chip folders.

    Returns:
        JSON-ready weight object with ``tile`` 256.

    Raises:
        ValueError: If the LightGBM extra is not installed, or the booster is
            not a single model per iteration.
        FileNotFoundError: If a class directory or its chips are missing.
    """
    try:
        import lightgbm as lgb
    except (ImportError, OSError) as exc:
        raise ValueError("LightGBM is unavailable. Install the lightgbm extra.") from exc
    if len(_CLASSES) != 2:
        raise ValueError("lgbm_clouds trains a binary model and needs exactly two classes")
    features: list[NDArray[np.float64]] = []
    labels: list[NDArray[np.int64]] = []
    for index, name in enumerate(_CLASSES):
        class_dir = data_dir / name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {name}")
        pixels = _class_pixels(class_dir)
        features.append(pixels)
        labels.append(np.full(len(pixels), index, dtype=np.int64))
    booster = lgb.train(
        {
            "objective": "binary",
            "verbosity": -1,
            "num_leaves": 4,
            "min_data_in_leaf": 1,
            "num_threads": 1,
        },
        lgb.Dataset(np.concatenate(features, axis=0), label=np.concatenate(labels)),
        num_boost_round=6,
    )
    if int(booster.num_model_per_iteration()) != 1:
        raise ValueError("Trained LightGBM model must have num_model_per_iteration == 1")
    model: Any = booster.model_to_string()
    return {
        "runtime": "lightgbm",
        "classes": list(_CLASSES),
        "features": ["r", "g", "b"],
        "tile": _TILE,
        "model": model,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fit a LightGBM cloud model from RGB chips.")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    data_dir = args.data_dir or _default_data_dir()
    out = args.out or _default_out()
    payload = fit_lightgbm(data_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(out.as_posix())


if __name__ == "__main__":
    main(sys.argv[1:])
