"""Fit per-class mean RGB centroids from acquired PNG chips."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

from atlas.models.base import mean_rgb, parse_plugin_toml, repo_root

_PLUGIN = Path(__file__).with_name("plugin.toml")
_CLASSES = parse_plugin_toml(_PLUGIN.read_text(encoding="utf-8")).output.classes


def _default_data_dir() -> Path:
    root = repo_root()
    if root is None:
        raise SystemExit("No checkout found; pass --data-dir")
    return root / "local" / "models" / "classify_chip" / "train"


def _default_out() -> Path:
    root = repo_root()
    if root is None:
        raise SystemExit("No checkout found; pass --out")
    return root / "local" / "models" / "classify_chip" / "centroids.json"


def _chips(class_dir: Path) -> list[Path]:
    return sorted(class_dir.glob("*.png")) + sorted(class_dir.glob("*.jpg"))


def fit_centroids(data_dir: Path, size: int = 224) -> dict[str, list[float]]:
    """Average mean RGB per class subdirectory of ``data_dir``."""
    sums: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    counts: dict[str, int] = defaultdict(int)
    for name in _CLASSES:
        class_dir = data_dir / name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {name}")
        for path in _chips(class_dir):
            with Image.open(path) as image:
                mean = mean_rgb(image, size=size)
            for i, value in enumerate(mean):
                sums[name][i] += value
            counts[name] += 1
        if counts[name] == 0:
            raise FileNotFoundError(f"No PNG chips for class {name}")
    return {name: [sums[name][i] / counts[name] for i in range(3)] for name in _CLASSES}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--size", type=int, default=224)
    args = parser.parse_args(argv)
    data_dir = args.data_dir or _default_data_dir()
    out = args.out or _default_out()
    centroids = fit_centroids(data_dir, size=args.size)
    payload = {"classes": _CLASSES, "centroids": centroids}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(out.as_posix())


if __name__ == "__main__":
    main(sys.argv[1:])
