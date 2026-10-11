from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from atlas.models.base import parse_plugin_toml, repo_root, weights_root

_PLUGIN = Path(__file__).with_name("plugin.toml")
_RELATIVE_WEIGHT = parse_plugin_toml(_PLUGIN.read_text(encoding="utf-8")).weight


def _default_trained() -> Path:
    root = repo_root()
    if root is None:
        raise SystemExit("No checkout found; pass --model")
    return root / "local" / "models" / "lgbm_clouds" / "model.json"


def export_weight(model_path: Path, destination: Path) -> str:
    """Write the weight contract to ``destination`` and return its sha256.

    Paste the digest into ``sha256`` in ``plugin.toml`` to pin the file.
    Weights are not committed; an empty ``sha256`` leaves the file unpinned.
    The LightGBM ``model`` string scores raw 0-255 RGB features.

    Args:
        model_path: JSON written by ``train.py``.
        destination: File under the weights root, usually ``lgbm_clouds/model.json``.

    Returns:
        Hex sha256 of the bytes written to ``destination``.
    """
    payload = json.loads(model_path.read_text(encoding="utf-8"))
    body = json.dumps(payload, indent=2) + "\n"
    data = body.encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Copy a trained LightGBM JSON into the weights directory and print its sha256."
        )
    )
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    model = args.model or _default_trained()
    destination = args.out or (weights_root() / _RELATIVE_WEIGHT)
    print(export_weight(model, destination))


if __name__ == "__main__":
    main(sys.argv[1:])
