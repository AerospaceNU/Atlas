from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from atlas.models.base import parse_plugin_toml, replace_weight_bytes, repo_root, weights_root

_PLUGIN = Path(__file__).with_name("plugin.toml")
_RELATIVE_WEIGHT = parse_plugin_toml(_PLUGIN.read_text(encoding="utf-8")).weight


def _default_trained() -> Path:
    root = repo_root()
    if root is None:
        raise SystemExit("No checkout found; pass --model")
    return root / "local" / "models" / "unet_water" / "model.json"


def export_weight(model_path: Path, destination: Path) -> str:
    """Write the weight contract to ``destination`` and return its sha256.

    The file is written to a temporary sibling and moved into place with
    ``os.replace``. A symlink at ``destination`` is refused. Paste the digest
    into ``sha256`` in ``plugin.toml``, then restart to load it.

    Args:
        model_path: JSON written by ``train.py``.
        destination: File under the weights root, usually ``unet_water/model.json``.

    Returns:
        Hex sha256 of the bytes written to ``destination``.
    """
    payload = json.loads(model_path.read_text(encoding="utf-8"))
    body = json.dumps(payload, indent=2) + "\n"
    data = body.encode("utf-8")
    replace_weight_bytes(destination, data)
    return hashlib.sha256(data).hexdigest()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Copy a trained U-Net JSON into the weights directory and print its sha256."
    )
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    model = args.model or _default_trained()
    destination = args.out or (weights_root() / _RELATIVE_WEIGHT)
    digest = export_weight(model, destination)
    print(f"restart to load {digest}")


if __name__ == "__main__":
    main(sys.argv[1:])
