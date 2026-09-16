"""Copy trained centroids into the repo-root weights directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from atlas.models.base import repo_root, weights_root

_RELATIVE_WEIGHT = "classify_chip/model.json"


def _default_centroids() -> Path:
    root = repo_root()
    if root is None:
        raise SystemExit("No checkout found; pass --centroids")
    return root / "local" / "models" / "classify_chip" / "centroids.json"


def export_weight(centroids_path: Path, destination: Path) -> str:
    """Write ``destination`` and return the sha256 hex digest."""
    payload = json.loads(centroids_path.read_text(encoding="utf-8"))
    body = json.dumps(payload, indent=2) + "\n"
    data = body.encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--centroids", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    centroids = args.centroids or _default_centroids()
    destination = args.out or (weights_root() / _RELATIVE_WEIGHT)
    digest = export_weight(centroids, destination)
    print(digest)


if __name__ == "__main__":
    main(sys.argv[1:])
