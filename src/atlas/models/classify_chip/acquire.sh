#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
root="$here"
while [[ "$root" != "/" && ! -f "$root/pyproject.toml" ]]; do
  root="$(dirname "$root")"
done
if [[ ! -f "$root/pyproject.toml" ]]; then
  echo "Could not find repo root (pyproject.toml)" >&2
  exit 1
fi

data="$root/local/models/classify_chip"
if command -v uv >/dev/null 2>&1; then
  py=(uv run python)
else
  py=(python3)
fi

"${py[@]}" - "$data" <<'PY'
from pathlib import Path
import sys

from PIL import Image

data = Path(sys.argv[1])
colors = {
    "bare": (180, 140, 90),
    "built-up": (128, 128, 128),
    "water": (30, 90, 180),
    "vegetation": (40, 160, 60),
    "cloud": (245, 245, 245),
}
for name, color in colors.items():
    class_dir = data / name
    class_dir.mkdir(parents=True, exist_ok=True)
    for index in range(3):
        Image.new("RGB", (224, 224), color).save(class_dir / f"{index}.png")
print(data)
PY
