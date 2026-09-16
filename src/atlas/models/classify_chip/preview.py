"""Render a labeled inference sheet for classify_chip chips."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.tools import default_registry
from atlas.models.base import parse_plugin_toml, repo_root

_PLUGIN = Path(__file__).with_name("plugin.toml")
_COLORS: dict[str, tuple[int, int, int]] = {
    "bare": (180, 140, 90),
    "built-up": (128, 128, 128),
    "water": (30, 90, 180),
    "vegetation": (40, 160, 60),
    "cloud": (245, 245, 245),
}
_CHIP = 160
_PAD = 16
_LABEL_H = 48


def _font(size: int) -> Any:
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        candidate = Path(path)
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _default_out() -> Path:
    root = repo_root()
    if root is None:
        raise SystemExit("No checkout found; pass --out")
    return root / "local" / "models" / "classify_chip" / "preview.png"


def render_preview(out: Path, fixtures: Path | None) -> Path:
    """Classify one synthetic chip per class and write a labeled PNG."""
    spec = parse_plugin_toml(_PLUGIN.read_text(encoding="utf-8"))
    classes = spec.output.classes
    root = repo_root()
    workspace = (root or out.parent) / "local" / "models" / "classify_chip" / "preview_workspace"
    if workspace.exists():
        for stale in workspace.rglob("*"):
            if stale.is_file():
                stale.unlink()
    store = LocalArtifactStore(workspace)
    registry = default_registry()
    cells: list[tuple[str, str, float, Image.Image]] = []
    for name in classes:
        color = _COLORS[name]
        chip = Image.new("RGB", (224, 224), color)
        relative = f"{name}.png"
        chip.save(store.root / relative)
        if fixtures is not None:
            fixtures.mkdir(parents=True, exist_ok=True)
            chip.resize((128, 128), Image.Resampling.BOX).save(fixtures / f"{name}.png")
        result = registry.execute(
            spec.name,
            {"path": relative, "output_path": f"artifacts/{name}.json"},
            store,
        )
        payload = json.loads((store.root / f"artifacts/{name}.json").read_text(encoding="utf-8"))
        label = str(payload["label"])
        score = float(payload["scores"][label])
        cells.append((name, label, score, chip.copy()))
        if not result.artifacts:
            raise RuntimeError(f"classify_chip produced no artifact for {name}")

    width = len(cells) * (_CHIP + _PAD) + _PAD
    height = _PAD + _CHIP + _LABEL_H + _PAD
    sheet = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    title_font = _font(14)
    for index, (expected, predicted, score, chip) in enumerate(cells):
        x = _PAD + index * (_CHIP + _PAD)
        y = _PAD
        sheet.paste(chip.resize((_CHIP, _CHIP), Image.Resampling.BOX), (x, y))
        ok = expected == predicted
        caption = f"{predicted} {score:.2f}"
        draw.text((x, y + _CHIP + 6), expected, fill=(40, 40, 40), font=title_font)
        draw.text(
            (x, y + _CHIP + 24),
            caption,
            fill=(20, 120, 40) if ok else (180, 30, 30),
            font=title_font,
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=None,
        help="Optional directory for 128px class chips (repo samples).",
    )
    args = parser.parse_args(argv)
    path = render_preview(args.out or _default_out(), args.fixtures)
    print(path.as_posix())


if __name__ == "__main__":
    main(sys.argv[1:])
