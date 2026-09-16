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
_CHIP = 128
_PAD = 12
_LABEL_H = 44


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


def _val_dir() -> Path:
    root = repo_root()
    if root is None:
        raise SystemExit("No checkout found; pass --data-dir")
    return root / "local" / "models" / "classify_chip" / "val"


def _list_chips(class_dir: Path) -> list[Path]:
    return sorted(class_dir.glob("*.png")) + sorted(class_dir.glob("*.jpg"))


def render_preview(
    out: Path,
    data_dir: Path,
    *,
    per_class: int,
    fixtures: Path | None,
) -> Path:
    """Classify held-out chips through the agent tool and write a labeled PNG."""
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
        chips = _list_chips(data_dir / name)
        if not chips:
            raise FileNotFoundError(
                f"No val chips for {name}. Run acquire.sh (satellite) then train/export."
            )
        if fixtures is not None:
            fixtures.mkdir(parents=True, exist_ok=True)
            with Image.open(chips[0]) as first:
                first.convert("RGB").resize((128, 128), Image.Resampling.BOX).save(
                    fixtures / f"{name}.png"
                )
        for index, source in enumerate(chips[:per_class]):
            relative = f"{name}_{index}.png"
            with Image.open(source) as image:
                rgb = image.convert("RGB")
                rgb.save(store.root / relative)
                chip = rgb.copy()
            result = registry.execute(
                spec.name,
                {"path": relative, "output_path": f"artifacts/{name}_{index}.json"},
                store,
            )
            payload = json.loads(
                (store.root / f"artifacts/{name}_{index}.json").read_text(encoding="utf-8")
            )
            label = str(payload["label"])
            score = float(payload["scores"][label])
            cells.append((name, label, score, chip))
            if not result.artifacts:
                raise RuntimeError(f"classify_chip produced no artifact for {relative}")

    columns = min(len(cells), 5)
    rows = (len(cells) + columns - 1) // columns
    width = columns * (_CHIP + _PAD) + _PAD
    height = rows * (_CHIP + _LABEL_H) + _PAD
    sheet = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    title_font = _font(13)
    for index, (expected, predicted, score, chip) in enumerate(cells):
        col = index % columns
        row = index // columns
        x = _PAD + col * (_CHIP + _PAD)
        y = _PAD + row * (_CHIP + _LABEL_H)
        sheet.paste(chip.resize((_CHIP, _CHIP), Image.Resampling.BOX), (x, y))
        ok = expected == predicted
        caption = f"{predicted} {score:.2f}"
        draw.text((x, y + _CHIP + 4), expected, fill=(40, 40, 40), font=title_font)
        draw.text(
            (x, y + _CHIP + 22),
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
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--per-class", type=int, default=3)
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=None,
        help="Optional directory for 128px class chips (repo samples).",
    )
    args = parser.parse_args(argv)
    path = render_preview(
        args.out or _default_out(),
        args.data_dir or _val_dir(),
        per_class=args.per_class,
        fixtures=args.fixtures,
    )
    print(path.as_posix())


if __name__ == "__main__":
    main(sys.argv[1:])
