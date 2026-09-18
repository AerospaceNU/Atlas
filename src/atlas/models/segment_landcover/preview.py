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
_THUMB = 256
_PAD = 12
_LABEL_H = 28


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
    return root / "local" / "models" / "segment_landcover" / "preview.png"


def _val_dir() -> Path:
    root = repo_root()
    if root is None:
        raise SystemExit("No checkout found; pass --data-dir")
    return root / "local" / "models" / "segment_landcover" / "val"


def _list_chips(class_dir: Path) -> list[Path]:
    return sorted(class_dir.glob("*.png")) + sorted(class_dir.glob("*.jpg"))


def _thumb(image: Image.Image) -> Image.Image:
    """Downscale only for the contact sheet. Inference already ran at native size."""
    fitted = image.convert("RGB").copy()
    fitted.thumbnail((_THUMB, _THUMB))
    canvas = Image.new("RGB", (_THUMB, _THUMB), (20, 20, 20))
    x = (_THUMB - fitted.width) // 2
    y = (_THUMB - fitted.height) // 2
    canvas.paste(fitted, (x, y))
    return canvas


def render_preview(
    out: Path,
    data_dir: Path,
    *,
    per_class: int,
    fixtures: Path | None,
) -> Path:
    """Segment held-out scenes at native size; sheet is RGB | overlay thumbnails."""
    spec = parse_plugin_toml(_PLUGIN.read_text(encoding="utf-8"))
    classes = spec.output.classes
    root = repo_root()
    workspace = (
        (root or out.parent) / "local" / "models" / "segment_landcover" / "preview_workspace"
    )
    if workspace.exists():
        for stale in workspace.rglob("*"):
            if stale.is_file():
                stale.unlink()
    store = LocalArtifactStore(workspace)
    registry = default_registry()
    cells: list[tuple[str, str, Image.Image, Image.Image]] = []
    for name in classes:
        chips = _list_chips(data_dir / name)
        if not chips:
            raise FileNotFoundError(f"No val scenes for {name}. Run acquire.sh then train/export.")
        if fixtures is not None:
            fixtures.mkdir(parents=True, exist_ok=True)
            with Image.open(chips[0]) as first:
                first.convert("RGB").save(fixtures / f"{name}.png")
        for index, source in enumerate(chips[:per_class]):
            relative = f"{name}_{index}.png"
            with Image.open(source) as image:
                rgb = image.convert("RGB")
                rgb.save(store.root / relative)
                original = rgb.copy()
            result = registry.execute(
                spec.name,
                {
                    "path": relative,
                    "labels_path": f"artifacts/{name}_{index}_labels.png",
                    "overlay_path": f"artifacts/{name}_{index}_overlay.png",
                    "json_path": f"artifacts/{name}_{index}.json",
                },
                store,
            )
            payload = json.loads(
                (store.root / f"artifacts/{name}_{index}.json").read_text(encoding="utf-8")
            )
            if payload["width"] != original.size[0] or payload["height"] != original.size[1]:
                raise RuntimeError("segment_landcover resized the input")
            with Image.open(store.root / payload["overlay"]) as overlay_im:
                overlay = overlay_im.convert("RGB")
            cells.append((name, f"{payload['width']}x{payload['height']}", original, overlay))
            if not result.artifacts:
                raise RuntimeError(f"segment_landcover produced no artifact for {relative}")

    columns = min(len(cells), 5)
    rows = (len(cells) + columns - 1) // columns
    cell_w = _THUMB * 2 + 4
    width = columns * (cell_w + _PAD) + _PAD
    height = rows * (_THUMB + _LABEL_H) + _PAD
    sheet = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    title_font = _font(13)
    for index, (expected, size_text, original, overlay) in enumerate(cells):
        col = index % columns
        row = index // columns
        x = _PAD + col * (cell_w + _PAD)
        y = _PAD + row * (_THUMB + _LABEL_H)
        sheet.paste(_thumb(original), (x, y))
        sheet.paste(_thumb(overlay), (x + _THUMB + 4, y))
        draw.text(
            (x, y + _THUMB + 6),
            f"{expected} {size_text}",
            fill=(40, 40, 40),
            font=title_font,
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Render native-resolution overlays for held-out satellite scenes."
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--per-class", type=int, default=1)
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=None,
        help="Optional directory for native-size class samples.",
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
