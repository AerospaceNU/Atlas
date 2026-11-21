"""Map rendering: stretched layers, composites, class overlays, legend and scale bar.

Pillow and numpy only (no rasterio). Invariants: decimation is an integer factor applied
before colouring, so an RGB array is never allocated at full size; legend hectares and the
valid fraction always come from the full-resolution arrays; the PNG is always written to
disk and `attach` only decides whether the tool lists it as an image artifact.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from atlas.geo.errors import GeoError, fix
from atlas.geo.results import grid_with_histogram, json_safe, sig4

if TYPE_CHECKING:
    from collections.abc import Sequence

    from atlas.geo.stack import LayerMeta, Stack

__all__ = [
    "CLOUD_RGB",
    "DIVERGING",
    "NODATA_RGB",
    "OKABE_ITO",
    "SEQUENTIAL",
    "RenderResult",
    "attach_enabled",
    "lut",
    "render_class_overlay",
    "render_map",
]

SCHEMA = "atlas.figure/1"
MAX_PNG_BYTES = 250_000
MAX_TEXT_CHARS = 1_100
MAX_LEGEND_ROWS = 12
_SAMPLE_CELLS = 1_000_000

SEQUENTIAL: tuple[tuple[int, int, int], ...] = (
    (68, 1, 84),
    (71, 44, 122),
    (59, 81, 139),
    (44, 113, 142),
    (33, 144, 141),
    (39, 173, 129),
    (92, 200, 99),
    (170, 220, 50),
    (253, 231, 37),
)
DIVERGING: tuple[tuple[int, int, int], ...] = (
    (103, 0, 31),
    (178, 24, 43),
    (214, 96, 77),
    (244, 165, 130),
    (247, 247, 247),
    (146, 197, 222),
    (67, 147, 195),
    (33, 102, 172),
    (5, 48, 97),
)
OKABE_ITO: tuple[tuple[int, int, int], ...] = (
    (230, 159, 0),
    (86, 180, 233),
    (0, 158, 115),
    (240, 228, 66),
    (0, 114, 178),
    (213, 94, 0),
    (204, 121, 167),
    (0, 0, 0),
)
NODATA_RGB = (0xB0, 0xB0, 0xB0)
CLOUD_RGB = (0xD8, 0xD8, 0xD8)
_COMPOSITES = {
    "truecolor": ("red", "green", "blue"),
    "falsecolor": ("nir", "red", "green"),
    "swir": ("swir22", "nir", "red"),
}
_CLOUDY = frozenset({"cloud", "cloud_shadow", "cirrus"})


@dataclass(frozen=True)
class RenderResult:
    """A rendered figure: attachable PNG (or None), sidecar, stretch, legend and text."""

    png: Path | None
    sidecar: Path
    lo_hi: tuple[float, float] | None
    legend: list[tuple[str, str, float | None]]
    text: str
    width: int
    height: int
    valid_fraction: float = 1.0
    figure: Path | None = None


def attach_enabled() -> bool:
    """Return False iff ATLAS_ATTACH_IMAGES is 0, false or no; read on every call."""
    return os.environ.get("ATLAS_ATTACH_IMAGES", "").strip().lower() not in {"0", "false", "no"}


# -- colour tables ---------------------------------------------------------------------


@lru_cache(maxsize=4)
def _lut_cached(name: str) -> NDArray[np.uint8]:
    stops = np.asarray(DIVERGING if name == "diverging" else SEQUENTIAL, dtype=np.float64)
    pos = np.linspace(0.0, 1.0, len(stops))
    grid = np.linspace(0.0, 1.0, 256)
    table = np.stack([np.interp(grid, pos, stops[:, k]) for k in range(3)], axis=1)
    return np.rint(table).astype(np.uint8)


def lut(name: str) -> NDArray[np.uint8]:
    """Return the 256x3 colour table for "sequential" or "diverging"."""
    return _lut_cached("diverging" if name == "diverging" else "sequential")


def _hex(rgb: Sequence[int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*(int(v) for v in rgb))


def _class_colors(codes: Sequence[int]) -> dict[int, tuple[int, int, int]]:
    return {c: OKABE_ITO[i % len(OKABE_ITO)] for i, c in enumerate(sorted(set(codes)))}


# -- fonts and layout metrics -----------------------------------------------------------


def _font(size: int) -> Any:
    """Return a font near `size` pixels; never raises."""
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        try:
            return ImageFont.load_default()
        except Exception:
            return None


def _height(font: Any) -> int:
    try:
        box = font.getbbox("Hg")
        return max(int(box[3] - box[1]), 1)
    except Exception:
        return 10


def _width(font: Any, text: str) -> int:
    try:
        wide: int = math.ceil(font.getlength(text))
        return wide
    except Exception:
        return 6 * len(text)


def _size_honoured() -> bool:
    return _height(_font(10)) != _height(_font(20))


@dataclass(frozen=True)
class _Metrics:
    font: Any
    h: int
    pad: int
    swatch: int
    bar: int

    @property
    def line(self) -> int:
        return self.h + max(2, self.pad // 2)


def _metrics(side: int) -> _Metrics:
    """Layout units scaled from the measured glyph height, so an ignored size still fits."""
    font = _font(12 if side <= 900 else 14)
    h = _height(font)
    return _Metrics(font, h, max(3, h // 2 + 1), h, max(3, round(h * 0.4)))


def _fit(text: str, m: _Metrics, max_w: int) -> str:
    if _width(m.font, text) <= max_w:
        return text
    while len(text) > 2 and _width(m.font, text + "..") > max_w:
        text = text[:-1]
    return text + ".."


# -- layer plumbing -----------------------------------------------------------------------


def _resolve(stack: Stack, name: str, date: str | None) -> str:
    from atlas.geo.errors import StackError

    try:
        return stack.layer(name, date=date).name
    except StackError as exc:
        if "has no layer" not in str(exc):
            raise
        raise GeoError(
            fix(
                f"stack '{stack.id}' has no layer {name!r}; layers are {stack.names()}.",
                "render_map layers=[...] using those names",
            )
        ) from exc


def _plan_layers(
    stack: Stack, layers: Sequence[str], composite: str | None, date: str | None
) -> list[str]:
    if composite is not None:
        if composite not in _COMPOSITES:
            raise GeoError(
                fix(f"unknown composite {composite!r}.", f"use one of {list(_COMPOSITES)}")
            )
        wanted = list(_COMPOSITES[composite])
    else:
        wanted = list(layers)
    if len(wanted) not in (1, 3):
        raise GeoError(
            fix(
                f"give 1 layer or 3 (got {len(wanted)}); layers are {stack.names()}.",
                "render_map layers=[...] or composite=truecolor",
            )
        )
    return [_resolve(stack, n, date) for n in wanted]


def _open_other(stack: Stack, ref: str) -> Stack:
    from atlas.geo.stack import Stack as StackCls

    return StackCls.open(stack.analysis_dir / "stacks" / ref)


def _cloud_mask(stack: Stack) -> NDArray[np.bool_] | None:
    planes = stack.planes()
    hit = [planes[q.name] for q in stack.meta.qa if q.semantic in _CLOUDY]
    if not hit:
        return None
    out = np.zeros(stack.grid.shape, dtype=bool)
    for plane in hit:
        out |= plane
    return out


@dataclass
class _Src:
    stack: Stack
    names: list[str]
    arrays: list[Any]
    metas: list[LayerMeta]
    valid: NDArray[np.bool_]
    cloud: NDArray[np.bool_] | None


def _source(stack: Stack, names: Sequence[str]) -> _Src:
    metas = [stack.layer(n) for n in names]
    return _Src(
        stack,
        list(names),
        [stack.band(n) for n in names],
        metas,
        stack.valid("clear", layers=list(names)),
        _cloud_mask(stack),
    )


# -- stretch ---------------------------------------------------------------------------------


def _sample_values(arr: Any, valid: NDArray[np.bool_]) -> NDArray[np.float64]:
    step = max(1, math.ceil(math.sqrt(arr.shape[0] * arr.shape[1] / _SAMPLE_CELLS)))
    sample = np.asarray(arr[::step, ::step], dtype=np.float64)
    ok = valid[::step, ::step] & np.isfinite(sample)
    return sample[ok]


def _stretch(values: NDArray[np.float64], spec: str) -> tuple[float, float, str | None]:
    """Return `(lo, hi, warning)` for "p2-98", "minmax" or "lo,hi" over finite values."""
    key = spec.strip().lower()
    if "," in key:
        try:
            lo_s, hi_s = key.split(",", 1)
            lo, hi = float(lo_s), float(hi_s)
        except ValueError as exc:
            raise GeoError(
                fix(f"bad stretch {spec!r}.", 'use "p2-98", "minmax" or "lo,hi"')
            ) from exc
        if not hi > lo:
            raise GeoError(fix(f"stretch needs lo < hi, got {spec!r}.", 'use "lo,hi" with lo < hi'))
        return lo, hi, None
    if key not in {"p2-98", "minmax"}:
        raise GeoError(fix(f"bad stretch {spec!r}.", 'use "p2-98", "minmax" or "lo,hi"'))
    if values.size == 0:
        return 0.0, 0.0, "no valid pixels"
    if key == "minmax":
        lo, hi = float(values.min()), float(values.max())
    else:
        lo, hi = (float(v) for v in np.nanpercentile(values, [2, 98]))
    if hi <= lo:
        eps = max(abs(hi) * 1e-6, 1e-6)
        return hi - eps, hi, "constant layer: stretch is degenerate"
    return lo, hi, None


# -- decimation --------------------------------------------------------------------------------


def _block_mean(arr: Any, valid: NDArray[np.bool_], d: int) -> NDArray[np.float32]:
    """NaN-aware block mean over d x d blocks (partial edge blocks kept), in row strips."""
    height, width = arr.shape
    cols = np.arange(0, width, d)
    out = np.empty((math.ceil(height / d), len(cols)), dtype=np.float32)
    strip = d * max(1, 256)
    for r0 in range(0, height, strip):
        r1 = min(r0 + strip, height)
        ok = valid[r0:r1] & np.isfinite(arr[r0:r1])
        x = np.where(ok, np.asarray(arr[r0:r1], dtype=np.float64), 0.0)
        rows = np.arange(0, r1 - r0, d)
        sums = np.add.reduceat(np.add.reduceat(x, rows, axis=0), cols, axis=1)
        cnt = np.add.reduceat(np.add.reduceat(ok.astype(np.float64), rows, axis=0), cols, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(cnt > 0, sums / np.maximum(cnt, 1), np.nan)
        out[r0 // d : r0 // d + mean.shape[0]] = mean
    return out


def _nearest(arr: Any, d: int) -> NDArray[Any]:
    return np.asarray(arr[::d, ::d])


def _paint_invalid(rgb: NDArray[np.uint8], bad: NDArray[np.bool_], src: _Src, d: int) -> None:
    rgb[bad] = NODATA_RGB
    if src.cloud is not None:
        rgb[bad & _nearest(src.cloud, d)] = CLOUD_RGB


# -- panels ----------------------------------------------------------------------------------


def _cont_panel(
    src: _Src, lo: float, hi: float, table: NDArray[np.uint8], d: int
) -> NDArray[np.uint8]:
    dec = _block_mean(src.arrays[0], src.valid, d)
    ok = np.isfinite(dec)
    t = np.clip((np.where(ok, dec, lo) - lo) / (hi - lo), 0.0, 1.0)
    rgb = table[np.rint(t * 255).astype(np.intp)]
    _paint_invalid(rgb, ~ok, src, d)
    return rgb


def _rgb_panel(src: _Src, ranges: Sequence[tuple[float, float]], d: int) -> NDArray[np.uint8]:
    chans = []
    ok_all: NDArray[np.bool_] | None = None
    for arr, (lo, hi) in zip(src.arrays, ranges, strict=True):
        dec = _block_mean(arr, src.valid, d)
        ok = np.isfinite(dec)
        ok_all = ok if ok_all is None else ok_all & ok
        t = np.clip((np.where(ok, dec, lo) - lo) / (hi - lo), 0.0, 1.0)
        chans.append(np.rint(t * 255).astype(np.uint8))
    rgb = np.stack(chans, axis=-1)
    assert ok_all is not None
    _paint_invalid(rgb, ~ok_all, src, d)
    return rgb


def _codes(src: _Src, d: int) -> NDArray[np.int64]:
    """Decimated class codes (nearest); -1 where invalid."""
    arr = _nearest(src.arrays[0], d)
    ok = _nearest(src.valid, d) & np.isfinite(arr)
    return np.where(ok, np.rint(np.where(ok, arr, 0)), -1).astype(np.int64)


def _full_counts(src: _Src) -> dict[int, int]:
    """Valid pixel count per class code at FULL resolution."""
    arr = src.arrays[0]
    found: dict[int, int] = {}
    for r0 in range(0, arr.shape[0], 512):
        chunk = np.asarray(arr[r0 : r0 + 512])
        ok = src.valid[r0 : r0 + 512] & np.isfinite(chunk)
        vals, cnt = np.unique(np.rint(chunk[ok]).astype(np.int64), return_counts=True)
        for v, n in zip(vals.tolist(), cnt.tolist(), strict=True):
            found[v] = found.get(v, 0) + n
    return found


def _cat_panel(src: _Src, colors: dict[int, tuple[int, int, int]], d: int) -> NDArray[np.uint8]:
    codes = _codes(src, d)
    rgb = np.zeros((*codes.shape, 3), dtype=np.uint8)
    _paint_invalid(rgb, codes < 0, src, d)
    for code, color in colors.items():
        rgb[codes == code] = color
    return rgb


def _outline(rgb: NDArray[np.uint8], codes: NDArray[np.int64], colors: dict[int, Any]) -> None:
    """Paint a 1-px outline: valid pixels with a 4-neighbour of another code."""
    edge = np.zeros(codes.shape, dtype=bool)
    edge[:, 1:] |= codes[:, 1:] != codes[:, :-1]
    edge[:, :-1] |= codes[:, :-1] != codes[:, 1:]
    edge[1:, :] |= codes[1:, :] != codes[:-1, :]
    edge[:-1, :] |= codes[:-1, :] != codes[1:, :]
    edge &= codes >= 0
    for code, color in colors.items():
        rgb[edge & (codes == code)] = color


def _fill(rgb: NDArray[np.uint8], codes: NDArray[np.int64], colors: dict[int, Any]) -> None:
    for code, color in colors.items():
        sel = codes == code
        rgb[sel] = ((rgb[sel].astype(np.float64) + np.asarray(color)) / 2).astype(np.uint8)


# -- legend, scale bar, composition ----------------------------------------------------------


@dataclass
class _Legend:
    title: str
    rows: list[tuple[str, str, float | None]]
    more: int = 0
    gradient: tuple[float, float, str] | None = None
    footer: str = ""


def _class_legend(
    src: _Src, colors: dict[int, tuple[int, int, int]], counts: dict[int, int], title: str
) -> _Legend:
    names = src.metas[0].classes or {}
    area = src.stack.pixel_area_m2 / 1e4
    present = sorted(counts)
    shown = present
    if len(present) > MAX_LEGEND_ROWS:
        top = sorted(present, key=lambda c: -counts[c])[:MAX_LEGEND_ROWS]
        shown = sorted(top)
    rows: list[tuple[str, str, float | None]] = [
        (str(names.get(c, f"class {c}")), _hex(colors[c]), float(counts[c] * area)) for c in shown
    ]
    return _Legend(title, rows, more=len(present) - len(shown))


def _row_text(row: tuple[str, str, float | None]) -> str:
    return row[0] if row[2] is None else f"{row[0]} {sig4(row[2])} ha"


def _legend_size(leg: _Legend, m: _Metrics, side: int) -> tuple[int, int]:
    texts = [leg.title, leg.footer, *(_row_text(r) for r in leg.rows)]
    if leg.more:
        texts.append(f"+{leg.more} more")
    wide = max(_width(m.font, t) for t in texts if t) + m.swatch + 3 * m.pad
    width = min(max(wide, 8 * m.h), int(side * 0.4))
    rows = len(leg.rows) + (1 if leg.more else 0) + (2 if leg.gradient else 0)
    height = 2 * m.pad + m.line * (1 + rows + (1 if leg.footer else 0))
    return width, height


def _draw_legend(draw: ImageDraw.ImageDraw, leg: _Legend, m: _Metrics, x0: int, width: int) -> None:
    x, y, room = x0 + m.pad, m.pad, width - 2 * m.pad
    draw.text((x, y), _fit(leg.title, m, room), fill=(0, 0, 0), font=m.font)
    y += m.line
    if leg.gradient is not None:
        lo, hi, name = leg.gradient
        table = lut(name)
        bar_w = max(room, 8)
        for i in range(bar_w):
            col = tuple(int(v) for v in table[round(i / max(bar_w - 1, 1) * 255)])
            draw.line([(x + i, y), (x + i, y + m.swatch)], fill=col)
        y += m.swatch + 2
        draw.text((x, y), _fit(sig4(lo), m, room // 2), fill=(0, 0, 0), font=m.font)
        hi_t = _fit(sig4(hi), m, room // 2)
        draw.text((x + room - _width(m.font, hi_t), y), hi_t, fill=(0, 0, 0), font=m.font)
        y += m.line
    for label, color, ha in leg.rows:
        rgb = tuple(int(color[i : i + 2], 16) for i in (1, 3, 5))
        draw.rectangle([x, y, x + m.swatch - 1, y + m.swatch - 1], fill=rgb, outline=(80, 80, 80))
        text = _fit(_row_text((label, color, ha)), m, room - m.swatch - m.pad)
        draw.text((x + m.swatch + m.pad, y), text, fill=(0, 0, 0), font=m.font)
        y += m.line
    if leg.more:
        draw.text((x, y), f"+{leg.more} more", fill=(0, 0, 0), font=m.font)
        y += m.line
    if leg.footer:
        draw.text((x, y), _fit(leg.footer, m, room), fill=(60, 60, 60), font=m.font)


def _nice_length(limit_m: float) -> float:
    """Largest n * 10^k metres (n in 1, 2, 5) not above `limit_m`; 0 when limit <= 0."""
    if limit_m <= 0:
        return 0.0
    k = math.floor(math.log10(limit_m))
    for exp in (k, k - 1):
        for n in (5, 2, 1):
            if n * 10.0**exp <= limit_m:
                return n * 10.0**exp
    return 0.0


def _scale_bar_px(xres: float, d: int, panel_w: int) -> tuple[float, int]:
    """Return `(nice_m, pixels)`; pixels = round(nice_m * s / xres) with s = 1 / d."""
    nice = _nice_length(0.2 * panel_w * xres * d)
    return nice, round(nice * (1.0 / d) / xres) if nice else 0


def _length_label(metres: float) -> str:
    return f"{sig4(metres / 1000)} km" if metres >= 1000 else f"{sig4(metres)} m"


def _draw_scale_bar(
    img: Image.Image, m: _Metrics, xres: float, d: int, panel_w: int, panel_h: int
) -> float | None:
    nice, px = _scale_bar_px(xres, d, panel_w)
    if nice <= 0 or px < 2:
        return None
    draw = ImageDraw.Draw(img, "RGBA")
    label = _length_label(nice)
    box_w = max(px, _width(m.font, label)) + 2 * m.pad
    box_h = m.h + m.bar + 3 * m.pad
    x0, y1 = m.pad, panel_h - m.pad
    if box_w > panel_w - 2 * m.pad or box_h > panel_h - 2 * m.pad:
        return None
    y0 = y1 - box_h
    draw.rectangle([x0, y0, x0 + box_w, y1], fill=(255, 255, 255, 190))
    draw.text((x0 + m.pad, y0 + m.pad // 2), label, fill=(0, 0, 0), font=m.font)
    by = y0 + m.pad + m.h + m.pad // 2
    draw.rectangle([x0 + m.pad, by, x0 + m.pad + px - 1, by + m.bar - 1], fill=(0, 0, 0))
    return nice


def _draw_north(img: Image.Image, m: _Metrics, panel_w: int, panel_h: int) -> None:
    size = round(m.h * 1.6)
    if panel_w < size * 4 or panel_h < (size + m.h) * 3:
        return
    draw = ImageDraw.Draw(img, "RGBA")
    cx, top = panel_w - m.pad - size // 2 - 2, m.pad
    draw.rectangle(
        [cx - size // 2 - 2, top - 1, cx + size // 2 + 2, top + m.h + size + 3],
        fill=(255, 255, 255, 190),
    )
    draw.text((cx - _width(m.font, "N") // 2, top), "N", fill=(0, 0, 0), font=m.font)
    base = top + m.h + size
    draw.polygon(
        [(cx, top + m.h + 2), (cx - size // 3, base), (cx + size // 3, base)], fill=(0, 0, 0)
    )


def _encode(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _quantize(img: Image.Image, colors: int) -> Image.Image:
    """Quantize to `colors` entries while keeping pure black and white (text, scale bar)."""
    base = img.quantize(colors=colors - 2)
    palette = list((base.getpalette() or [])[: 3 * (colors - 2)])
    palette += [0, 0, 0, 255, 255, 255]
    palette += [0] * (768 - len(palette))
    ref = Image.new("P", (1, 1))
    ref.putpalette(palette)
    return img.quantize(palette=ref, dither=Image.Dither.NONE)


def _shrink_png(img: Image.Image) -> tuple[bytes, int]:
    """Encode; above 250 KB quantize to 64, then 32 colours, then halve once.

    Returns the bytes and the downscale factor applied (1 or 2).
    """
    data = _encode(img)
    if len(data) <= MAX_PNG_BYTES:
        return data, 1
    for colors in (64, 32):
        data = _encode(_quantize(img, colors))
        if len(data) <= MAX_PNG_BYTES:
            return data, 1
    half = img.resize((max(img.width // 2, 1), max(img.height // 2, 1)), Image.Resampling.LANCZOS)
    return _encode(_quantize(half, 32)), 2


# -- orchestration -----------------------------------------------------------------------------


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _tag(text: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "-" for c in text)[:48]


def _h6(*parts: object) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:6]


def _full_valid_fraction(valid: NDArray[np.bool_]) -> float:
    return float(valid.mean()) if valid.size else 0.0


def _text(
    head: list[str],
    arr: Any,
    valid: NDArray[np.bool_],
    classes: dict[int, str] | None,
) -> str:
    masked = np.where(valid, np.asarray(arr, dtype=np.float32), np.nan)
    first = "\n".join(head)
    room = max(MAX_TEXT_CHARS - len(first) - 1, 120)
    body = grid_with_histogram(masked, classes=classes, max_chars=min(room, 700))
    return (first + "\n" + body)[:MAX_TEXT_CHARS]


@dataclass
class _Plan:
    """What to draw: panels' builder inputs resolved from the arguments."""

    kind: str
    sources: list[_Src]
    layer_label: str
    legend: _Legend
    lo_hi: tuple[float, float] | None
    ranges: list[tuple[float, float]]
    table_name: str
    colors: dict[int, tuple[int, int, int]]
    warnings: list[str]
    text_arr: Any
    text_valid: NDArray[np.bool_]
    text_classes: dict[int, str] | None
    stretch_line: str


def _cont_plan(
    srcs: list[_Src], stretch: str, cmap: str, label: str, empty_ok: bool = True
) -> _Plan:
    values = np.concatenate([_sample_values(s.arrays[0], s.valid) for s in srcs])
    lo, hi, warn = _stretch(values, stretch)
    warnings = [warn] if warn else []
    if cmap == "diverging" and values.size:
        lim = max(abs(lo), abs(hi))
        lo, hi = -lim, lim
    lo_hi = None if values.size == 0 and warn else (lo, hi)
    src = srcs[0]
    frac = _full_valid_fraction(src.valid)
    legend = _Legend(
        label,
        [],
        gradient=(lo, hi, cmap),
        footer=f"valid {sig4(100 * frac)}%",
    )
    return _Plan(
        "cont", srcs, label, legend, lo_hi, [(lo, hi)], cmap, {}, warnings,
        src.arrays[0], src.valid, None, f"stretch {sig4(lo)}..{sig4(hi)}",
    )  # fmt: skip


def _rgb_plan(src: _Src, stretch: str) -> _Plan:
    ranges: list[tuple[float, float]] = []
    warnings: list[str] = []
    rows: list[tuple[str, str, float | None]] = []
    for name, arr, tint in zip(
        src.names, src.arrays, ("#ff0000", "#00aa00", "#0000ff"), strict=True
    ):
        lo, hi, warn = _stretch(_sample_values(arr, src.valid), stretch)
        if warn:
            warnings.append(f"{name}: {warn}")
        ranges.append((lo, hi))
        rows.append((f"{name} {sig4(lo)}..{sig4(hi)}", tint, None))
    frac = _full_valid_fraction(src.valid)
    legend = _Legend("RGB", rows, footer=f"valid {sig4(100 * frac)}%")
    lo_hi = (min(r[0] for r in ranges), max(r[1] for r in ranges))
    line = "stretch " + ", ".join(f"{sig4(lo)}..{sig4(hi)}" for lo, hi in ranges)
    return _Plan(
        "rgb", [src], ",".join(src.names), legend, lo_hi, ranges, "", {}, warnings,
        src.arrays[0], src.valid, None, line,
    )  # fmt: skip


def _cat_plan(src: _Src) -> _Plan:
    counts = _full_counts(src)
    names = src.metas[0].classes or {}
    colors = _class_colors([*names, *counts])
    legend = _class_legend(src, colors, counts, src.names[0])
    frac = _full_valid_fraction(src.valid)
    legend.footer = f"valid {sig4(100 * frac)}%"
    shown = {c: str(names.get(c, f"class {c}")) for c in sorted(counts)[:MAX_LEGEND_ROWS]}
    return _Plan(
        "cat", [src], src.names[0], legend, None, [], "categorical", colors, [],
        src.arrays[0], src.valid, shown, "classes",
    )  # fmt: skip


def _make_plan(srcs: list[_Src], names: list[str], stretch: str, colormap: str | None) -> _Plan:
    src = srcs[0]
    if len(names) == 3:
        return _rgb_plan(src, stretch)
    cmap = colormap or ("categorical" if src.metas[0].kind == "categorical" else "sequential")
    if cmap == "categorical":
        return _cat_plan(src)
    return _cont_plan(srcs, stretch, cmap, names[0])


def _panel(plan: _Plan, src: _Src, d: int) -> NDArray[np.uint8]:
    if plan.kind == "rgb":
        return _rgb_panel(src, plan.ranges, d)
    if plan.kind == "cat":
        return _cat_panel(src, plan.colors, d)
    assert plan.lo_hi is not None or src.valid.sum() == 0
    lo, hi = plan.ranges[0]
    return _cont_panel(src, lo, hi, lut(plan.table_name), d)


def _compose(
    panels: list[NDArray[np.uint8]],
    legend: _Legend | None,
    m: _Metrics,
    *,
    caption: str,
    xres: float,
    d: int,
    scale_bar: bool,
    side: int,
) -> tuple[Image.Image, float | None]:
    ih, iw = panels[0].shape[:2]
    cap_h = m.h + 2 * m.pad
    lw, lh = _legend_size(legend, m, side) if legend else (0, 0)
    width = iw * len(panels) + lw
    height = max(ih + cap_h, lh)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    bar_m: float | None = None
    for i, arr in enumerate(panels):
        tile = Image.fromarray(arr, "RGB")
        if i == 0:
            if scale_bar:
                bar_m = _draw_scale_bar(tile, m, xres, d, iw, ih)
            _draw_north(tile, m, iw, ih)
        canvas.paste(tile, (i * iw, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((m.pad, ih + m.pad), _fit(caption, m, width - 2 * m.pad), fill=(0, 0, 0), font=m.font)
    if legend:
        draw.line([(iw * len(panels), 0), (iw * len(panels), height)], fill=(200, 200, 200))
        _draw_legend(draw, legend, m, iw * len(panels), lw)
    return canvas, bar_m


def _choose_d(shape: tuple[int, int], budget_w: int, budget_h: int) -> int:
    return max(1, math.ceil(max(shape[1] / max(budget_w, 1), shape[0] / max(budget_h, 1))))


def _finish(
    stack: Stack,
    plan: _Plan,
    panels_fn: Any,
    *,
    names: list[str],
    side: int,
    npanels: int,
    legend_on: bool,
    scale_bar: bool,
    attach: bool,
    path: Path | None,
    caption_extra: str,
    identity: dict[str, Any],
    colormap: str | None,
) -> RenderResult:
    m = _metrics(side)
    leg = plan.legend if legend_on else None
    lw = _legend_size(leg, m, side)[0] if leg else 0
    cap_h = m.h + 2 * m.pad
    d = _choose_d(stack.grid.shape, (side - lw) // npanels, side - cap_h)
    panels = panels_fn(d)
    date_part = plan.stretch_line
    caption = f"{stack.id} | {plan.layer_label}{caption_extra} | {date_part}"
    canvas, bar_m = _compose(
        panels, leg, m, caption=caption, xres=stack.grid.xres, d=d, scale_bar=scale_bar, side=side
    )
    if max(canvas.size) > side:
        factor = side / max(canvas.size)
        canvas = canvas.resize(
            (max(1, round(canvas.width * factor)), max(1, round(canvas.height * factor))),
            Image.Resampling.LANCZOS,
        )
    data, shrink = _shrink_png(canvas)
    out = path or (
        stack.analysis_dir
        / "figures"
        / f"{_tag(stack.id)}-{_tag(plan.layer_label)}-{_h6(identity)}.png"
    )
    sidecar = out.with_suffix(".json")
    scale = stack.grid.xres * d * shrink
    size = Image.open(io.BytesIO(data)).size
    frac = _full_valid_fraction(plan.sources[0].valid)
    warnings = list(plan.warnings)
    if plan.sources[0].valid.size and not plan.sources[0].valid.any():
        warnings.insert(0, "no valid pixels")
    head = [f"warning: {w}" for w in warnings]
    head.append(
        f"{plan.layer_label} of {stack.id}: shown {size[0]}x{size[1]} px, "
        f"{sig4(scale)} m/px (1 in {d * shrink} pixels)"
    )
    head.append(plan.stretch_line)
    if plan.legend.rows and plan.kind != "rgb":
        head.append("; ".join(_row_text(r) for r in plan.legend.rows))
        if plan.legend.more:
            head.append(f"+{plan.legend.more} more classes")
    head.append(f"valid {sig4(100 * frac)}%")
    text = _text(head, plan.text_arr, plan.text_valid, plan.text_classes)
    _write_atomic(out, data)
    body = {
        "schema": SCHEMA,
        "stack_id": stack.id,
        "layers": names,
        "lo_hi": plan.lo_hi,
        "stretch": plan.stretch_line,
        "colormap": colormap or plan.table_name,
        "legend": [list(r) for r in plan.legend.rows],
        "size": [size[0], size[1]],
        "scale_m_per_px": scale,
        "scale_bar_m": bar_m,
        "png_bytes": len(data),
        "text": text,
    }
    _write_atomic(sidecar, (json.dumps(json_safe(body), indent=2) + "\n").encode())
    return RenderResult(
        out if attach and attach_enabled() else None,
        sidecar,
        plan.lo_hi,
        list(plan.legend.rows),
        text,
        size[0],
        size[1],
        frac,
        out,
    )


def _clamp_side(max_side: int) -> int:
    return int(min(max(max_side, 64), 1200))


def render_map(
    stack: Stack,
    layers: Sequence[str],
    *,
    composite: str | None = None,
    stretch: str = "p2-98",
    colormap: str | None = None,
    overlay: str | None = None,
    compare: Stack | str | None = None,
    legend: bool = True,
    scale_bar: bool = True,
    max_side: int = 800,
    attach: bool = True,
    path: Path | None = None,
    date: str | None = None,
) -> RenderResult:
    """Render one layer, a 3-layer composite, an overlay or a two-stack comparison to PNG.

    The PNG and a `<fig>.json` sidecar are always written; `attach` only controls
    whether `RenderResult.png` (the attachable artifact) is set.

    Args:
        stack: Source stack.
        layers: One layer, or three for an RGB composite (ignored when `composite` is set).
        composite: "truecolor", "falsecolor" or "swir".
        stretch: "p2-98", "minmax" or "lo,hi".
        colormap: "sequential", "diverging" or "categorical"; default by layer kind.
        overlay: Categorical layer in the same stack drawn as a 1-px outline, or
            `fill:<layer>` for 50% fill.
        compare: A second stack on the same grid; two panels share one stretch.
        legend: Draw the legend panel.
        scale_bar: Draw the scale bar.
        max_side: Longest output side including the legend (clamped to 64..1200).
        attach: Whether the result's `png` is attachable.
        path: Explicit PNG path; default under `analysis/figures`.
        date: YYYYMMDD resolving `name@date`.

    Returns:
        The render result.

    Raises:
        GeoError: Missing layer, bad stretch, or mismatched compare grid.
    """
    side = _clamp_side(max_side)
    names = _plan_layers(stack, layers, composite, date)
    other: Stack | None = None
    if compare is not None:
        if len(names) != 1:
            raise GeoError(
                fix(
                    "compare needs a single layer.", "render_map layers=[one layer] compare=<stack>"
                )
            )
        other = compare if not isinstance(compare, str) else _open_other(stack, compare)
        if not other.grid.same_as(stack.grid):
            raise GeoError(
                fix(
                    f"compare stack '{other.id}' is on a different grid.",
                    "prepare_stack both on the same grid",
                )
            )
        other_name = _resolve(other, names[0], None)
    srcs = [_source(stack, names)]
    if other is not None:
        srcs.append(_source(other, [other_name]))
    plan = _make_plan(srcs, names, stretch, colormap)
    ov: tuple[_Src, str, dict[int, tuple[int, int, int]], bool] | None = None
    if overlay:
        fill = overlay.startswith("fill:")
        ov_name = _resolve(stack, overlay.split(":", 1)[1] if fill else overlay, None)
        ov_src = _source(stack, [ov_name])
        counts = _full_counts(ov_src)
        colors = _class_colors([*(ov_src.metas[0].classes or {}), *counts])
        plan.legend = _class_legend(ov_src, colors, counts, f"overlay {ov_name}")
        plan.legend.footer = f"valid {sig4(100 * _full_valid_fraction(srcs[0].valid))}%"
        ov = (ov_src, ov_name, colors, fill)

    def panels(d: int) -> list[NDArray[np.uint8]]:
        out = [_panel(plan, s, d) for s in srcs]
        if ov is not None:
            codes = _codes(ov[0], d)
            (_fill if ov[3] else _outline)(out[0], codes, ov[2])
        return out

    identity = {
        "stack": stack.id, "layers": names, "stretch": stretch, "colormap": colormap,
        "overlay": overlay, "compare": other.id if other else None, "size": side, "date": date,
    }  # fmt: skip
    extra = f" vs {other.id}" if other is not None else ""
    if ov is not None:
        extra += f" + {ov[1]}"
    return _finish(
        stack, plan, panels, names=names, side=side, npanels=len(srcs), legend_on=legend,
        scale_bar=scale_bar, attach=attach, path=path, caption_extra=extra, identity=identity,
        colormap=colormap,
    )  # fmt: skip


def _base_layers(product: Stack, base: Stack, base_layers: Sequence[str]) -> list[str]:
    """Pick 3 RGB layers (explicit, else red/green/blue), else one reflectance layer, else none."""
    if base_layers:
        return [_resolve(base, n, None) for n in base_layers]
    have = base.names()
    if all(n in have for n in _COMPOSITES["truecolor"]):
        return list(_COMPOSITES["truecolor"])
    for meta in base.meta.layers:
        if meta.kind == "reflectance":
            return [meta.name]
    return []


def render_class_overlay(
    product: Stack,
    layer: str,
    *,
    base: Stack | None = None,
    base_layers: Sequence[str] = (),
    max_side: int = 800,
    attach: bool = True,
    path: Path | None = None,
) -> RenderResult:
    """Draw a class layer as 1-px outlines over an RGB (or grey) base image.

    Legend rows carry hectares counted over the valid pixels at full resolution.

    Args:
        product: Stack holding the categorical `layer`.
        layer: Class layer name.
        base: Stack supplying the base image; default the product itself.
        base_layers: Base layers (1 or 3); default red/green/blue, else the first reflectance.
        max_side: Longest output side (clamped 64..1200).
        attach: Whether the result's `png` is attachable.
        path: Explicit PNG path.

    Returns:
        The render result.

    Raises:
        GeoError: Missing layer or a base on a different grid.
    """
    side = _clamp_side(max_side)
    cls_name = _resolve(product, layer, None)
    base_stack = base or product
    if not base_stack.grid.same_as(product.grid):
        raise GeoError(fix("base stack is on a different grid.", "prepare_stack both on one grid"))
    cls_src = _source(product, [cls_name])
    counts = _full_counts(cls_src)
    colors = _class_colors([*(cls_src.metas[0].classes or {}), *counts])
    bnames = _base_layers(product, base_stack, base_layers)
    base_src = _source(base_stack, bnames) if bnames else None
    if base_src is not None and len(bnames) == 3:
        plan = _rgb_plan(base_src, "p2-98")
    elif base_src is not None and len(bnames) == 1:
        plan = _cont_plan([base_src], "p2-98", "sequential", bnames[0])
    else:
        plan = _cat_plan(cls_src)
    plan.legend = _class_legend(cls_src, colors, counts, cls_name)
    plan.legend.footer = f"valid {sig4(100 * _full_valid_fraction(cls_src.valid))}%"
    plan.layer_label = cls_name
    plan.text_arr, plan.text_valid = cls_src.arrays[0], cls_src.valid
    plan.text_classes = {
        c: str((cls_src.metas[0].classes or {}).get(c, f"class {c}"))
        for c in sorted(counts)[:MAX_LEGEND_ROWS]
    }
    plan.stretch_line = f"base {','.join(bnames) or 'grey'}"

    def panels(d: int) -> list[NDArray[np.uint8]]:
        if base_src is not None and len(bnames) == 3:
            rgb = _rgb_panel(base_src, plan.ranges, d)
        elif base_src is not None:
            gray = _cont_panel(base_src, *plan.ranges[0], lut("sequential"), d)
            lum = gray.mean(axis=2, keepdims=True)
            rgb = np.repeat(lum, 3, axis=2).astype(np.uint8)
        else:
            shape = _codes(cls_src, d).shape
            rgb = np.full((*shape, 3), 200, dtype=np.uint8)
        _outline(rgb, _codes(cls_src, d), colors)
        return [rgb]

    identity = {"stack": product.id, "layer": cls_name, "base": bnames, "size": side}
    return _finish(
        product, plan, panels, names=[cls_name], side=side, npanels=1, legend_on=True,
        scale_bar=True, attach=attach, path=path, caption_extra="", identity=identity,
        colormap="categorical",
    )  # fmt: skip
