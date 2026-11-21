"""Small Pillow charts: line, bar and histogram PNGs.

Invariants: Pillow and numpy only; callers own sidecars; a figure never exceeds 900x500.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from atlas.geo import render as _render
from atlas.geo.errors import GeoError, fix
from atlas.geo.results import sig4

__all__ = ["bar_chart", "histogram", "line_chart"]

MAX_SIZE = (900, 500)
_LABEL_MAX = 24
_GRID = (225, 225, 225)
_AXIS = (60, 60, 60)
_INK = (0, 0, 0)


def _size(size: tuple[int, int]) -> tuple[int, int]:
    return (
        int(min(max(size[0], 240), MAX_SIZE[0])),
        int(min(max(size[1], 160), MAX_SIZE[1])),
    )


def _nice_ticks(lo: float, hi: float, target: int = 5) -> list[float]:
    """Ticks at 1, 2 or 5 times a power of ten covering `[lo, hi]`."""
    if not math.isfinite(lo) or not math.isfinite(hi):
        return [0.0, 1.0]
    if hi <= lo:
        hi = lo + (abs(lo) * 0.1 or 1.0)
    raw = (hi - lo) / max(target, 1)
    mag = 10.0 ** math.floor(math.log10(raw))
    step = next(n * mag for n in (1, 2, 5, 10) if n * mag >= raw)
    first = math.floor(lo / step + 1e-9) * step
    ticks = []
    value = first
    while value <= hi + step * 1e-9:
        ticks.append(round(value, 12))
        value += step
    return ticks


def _truncate(label: str) -> str:
    return label if len(label) <= _LABEL_MAX else label[: _LABEL_MAX - 2] + ".."


def _rotated(text: str, m: Any, angle: int) -> Image.Image:
    box = Image.new("RGBA", (_render._width(m.font, text) + 4, m.h + m.pad * 2), (0, 0, 0, 0))
    ImageDraw.Draw(box).text((2, 0), text, fill=_INK, font=m.font)
    return box.rotate(angle, expand=True)


class _Frame:
    """Axes box, grid, ticks, title and y label over a white canvas."""

    def __init__(self, size: tuple[int, int], title: str, ylabel: str, ylim: tuple[float, float]):
        self.w, self.h = _size(size)
        self.m = _render._metrics(self.w)
        self.img = Image.new("RGB", (self.w, self.h), (255, 255, 255))
        self.draw = ImageDraw.Draw(self.img)
        self.yticks = _nice_ticks(*ylim)
        self.lo, self.hi = min(ylim[0], self.yticks[0]), max(ylim[1], self.yticks[-1])
        if self.hi <= self.lo:
            self.hi = self.lo + 1.0
        m = self.m
        ylabels = [sig4(t) for t in self.yticks]
        label_w = max(_render._width(m.font, t) for t in ylabels)
        self.left = label_w + (m.h + 3 * m.pad if ylabel else 2 * m.pad) + m.pad
        self.right = self.w - 2 * m.pad
        self.top = m.h + 3 * m.pad
        self.bottom = self.h - (m.h + 3 * m.pad)
        self.draw.text(
            (m.pad, m.pad), _render._fit(title, m, self.w - 2 * m.pad), fill=_INK, font=m.font
        )
        if ylabel:
            tag = _rotated(_render._fit(ylabel, m, self.bottom - self.top), m, 90)
            self.img.paste(tag, (m.pad, (self.top + self.bottom - tag.height) // 2), tag)
        for tick, text in zip(self.yticks, ylabels, strict=True):
            y = self.y(tick)
            self.draw.line([(self.left, y), (self.right, y)], fill=_GRID)
            tw = _render._width(m.font, text)
            self.draw.text((self.left - tw - m.pad, y - m.h // 2), text, fill=_INK, font=m.font)
        self.draw.line([(self.left, self.top), (self.left, self.bottom)], fill=_AXIS)
        self.draw.line([(self.left, self.bottom), (self.right, self.bottom)], fill=_AXIS)

    def y(self, value: float) -> float:
        frac = (value - self.lo) / (self.hi - self.lo)
        return self.bottom - frac * (self.bottom - self.top)

    def xlabel(self, text: str, cx: float, *, rotate: bool = False) -> None:
        m = self.m
        if rotate:
            tag = _rotated(_truncate(text), m, 90)
            self.img.paste(tag, (round(cx - tag.width / 2), self.bottom + 2), tag)
        else:
            tw = _render._width(m.font, text)
            x0 = min(max(cx - tw / 2, 0), self.w - tw - 1)
            self.draw.text((x0, self.bottom + m.pad), text, fill=_INK, font=m.font)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        data, _ = _render._shrink_png(self.img)
        _render._write_atomic(path, data)
        return path


def _color(i: int) -> tuple[int, int, int]:
    return _render.OKABE_ITO[i % len(_render.OKABE_ITO)]


def _floats(values: Sequence[Any]) -> list[float]:
    out = []
    for v in values:
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            out.append(math.nan)
    return out


def _span(arrays: Sequence[Sequence[float]]) -> tuple[float, float]:
    flat = np.asarray([v for a in arrays for v in a], dtype=np.float64)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return 0.0, 1.0
    return float(flat.min()), float(flat.max())


def _x_positions(x: Sequence[Any]) -> tuple[list[float], list[str]]:
    nums = _floats(x)
    if all(math.isfinite(v) for v in nums) and not any(isinstance(v, str) for v in x):
        return nums, [sig4(v) for v in nums]
    return [float(i) for i in range(len(x))], [str(v) for v in x]


def _trend_line(xs: list[float], ys: list[float]) -> tuple[float, float] | None:
    pts = [(a, b) for a, b in zip(xs, ys, strict=True) if math.isfinite(a) and math.isfinite(b)]
    if len(pts) < 2:
        return None
    arr = np.asarray(pts)
    if float(arr[:, 0].max() - arr[:, 0].min()) == 0.0:
        return None
    slope, icpt = np.polyfit(arr[:, 0], arr[:, 1], 1)
    return float(slope), float(icpt)


def line_chart(
    x: Sequence[Any],
    series: Mapping[str, Sequence[float]],
    *,
    title: str,
    ylabel: str = "",
    path: Path,
    size: tuple[int, int] = (800, 450),
    trend: str | None = None,
) -> Path:
    """Draw one or more series against x; NaN breaks a line; markers when <= 12 points.

    Args:
        x: Numbers or labels (dates); labels are spaced by index.
        series: Name to values, same length as `x`.
        title: Chart title.
        ylabel: Y axis label.
        path: Output PNG path.
        size: Requested size, clamped to 900x500.
        trend: "linear" draws a dashed least-squares line per series.

    Returns:
        The PNG path.

    Raises:
        GeoError: Empty or ragged input, or an unknown trend.
    """
    if trend not in (None, "linear"):
        raise GeoError(fix(f"unknown trend {trend!r}.", 'use trend="linear" or omit it'))
    if not series or not len(x):
        raise GeoError(fix("line_chart needs data.", "pass x and at least one series"))
    data = {k: _floats(v) for k, v in series.items()}
    if any(len(v) != len(x) for v in data.values()):
        raise GeoError(fix("series length differs from x.", "pass series as long as x"))
    xs, xlabels = _x_positions(x)
    lo, hi = _span(list(data.values()))
    frame = _Frame(size, title, ylabel, (lo, hi))
    xmin, xmax = min(xs), max(xs)
    xmax = xmax if xmax > xmin else xmin + 1.0
    span = frame.right - frame.left - 2 * frame.m.pad

    def px(v: float) -> float:
        return frame.left + frame.m.pad + (v - xmin) / (xmax - xmin) * span

    ticks = max(
        2, min(len(xs), span // max(_render._width(frame.m.font, max(xlabels, key=len)) + 8, 1))
    )
    for k in sorted({round(i * (len(xs) - 1) / (ticks - 1)) for i in range(int(ticks))}):
        frame.xlabel(_truncate(xlabels[k]), px(xs[k]))
    for i, ys in enumerate(data.values()):
        color = _color(i)
        run: list[tuple[float, float]] = []
        for a, b in [*zip(xs, ys, strict=True), (math.nan, math.nan)]:
            if math.isfinite(b) and math.isfinite(a):
                run.append((px(a), frame.y(b)))
                continue
            if len(run) > 1:
                frame.draw.line(run, fill=color, width=2)
            run = []
        if len(xs) <= 12:
            for a, b in zip(xs, ys, strict=True):
                if math.isfinite(b):
                    cx, cy = px(a), frame.y(b)
                    frame.draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=color)
        fit = _trend_line(xs, ys) if trend == "linear" else None
        if fit is not None:
            slope, icpt = fit
            p0, p1 = (
                (px(xmin), frame.y(slope * xmin + icpt)),
                (px(xmax), frame.y(slope * xmax + icpt)),
            )
            for t in range(0, 20, 2):
                seg0 = (p0[0] + (p1[0] - p0[0]) * t / 20, p0[1] + (p1[1] - p0[1]) * t / 20)
                seg1 = (
                    p0[0] + (p1[0] - p0[0]) * (t + 1) / 20,
                    p0[1] + (p1[1] - p0[1]) * (t + 1) / 20,
                )
                frame.draw.line([seg0, seg1], fill=color, width=1)
    if len(data) > 1:
        y = frame.top + frame.m.pad
        for i, name in enumerate(data):
            text = _truncate(name)
            x0 = frame.right - _render._width(frame.m.font, text) - frame.m.swatch - 3 * frame.m.pad
            frame.draw.rectangle(
                [x0, y, x0 + frame.m.swatch - 1, y + frame.m.swatch - 1], fill=_color(i)
            )
            frame.draw.text(
                (x0 + frame.m.swatch + frame.m.pad, y), text, fill=_INK, font=frame.m.font
            )
            y += frame.m.line
    return frame.save(path)


def bar_chart(
    labels: Sequence[str],
    values: Sequence[float],
    *,
    title: str,
    ylabel: str = "",
    path: Path,
    size: tuple[int, int] = (800, 450),
    colors: Sequence[str] | None = None,
) -> Path:
    """Draw one bar per label (labels cut to 24 characters with "..").

    Args:
        labels: Category labels.
        values: Bar heights; NaN bars are skipped.
        title: Chart title.
        ylabel: Y axis label.
        path: Output PNG path.
        size: Requested size, clamped to 900x500.
        colors: Optional "#rrggbb" per bar; default the Okabe-Ito cycle.

    Returns:
        The PNG path.

    Raises:
        GeoError: Empty or ragged input.
    """
    ys = _floats(values)
    if not ys or len(labels) != len(ys):
        raise GeoError(fix("bar_chart needs equal-length labels and values.", "pass both lists"))
    lo, hi = _span([ys])
    frame = _Frame(size, title, ylabel, (min(lo, 0.0), max(hi, 0.0)))
    slot = (frame.right - frame.left) / len(ys)
    base = frame.y(0.0)
    for i, (label, v) in enumerate(zip(labels, ys, strict=True)):
        cx = frame.left + slot * (i + 0.5)
        if math.isfinite(v):
            rgb: tuple[int, ...] = _color(i)
            if colors is not None and i < len(colors) and len(colors[i]) == 7:
                rgb = tuple(int(colors[i][k : k + 2], 16) for k in (1, 3, 5))
            x0, x1 = cx - slot * 0.35, cx + slot * 0.35
            frame.draw.rectangle([x0, min(base, frame.y(v)), x1, max(base, frame.y(v))], fill=rgb)
        text = _truncate(label)
        fits = _render._width(frame.m.font, text) <= slot - 2
        frame.xlabel(text, cx, rotate=not fits)
    return frame.save(path)


def histogram(
    values: Sequence[float],
    *,
    bins: int = 20,
    title: str,
    xlabel: str = "",
    path: Path,
    size: tuple[int, int] = (800, 450),
) -> Path:
    """Draw a histogram of the finite values.

    Args:
        values: Samples; NaN and inf are dropped.
        bins: Number of bins (1..200).
        title: Chart title.
        xlabel: X axis label (drawn under the axis).
        path: Output PNG path.
        size: Requested size, clamped to 900x500.

    Returns:
        The PNG path.

    Raises:
        GeoError: No finite values.
    """
    arr = np.asarray(_floats(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise GeoError(fix("histogram has no finite values.", "pass finite numbers"))
    counts, edges = np.histogram(arr, bins=int(min(max(bins, 1), 200)))
    frame = _Frame(size, title, "count", (0.0, float(counts.max())))
    xmin, xmax = float(edges[0]), float(edges[-1])
    xmax = xmax if xmax > xmin else xmin + 1.0
    width = frame.right - frame.left
    for i, n in enumerate(counts):
        x0 = frame.left + (float(edges[i]) - xmin) / (xmax - xmin) * width
        x1 = frame.left + (float(edges[i + 1]) - xmin) / (xmax - xmin) * width
        frame.draw.rectangle([x0, frame.y(float(n)), max(x1 - 1, x0), frame.bottom], fill=_color(4))
    for tick in _nice_ticks(xmin, xmax, 6):
        if xmin <= tick <= xmax:
            frame.xlabel(sig4(tick), frame.left + (tick - xmin) / (xmax - xmin) * width)
    if xlabel:
        tw = _render._width(frame.m.font, xlabel)
        frame.draw.text(
            ((frame.left + frame.right - tw) / 2, frame.h - frame.m.h - frame.m.pad // 2),
            xlabel, fill=_INK, font=frame.m.font,
        )  # fmt: skip
    return frame.save(path)
