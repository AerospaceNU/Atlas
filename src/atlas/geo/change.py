"""Change detection between two dates of one layer: difference, ratio, dB ratio, transitions.

Invariants: only pixels valid in both dates count; small blobs are removed per class
(8-connected) and reported, never silently; no raster file is read here beyond ``Stack``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from atlas.geo.classify import classify_layer
from atlas.geo.errors import GeoError, GridMismatch, fix
from atlas.geo.stack import Stack
from atlas.geo.zonal import class_rows

__all__ = ["METHODS", "ChangeOutput", "Pair", "detect_change", "pair_valid", "resolve_pair"]

METHODS = ("difference", "ratio", "log_ratio_db", "transitions")
_HINT = {"difference": "0.1", "ratio": "1.5", "log_ratio_db": "3.0"}
_MAX_CODES = 512
_TOP = 12


@dataclass(frozen=True)
class Pair:
    """Two layer views on one grid: which stack and layer name hold each date."""

    pre_stack: Stack
    post_stack: Stack
    pre_layer: str
    post_layer: str
    pre_label: str
    post_label: str
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ChangeOutput:
    """Change layers and tables.

    Attributes:
        value: Continuous change (None for transitions).
        classes: Codes (float32, NaN invalid): 0 stable, 1 gain, 2 loss (rules: 0 stable,
            i+1 rule i; transitions: 0 unchanged, 1 changed).
        class_names: Code to name.
        rows: Class rows ``{class, px, ha, pct_valid}``, or top transitions.
        matrix: Transition pixel counts (K x K), else None.
        labels: Transition class names, row/column order.
        dropped_blobs: Components removed by ``min_blob_px``.
        dropped_px: Pixels in those components.
        valid_px: Pixels valid in both dates (after dropping undefined ratios).
        notes: Warnings and remarks.
    """

    value: NDArray[np.float32] | None
    classes: NDArray[np.float32]
    class_names: dict[int, str]
    rows: list[dict[str, Any]]
    matrix: NDArray[np.int64] | None
    labels: list[str] | None
    dropped_blobs: int
    dropped_px: int
    valid_px: int
    notes: list[str]


def _date_key(value: str) -> str:
    return value.replace("-", "")


def _dated(stack: Stack, layer: str) -> list[str]:
    found: list[str] = []
    for date in stack.dates():
        try:
            stack.layer(layer, date=date)
        except GeoError:
            continue
        found.append(date)
    return found


def resolve_pair(
    stack: Stack,
    layer: str,
    *,
    stack_b: Stack | None = None,
    pre: str | None = None,
    post: str | None = None,
) -> Pair:
    """Find the pre and post layers, from one multi-date stack or from two stacks.

    Args:
        stack: The (first) stack.
        layer: Layer base name present at both dates.
        stack_b: Second stack holding the post layer; must share the grid.
        pre: Pre date (YYYYMMDD); default earliest.
        post: Post date; default latest.

    Raises:
        GeoError: Fewer than two dates, unknown dates, or the same date twice.
        GridMismatch: The stacks are on different grids.
    """
    if stack_b is not None:
        if not stack.grid.same_as(stack_b.grid):
            a, b = stack.grid, stack_b.grid
            raise GridMismatch(
                fix(
                    f"stack_b is on a different grid ({b.crs} {b.width}x{b.height} vs "
                    f"{a.crs} {a.width}x{a.height}).",
                    "prepare_stack both with the same crs, resolution and bbox",
                )
            )
        first = stack.layer(layer, date=_date_key(pre) if pre else None)
        second = stack_b.layer(layer, date=_date_key(post) if post else None)
        return Pair(
            stack,
            stack_b,
            first.name,
            second.name,
            f"{stack.id}:{first.name}",
            f"{stack_b.id}:{second.name}",
        )
    dates = _dated(stack, layer)
    if len(dates) < 2:
        raise GeoError(
            fix(
                f"stack '{stack.id}' has {len(dates)} date(s) for {layer!r}; change needs two.",
                "prepare_stack scenes=[pre_scene, post_scene], or pass stack_b",
            )
        )
    d_pre = _date_key(pre) if pre else dates[0]
    d_post = _date_key(post) if post else dates[-1]
    for d in (d_pre, d_post):
        if d not in dates:
            raise GeoError(fix(f"no {layer!r} layer on date {d}.", f"use pre/post from {dates}"))
    if d_pre == d_post:
        raise GeoError(fix("pre and post are the same date.", f"use two of {dates}"))
    notes = []
    if stack.meta.qa:
        notes.append("QA planes are stack-wide; per-date validity is finite pixels plus the planes")
    return Pair(
        stack,
        stack,
        stack.layer(layer, date=d_pre).name,
        stack.layer(layer, date=d_post).name,
        d_pre,
        d_post,
        notes,
    )


def pair_valid(
    pair: Pair, mask: str | Sequence[str] = "clear", extra: Sequence[Stack] = ()
) -> NDArray[np.bool_]:
    """Return pixels valid in both dates (the intersection of each date's ``Stack.valid``)."""
    first = pair.pre_stack.valid(mask, layers=[pair.pre_layer], extra=extra)
    second = pair.post_stack.valid(mask, layers=[pair.post_layer], extra=extra)
    return np.asarray(first & second, dtype=bool)


def _check_delta(
    method: str, delta: float | None, rules: Sequence[Mapping[str, Any]] | None
) -> None:
    if rules:
        return
    if delta is None:
        raise GeoError(
            fix(
                f"{method} needs delta (or rules).",
                f"change_detect delta={_HINT.get(method, '0.1')} (or rules=[...])",
            )
        )
    if method == "ratio" and not delta > 1.0:
        raise GeoError(fix(f"ratio delta is a factor above 1, got {delta}.", "delta=1.5"))
    if method != "ratio" and not delta > 0.0:
        raise GeoError(fix(f"delta must be positive, got {delta}.", "delta=0.1"))


def _value(
    pre: NDArray[np.float32], post: NDArray[np.float32], method: str
) -> tuple[NDArray[np.float32], int]:
    """Return the change value and the count of pixels made undefined (ratio, dB)."""
    a, b = pre.astype(np.float64), post.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        if method == "difference":
            return (b - a).astype(np.float32), 0
        if method == "ratio":
            bad = a == 0
            out = np.where(bad, np.nan, b / np.where(bad, 1.0, a))
        else:
            bad = (a <= 0) | (b <= 0)
            out = np.where(
                bad, np.nan, 10.0 * np.log10(np.where(bad, 1.0, b) / np.where(bad, 1.0, a))
            )
    undefined = int(np.count_nonzero(bad & np.isfinite(a) & np.isfinite(b)))
    return out.astype(np.float32), undefined


def _label_codes(
    value: NDArray[np.float32],
    ok: NDArray[np.bool_],
    method: str,
    delta: float | None,
    rules: Sequence[Mapping[str, Any]] | None,
) -> tuple[NDArray[np.float32], dict[int, str]]:
    if rules:
        out = classify_layer(value, ok, rules=rules)
        names = {k: v for k, v in out.classes.items() if k != 0}
        names[0] = "stable"
        return np.nan_to_num(out.data, nan=0.0), names
    assert delta is not None
    if method == "ratio":
        loss_at, gain_at = 1.0 / delta, delta
    else:
        loss_at, gain_at = -delta, delta
    codes = np.zeros(value.shape, dtype=np.float32)
    codes[ok & (value >= gain_at)] = 1
    codes[ok & (value <= loss_at)] = 2
    return codes, {0: "stable", 1: "gain", 2: "loss"}


def _drop_small(
    codes: NDArray[np.float32], min_px: int, names: Mapping[int, str]
) -> tuple[int, int]:
    """Reset connected components smaller than ``min_px`` to 0 in place; return counts."""
    blobs = dropped = 0
    structure = np.ones((3, 3), dtype=bool)
    for code in (c for c in names if c != 0):
        labelled, n = ndimage.label(codes == code, structure=structure)
        if n == 0:
            continue
        sizes = np.bincount(labelled.ravel())
        small = np.flatnonzero(sizes < min_px)
        small = small[small != 0]
        if small.size:
            codes[np.isin(labelled, small)] = 0
            blobs += int(small.size)
            dropped += int(sizes[small].sum())
    return blobs, dropped


def _transitions(
    pre: NDArray[np.float32],
    post: NDArray[np.float32],
    ok: NDArray[np.bool_],
    px_area_ha: float,
    names: Mapping[int, str] | None,
) -> ChangeOutput:
    a, b = pre[ok].astype(np.int64), post[ok].astype(np.int64)
    union = np.unique(np.concatenate([a, b]))
    k = int(union.size)
    if k > _MAX_CODES:
        raise GeoError(
            fix(
                f"layer has {k} distinct codes; transitions needs a categorical layer.",
                "classify it first",
            )
        )
    flat = np.bincount(np.searchsorted(union, a) * k + np.searchsorted(union, b), minlength=k * k)
    matrix = flat.reshape(k, k).astype(np.int64)
    n_valid = int(ok.sum())
    assert int(matrix.sum()) == n_valid
    labels = [(names or {}).get(int(c), str(int(c))) for c in union]
    changed = np.full(pre.shape, np.nan, dtype=np.float32)
    changed[ok] = (pre[ok] != post[ok]).astype(np.float32)
    order = np.argsort(-matrix, axis=None, kind="stable")
    rows = [
        {
            "from": labels[int(i) // k],
            "to": labels[int(i) % k],
            "px": int(matrix.flat[i]),
            "ha": int(matrix.flat[i]) * px_area_ha,
        }
        for i in order[:_TOP]
        if matrix.flat[i] > 0
    ]
    return ChangeOutput(
        None, changed, {0: "unchanged", 1: "changed"}, rows, matrix, labels, 0, 0, n_valid, []
    )


def detect_change(
    pre: NDArray[np.float32],
    post: NDArray[np.float32],
    valid: NDArray[np.bool_],
    *,
    method: str,
    pixel_area_m2: float,
    delta: float | None = None,
    rules: Sequence[Mapping[str, Any]] | None = None,
    min_blob_px: int = 0,
    categorical_names: Mapping[int, str] | None = None,
) -> ChangeOutput:
    """Compare two arrays over pixels valid in both dates.

    Args:
        pre: Earlier values (H, W).
        post: Later values.
        valid: Intersected validity of both dates.
        method: One of ``METHODS``.
        pixel_area_m2: Pixel area for hectares.
        delta: Threshold: difference and dB magnitude, or ratio factor above 1.
        rules: ``[{name, min, max}]`` on the change value instead of ``delta``.
        min_blob_px: Remove gain/loss components smaller than this many pixels.
        categorical_names: Code names for transitions.

    Returns:
        A ``ChangeOutput``.

    Raises:
        GeoError: Unknown method, missing delta, or too many categorical codes.
    """
    if method not in METHODS:
        raise GeoError(fix(f"unknown method {method!r}.", f"use one of {list(METHODS)}"))
    area_ha = pixel_area_m2 / 1e4
    ok = np.asarray(valid, dtype=bool) & np.isfinite(pre) & np.isfinite(post)
    if method == "transitions":
        return _transitions(pre, post, ok, area_ha, categorical_names)
    _check_delta(method, delta, rules)
    value, undefined = _value(pre, post, method)
    notes: list[str] = []
    if undefined:
        what = "pre == 0" if method == "ratio" else "a value <= 0"
        notes.append(f"{undefined} px dropped as nodata ({what} in the {method} formula)")
    ok &= np.isfinite(value)
    codes, names = _label_codes(value, ok, method, delta, rules)
    blobs = dropped = 0
    if min_blob_px > 1:
        blobs, dropped = _drop_small(codes, min_blob_px, names)
    out = np.where(ok, codes, np.nan).astype(np.float32)
    shown = np.where(ok, value, np.nan).astype(np.float32)
    rows = class_rows(out, ok, names, area_ha)
    by_code = {int(r["code"]): r for r in rows}
    ordered = [
        {
            "class": names[c],
            "px": by_code[c]["px"] if c in by_code else 0,
            "ha": by_code[c]["ha"] if c in by_code else 0.0,
            "pct_valid": by_code[c]["pct_valid"] if c in by_code else 0.0,
        }
        for c in sorted(names, key=lambda c: (c == 0, c))
    ]
    return ChangeOutput(
        shown, out, names, ordered, None, None, blobs, dropped, int(ok.sum()), notes
    )
