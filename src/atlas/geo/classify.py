"""Threshold, rule and Otsu classification of one continuous layer.

Invariants: intervals are half-open (``lo <= v < hi``); only pixels in ``valid`` receive a
code; nothing here reads files or executes caller-supplied code.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from atlas.geo.errors import GeoError, fix

__all__ = ["PRESETS", "ClassifyOutput", "classify_layer", "otsu_threshold"]

PRESETS: dict[str, tuple[tuple[float, ...], tuple[str, ...]]] = {
    "ndvi": ((0.2, 0.5), ("bare", "sparse_veg", "dense_veg")),
    "mndwi_water": ((0.0,), ("land", "water")),
    "dnbr": (
        (-0.10, 0.10, 0.27, 0.44, 0.66),
        ("regrowth", "unburned", "low", "moderate_low", "moderate_high", "high"),
    ),
    "sar_db_water": ((-15.0,), ("water", "land")),
}
PRESET_UNITS = {"sar_db_water": "dB"}
_FIX = "classify stack=S layer=ndvi preset=ndvi"


@dataclass(frozen=True)
class ClassifyOutput:
    """Codes (float32, NaN where invalid), their names and what was used."""

    data: NDArray[np.float32]
    classes: dict[int, str]
    threshold: float | None = None
    notes: list[str] = field(default_factory=list)


def otsu_threshold(values: NDArray[Any], bins: int = 256) -> float:
    """Return the Otsu threshold of the finite values.

    Args:
        values: Any-shaped array; non-finite entries are ignored.
        bins: Histogram bins over ``[min, max]``.

    Returns:
        The bin edge that maximises between-class variance (the middle of a flat optimum).

    Raises:
        GeoError: Fewer than two distinct finite values.
    """
    flat = np.asarray(values, dtype=np.float64).ravel()
    flat = flat[np.isfinite(flat)]
    if flat.size == 0 or float(flat.min()) == float(flat.max()):
        raise GeoError(
            fix("otsu needs at least two distinct valid values.", "use thresholds=[...] instead")
        )
    hist, edges = np.histogram(flat, bins=bins, range=(float(flat.min()), float(flat.max())))
    weight = hist.astype(np.float64)
    total = weight.sum()
    centres = (edges[:-1] + edges[1:]) / 2.0
    w0 = np.cumsum(weight)
    w1 = total - w0
    s0 = np.cumsum(weight * centres)
    mu0 = np.divide(s0, w0, out=np.zeros_like(s0), where=w0 > 0)
    mu1 = np.divide(s0[-1] - s0, w1, out=np.zeros_like(s0), where=w1 > 0)
    between = w0 * w1 * (mu0 - mu1) ** 2
    between[(w0 == 0) | (w1 == 0)] = -1.0
    best = np.flatnonzero(between >= between.max() * (1.0 - 1e-12))
    return float((edges[int(best[0]) + 1] + edges[int(best[-1]) + 1]) / 2.0)


def _choose(
    rules: Sequence[Mapping[str, Any]] | None,
    thresholds: Sequence[float] | None,
    preset: str | None,
    method: str | None,
) -> str:
    given = [
        k
        for k, v in (
            ("rules", rules),
            ("thresholds", thresholds),
            ("preset", preset),
            ("method", method),
        )
        if v
    ]
    if len(given) != 1:
        raise GeoError(
            fix(
                f"give exactly one of rules, thresholds, preset or method (got {given or 'none'}).",
                _FIX,
            )
        )
    if given[0] == "method" and method != "otsu":
        raise GeoError(fix(f"unknown method {method!r}.", "method=otsu"))
    if given[0] == "preset" and preset not in PRESETS:
        raise GeoError(fix(f"unknown preset {preset!r}.", f"use one of {sorted(PRESETS)}"))
    return given[0]


def _bounds(rule: Mapping[str, Any]) -> tuple[float, float]:
    lo, hi = rule.get("min"), rule.get("max")
    return (-math.inf if lo is None else float(lo)), (math.inf if hi is None else float(hi))


def _by_rules(
    values: NDArray[np.float32], ok: NDArray[np.bool_], rules: Sequence[Mapping[str, Any]]
) -> tuple[NDArray[np.float32], dict[int, str]]:
    codes = np.zeros(values.shape, dtype=np.float32)
    done = np.zeros(values.shape, dtype=bool)
    names: dict[int, str] = {}
    for i, rule in enumerate(rules, start=1):
        lo, hi = _bounds(rule)
        hit = ok & ~done & (values >= lo) & (values < hi)
        codes[hit] = i
        done |= hit
        names[i] = str(rule.get("name") or f"class_{i}")
    if int((ok & ~done).sum()) > 0:
        names[0] = "other"
    return codes, names


def _by_thresholds(
    values: NDArray[np.float32], thresholds: Sequence[float], labels: Sequence[str] | None
) -> tuple[NDArray[np.float32], dict[int, str]]:
    cuts = [float(t) for t in thresholds]
    if any(b <= a for a, b in itertools.pairwise(cuts)):
        raise GeoError(fix("thresholds must be strictly ascending.", "thresholds=[0.2, 0.5]"))
    names = list(labels) if labels else [f"class_{i + 1}" for i in range(len(cuts) + 1)]
    if len(names) != len(cuts) + 1:
        raise GeoError(
            fix(
                f"{len(cuts)} thresholds need {len(cuts) + 1} labels, got {len(names)}.",
                f"pass labels with {len(cuts) + 1} names",
            )
        )
    codes = (np.digitize(values, cuts) + 1).astype(np.float32)
    return codes, {i + 1: n for i, n in enumerate(names)}


def classify_layer(
    band: NDArray[np.float32],
    valid: NDArray[np.bool_],
    *,
    rules: Sequence[Mapping[str, Any]] | None = None,
    thresholds: Sequence[float] | None = None,
    labels: Sequence[str] | None = None,
    preset: str | None = None,
    method: str | None = None,
) -> ClassifyOutput:
    """Classify a continuous layer into integer codes.

    Exactly one of ``rules``, ``thresholds``, ``preset`` or ``method="otsu"`` is used. Rules
    give codes 1..n (first match wins); valid pixels matching none get code 0 "other".
    Thresholds give ``digitize + 1`` codes.

    Args:
        band: (H, W) values.
        valid: (H, W) usable-pixel mask; others become NaN.
        rules: ``[{name, min, max}]``, bounds optional.
        thresholds: Ascending cut points.
        labels: One more name than ``thresholds``.
        preset: A key of ``PRESETS``.
        method: ``"otsu"`` for a two-class data-driven split.

    Returns:
        The codes, class names, the threshold (single split only) and notes.

    Raises:
        GeoError: Not exactly one scheme, bad thresholds, or Otsu on a constant layer.
    """
    kind = _choose(rules, thresholds, preset, method)
    values = np.asarray(band, dtype=np.float32)
    ok = np.asarray(valid, dtype=bool) & np.isfinite(values)
    threshold: float | None = None
    notes: list[str] = []
    if kind == "rules":
        assert rules is not None
        codes, names = _by_rules(values, ok, rules)
    else:
        if kind == "preset":
            assert preset is not None
            cuts, preset_labels = PRESETS[preset]
            use, use_labels = list(cuts), list(preset_labels)
        elif kind == "method":
            cut = otsu_threshold(values[ok])
            use, use_labels = [cut], ["low", "high"]
            notes.append("otsu assumes two modes; check the histogram")
        else:
            assert thresholds is not None
            use, use_labels = list(thresholds), list(labels) if labels else []
        codes, names = _by_thresholds(values, use, use_labels or None)
        threshold = float(use[0]) if len(use) == 1 else None
    out = np.where(ok, codes, np.nan).astype(np.float32)
    return ClassifyOutput(out, names, threshold, notes)
