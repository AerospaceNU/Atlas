"""Layer statistics over valid pixels, histograms and QA fractions.

Invariant: every number shown to the model is formatted by ``results.sig4``; nothing here
stacks a whole cube, layers are read one at a time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from atlas.geo.stack import Stack

__all__ = ["LayerStats", "histogram", "layer_stats", "qa_fractions", "stats_row"]

SUBSAMPLE_ABOVE = 4_000_000


@dataclass
class LayerStats:
    """Summary of one layer over its valid pixels."""

    name: str
    unit: str
    n_valid: int
    n_total: int
    valid_pct: float
    min: float | None
    p2: float | None
    median: float | None
    p98: float | None
    max: float | None
    mean: float | None
    sampled: bool = False


def layer_stats(
    arr: NDArray[Any], valid: NDArray[np.bool_] | None, *, name: str, unit: str
) -> LayerStats:
    """Summarise ``arr[valid & isfinite]``; an empty selection gives None statistics.

    Above 4e6 valid pixels percentiles and median use a deterministic stride subsample
    (``sampled`` is set); min, max and mean stay exact.
    """
    total = int(arr.size)
    keep = np.isfinite(arr) if valid is None else (valid & np.isfinite(arr))
    values = np.asarray(arr[keep], dtype=np.float32)
    n = int(values.size)
    pct = 100.0 * n / total if total else 0.0
    if n == 0:
        return LayerStats(name, unit, 0, total, pct, None, None, None, None, None, None)
    sampled = n > SUBSAMPLE_ABOVE
    part = values[:: math.ceil(n / SUBSAMPLE_ABOVE)] if sampled else values
    p2, med, p98 = (float(v) for v in np.percentile(part, [2, 50, 98]))
    return LayerStats(
        name,
        unit,
        n,
        total,
        pct,
        float(values.min()),
        p2,
        med,
        p98,
        float(values.max()),
        float(values.mean(dtype=np.float64)),
        sampled,
    )


def histogram(
    arr: NDArray[Any], valid: NDArray[np.bool_] | None, bins: int = 10
) -> list[dict[str, float]]:
    """Return histogram bins of valid pixels as ``lo``, ``hi``, ``n``, ``pct``; bins clamp to 2..50."""
    bins = max(2, min(int(bins), 50))
    keep = np.isfinite(arr) if valid is None else (valid & np.isfinite(arr))
    values = np.asarray(arr[keep], dtype=np.float32)
    if values.size == 0:
        return []
    if values.size > SUBSAMPLE_ABOVE:
        values = values[:: math.ceil(values.size / SUBSAMPLE_ABOVE)]
    counts, edges = np.histogram(values, bins=bins)
    total = int(counts.sum()) or 1
    return [
        {
            "lo": float(edges[i]),
            "hi": float(edges[i + 1]),
            "n": float(counts[i]),
            "pct": 100.0 * float(counts[i]) / total,
        }
        for i in range(len(counts))
    ]


def qa_fractions(stack: Stack) -> dict[str, float]:
    """Return the flagged fraction of every QA plane (metadata value, else the plane mean)."""
    out: dict[str, float] = {}
    for q in stack.meta.qa:
        out[q.name] = (
            float(q.fraction) if q.fraction is not None else float(stack.plane(q.name).mean())
        )
    return out


def stats_row(s: LayerStats, *, full: bool = False) -> dict[str, str]:
    """Format a statistics row with ``sig4``; ``full`` adds counts, p2, p98 and mean."""
    from atlas.geo.results import sig4

    row = {
        "layer": s.name,
        "unit": s.unit,
        "valid%": sig4(s.valid_pct),
        "min": sig4(s.min),
        "median": sig4(s.median),
        "max": sig4(s.max),
    }
    if full:
        row.update(
            {
                "p2": sig4(s.p2),
                "p98": sig4(s.p98),
                "mean": sig4(s.mean),
                "n_valid": sig4(s.n_valid),
                "n_total": sig4(s.n_total),
            }
        )
    return row
