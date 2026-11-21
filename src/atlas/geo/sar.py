"""SAR preprocessing on linear power: despeckling and conversion to dB.

Invariant: inputs are LINEAR power with NaN as nodata. ``to_db`` never despeckles; the order
(despeckle, then dB) is the caller's. Holes are never filled.
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray
from scipy import ndimage

from atlas.geo._lazy import check_cancel
from atlas.geo.errors import GeoError, fix

__all__ = ["LEE_ENL", "despeckle", "to_db"]

LEE_ENL = 5.0
"""Equivalent number of looks assumed by the Lee filter: a fixed assumption, not estimated."""

_MEDIAN_STRIP = 64
_LEE_STRIP = 512
_LEE_WINDOW = 5
_LEE_HALO = _LEE_WINDOW // 2


def to_db(linear: NDArray[np.floating]) -> NDArray[np.float32]:
    """Convert linear power to decibels.

    Args:
        linear: Linear power.

    Returns:
        ``10 * log10(x)`` as float32; NaN where x is NaN or <= 0.
    """
    x = np.asarray(linear, dtype=np.float32)
    out = np.full(x.shape, np.nan, dtype=np.float32)
    ok = x > 0
    out[ok] = np.float32(10.0) * np.log10(x[ok])
    return out


def _median3(x: NDArray[np.float32]) -> NDArray[np.float32]:
    height, width = x.shape
    padded = np.pad(x, 1, mode="constant", constant_values=np.nan)
    out = np.full(x.shape, np.nan, dtype=np.float32)
    for r0 in range(0, height, _MEDIAN_STRIP):
        check_cancel()
        r1 = min(r0 + _MEDIAN_STRIP, height)
        view = sliding_window_view(padded[r0 : r1 + 2], (3, 3)).reshape(r1 - r0, width, 9)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out[r0:r1] = np.nanmedian(view, axis=-1)
    out[~np.isfinite(x)] = np.nan
    return out


def _lee_strip(x: NDArray[np.float32], r0: int, r1: int, enl: float) -> NDArray[np.float32]:
    height = x.shape[0]
    rows = np.clip(np.arange(r0 - _LEE_HALO, r1 + _LEE_HALO), 0, height - 1)
    sub = x[rows].astype(np.float64)
    valid = np.isfinite(sub)
    x0 = np.where(valid, sub, 0.0)
    v = valid.astype(np.float64)
    size = _LEE_WINDOW
    f_v = ndimage.uniform_filter(v, size=size, mode="nearest")
    f_x = ndimage.uniform_filter(x0, size=size, mode="nearest")
    f_x2 = ndimage.uniform_filter(x0 * x0, size=size, mode="nearest")
    live = f_v >= 0.5 / (size * size)
    safe = np.where(live, f_v, 1.0)
    mean = f_x / safe
    var = np.clip(f_x2 / safe - mean * mean, 0.0, None)
    cu2 = 1.0 / enl
    with np.errstate(divide="ignore", invalid="ignore"):
        ci2 = np.where(mean > 0, var / (mean * mean), np.inf)
        k = np.where(mean > 0, np.clip(1.0 - cu2 / ci2, 0.0, 1.0), 0.0)
    k = np.where(np.isfinite(k), k, 0.0)
    out = mean + k * (x0 - mean)
    out = np.where(valid & live, out, np.nan)
    return np.asarray(out[_LEE_HALO : _LEE_HALO + (r1 - r0)], dtype=np.float32)


def _lee5(x: NDArray[np.float32], enl: float) -> NDArray[np.float32]:
    out = np.empty(x.shape, dtype=np.float32)
    for r0 in range(0, x.shape[0], _LEE_STRIP):
        check_cancel()
        r1 = min(r0 + _LEE_STRIP, x.shape[0])
        out[r0:r1] = _lee_strip(x, r0, r1, enl)
    return out


def despeckle(
    linear: NDArray[np.floating],
    method: Literal["none", "median3", "lee5"],
    *,
    enl: float = LEE_ENL,
) -> NDArray[np.float32]:
    """Reduce speckle in a linear-power image.

    Args:
        linear: 2-D linear power, NaN = nodata.
        method: "none" (copy), "median3" (NaN-aware 3x3 median) or "lee5" (NaN-aware 5x5
            Lee filter with a fixed ENL).
        enl: Equivalent number of looks for lee5; defaults to ``LEE_ENL``.

    Returns:
        A new float32 array, NaN wherever the input is NaN.

    Raises:
        GeoError: Unknown method, non-2-D input, or non-positive enl.
    """
    x = np.asarray(linear, dtype=np.float32)
    if x.ndim != 2:
        raise GeoError(fix(f"despeckle needs a 2-D array, got {x.ndim}-D.", "pass one band"))
    if method == "none":
        return x.copy()
    if method == "median3":
        return _median3(x)
    if method == "lee5":
        if not enl > 0:
            raise GeoError(fix(f"enl={enl} must be positive.", "use the default enl"))
        return _lee5(x, float(enl))
    raise GeoError(fix(f"unknown despeckle method {method!r}.", "use none, median3 or lee5"))
