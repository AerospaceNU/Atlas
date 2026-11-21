"""QA planes: bool masks decoded from QA bands via the shared preset table.

Invariant: bit and code numbers live only in ``chipmeta.QA_PRESETS``; the semantic vocabulary
lives in ``stack.py``. Planes are returned beside the data and never burned into it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from atlas.geo.chipmeta import QA_PRESETS, apply_rule
from atlas.geo.errors import GeoError, fix
from atlas.geo.grid import Grid

__all__ = ["dilate", "outside_aoi_plane", "plane_fractions", "planes_from_qa"]

DILATED = ("cloud", "cloud_shadow", "cirrus")


def _disc(px: int) -> NDArray[np.bool_]:
    ax = np.arange(-px, px + 1)
    return np.asarray(ax[:, None] ** 2 + ax[None, :] ** 2 <= px * px, dtype=np.bool_)


def dilate(mask: NDArray[np.bool_], px: int) -> NDArray[np.bool_]:
    """Grow a mask by a disc of radius ``px`` pixels without wrapping at the image edge.

    Args:
        mask: Bool (H, W).
        px: Radius in pixels; 0 returns the mask unchanged.

    Returns:
        The dilated mask.

    Raises:
        GeoError: ``px`` is negative.
    """
    if px < 0:
        raise GeoError(fix(f"dilate_px={px} is negative.", "use dilate_px=0 or a positive int"))
    if px == 0 or not mask.any():
        return mask
    grown = ndimage.binary_dilation(mask, structure=_disc(px), border_value=0)
    return np.asarray(grown, dtype=np.bool_)


def planes_from_qa(
    raw: NDArray[Any], preset: str, *, dilate_px: int = 0
) -> dict[str, NDArray[np.bool_]]:
    """Decode a QA band into named bool planes.

    Args:
        raw: Integer QA band, or float32 with NaN (a nearest-resampled QA band). NaN counts
            as ``nodata`` and as 0 for every other plane.
        preset: Key of ``chipmeta.QA_PRESETS``.
        dilate_px: Radius (pixels) grown around cloud, cloud_shadow and cirrus. Nodata is
            never un-flagged by the growth.

    Returns:
        A plane for each semantic the preset defines (and ``nodata`` when raw has NaN).

    Raises:
        GeoError: Unknown preset or negative ``dilate_px``.
    """
    rules = QA_PRESETS.get(preset)
    if rules is None:
        raise GeoError(
            fix(
                f"unknown QA preset {preset!r}.",
                f"use one of {', '.join(sorted(QA_PRESETS))} or qa='none'",
            )
        )
    if dilate_px < 0:
        raise GeoError(fix(f"dilate_px={dilate_px} is negative.", "use dilate_px=0 or 2"))
    arr = np.asarray(raw)
    nan = ~np.isfinite(arr) if arr.dtype.kind == "f" else np.zeros(arr.shape, dtype=bool)
    ints = np.where(nan, 0, arr).astype(np.uint32)
    nodata = nan.copy()
    if "nodata" in rules:
        nodata |= apply_rule(rules["nodata"], ints)
    planes: dict[str, NDArray[np.bool_]] = {}
    if "nodata" in rules or nan.any():
        planes["nodata"] = nodata
    for name, rule in rules.items():
        if name != "nodata":
            planes[name] = apply_rule(rule, ints) & ~nodata
    for name in DILATED:
        if name in planes:
            planes[name] = dilate(planes[name], dilate_px) & ~nodata
    return planes


def outside_aoi_plane(grid: Grid, polygon_lonlat: Sequence[Any] | None) -> NDArray[np.bool_] | None:
    """Return the plane of pixels outside a lon/lat polygon, or None when there is none."""
    if polygon_lonlat is None:
        return None
    return np.asarray(~grid.rasterize(polygon_lonlat), dtype=np.bool_)


def plane_fractions(
    planes: Mapping[str, NDArray[np.bool_]], aoi: NDArray[np.bool_] | None = None
) -> dict[str, float]:
    """Return the raw fraction of True pixels per plane.

    Args:
        planes: Named bool planes.
        aoi: Bool (H, W) mask of pixels that count (e.g. inside the polygon); None counts all.

    Returns:
        Fractions in [0, 1] (unrounded; callers format them with ``results.sig4``).
    """
    out: dict[str, float] = {}
    for name, plane in planes.items():
        if aoi is None:
            total = plane.size
            hit = int(plane.sum())
        else:
            total = int(aoi.sum())
            hit = int((plane & aoi).sum())
        out[name] = hit / total if total else 0.0
    return out
