"""Resampling between pixel grids, plus flag-preserving mask resampling.

Invariant: pure functions on arrays and ``Grid``; NaN is nodata. Branch order is fixed:
identical grid, integer-ratio aligned same-CRS (exact index arithmetic), other same-CRS
(``scipy.ndimage.map_coordinates``), then ``rasterio.warp`` (geo extra) for cross-CRS work.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from atlas.geo._lazy import check_cancel, require_rasterio
from atlas.geo.errors import GeoError, fix
from atlas.geo.grid import Grid

__all__ = ["ResampleInfo", "resample_mask", "resample_to"]

_TOL = 1e-6
_STRIP_ROWS = 256
_MAP_STRIP = 512
_DEG_M = 111_320.0
Array = NDArray[np.float32]


@dataclass(frozen=True)
class ResampleInfo:
    """How a layer was resampled.

    Attributes:
        method: none, crop, replicate, block_mean, nearest, bilinear, average or similar.
        note: One-line human note (rules applied, caveats); empty when nothing to say.
        uncovered_fraction: Share of destination pixels with no source footprint.
    """

    method: str
    note: str
    uncovered_fraction: float


@dataclass(frozen=True)
class _Axis:
    kind: str  # "same", "coarse" (source coarser), "fine" (source finer)
    k: int
    off: int
    n_src: int
    n_dst: int


def _integer(value: float) -> int | None:
    rounded = round(value)
    return int(rounded) if abs(value - rounded) <= _TOL else None


def _axis(src_res: float, dst_res: float, off_map: float, n_src: int, n_dst: int) -> _Axis | None:
    if abs(src_res - dst_res) <= _TOL * max(src_res, dst_res):
        off = _integer(off_map / src_res)
        return None if off is None else _Axis("same", 1, off, n_src, n_dst)
    if src_res > dst_res:
        k = _integer(src_res / dst_res)
        off = _integer(off_map / dst_res)
        if k is None or off is None or k < 2:
            return None
        return _Axis("coarse", k, off, n_src, n_dst)
    k = _integer(dst_res / src_res)
    off = _integer(off_map / src_res)
    if k is None or off is None or k < 2:
        return None
    return _Axis("fine", k, off, n_src, n_dst)


def _axes(src: Grid, dst: Grid) -> tuple[_Axis, _Axis] | None:
    """Return per-axis integer plans for aligned same-CRS grids, else None."""
    rows = _axis(src.yres, dst.yres, src.transform[5] - dst.transform[5], src.height, dst.height)
    cols = _axis(src.xres, dst.xres, dst.transform[2] - src.transform[2], src.width, dst.width)
    if rows is None or cols is None:
        return None
    kinds = {rows.kind, cols.kind} - {"same"}
    return (rows, cols) if len(kinds) <= 1 else None


def _coverage(axis: _Axis) -> NDArray[np.bool_]:
    j = np.arange(axis.n_dst)
    if axis.kind == "fine":
        lo, hi = axis.off + j * axis.k, axis.off + (j + 1) * axis.k
        return np.asarray((hi > 0) & (lo < axis.n_src), dtype=np.bool_)
    idx = (j + axis.off) // axis.k
    return np.asarray((idx >= 0) & (idx < axis.n_src), dtype=np.bool_)


def _gather(data: Array, rows: _Axis, cols: _Axis) -> Array:
    ri = (np.arange(rows.n_dst) + rows.off) // rows.k
    ci = (np.arange(cols.n_dst) + cols.off) // cols.k
    ok_r = (ri >= 0) & (ri < rows.n_src)
    ok_c = (ci >= 0) & (ci < cols.n_src)
    out = np.full((rows.n_dst, cols.n_dst), np.nan, dtype=np.float32)
    if ok_r.any() and ok_c.any():
        out[np.ix_(np.flatnonzero(ok_r), np.flatnonzero(ok_c))] = data[np.ix_(ri[ok_r], ci[ok_c])]
    return out


def _block_strip(
    data: Array, i0: int, i1: int, rows: _Axis, cols: _Axis, mode: str, min_valid: float
) -> Array:
    kr, kc = rows.k, cols.k
    a, b = rows.off + i0 * kr, rows.off + i1 * kr
    ca, cb = cols.off, cols.off + cols.n_dst * kc
    buf = np.full((b - a, cb - ca), np.nan, dtype=np.float32)
    ra, rb = max(a, 0), min(b, rows.n_src)
    cc0, cc1 = max(ca, 0), min(cb, cols.n_src)
    if rb > ra and cc1 > cc0:
        buf[ra - a : rb - a, cc0 - ca : cc1 - ca] = data[ra:rb, cc0:cc1]
    if mode == "nearest":
        return np.ascontiguousarray(buf[kr // 2 :: kr, kc // 2 :: kc])
    blocks = buf.reshape(i1 - i0, kr, cols.n_dst, kc)
    valid = np.isfinite(blocks)
    total = np.where(valid, blocks, 0.0).sum(axis=(1, 3), dtype=np.float64)
    count = valid.sum(axis=(1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / np.maximum(count, 1)
    keep = count >= min_valid * kr * kc
    keep &= count > 0
    return np.where(keep, mean, np.nan).astype(np.float32)


def _block(
    data: Array, rows: _Axis, cols: _Axis, categorical: bool, min_valid: float
) -> tuple[Array, str]:
    mode = "nearest" if categorical else "block_mean"
    out = np.empty((rows.n_dst, cols.n_dst), dtype=np.float32)
    for i0 in range(0, rows.n_dst, _STRIP_ROWS):
        check_cancel()
        i1 = min(i0 + _STRIP_ROWS, rows.n_dst)
        out[i0:i1] = _block_strip(data, i0, i1, rows, cols, mode, min_valid)
    return out, mode


def _integer_path(
    data: Array, src: Grid, dst: Grid, categorical: bool, min_valid: float
) -> tuple[Array, ResampleInfo] | None:
    axes = _axes(src, dst)
    if axes is None:
        return None
    rows, cols = axes
    cover = _coverage(rows)[:, None] & _coverage(cols)[None, :]
    uncovered = float(1.0 - cover.mean())
    kinds = {rows.kind, cols.kind}
    if "fine" in kinds:
        out, method = _block(data, rows, cols, categorical, min_valid)
        note = (
            f"nearest pixel at block offset k//2 ({max(rows.k, cols.k)}x downsample)"
            if categorical
            else f"block mean, NaN if under {min_valid:.0%} of the block is valid"
        )
        return out, ResampleInfo(method, note, uncovered)
    out = _gather(data, rows, cols)
    if "coarse" in kinds:
        return out, ResampleInfo("replicate", f"replicated {max(rows.k, cols.k)}x", uncovered)
    return out, ResampleInfo("crop", "", uncovered)


def _centres(grid: Grid) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    cols, _ = grid.xy(np.zeros(grid.width), np.arange(grid.width))
    _, rows = grid.xy(np.arange(grid.height), np.zeros(grid.height))
    return cols, rows


def _footprint(src: Grid, dst: Grid) -> NDArray[np.bool_]:
    xs, ys = _centres(dst)
    west, south, east, north = src.bounds
    return np.asarray(
        ((ys >= south) & (ys <= north))[:, None] & ((xs >= west) & (xs <= east))[None, :],
        dtype=np.bool_,
    )


def _map_continuous(data: Array, coords: NDArray[np.float64]) -> Array:
    valid = np.isfinite(data)
    zero = np.where(valid, data, 0.0).astype(np.float32)
    w = ndimage.map_coordinates(
        valid.astype(np.float32), coords, order=1, mode="grid-constant", cval=0.0, prefilter=False
    )
    v = ndimage.map_coordinates(
        zero, coords, order=1, mode="grid-constant", cval=0.0, prefilter=False
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(w >= 0.5, v / np.maximum(w, 1e-12), np.nan).astype(np.float32)


def _map_path(data: Array, src: Grid, dst: Grid, categorical: bool) -> Array:
    xs, ys = _centres(dst)
    c = (xs - src.transform[2]) / src.xres - 0.5
    out = np.empty(dst.shape, dtype=np.float32)
    for i0 in range(0, dst.height, _MAP_STRIP):
        check_cancel()
        i1 = min(i0 + _MAP_STRIP, dst.height)
        r = (src.transform[5] - ys[i0:i1]) / src.yres - 0.5
        coords = np.stack(np.meshgrid(r, c, indexing="ij"))
        if categorical:
            out[i0:i1] = ndimage.map_coordinates(
                data, coords, order=0, mode="grid-constant", cval=np.nan, prefilter=False
            )
        else:
            out[i0:i1] = _map_continuous(data, coords)
    return out


def _have_rasterio() -> bool:
    try:
        require_rasterio()
    except GeoError:
        return False
    return True


def _metre_res(grid: Grid) -> float:
    from atlas.data.cog import GEOGRAPHIC_EPSG

    res = math.sqrt(grid.xres * grid.yres)
    if grid.epsg in GEOGRAPHIC_EPSG:
        lat = grid.bounds[3] - 0.5 * grid.height * grid.yres
        return res * _DEG_M * max(math.cos(math.radians(lat)), 0.01)
    return res


def _downsample_ratio(src: Grid, dst: Grid) -> float:
    return _metre_res(dst) / _metre_res(src)


def _is_geographic(grid: Grid) -> bool:
    from atlas.data.cog import GEOGRAPHIC_EPSG

    return grid.epsg in GEOGRAPHIC_EPSG


def _reproject(data: Array, src: Grid, dst: Grid, categorical: bool) -> tuple[Array, Array, str]:
    rio = require_rasterio()
    from rasterio.transform import Affine
    from rasterio.warp import Resampling, reproject

    ratio = _downsample_ratio(src, dst)
    if categorical:
        method, resampling = "nearest", Resampling.nearest
    elif ratio > 2.0:
        method, resampling = "average", Resampling.average
    else:
        method, resampling = "bilinear", Resampling.bilinear
    src_t, dst_t = Affine(*src.transform), Affine(*dst.transform)
    out = np.full(dst.shape, np.nan, dtype=np.float32)
    reproject(
        np.ascontiguousarray(data),
        out,
        src_transform=src_t,
        src_crs=rio.crs.CRS.from_string(src.crs),
        src_nodata=float("nan"),
        dst_transform=dst_t,
        dst_crs=rio.crs.CRS.from_string(dst.crs),
        dst_nodata=float("nan"),
        resampling=resampling,
        init_dest_nodata=True,
    )
    cover = np.zeros(dst.shape, dtype=np.uint8)
    reproject(
        np.ones(src.shape, dtype=np.uint8),
        cover,
        src_transform=src_t,
        src_crs=rio.crs.CRS.from_string(src.crs),
        src_nodata=0,
        dst_transform=dst_t,
        dst_crs=rio.crs.CRS.from_string(dst.crs),
        dst_nodata=0,
        resampling=Resampling.nearest,
    )
    return out, np.asarray(cover > 0, dtype=np.float32), method


def _check_input(data: Array, src: Grid) -> Array:
    arr = np.asarray(data, dtype=np.float32)
    if arr.shape != src.shape:
        raise GeoError(
            fix(
                f"data shape {arr.shape} does not match source grid {src.shape}.",
                "pass the grid the array was read with",
            )
        )
    return arr


def _same_crs_branch(
    arr: Array, src: Grid, dst: Grid, categorical: bool
) -> tuple[Array, ResampleInfo]:
    ratio = _downsample_ratio(src, dst)
    if ratio > 1.5 and _have_rasterio():
        return _warp_branch(arr, src, dst, categorical)
    method = "nearest" if categorical else "bilinear"
    note = "centre-aligned " + method
    if ratio > 1.5:
        note += "; aliased: install geo extra"
    out = _map_path(arr, src, dst, categorical)
    uncovered = float(1.0 - _footprint(src, dst).mean())
    return out, ResampleInfo("map_" + method, note, uncovered)


def _warp_branch(arr: Array, src: Grid, dst: Grid, categorical: bool) -> tuple[Array, ResampleInfo]:
    out, cover, method = _reproject(arr, src, dst, categorical)
    note = f"rasterio.warp {method} {src.crs} -> {dst.crs}"
    return out, ResampleInfo(method, note, float(1.0 - cover.mean()))


def resample_to(
    data: NDArray[np.floating],
    src: Grid,
    dst: Grid,
    *,
    categorical: bool,
    min_valid: float = 0.5,
) -> tuple[NDArray[np.float32], ResampleInfo]:
    """Resample a 2-D layer from ``src`` onto ``dst``.

    Args:
        data: Float (H, W) on ``src``; NaN = nodata.
        src: Source grid.
        dst: Destination grid.
        categorical: Class codes: nearest only, never blended.
        min_valid: Block-mean rule: NaN when less than this share of the block is valid.

    Returns:
        The resampled array and how it was made. Uncovered destination pixels are NaN.

    Raises:
        GeoError: Shape mismatch with ``src``.
        MissingExtraError: The route needs the geo extra (cross-CRS or geographic source).
    """
    check_cancel()
    arr = _check_input(data, src)
    if dst.same_as(src):
        return arr.copy(), ResampleInfo("none", "", 0.0)
    if src.crs == dst.crs:
        hit = _integer_path(arr, src, dst, categorical, min_valid)
        if hit is not None:
            return hit
        if not _is_geographic(src):
            return _same_crs_branch(arr, src, dst, categorical)
    return _warp_branch(arr, src, dst, categorical)


def _mask_any(mask: NDArray[np.bool_], src: Grid, dst: Grid) -> NDArray[np.bool_]:
    flags = mask.astype(np.float32)
    downsample = _downsample_ratio(src, dst) > 1.0 + _TOL
    out, _ = resample_to(flags, src, dst, categorical=not downsample, min_valid=_TOL)
    with np.errstate(invalid="ignore"):
        return np.asarray(out > 0.0 if downsample else out >= 0.5, dtype=np.bool_)


def resample_mask(mask: NDArray[np.bool_], src: Grid, dst: Grid) -> NDArray[np.bool_]:
    """Resample a bool mask: any-flag on downsample, nearest or replicate otherwise.

    Args:
        mask: Bool (H, W) on ``src``.
        src: Source grid.
        dst: Destination grid.

    Returns:
        Bool (dst.height, dst.width); uncovered area is False.
    """
    if dst.same_as(src):
        return mask.copy()
    return _mask_any(mask, src, dst)
