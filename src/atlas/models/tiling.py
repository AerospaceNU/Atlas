"""Overlap-discard tiling for per-pixel segmentation models (pure numpy).

Invariants: no optional extra is imported here. Edge padding plus margin discarding makes
local (convolutional) models tile-invariant away from the image border; models that
z-score inside the graph are tile-extent dependent, which this module cannot remove.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["Tile", "plan_tiles", "run_single", "run_tiled"]

ReadFn = Callable[[int, int, int, int], NDArray[np.float32]]
ModelFn = Callable[[NDArray[np.float32]], NDArray[np.floating]]


@dataclass(frozen=True)
class Tile:
    """One output core region in image coordinates (half-open)."""

    r0: int
    r1: int
    c0: int
    c1: int


def plan_tiles(h: int, w: int, size: int, margin: int) -> list[Tile]:
    """Partition an (h, w) image into cores of size - 2 * margin, row-major.

    Args:
        h: Image rows.
        w: Image columns.
        size: Model input edge length.
        margin: Pixels discarded on every side of a tile.

    Returns:
        Cores that partition the image; edge cores are clipped.

    Raises:
        ValueError: If the core size is not positive.
    """
    core = size - 2 * margin
    if core <= 0:
        raise ValueError(f"tile size {size} must exceed 2 * margin ({margin})")
    return [
        Tile(r, min(r + core, h), c, min(c + core, w))
        for r in range(0, h, core)
        for c in range(0, w, core)
    ]


def _check_cancel(deadline: float | None) -> None:
    from atlas.geo._lazy import check_cancel

    check_cancel(deadline)


def _edge_pad(
    block: NDArray[np.float32], top: int, bottom: int, left: int, right: int
) -> NDArray[np.float32]:
    if not (top or bottom or left or right):
        return block
    pad = ((0, 0), (top, bottom), (left, right))
    return np.pad(block, pad, mode="edge")


def _run_block(
    block: NDArray[np.float32], fn: ModelFn, core: tuple[slice, slice]
) -> tuple[NDArray[np.float32] | None, NDArray[np.bool_]]:
    """Run `fn` on one padded block; return (codes or None if the core is all NaN, nan mask)."""
    nan_in = np.isnan(block).all(axis=0)
    if nan_in[core].all():
        return None, nan_in
    clean = np.ascontiguousarray(np.where(np.isnan(block), np.float32(0), block))
    result = np.asarray(fn(clean[None]), dtype=np.float32)
    return result, nan_in


def run_tiled(
    read: ReadFn,
    fn: ModelFn,
    out: NDArray[np.float32],
    *,
    size: int,
    margin: int,
    deadline: float | None = None,
    pad_multiple: int = 1,
) -> int:
    """Run `fn` over overlapping tiles, keeping only each tile's core.

    Args:
        read: Returns a (C, h, w) float32 window for [r0, r1) x [c0, c1); may hold NaN.
        fn: Maps a (1, C, size, size) float32 block (NaN already 0) to (size, size) codes.
        out: (H, W) float32 output (a memmap streams rows to disk); NaN where no data.
        size: Model input edge length.
        margin: Pixels discarded around each tile.
        deadline: Absolute `time.monotonic()` cap passed to `check_cancel`.
        pad_multiple: `size` must be a multiple of this.

    Returns:
        Number of tiles processed (including skipped all-NaN tiles).

    Raises:
        ValueError: If the geometry is invalid.
        GeoCancelled: Cancellation or deadline, checked at the top of every tile.
    """
    if size % pad_multiple:
        raise ValueError(f"tile size {size} must be a multiple of {pad_multiple}")
    h, w = out.shape
    tiles = plan_tiles(h, w, size, margin)
    for tile in tiles:
        _check_cancel(deadline)
        _run_tile(read, fn, out, tile, size, margin, (h, w))
    return len(tiles)


def _run_tile(
    read: ReadFn,
    fn: ModelFn,
    out: NDArray[np.float32],
    tile: Tile,
    size: int,
    margin: int,
    shape: tuple[int, int],
) -> None:
    h, w = shape
    rs, re = max(0, tile.r0 - margin), min(h, tile.r0 - margin + size)
    cs, ce = max(0, tile.c0 - margin), min(w, tile.c0 - margin + size)
    block = np.asarray(read(rs, re, cs, ce), dtype=np.float32)
    top = rs - (tile.r0 - margin)
    left = cs - (tile.c0 - margin)
    block = _edge_pad(block, top, size - top - (re - rs), left, size - left - (ce - cs))
    th, tw = tile.r1 - tile.r0, tile.c1 - tile.c0
    core_slice = (slice(margin, margin + th), slice(margin, margin + tw))
    result, nan_in = _run_block(block, fn, core_slice)
    core_nan = nan_in[core_slice]
    if result is None:
        out[tile.r0 : tile.r1, tile.c0 : tile.c1] = np.nan
        return
    core = np.array(result[core_slice], dtype=np.float32)
    core[core_nan] = np.nan
    out[tile.r0 : tile.r1, tile.c0 : tile.c1] = core


def run_single(
    read: ReadFn,
    fn: ModelFn,
    out: NDArray[np.float32],
    *,
    deadline: float | None = None,
    pad_multiple: int = 32,
) -> None:
    """Run `fn` once on the whole image, edge-padded up to a multiple of `pad_multiple`.

    Args:
        read: As for `run_tiled`.
        fn: Maps (1, C, H', W') to (H', W') codes.
        out: (H, W) float32 output.
        deadline: Absolute `time.monotonic()` cap.
        pad_multiple: Padding multiple for both axes.

    Raises:
        GeoCancelled: Cancellation or deadline before the run starts.
    """
    _check_cancel(deadline)
    h, w = out.shape
    block = np.asarray(read(0, h, 0, w), dtype=np.float32)
    ph = -h % pad_multiple
    pw = -w % pad_multiple
    block = _edge_pad(block, 0, ph, 0, pw)
    result, nan_in = _run_block(block, fn, (slice(0, h), slice(0, w)))
    if result is None:
        out[:] = np.nan
        return
    cropped = np.array(result[:h, :w], dtype=np.float32)
    cropped[nan_in[:h, :w]] = np.nan
    out[:] = cropped
