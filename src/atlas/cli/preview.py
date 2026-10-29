"""Display previews for downloaded GeoTIFFs that viewers show dark or not at all.

A 16-bit, signed, or float GeoTIFF opens black or fails in most image
viewers, and a 4-band 8-bit GeoTIFF (NAIP RGBN) is drawn with the wrong
colors. Next to such a file the download writes ``<stem>.preview.png``.

Invariants:

* The data file is never modified. The preview is a separate 8-bit PNG and
  is not a scene asset; analysis reads the GeoTIFF.
* Each band is stretched linearly from its 2nd to its 98th percentile of
  valid pixels to 0-255. Valid excludes NaN and infinities, the GDAL nodata
  value, and, when nodata is unset and more than half the band is zero, the
  zero fill. Invalid pixels are black.
* One or two bands give a grayscale PNG from the first band; three or more
  give RGB from the first three bands.
* The longer side is at most ``MAX_SIDE`` pixels, downsampled by
  nearest-neighbor striding.
* 8-bit palette images (class maps with a ColorMap) and 8-bit three-band
  images already display, so they get no preview, and a stale preview
  beside them is removed.
* A windowed chip is previewed from the pixels in memory. A whole-file
  GeoTIFF is previewed from its smallest overview whose longer side is at
  least ``OVERVIEW_MIN_SIDE`` (``read_overview``), so a 1.8 GB file is not
  decoded whole. Such a preview may be low resolution.
* A preview failure never fails the download; the caller records the text.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from atlas.data.cog import CogArray, CogRead, read_overview

PREVIEW_SUFFIX = ".preview.png"
MAX_SIDE = 2048
OVERVIEW_MIN_SIDE = 1024
LOW_PERCENTILE = 2.0
HIGH_PERCENTILE = 98.0
# Share of zeros above which an unset-nodata band treats zero as fill.
ZERO_FILL_FRACTION = 0.5


def preview_path(data_path: Path) -> Path:
    """``<dir>/<stem>.preview.png`` beside a data file."""
    return data_path.with_name(data_path.stem + PREVIEW_SUFFIX)


def needs_preview(array: CogArray, colormap: tuple[int, ...] | None = None) -> bool:
    """False for an 8-bit palette image or an 8-bit three-band image."""
    if array.dtype != np.uint8:
        return True
    if colormap is not None:
        return False
    return int(array.shape[0]) != 3


def render_preview(array: CogArray, nodata: str | None = None) -> NDArray[np.uint8]:
    """Return a ``(height, width)`` or ``(height, width, 3)`` uint8 display image.

    ``array`` has shape ``(bands, height, width)``. It is not modified.
    """
    if array.ndim != 3 or min(array.shape) < 1:
        raise ValueError("preview needs a non-empty (bands, height, width) array")
    step = max(1, math.ceil(max(array.shape[1], array.shape[2]) / MAX_SIDE))
    small = array[:, ::step, ::step]
    fill = _nodata_value(nodata)
    bands = [0] if small.shape[0] < 3 else [0, 1, 2]
    channels = [_stretch(small[index], fill) for index in bands]
    if len(channels) == 1:
        return channels[0]
    return np.stack(channels, axis=-1)


def write_preview(
    data_path: Path,
    array: CogArray,
    *,
    nodata: str | None = None,
    colormap: tuple[int, ...] | None = None,
) -> Path | None:
    """Write the preview for ``data_path`` from its pixels, or remove a stale one.

    Returns the preview path, or None when the image already displays. The
    PNG is written to a temporary name and renamed, so a failure leaves no
    partial preview.
    """
    dest = preview_path(data_path)
    if not needs_preview(array, colormap):
        _remove(dest)
        return None
    image = render_preview(array, nodata)
    tmp = dest.with_name(dest.name + ".part")
    # A leftover temporary name, or a symlink planted there, is not written through.
    _remove(tmp)
    try:
        Image.fromarray(image).save(tmp, format="PNG")
        os.replace(tmp, dest)
    except BaseException:
        _remove(tmp)
        raise
    return dest


def write_read_preview(data_path: Path, read: CogRead) -> Path | None:
    """Preview a windowed chip from the ``CogRead`` that was encoded to it."""
    return write_preview(data_path, read.array, nodata=read.nodata, colormap=read.colormap)


async def write_file_preview(data_path: Path) -> Path | None:
    """Preview a whole-file GeoTIFF from one overview level, read-only.

    Raises:
        CogError: The file is not a tiled GeoTIFF this reader decodes, or its
            smallest usable level is over the pixel cap.
    """
    read = await read_overview(data_path, min_long_side=OVERVIEW_MIN_SIDE)
    return write_read_preview(data_path, read)


def clear_preview(data_path: Path) -> None:
    """Remove the preview beside ``data_path`` so it does not outlive a failed rewrite."""
    _remove(preview_path(data_path))


def _remove(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()


def _nodata_value(nodata: str | None) -> float | None:
    if nodata is None:
        return None
    try:
        return float(nodata.strip())
    except ValueError:
        return None


def _in_band_dtype(nodata: float, dtype: np.dtype[Any]) -> np.generic | None:
    """``nodata`` as the band stores it, or None when the band cannot hold it.

    GDAL compares nodata in the band's own type. A float32 fill written as
    shortened text (``-3.4028235e+38``) rounds just past float32 and is
    clamped to the type's extreme so it still matches.
    """
    if np.issubdtype(dtype, np.floating):
        info = np.finfo(dtype)
        clamped = min(max(nodata, float(info.min)), float(info.max))
        value: np.generic = dtype.type(clamped)
        return value
    info_i = np.iinfo(dtype)
    if nodata != round(nodata) or not info_i.min <= nodata <= info_i.max:
        return None
    whole: np.generic = dtype.type(int(nodata))
    return whole


def _stretch(band: NDArray[np.generic], nodata: float | None) -> NDArray[np.uint8]:
    """Scale one band's 2nd-98th percentile of valid pixels to 0-255."""
    values = band.astype(np.float64)
    valid = np.isfinite(values)
    if nodata is not None and math.isfinite(nodata):
        fill = _in_band_dtype(nodata, band.dtype)
        if fill is not None:
            valid &= band != fill
    elif nodata is None:
        zeros = values == 0
        if zeros.mean() > ZERO_FILL_FRACTION:
            valid &= ~zeros
    out = np.zeros(values.shape, dtype=np.uint8)
    if not valid.any():
        return out
    low, high = np.percentile(values[valid], [LOW_PERCENTILE, HIGH_PERCENTILE])
    span = high - low
    if span <= 0:
        out[valid] = 128
        return out
    scaled = np.clip((values[valid] - low) / span * 255.0, 0.0, 255.0)
    out[valid] = np.round(scaled).astype(np.uint8)
    return out
