"""Raw digital numbers to physical units: calibration records and their application.

Invariant: ``apply`` always computes ``data * scale + offset`` with the stored values, and
nodata is turned into NaN before scaling, never scaled itself. ``offset_applied`` is
informational only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from atlas.geo.errors import GeoError, fix

if TYPE_CHECKING:
    from atlas.geo.chipmeta import ChipMeta

__all__ = ["Calibration", "apply", "from_sidecar", "from_user"]


@dataclass(frozen=True)
class Calibration:
    """How to turn raw pixel values into physical units.

    Attributes:
        scale: Multiplier.
        offset: Added after scaling.
        offset_applied: True when an upstream already baked an offset in (informational;
            the sidecar then stores ``offset=0``).
        nodata_in: Raw value meaning "no data", or None when none is declared.
        dtype_in: Source dtype name, for reporting.
        valid_range: Inclusive raw-DN range; values outside become NaN.
        source: Where the record came from.
    """

    scale: float
    offset: float
    offset_applied: bool
    nodata_in: float | None
    dtype_in: str
    valid_range: tuple[float, float] | None
    source: Literal["sidecar", "user", "none"]


def from_sidecar(meta: ChipMeta, *, file: str = "") -> Calibration:
    """Build a calibration from a chip sidecar.

    Args:
        meta: The chip sidecar.
        file: Chip file name, used in the error message.

    Returns:
        The calibration. The file nodata is the fallback when the record declares none.

    Raises:
        GeoError: The sidecar has no calibration record.
    """
    cal = meta.calibration
    if not cal.source:
        label = file or meta.scene_id or "<chip>"
        raise GeoError(
            fix(
                f"chip {label} has no calibration.",
                "prepare_stack files=[{path,scale,offset,nodata,unit,band}]",
            )
        )
    nodata = cal.nodata_in if cal.nodata_in is not None else meta.raster.nodata
    valid = None
    if cal.valid_range is not None and len(cal.valid_range) == 2:
        valid = (float(cal.valid_range[0]), float(cal.valid_range[1]))
    return Calibration(
        scale=float(cal.scale),
        offset=float(cal.offset),
        offset_applied=bool(cal.offset_applied),
        nodata_in=None if nodata is None else float(nodata),
        dtype_in=cal.dtype_in or meta.raster.dtype or "",
        valid_range=valid,
        source="sidecar",
    )


def from_user(
    *,
    scale: float | None,
    offset: float | None,
    nodata: float | None,
    unit: str | None,
    band: str | None,
    dtype: str,
) -> Calibration:
    """Build a calibration from user-supplied fields.

    Args:
        scale: Multiplier; defaults to 1 only for float dtypes.
        offset: Additive term; defaults to 0 only for float dtypes.
        nodata: Raw nodata value, or None for none declared.
        unit: Physical unit of the result (required).
        band: Band identity (required).
        dtype: Source dtype name.

    Returns:
        The calibration.

    Raises:
        GeoError: Required fields are missing (all listed together), or scale is zero or
            not finite.
    """
    is_float = np.dtype(dtype).kind == "f"
    missing = [n for n, v in (("unit", unit), ("band", band)) if not v]
    if scale is None and not is_float:
        missing.append("scale")
    if offset is None and not is_float:
        missing.append("offset")
    if missing:
        raise GeoError(
            fix(
                f"user raster ({dtype}) is missing {', '.join(missing)}.",
                "prepare_stack files=[{path,band:'red',unit:'reflectance',scale:0.0001,"
                "offset:0,nodata:0}]",
            )
        )
    s = 1.0 if scale is None else float(scale)
    o = 0.0 if offset is None else float(offset)
    if not math.isfinite(s) or s == 0.0 or not math.isfinite(o):
        raise GeoError(
            fix(f"scale={scale} offset={offset} must be finite and scale non-zero.", "pass scale=1")
        )
    return Calibration(
        scale=s,
        offset=o,
        offset_applied=False,
        nodata_in=None if nodata is None else float(nodata),
        dtype_in=dtype,
        valid_range=None,
        source="user",
    )


def _bad_mask(data: NDArray[Any], cal: Calibration) -> NDArray[np.bool_]:
    bad = np.zeros(data.shape, dtype=bool)
    if data.dtype.kind == "f":
        bad |= ~np.isfinite(data)
    nodata = cal.nodata_in
    if nodata is not None and not math.isnan(nodata):
        bad |= data == nodata
    if cal.valid_range is not None:
        lo, hi = cal.valid_range
        bad |= (data < lo) | (data > hi)
    return bad


def apply(data: NDArray[Any], cal: Calibration) -> NDArray[np.float32]:
    """Return physical values as float32 with nodata and out-of-range pixels as NaN.

    Args:
        data: Raw pixels of any numeric dtype.
        cal: The calibration.

    Returns:
        A new float32 array: ``data * scale + offset``, NaN where the raw value is nodata,
        outside ``valid_range``, or non-finite.
    """
    bad = _bad_mask(data, cal)
    out: NDArray[np.float32] = data.astype(np.float32) * np.float32(cal.scale) + np.float32(
        cal.offset
    )
    if bad.any():
        out[bad] = np.nan
    return out
