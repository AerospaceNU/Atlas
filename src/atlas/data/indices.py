"""NDVI, McFeeters NDWI, and NBR from surface reflectance.

All three stay uncomputed until both red and NIR are present. NDVI is
``(NIR - red) / (NIR + red)``. NDWI is the McFeeters index, named
``ndwi_mcfeeters``, ``(green - NIR) / (green + NIR)``. NBR is
``(NIR - swir22) / (NIR + swir22)``. A zero denominator is NaN, and so is
any pixel whose input band is nodata.

Inputs must be surface reflectance. Stored digital numbers can be converted
with per-band scale and offset, ``reflectance = stored * scale + offset``.
Sentinel-2 L2A publishes an additive offset of -1000 and a scale of 1e-4,
which is that form with ``offset=-1000 * 1e-4``. Landsat Collection 2
surface reflectance uses scale ``2.75e-5`` and offset ``-0.2`` directly.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any, Final, NamedTuple

import numpy as np
from numpy.typing import NDArray


class _Formula(NamedTuple):
    name: str
    left: str
    right: str


# Insertion order is the order both public functions return.
_FORMULAS: Final[tuple[_Formula, ...]] = (
    _Formula("ndvi", "nir", "red"),
    _Formula("ndwi_mcfeeters", "green", "nir"),
    _Formula("nbr", "nir", "swir22"),
)
_GATE: Final[frozenset[str]] = frozenset({"red", "nir"})
_BAND_ORDER: Final[tuple[str, ...]] = ("nir", "red", "green", "swir22")


def available_indices(band_names: Iterable[str]) -> tuple[str, ...]:
    """Return the index names these bands can compute.

    Args:
        band_names: Band names, compared case-insensitively. A mapping such as
            ``Scene.band_aliases`` contributes its keys.

    Returns:
        ``ndvi``, ``ndwi_mcfeeters``, and ``nbr``, in that order. Empty unless
        both red and NIR are present. McFeeters NDWI also needs green, and
        NBR also needs swir22.

    Raises:
        TypeError: if ``band_names`` is a string, or a name is not a string.
    """
    names = _names(band_names)
    if not names >= _GATE:
        return ()
    return tuple(
        formula.name for formula in _FORMULAS if formula.left in names and formula.right in names
    )


def compute_indices(
    bands: Mapping[str, NDArray[Any]],
    *,
    nodata: float | Mapping[str, float] | None = None,
    mask: NDArray[Any] | Mapping[str, NDArray[Any]] | None = None,
    scale: float | Mapping[str, float] | None = None,
    offset: float | Mapping[str, float] | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Compute NDVI, McFeeters NDWI, and NBR once red and NIR are available.

    Inputs must be surface reflectance. Pass ``scale`` and ``offset`` when the
    arrays are still digital numbers; each band becomes
    ``stored * scale + offset`` before the index is computed. Nodata is read
    from the stored values, before that conversion.

    Args:
        bands: Band name to array. Names are compared case-insensitively.
            Bands that no returned index uses are ignored.
        nodata: Fill value, or band name to fill value. A pixel equal to that
            value on an input band is NaN in every index that uses the band.
        mask: Boolean array, or band name to boolean array. ``True`` marks
            nodata. A single array applies to every band that is used.
        scale: Multiply stored values. One number applies to every used band.
        offset: Add after scaling. One number applies to every used band.
            Sentinel-2 L2A's published offset of -1000 with scale 1e-4 is
            ``offset=-1000 * 1e-4``. Landsat Collection 2 uses scale
            ``2.75e-5`` and offset ``-0.2``.

    Returns:
        Index name to a float64 array with the same shape as NIR. Empty when
        red or NIR is absent. ``ndwi_mcfeeters`` is included when green is
        present, and ``nbr`` when swir22 is present.

    Raises:
        TypeError: if a name is not a string, a used band is not a numeric
            array, or a mask is not boolean.
        ValueError: if two names collapse to one band, used bands do not share
            a shape, or scale, offset, nodata, or mask names an unused band.
    """
    collected = _collect(bands)
    chosen = available_indices(collected)
    if not chosen:
        return {}

    used = _used_bands(chosen)
    nodata_map = _number_map("nodata", nodata, used)
    scale_map = _number_map("scale", scale, used)
    offset_map = _number_map("offset", offset, used)
    mask_map = _mask_map(mask, used)

    prepared: dict[str, NDArray[np.float64]] = {}
    shape: tuple[int, ...] | None = None
    for name in _BAND_ORDER:
        if name not in used:
            continue
        array = _as_float64(name, collected[name])
        if shape is None:
            shape = array.shape
        elif array.shape != shape:
            raise ValueError(f"bands must share a shape, got {name} {array.shape} and {shape}")
        invalid = ~np.isfinite(array)
        if name in nodata_map:
            invalid |= array == nodata_map[name]
        band_mask = mask_map.get(name)
        if band_mask is not None:
            if band_mask.shape != array.shape:
                raise ValueError(
                    f"{name} mask shape {band_mask.shape} does not match {array.shape}"
                )
            invalid |= band_mask
        band_scale = scale_map.get(name, 1.0)
        band_offset = offset_map.get(name, 0.0)
        prepared[name] = array * band_scale + band_offset
        prepared[name][invalid] = np.nan

    return {
        formula.name: normalized_difference(prepared[formula.left], prepared[formula.right])
        for formula in _FORMULAS
        if formula.name in chosen
    }


def normalized_difference(
    left: NDArray[np.float64],
    right: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return ``(left - right) / (left + right)`` as float64.

    Args:
        left: Band on the left of both the difference and the sum.
        right: Band on the right of both the difference and the sum.

    Returns:
        Float64 array of the same shape. Where ``left + right`` is 0, the
        value is NaN. Inputs are not modified.

    Raises:
        ValueError: if the shapes differ.
    """
    if left.shape != right.shape:
        raise ValueError(f"bands must share a shape, got {left.shape} and {right.shape}")
    numerator = left - right
    denominator = left + right
    out = np.full(left.shape, np.nan, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(numerator, denominator, out=out, where=denominator != 0)
    return out


def _names(band_names: Iterable[str]) -> set[str]:
    """Lowercase the band names, dropping blanks."""
    if isinstance(band_names, str):
        raise TypeError("band_names must be a collection of names, not a string")
    names: set[str] = set()
    for raw in band_names:
        name = _normalize(raw)
        if name:
            names.add(name)
    return names


def _collect(bands: Mapping[str, NDArray[Any]]) -> dict[str, NDArray[Any]]:
    """Map each normalized band name to its array."""
    collected: dict[str, NDArray[Any]] = {}
    for raw, value in bands.items():
        name = _normalize(raw)
        if not name:
            continue
        if name in collected:
            raise ValueError(f"duplicate band {name!r}")
        collected[name] = value
    return collected


def _normalize(raw: object) -> str:
    """Return a band name ready to compare."""
    if not isinstance(raw, str):
        raise TypeError(f"band name must be a string, got {type(raw).__name__}")
    return raw.strip().lower()


def _used_bands(chosen: tuple[str, ...]) -> set[str]:
    """Return the band names the selected indices actually read."""
    used: set[str] = set()
    chosen_names = set(chosen)
    for formula in _FORMULAS:
        if formula.name in chosen_names:
            used.add(formula.left)
            used.add(formula.right)
    return used


def _number_map(label: str, raw: object, used: set[str]) -> dict[str, float]:
    """Normalize one number, or a band-to-number mapping, onto ``used`` bands."""
    if raw is None:
        return {}
    if isinstance(raw, Mapping):
        mapped: dict[str, float] = {}
        for key, value in raw.items():
            name = _normalize(key)
            if not name or name not in used:
                raise ValueError(f"unknown band {name!r} in {label}")
            if name in mapped:
                raise ValueError(f"duplicate {label} for {name!r}")
            mapped[name] = _finite_number(label, value)
        return mapped
    return dict.fromkeys(used, _finite_number(label, raw))


def _mask_map(
    raw: NDArray[Any] | Mapping[str, NDArray[Any]] | None,
    used: set[str],
) -> dict[str, NDArray[np.bool_]]:
    """Normalize a nodata mask. ``True`` means the pixel is nodata."""
    if raw is None:
        return {}
    if isinstance(raw, np.ndarray):
        mask = _as_bool_mask("mask", raw)
        return {name: mask for name in used}
    if isinstance(raw, Mapping):
        mapped: dict[str, NDArray[np.bool_]] = {}
        for key, value in raw.items():
            name = _normalize(key)
            if not name or name not in used:
                raise ValueError(f"unknown band {name!r} in mask")
            if name in mapped:
                raise ValueError(f"duplicate mask for {name!r}")
            mapped[name] = _as_bool_mask(name, value)
        return mapped
    raise TypeError("mask must be a boolean array or a mapping of boolean arrays")


def _as_bool_mask(name: str, value: object) -> NDArray[np.bool_]:
    """Return a boolean nodata mask."""
    if isinstance(value, np.ndarray) and value.dtype == np.bool_:
        return np.array(value, dtype=np.bool_, copy=True)
    raise TypeError(f"{name} mask must be a boolean array")


def _finite_number(label: str, raw: object) -> float:
    """Return a finite float, rejecting booleans and non-numbers."""
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        raise TypeError(f"{label} must be a real number")
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def _as_float64(name: str, value: object) -> NDArray[np.float64]:
    """Copy a numeric array to float64 without changing the original."""
    if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
        return np.array(value, dtype=np.float64, copy=True)
    raise TypeError(f"{name} must be a numeric array")
