"""NDVI, NDWI, and NBR from reflectance arrays.

All three stay uncomputed until both red and NIR are present. NDVI is
``(NIR - red) / (NIR + red)``. NDWI is McFeeters,
``(green - NIR) / (green + NIR)``. NBR is
``(NIR - swir22) / (NIR + swir22)``. A zero denominator is NaN.
"""

from __future__ import annotations

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
    _Formula("ndwi", "green", "nir"),
    _Formula("nbr", "nir", "swir22"),
)
_GATE: Final[frozenset[str]] = frozenset({"red", "nir"})


def available_indices(band_names: Iterable[str]) -> tuple[str, ...]:
    """Return the index names these bands can compute.

    Args:
        band_names: Band names, compared case-insensitively. A mapping such as
            ``Scene.band_aliases`` contributes its keys.

    Returns:
        ``ndvi``, ``ndwi``, and ``nbr``, in that order. Empty unless both red
        and NIR are present. NDWI also needs green, and NBR also needs swir22.

    Raises:
        TypeError: if ``band_names`` is a string, or a name is not a string.
    """
    names = _names(band_names)
    if not names >= _GATE:
        return ()
    return tuple(
        formula.name for formula in _FORMULAS if formula.left in names and formula.right in names
    )


def compute_indices(bands: Mapping[str, NDArray[Any]]) -> dict[str, NDArray[np.float64]]:
    """Compute NDVI, NDWI, and NBR once red and NIR are available.

    Args:
        bands: Band name to reflectance array. Names are compared
            case-insensitively. Bands that no returned index uses are ignored.

    Returns:
        Index name to a float64 array with the same shape as NIR. Empty when
        red or NIR is absent. NDWI is included when green is present, and NBR
        when swir22 is present.

    Raises:
        TypeError: if a name is not a string, or a band used by a returned
            index is not a numeric array.
        ValueError: if two names collapse to one band, or used bands do not
            share a shape.
    """
    collected = _collect(bands)
    chosen = available_indices(collected)
    if not chosen:
        return {}

    prepared: dict[str, NDArray[np.float64]] = {}
    shape: tuple[int, ...] | None = None
    for formula in _FORMULAS:
        if formula.name not in chosen:
            continue
        for name in (formula.left, formula.right):
            if name in prepared:
                continue
            array = _as_float64(name, collected[name])
            if shape is None:
                shape = array.shape
            elif array.shape != shape:
                raise ValueError(f"bands must share a shape, got {name} {array.shape} and {shape}")
            prepared[name] = array

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


def _as_float64(name: str, value: object) -> NDArray[np.float64]:
    """Copy a numeric array to float64 without changing the original."""
    if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
        return np.asarray(value, dtype=np.float64)
    raise TypeError(f"{name} must be a numeric array")
