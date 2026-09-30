"""Resolve common band names to catalog-native asset keys."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Final, NamedTuple


class BandSpec(NamedTuple):
    """Match rule for one band name. Wavelengths are micrometres."""

    target: float
    tolerance: float
    names: frozenset[str]


class Candidate(NamedTuple):
    """One single-band asset and the two facts used to match it."""

    key: str
    wavelength: float | None
    name: str | None


# Catalogs disagree about common_name -- S30 tags its broad 0.835 band "nir",
# L30 its narrow 0.86 band -- so match on wavelength and keep names a fallback.
# Each tolerance is bounded by real bands, so widening one changes the winner:
#   red  admits L30 B04 at 0.650 (0.015), rejects S2 B05 at 0.7039 (0.039)
#   nir  admits S30 B08 at 0.8351 (0.030), rejects S30 B05 at 0.7039 (0.161)
BANDS: Final[Mapping[str, BandSpec]] = {
    "red": BandSpec(0.665, 0.03, frozenset({"red"})),
    "nir": BandSpec(0.865, 0.05, frozenset({"nir", "nir08"})),
}


def resolve_bands(raw_assets: Mapping[str, Any]) -> dict[str, str]:
    """Map each band name in `BANDS` to the asset key that holds it.

    Returns:
        Band name to asset key. Names that match nothing are absent.
    """
    candidates = _candidates(raw_assets)
    if not candidates:
        return {}
    resolved: dict[str, str] = {}
    for alias, spec in BANDS.items():
        key = _match_by_wavelength(candidates, spec)
        if key is None:
            key = _match_by_name(candidates, spec)
        if key is not None:
            resolved[alias] = key
    return resolved


def _candidates(raw_assets: Mapping[str, Any]) -> list[Candidate]:
    """Extract single-band assets and the two facts used to match them."""
    found: list[Candidate] = []
    for key, raw in raw_assets.items():
        if not isinstance(raw, dict):
            continue
        bands = raw.get("eo:bands")
        # Multi-entry rejects RGB products like Sentinel-2 `visual`, which
        # advertises a red band and would otherwise win `red`.
        if not isinstance(bands, list) or len(bands) != 1:
            continue
        entry = bands[0]
        if not isinstance(entry, dict):
            continue
        found.append(
            Candidate(
                key=str(key),
                wavelength=_micrometres(entry.get("center_wavelength")),
                name=_common_name(entry.get("common_name")),
            )
        )
    return found


def _match_by_wavelength(candidates: list[Candidate], spec: BandSpec) -> str | None:
    """Return the in-tolerance candidate closest to `spec.target`."""
    best: tuple[float, str] | None = None
    for candidate in candidates:
        if candidate.wavelength is None:
            continue
        distance = abs(candidate.wavelength - spec.target)
        if distance > spec.tolerance:
            continue
        # Key breaks a tie, so serialization order cannot change the result.
        scored = (distance, candidate.key)
        if best is None or scored < best:
            best = scored
    return None if best is None else best[1]


def _match_by_name(candidates: list[Candidate], spec: BandSpec) -> str | None:
    """Return the candidate whose common name is accepted by `spec`."""
    matches = [c.key for c in candidates if c.name is not None and c.name in spec.names]
    return min(matches) if matches else None


def _micrometres(raw: object) -> float | None:
    """Return a usable center wavelength, or None for anything else."""
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        return None
    value = float(raw)
    return value if math.isfinite(value) and value > 0 else None


def _common_name(raw: object) -> str | None:
    """Return a lowercased common name, or None for anything else."""
    if not isinstance(raw, str):
        return None
    return raw.strip().lower() or None
