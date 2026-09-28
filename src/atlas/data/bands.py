"""Resolve catalog-native asset keys to common band names.

STAC assets keep whatever key the catalog chose, and those keys are not
portable: ``B05`` is near-infrared in HLS L30 and red edge in Sentinel-2.
``common_name`` is no better, because catalogs disagree about it -- HLS S30
tags its broad 0.835 um band ``nir`` while L30 tags its narrow 0.86 um band
the same way, so trusting the label pairs two different measurements.

Matching on ``center_wavelength`` avoids both problems: a wavelength is a
physical quantity, comparable across catalogs, and it needs no per-collection
table. Nothing here knows which collection it is looking at.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Final, NamedTuple


class BandSpec(NamedTuple):
    """Match rule for one common band name.

    Attributes:
        target: Center wavelength in micrometres.
        tolerance: Maximum absolute distance from ``target`` to stay eligible.
        names: Accepted ``eo:bands`` common names, for catalogs that publish
            no wavelength.
    """

    target: float
    tolerance: float
    names: frozenset[str]


class Candidate(NamedTuple):
    """One single-band asset and the two facts used to match it.

    Attributes:
        key: The catalog's own asset key, returned as the match result.
        wavelength: Center wavelength in micrometres, or ``None``.
        name: Lowercased ``common_name``, or ``None``.
    """

    key: str
    wavelength: float | None
    name: str | None


# Each tolerance is bounded on both sides by real bands and sits inside that
# interval, so widening one silently changes which band wins:
#   red  admits L30 B04 at 0.650 (0.015), rejects S2 B05 at 0.7039 (0.039)
#   nir  admits S30 B08 at 0.8351 (0.030), rejects S30 B05 at 0.7039 (0.161)
BANDS: Final[Mapping[str, BandSpec]] = {
    "red": BandSpec(0.665, 0.03, frozenset({"red"})),
    "nir": BandSpec(0.865, 0.05, frozenset({"nir", "nir08"})),
}


def resolve_bands(raw_assets: Mapping[str, Any]) -> dict[str, str]:
    """Map each known band name to the asset key that holds it.

    Wavelength is tried first and the name set is only consulted when no
    candidate falls inside the tolerance, so a catalog that publishes both
    is never decided by its labels.

    Args:
        raw_assets: STAC ``assets`` object, asset key to asset dict.

    Returns:
        Band name to catalog asset key, e.g. ``{"red": "B04", "nir": "B8A"}``.
        Names that match nothing are absent rather than ``None``.
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
    """Extract single-band assets and the two facts used to match them.

    Assets whose ``eo:bands`` is absent or holds more than one entry are
    skipped: multi-band products such as Sentinel-2 ``visual`` would otherwise
    match ``red``, and service assets such as ``tilejson`` carry no band
    metadata at all.

    Args:
        raw_assets: STAC ``assets`` object, asset key to asset dict.

    Returns:
        One candidate per single-band asset, in input order. Either fact may
        be ``None``; quality masks such as ``Fmask`` have neither.
    """
    found: list[Candidate] = []
    for key, raw in raw_assets.items():
        if not isinstance(raw, dict):
            continue
        bands = raw.get("eo:bands")
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
    """Return the in-tolerance candidate closest to ``spec.target``.

    Args:
        candidates: Candidates for one scene; those without a wavelength are
            skipped rather than falling through to their name.
        spec: The band being resolved.

    Returns:
        The winning asset key, or ``None`` if nothing is inside the tolerance.
    """
    best: tuple[float, str] | None = None
    for candidate in candidates:
        if candidate.wavelength is None:
            continue
        distance = abs(candidate.wavelength - spec.target)
        if distance > spec.tolerance:
            continue
        # Key breaks an exact distance tie so the result does not depend on
        # the order the catalog happened to serialize its assets in.
        scored = (distance, candidate.key)
        if best is None or scored < best:
            best = scored
    return None if best is None else best[1]


def _match_by_name(candidates: list[Candidate], spec: BandSpec) -> str | None:
    """Return the candidate whose ``common_name`` is accepted by ``spec``.

    Args:
        candidates: Candidates for one scene.
        spec: The band being resolved.

    Returns:
        The winning asset key, or ``None`` if no name matches.
    """
    matches = [c.key for c in candidates if c.name is not None and c.name in spec.names]
    return min(matches) if matches else None


def _micrometres(raw: object) -> float | None:
    """Return a usable center wavelength, or ``None`` for anything else."""
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        return None
    value = float(raw)
    return value if math.isfinite(value) and value > 0 else None


def _common_name(raw: object) -> str | None:
    """Return a lowercased common name, or ``None`` for anything else."""
    if not isinstance(raw, str):
        return None
    return raw.strip().lower() or None
