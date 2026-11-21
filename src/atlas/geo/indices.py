"""Spectral and SAR index catalogue: pure data plus name binding over a stack.

Invariant: expressions are plain band-math text for ``atlas.geo.expr``; nothing here executes
code. Only stdlib is imported at module level.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, TypedDict

from atlas.geo.errors import ExprError, fix

if TYPE_CHECKING:
    from atlas.geo.stack import Stack

__all__ = [
    "INDEX_CATALOGUE",
    "IndexSpec",
    "bind_dates",
    "bind_names",
    "index_spec",
    "stack_dates",
]

_SAR = frozenset({"vv", "vh"})
_DATE_RE = re.compile(r"@(\d{8})$")


class IndexSpec(TypedDict):
    """One catalogue entry."""

    expr: str
    bands: tuple[str, ...]
    unit: str
    range: tuple[float, float]
    citation: str
    note: str


INDEX_CATALOGUE: dict[str, IndexSpec] = {
    "ndvi": {
        "expr": "(nir-red)/(nir+red)",
        "bands": ("nir", "red"),
        "unit": "1",
        "range": (-1.0, 1.0),
        "citation": "Rouse et al. 1974, NASA/GSFC TN E74-10093",
        "note": "vegetation greenness",
    },
    "ndwi": {
        "expr": "(green-nir)/(green+nir)",
        "bands": ("green", "nir"),
        "unit": "1",
        "range": (-1.0, 1.0),
        "citation": "McFeeters 1996, Int. J. Remote Sens. 17(7)",
        "note": "open water",
    },
    "mndwi": {
        "expr": "(green-swir16)/(green+swir16)",
        "bands": ("green", "swir16"),
        "unit": "1",
        "range": (-1.0, 1.0),
        "citation": "Xu 2006, Int. J. Remote Sens. 27(14)",
        "note": "open water, suppresses built-up",
    },
    "nbr": {
        "expr": "(nir-swir22)/(nir+swir22)",
        "bands": ("nir", "swir22"),
        "unit": "1",
        "range": (-1.0, 1.0),
        "citation": "Key and Benson 2006, FIREMON LA-1-1",
        "note": "burn ratio",
    },
    "ndmi": {
        "expr": "(nir-swir16)/(nir+swir16)",
        "bands": ("nir", "swir16"),
        "unit": "1",
        "range": (-1.0, 1.0),
        "citation": "Gao 1996, Remote Sens. Environ. 58(3)",
        "note": "canopy moisture",
    },
    "evi": {
        "expr": "2.5*(nir-red)/(nir+6*red-7.5*blue+1)",
        "bands": ("nir", "red", "blue"),
        "unit": "1",
        "range": (-1.0, 1.0),
        "citation": "Huete et al. 2002, Remote Sens. Environ. 83",
        "note": "needs blue; reflectance 0-1",
    },
    "savi": {
        "expr": "1.5*(nir-red)/(nir+red+0.5)",
        "bands": ("nir", "red"),
        "unit": "1",
        "range": (-1.5, 1.5),
        "citation": "Huete 1988, Remote Sens. Environ. 25(3)",
        "note": "soil-adjusted, L=0.5: (1+L)*(nir-red)/(nir+red+L)",
    },
    "bsi": {
        "expr": "((swir16+red)-(nir+blue))/((swir16+red)+(nir+blue))",
        "bands": ("swir16", "red", "nir", "blue"),
        "unit": "1",
        "range": (-1.0, 1.0),
        "citation": "Rikimaru et al. 2002, Trop. Ecol. 43(1-2)",
        "note": "bare soil; needs blue",
    },
    "ndbi": {
        "expr": "(swir16-nir)/(swir16+nir)",
        "bands": ("swir16", "nir"),
        "unit": "1",
        "range": (-1.0, 1.0),
        "citation": "Zha et al. 2003, Int. J. Remote Sens. 24(3)",
        "note": "built-up",
    },
    "ndsi": {
        "expr": "(green-swir16)/(green+swir16)",
        "bands": ("green", "swir16"),
        "unit": "1",
        "range": (-1.0, 1.0),
        "citation": "Hall et al. 1995, Remote Sens. Environ. 54(2)",
        "note": "snow; same formula as mndwi, read it as snow above about 0.4",
    },
    "vv_vh_ratio": {
        "expr": "vv-vh",
        "bands": ("vv", "vh"),
        "unit": "dB",
        "range": (0.0, 25.0),
        "citation": "VV/VH cross-polarisation ratio; dB difference equals the linear ratio",
        "note": "Sentinel-1 dB layers only",
    },
}


def index_spec(name: str) -> IndexSpec:
    """Return a catalogue entry; the lookup is case-insensitive.

    Raises:
        ExprError: Unknown index; the message lists the catalogue.
    """
    spec = INDEX_CATALOGUE.get(str(name).strip().lower())
    if spec is None:
        raise ExprError(
            fix(
                f"unknown index {str(name)[:40]!r}; catalogue: {', '.join(INDEX_CATALOGUE)}.",
                "band_math expression='(nir-red)/(nir+red)'",
            )
        )
    return spec


def _check_kind(spec: IndexSpec, index: str, band: str, stack: Stack, layer: str) -> None:
    meta = stack.layer(layer)
    if band in _SAR:
        if meta.kind == "backscatter" and meta.unit == "dB":
            return
        raise ExprError(
            fix(
                f"{index} needs backscatter in dB; layer '{layer}' is {meta.kind} "
                f"({meta.unit or 'no unit'}).",
                f"prepare_stack bands={list(spec['bands'])} sar={{db:true}}",
            )
        )
    if meta.kind != "reflectance":
        raise ExprError(
            fix(
                f"{index} needs reflectance layers; '{layer}' is {meta.kind}.",
                f"prepare_stack bands={list(spec['bands'])}",
            )
        )


def _lookup(stack: Stack, band: str, date: str | None) -> str | None:
    names = stack.names()
    if date is None:
        return band if band in names else None
    wanted = f"{band}@{date}"
    return wanted if wanted in names else None


def bind_names(
    index: str, spec: IndexSpec, stack: Stack, date: str | None = None
) -> dict[str, str]:
    """Map each common name of an index to a layer name, checking layer kinds.

    Args:
        index: Catalogue name (for messages).
        spec: The catalogue entry.
        stack: Stack to bind in.
        date: YYYYMMDD for a dated layer set; None for plain names.

    Returns:
        Common name to layer name.

    Raises:
        ExprError: A band is missing or has the wrong kind or unit.
    """
    bound: dict[str, str] = {}
    missing: list[str] = []
    for band in spec["bands"]:
        layer = _lookup(stack, band, date)
        if layer is None:
            missing.append(band)
        else:
            bound[band] = layer
    if missing:
        have = ", ".join(stack.names()[:12])
        raise ExprError(
            fix(
                f"{index} needs {', '.join(missing)}; stack '{stack.id}' has {have}.",
                f"prepare_stack bands={list(spec['bands'])}",
            )
        )
    for band, layer in bound.items():
        _check_kind(spec, index, band, stack, layer)
    return bound


def stack_dates(stack: Stack) -> list[str]:
    """Return dates carried in layer names (``name@YYYYMMDD``), sorted."""
    found = {m.group(1) for n in stack.names() if (m := _DATE_RE.search(n))}
    return sorted(found)


def bind_dates(
    index: str, spec: IndexSpec, stack: Stack
) -> tuple[list[tuple[str | None, dict[str, str]]], list[str]]:
    """Bind an index for every date of a stack (one entry, date None, for plain names).

    Returns:
        (date, binding) pairs for the computable dates, and warnings for skipped dates.

    Raises:
        ExprError: No date can be computed, or a kind check fails.
    """
    dates: list[str | None] = [*stack_dates(stack)] if stack_dates(stack) else [None]
    out: list[tuple[str | None, dict[str, str]]] = []
    warnings: list[str] = []
    last: ExprError | None = None
    for date in dates:
        try:
            out.append((date, bind_names(index, spec, stack, date)))
        except ExprError as exc:
            if "needs backscatter" in str(exc) or "needs reflectance" in str(exc):
                raise
            last = exc
            warnings.append(f"{date}: skipped, missing band")
    if not out:
        assert last is not None
        raise last
    return out, warnings
