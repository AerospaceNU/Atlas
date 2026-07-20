"""Consolidation layer: fan-out search, scene selection, and mosaic rendering.

Turns a single AOI + time window into one provenance-tracked `CompiledProduct`
spanning every satellite source. Agent-agnostic — the same entry points back
both interactive tool calls and batch precompute.
"""

from __future__ import annotations

from atlas.compile.aggregate import aggregate
from atlas.compile.mosaic import build_mosaic
from atlas.compile.pipeline import consolidate
from atlas.compile.product import (
    CompiledProduct,
    MosaicResult,
    SceneCatalog,
    SourceScenes,
)
from atlas.compile.select import (
    coverage_fraction,
    dedup,
    filter_cloud,
    representative_cloud,
    select,
    sort_clearest,
)

__all__ = [
    "CompiledProduct",
    "MosaicResult",
    "SceneCatalog",
    "SourceScenes",
    "aggregate",
    "build_mosaic",
    "consolidate",
    "coverage_fraction",
    "dedup",
    "filter_cloud",
    "representative_cloud",
    "select",
    "sort_clearest",
]
