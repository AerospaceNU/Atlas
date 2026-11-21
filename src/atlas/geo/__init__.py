"""Satellite analysis core. Attributes load lazily so importing stays cheap."""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS: dict[str, str] = {
    "open_stack": "stack",
    "Stack": "stack",
    "Grid": "grid",
    "GeoError": "errors",
    "workspace": "paths",
    "eval_expr": "expr",
    "finish": "provenance",
    "class_table": "zonal",
    "render_map": "render",
    "write_geotiff": "export",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{module}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *_EXPORTS])
