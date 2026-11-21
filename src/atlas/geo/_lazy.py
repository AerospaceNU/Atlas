"""Lazy access to optional extras and the cooperative cancel shim.

Invariant: rasterio, pyproj and onnxruntime are imported only inside functions.
"""

from __future__ import annotations

import importlib
import time
from typing import Any

from atlas.geo.errors import GeoCancelled, MissingExtraError

__all__ = [
    "check_cancel",
    "have_pyproj",
    "require_onnxruntime",
    "require_pyproj",
    "require_rasterio",
]


def _require(module: str, extra: str) -> Any:
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise MissingExtraError(extra, module) from exc


def require_rasterio() -> Any:
    """Import rasterio or raise ``MissingExtraError`` naming the ``geo`` extra."""
    return _require("rasterio", "geo")


def require_pyproj() -> Any:
    """Import pyproj or raise ``MissingExtraError`` naming the ``geo`` extra."""
    return _require("pyproj", "geo")


def require_onnxruntime() -> Any:
    """Import onnxruntime or raise ``MissingExtraError`` naming the ``onnx`` extra."""
    return _require("onnxruntime", "onnx")


def have_pyproj() -> bool:
    """Return True when pyproj can be imported."""
    try:
        importlib.import_module("pyproj")
    except ImportError:
        return False
    return True


def check_cancel(deadline: float | None = None) -> None:
    """Raise ``GeoCancelled`` when the deadline passed or the session was cancelled.

    Args:
        deadline: Absolute ``time.monotonic()`` value, or None for no cap.

    Raises:
        GeoCancelled: Deadline exceeded or cancellation requested. A no-op for
            cancellation while ``atlas.agent.scope`` is absent.
    """
    if deadline is not None and time.monotonic() > deadline:
        raise GeoCancelled("deadline exceeded: cap reached")
    try:
        cancel_requested = importlib.import_module("atlas.agent.scope").cancel_requested
    except (ImportError, AttributeError):
        return
    if cancel_requested():
        raise GeoCancelled("cancelled")
