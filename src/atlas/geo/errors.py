"""Errors raised by ``atlas.geo``; tools catch them and return plain text."""

from __future__ import annotations

__all__ = [
    "ExprError",
    "GeoCancelled",
    "GeoError",
    "GridMismatch",
    "InputMismatch",
    "MissingExtraError",
    "StackError",
    "fix",
]


def fix(msg: str, cmd: str) -> str:
    """Append the corrective call or command to an error message.

    Args:
        msg: What is wrong.
        cmd: The exact corrective call or command.

    Returns:
        ``"<msg> Fix: <cmd>"``.
    """
    return f"{msg} Fix: {cmd}"


class GeoError(ValueError):
    """Base class: message says what is wrong and ends with ``Fix: ...``."""


class MissingExtraError(GeoError):
    """An optional dependency group is not installed."""

    def __init__(self, extra: str, module: str) -> None:
        self.extra = extra
        self.module = module
        super().__init__(fix(f"{module} is not installed.", f"uv sync --extra {extra}"))


class StackError(GeoError):
    """A stack is missing, malformed, mis-named or refused."""


class GridMismatch(GeoError):
    """Two rasters do not share a grid."""


class ExprError(GeoError):
    """A band-math expression is invalid."""


class InputMismatch(GeoError):
    """A tool or model input does not fit what it requires."""


class GeoCancelled(GeoError):
    """Work stopped on cancellation or a wall-clock cap.

    Tools test this before ``GeoError`` and return "cancelled".
    """
