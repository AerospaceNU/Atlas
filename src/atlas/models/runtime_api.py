"""Runtime contract types for model manifests v2.

Invariants: types and a lazy loader only; runtime modules import on demand so optional
extras are never imported here.
"""

from __future__ import annotations

import importlib
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from atlas.geo.stack import Stack
    from atlas.models.manifest import ModelManifest

__all__ = [
    "RUNTIMES",
    "ModelRuntime",
    "ParamValue",
    "RuntimeContext",
    "RuntimeOutput",
    "load_runtime",
]

RUNTIMES: dict[str, str] = {
    "rules": "atlas.models.runtime_rules:RulesRuntime",
    "onnx": "atlas.models.runtime_onnx:OnnxRuntime",
}

ParamValue = int | float | str | bool


@dataclass(frozen=True)
class RuntimeContext:
    """Inputs for one model run.

    Attributes:
        manifest: The validated manifest.
        stack: The input stack.
        layers_by_role: Layer names per role in ascending date order; pair models use
            index 0 as pre and -1 as post.
        params: Resolved parameter values.
        valid: Boolean validity mask on the stack grid.
        deadline: Absolute `time.monotonic()` deadline, or None.
        weights_path: Verified weights file for learned models, else None.
    """

    manifest: ModelManifest
    stack: Stack
    layers_by_role: Mapping[str, tuple[str, ...]]
    params: Mapping[str, ParamValue]
    valid: NDArray[np.bool_]
    deadline: float | None = None
    weights_path: Path | None = None

    def seconds_left(self) -> float | None:
        """Return seconds until the deadline, or None when there is none."""
        if self.deadline is None:
            return None
        return self.deadline - time.monotonic()


@dataclass
class RuntimeOutput:
    """Result of one model run."""

    layers: dict[str, NDArray[Any]] = field(default_factory=dict)
    qa_planes: dict[str, NDArray[np.bool_]] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


class ModelRuntime(Protocol):
    """A model backend: pure function of the context."""

    def run(self, ctx: RuntimeContext) -> RuntimeOutput: ...


def load_runtime(name: str) -> ModelRuntime:
    """Import and instantiate the runtime registered under `name`.

    Raises:
        ValueError: If `name` is not in `RUNTIMES`.
    """
    target = RUNTIMES.get(name)
    if target is None:
        known = ", ".join(repr(k) for k in RUNTIMES)
        raise ValueError(f"unknown model runtime {name!r}; known: {known}")
    module_name, _, cls_name = target.partition(":")
    cls = getattr(importlib.import_module(module_name), cls_name)
    runtime: ModelRuntime = cls()
    return runtime
