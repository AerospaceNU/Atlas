"""ONNX Runtime backend for learned segmentation models.

Invariants: onnxruntime is imported only inside functions through
`atlas.geo._lazy.require_onnxruntime`; CPUExecutionProvider only; inputs are never
resampled or rescaled here (enforcement is the caller's). Normalisation must happen
inside the graph because NaN becomes 0 before the network sees it.
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from atlas.geo.errors import GeoError, fix
from atlas.models import tiling, weights
from atlas.models.runtime_api import RuntimeContext, RuntimeOutput

if TYPE_CHECKING:
    from atlas.models.manifest import ModelManifest, OnnxRuntimeSpec

__all__ = ["MAX_PIXELS", "OnnxRuntime"]

MAX_PIXELS = 36_000_000
_TILED_COST = 1.3
_SESSIONS: dict[tuple[str, int, int, int], Any] = {}
_LOCK = threading.Lock()


def _thread_count(cfg: OnnxRuntimeSpec) -> int:
    return cfg.threads or min(8, os.cpu_count() or 1)


def _session(path: Path, threads: int) -> Any:
    """Return a cached InferenceSession for the file and thread count."""
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns, threads)
    with _LOCK:
        cached = _SESSIONS.get(key)
        if cached is not None:
            return cached
        from atlas.geo._lazy import require_onnxruntime

        ort = require_onnxruntime()
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        _SESSIONS[key] = session
        return session


def _check_names(session: Any, cfg: OnnxRuntimeSpec) -> None:
    inputs = [i.name for i in session.get_inputs()]
    outputs = [o.name for o in session.get_outputs()]
    if cfg.input_name not in inputs or cfg.output_name not in outputs:
        raise GeoError(
            fix(
                f"model graph has inputs {inputs} and outputs {outputs}, but the manifest "
                f"names {cfg.input_name!r} and {cfg.output_name!r}.",
                "report this as a packaging error; the weights do not match the manifest",
            )
        )


def _open_output(shape: tuple[int, int]) -> NDArray[np.float32]:
    """Create a float32 memmap whose backing file is unlinked immediately (POSIX)."""
    directory = Path(tempfile.mkdtemp(prefix="atlas-onnx-"))
    path = directory / "out.npy"
    out: NDArray[np.float32] = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float32, shape=shape
    )
    path.unlink()
    directory.rmdir()
    return out


def _sigmoid(x: NDArray[np.float32]) -> NDArray[np.float32]:
    return (1.0 / (1.0 + np.exp(-x.astype(np.float64)))).astype(np.float32)


class OnnxRuntime:
    """Run a packaged ONNX segmentation model on a stack.

    Args:
        manifest: Optional manifest; `ctx.manifest` is used when omitted.
    """

    def __init__(self, manifest: ModelManifest | None = None) -> None:
        self._manifest = manifest

    def run(self, ctx: RuntimeContext) -> RuntimeOutput:
        """Run the model over the whole stack grid.

        Args:
            ctx: Run context; layers are read through `ctx.stack`.

        Returns:
            Output layer (float32, NaN for no data), QA planes, extras and notes.

        Raises:
            GeoError: For oversize or too-slow runs, missing weights, an unsupported
                normalisation or head, or a graph that does not match the manifest.
            GeoCancelled: On cancellation or deadline.
        """
        manifest = self._manifest or ctx.manifest
        cfg = manifest.runtime.onnx
        if cfg is None:
            raise GeoError(fix(f"{manifest.name} has no onnx runtime table.", "report a bug"))
        h, w = ctx.stack.shape[1:]
        tiled = h * w > cfg.tile.max_single_pass
        self._refuse_oversize(manifest, cfg, h, w, tiled, ctx)
        self._check_normalisation(cfg)
        if cfg.head == "identity":
            raise GeoError(
                fix("head 'identity' is unsupported.", "use head = 'argmax' or 'sigmoid'")
            )
        path = self._weights(manifest)
        threads = _thread_count(cfg)
        session = _session(path, threads)
        _check_names(session, cfg)
        names = [ctx.layers_by_role[band][-1] for band in manifest.input.bands]
        bands = [ctx.stack.band(name) for name in names]

        def read(r0: int, r1: int, c0: int, c1: int) -> NDArray[np.float32]:
            return np.stack([b[r0:r1, c0:c1] for b in bands]).astype(np.float32, copy=False)

        fn = self._model_fn(session, cfg)
        out = _open_output((h, w))
        started = time.monotonic()
        if tiled:
            tiles = tiling.run_tiled(
                read,
                fn,
                out,
                size=cfg.tile.size,
                margin=cfg.tile.margin,
                deadline=ctx.deadline,
                pad_multiple=1,
            )
        else:
            tiling.run_single(read, fn, out, deadline=ctx.deadline)
            tiles = 1
        seconds = time.monotonic() - started
        notes = ["per-tile normalisation: tile-extent dependent"] if tiled else []
        return RuntimeOutput(
            layers={manifest.output.layer: out},
            qa_planes=self._planes(manifest, out),
            extras={"tiles": tiles, "threads": threads, "seconds": seconds, "tiled": tiled},
            notes=notes,
        )

    @staticmethod
    def _refuse_oversize(
        manifest: ModelManifest,
        cfg: OnnxRuntimeSpec,
        h: int,
        w: int,
        tiled: bool,
        ctx: RuntimeContext,
    ) -> None:
        megapixels = h * w / 1e6
        if h * w > MAX_PIXELS:
            raise GeoError(
                fix(
                    f"{manifest.name} input is {megapixels:.1f} MP; the limit is "
                    f"{MAX_PIXELS / 1e6:.0f} MP.",
                    "use a smaller bbox or a coarser resolution when preparing the stack",
                )
            )
        estimate = cfg.ms_per_mp * megapixels / 1000 * (_TILED_COST if tiled else 1.0)
        left = ctx.seconds_left()
        limit = cfg.max_seconds if left is None else min(cfg.max_seconds, left)
        if cfg.ms_per_mp > 0 and estimate > limit:
            raise GeoError(
                fix(
                    f"{manifest.name} on {megapixels:.1f} MP needs about {estimate:.0f} s; "
                    f"the cap is {limit:.0f} s.",
                    "use a smaller bbox or a coarser resolution when preparing the stack",
                )
            )

    @staticmethod
    def _check_normalisation(cfg: OnnxRuntimeSpec) -> None:
        if cfg.normalisation != "in_graph":
            label = cfg.normalisation if isinstance(cfg.normalisation, str) else "mean/std"
            raise GeoError(
                fix(
                    f"normalisation {label} unsupported by M-01; NaN becomes 0 before the "
                    "network, so only in_graph normalisation is exact.",
                    "use a model whose graph normalises its own input",
                )
            )

    @staticmethod
    def _weights(manifest: ModelManifest) -> Path:
        state = weights.status(manifest)
        if state.status != "ready" or state.path is None:
            raise GeoError(
                fix(
                    f"weights {state.status}.",
                    f"atlas models pull {manifest.name} (or --from PATH)",
                )
            )
        return state.path

    @staticmethod
    def _model_fn(session: Any, cfg: OnnxRuntimeSpec) -> Any:
        head = cfg.head

        def fn(block: NDArray[np.float32]) -> NDArray[np.float32]:
            raw: NDArray[np.float32] = np.asarray(
                session.run([cfg.output_name], {cfg.input_name: block})[0], dtype=np.float32
            )
            if head == "argmax":
                codes: NDArray[np.float32] = raw.argmax(axis=1)[0].astype(np.float32)
                return codes
            if head == "sigmoid" and raw.shape[1] == 1:
                return _sigmoid(raw[0, 0])
            raise GeoError(
                fix(
                    f"head {head!r} with {raw.shape[1]} output channels is unsupported.",
                    "use head = 'argmax', or 'sigmoid' with one channel",
                )
            )

        return fn

    @staticmethod
    def _planes(manifest: ModelManifest, out: NDArray[np.float32]) -> dict[str, NDArray[np.bool_]]:
        by_name = {name: code for code, name in manifest.output.classes.items()}
        planes: dict[str, NDArray[np.bool_]] = {}
        for plane, names in manifest.output.qa_planes.items():
            codes = [by_name[n] for n in names]
            planes[plane] = np.isin(out, codes)
        return planes
