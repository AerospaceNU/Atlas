"""Shared plumbing for the satellite analysis tools (private; discovery skips it).

Only pydantic, stdlib and ``atlas.agent.contracts`` load at import time; ``atlas.geo`` is
imported inside functions so the optional extras never load at discovery.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from atlas.agent.contracts import ToolResult

if TYPE_CHECKING:
    from atlas.agent.artifacts import LocalArtifactStore
    from atlas.geo.results import GeoResult
    from atlas.geo.stack import Stack

__all__ = ["analysis_dir", "error_result", "result_to_tool_result", "stack_from_args"]

_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp"})


def analysis_dir(store: LocalArtifactStore) -> Path:
    """Return the store's analysis directory, creating the standard subdirectories."""
    from atlas.geo import paths

    return paths.analysis_dir(store, ensure=True)


def stack_from_args(store: LocalArtifactStore, ref: str) -> Stack:
    """Open a stack by id, path or "latest".

    Raises:
        GeoError: The stack does not exist; the message lists existing ids.
    """
    from atlas.geo import paths
    from atlas.geo.errors import GeoError
    from atlas.geo.stack import Stack

    try:
        return Stack.open(ref, store=store)
    except GeoError as exc:
        if "no committed stack" not in str(exc):
            raise
        ids = ", ".join(paths.list_stack_ids(paths.analysis_dir(store))[:10]) or "none"
        raise GeoError(
            f"{str(exc).split(' Fix:')[0]} Fix: use an id from [{ids}] or "
            "prepare_stack to build one"
        ) from exc


def _attach_default() -> bool:
    import importlib

    try:
        render = importlib.import_module("atlas.geo.render")
    except ImportError:
        return True
    check = getattr(render, "attach_enabled", None)
    return bool(check()) if callable(check) else True


def _rel(store: LocalArtifactStore, path: Path) -> str:
    try:
        return store.relative(path)
    except (OSError, ValueError):
        return str(path)


def result_to_tool_result(
    res: GeoResult,
    store: LocalArtifactStore,
    *,
    tool: str = "",
    arguments: Mapping[str, Any] | None = None,
    stack_ids: Iterable[str] = (),
    bytes_fetched: int = 0,
    attach: bool | None = None,
) -> ToolResult:
    """Turn a ``GeoResult`` into a ``ToolResult``, recording the run.

    Order: spill the table, append the run record, write ``<run_id>.result.json``, render.
    A failure to record the run adds a warning and the tool still returns.

    Args:
        res: The analysis result.
        store: Session store.
        tool: Tool name for the run record.
        arguments: Tool arguments for the record.
        stack_ids: Stacks read or written.
        bytes_fetched: Network bytes spent.
        attach: Keep image artifacts in ``ToolResult.artifacts``; default follows
            ``render.attach_enabled()``.

    Returns:
        The tool result.
    """
    from atlas.geo import paths
    from atlas.geo.provenance import append_run
    from atlas.geo.results import spill_table, write_result_json

    adir = paths.analysis_dir(store)
    artifacts = [Path(a) for a in res.artifacts]
    try:
        csv_rel = spill_table(res, adir / "tables", base=store.root)
    except OSError as exc:
        csv_rel = None
        res.warnings.append(f"table not saved: {exc}")
    if csv_rel:
        artifacts.append(store.root / csv_rel)
    outputs = [_rel(store, p) for p in artifacts]
    try:
        res.run_id = append_run(
            adir,
            {
                "tool": tool,
                "arguments": dict(arguments or {}),
                "outputs": outputs,
                "stack_ids": list(stack_ids),
                "summary": res.numbers,
                "warnings": res.warnings,
                "bytes_fetched": bytes_fetched,
            },
        )
    except (OSError, TypeError, ValueError) as exc:
        res.run_id = None
        res.warnings.append(f"run not recorded: {exc}")
    if res.run_id:
        try:
            write_result_json(res, adir / "tables" / f"{res.run_id}.result.json", root=store.root)
        except (OSError, TypeError, ValueError) as exc:
            res.warnings.append(f"result.json not written: {exc}")
    keep_images = _attach_default() if attach is None else attach
    listed = [
        _rel(store, p) for p in artifacts if keep_images or p.suffix.lower() not in _IMAGE_SUFFIXES
    ]
    return ToolResult(text=res.render(), artifacts=listed)


def error_result(exc: Exception) -> ToolResult:
    """Return plain-text error output for a ``GeoError``: no traceback, no run record."""
    from atlas.geo.errors import GeoCancelled

    if isinstance(exc, GeoCancelled):
        return ToolResult(text="cancelled")
    return ToolResult(text=str(exc)[:1500])
