"""Provenance: file hashes, citations and the append-only run manifest (``runs.jsonl``)."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from atlas.geo.results import json_safe

__all__ = [
    "append_run",
    "bytes_fetched_total",
    "citations",
    "file_sha256",
    "finish",
    "read_runs",
    "runs_path",
]

_ID_RE = re.compile(r"^r(\d+)$")
_CITATIONS: tuple[tuple[str, str], ...] = (
    ("sentinel-1", "Contains modified Copernicus Sentinel data {year}"),
    ("sentinel1", "Contains modified Copernicus Sentinel data {year}"),
    ("sentinel-2", "Contains modified Copernicus Sentinel data {year}"),
    ("sentinel2", "Contains modified Copernicus Sentinel data {year}"),
    ("landsat", "Landsat data courtesy of the U.S. Geological Survey"),
    ("hls", "NASA HLS (Claverie et al. 2018)"),
    ("worldcover", "ESA WorldCover 10 m, CC BY 4.0"),
)


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    """Return the hex SHA-256 of a file, streamed."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def citations(source: str, *, year: int | str | None = None) -> list[str]:
    """Return attribution lines for a source id or collection; unknown gives [].

    Args:
        source: Source or collection name, matched by prefix.
        year: Year for Copernicus attributions; defaults to the current year.

    Returns:
        Citation strings.
    """
    name = source.lower()
    stamp = str(year) if year is not None else str(datetime.now(UTC).year)
    for prefix, text in _CITATIONS:
        if name.startswith(prefix):
            return [text.format(year=stamp)]
    return []


def runs_path(analysis_dir: Path) -> Path:
    """Return the manifest path inside an analysis directory."""
    return analysis_dir / "runs.jsonl"


def _lock(fd: int) -> None:
    try:
        import fcntl
    except ImportError:  # pragma: no cover - POSIX-only agent
        return
    fcntl.flock(fd, fcntl.LOCK_EX)


def _parse_lines(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_bytes().decode("utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _max_id(rows: Iterable[dict[str, Any]]) -> int:
    best = 0
    for row in rows:
        match = _ID_RE.match(str(row.get("id", "")))
        if match:
            best = max(best, int(match.group(1)))
    return best


def _ends_with_newline(path: Path) -> bool:
    size = path.stat().st_size if path.exists() else 0
    if size == 0:
        return True
    with open(path, "rb") as handle:
        handle.seek(-1, os.SEEK_END)
        return handle.read(1) == b"\n"


def append_run(analysis_dir: Path, record: dict[str, Any]) -> str:
    """Append a run record under an exclusive lock and return its id.

    The id is one past the highest valid id seen; torn lines are skipped and a missing
    trailing newline is repaired before appending. Any caller-supplied id is ignored.

    Args:
        analysis_dir: Analysis directory (created when missing).
        record: Fields overriding the defaults (tool, arguments, outputs, ...).

    Returns:
        The new id, ``rNNNN``.
    """
    analysis_dir.mkdir(parents=True, exist_ok=True)
    path = runs_path(analysis_dir)
    lock_fd = os.open(str(path) + ".lock", os.O_CREAT | os.O_RDWR, 0o644)
    try:
        _lock(lock_fd)
        run_id = f"r{_max_id(_parse_lines(path)) + 1:04d}"
        full: dict[str, Any] = {
            "id": run_id,
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "tool": "",
            "arguments": {},
            "outputs": [],
            "stack_ids": [],
            "summary": {},
            "warnings": [],
            "bytes_fetched": 0,
        }
        full.update({k: v for k, v in record.items() if k != "id"})
        full["summary"] = json_safe(full["summary"])
        line = json.dumps(full, separators=(",", ":"), default=str) + "\n"
        prefix = b"" if _ends_with_newline(path) else b"\n"
        with open(path, "ab") as handle:
            handle.write(prefix + line.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(lock_fd)
    return run_id


def read_runs(analysis_dir: Path) -> list[dict[str, Any]]:
    """Return valid run records in file order; bad lines are skipped."""
    return _parse_lines(runs_path(analysis_dir))


def bytes_fetched_total(analysis_dir: Path) -> int:
    """Return the total bytes fetched over all recorded runs."""
    return sum(int(r.get("bytes_fetched") or 0) for r in read_runs(analysis_dir))


def _script_arguments(root: Path) -> dict[str, Any]:
    argv0 = sys.argv[0] if sys.argv else ""
    if not argv0 or not os.path.isfile(argv0):
        return {}
    script = Path(argv0).resolve()
    try:
        shown = script.relative_to(root.resolve()).as_posix()
    except ValueError:
        shown = script.name
    return {"script": shown, "script_sha256": file_sha256(script)}


def finish(
    numbers: dict[str, Any] | None = None,
    *,
    outputs: Iterable[str | Path] = (),
    tool: str = "script",
) -> str:
    """Record a script run, print ``run: rNNNN`` and return the id.

    Args:
        numbers: Key numbers to store as the run summary.
        outputs: Files the script wrote.
        tool: Tool label for the record.

    Returns:
        The run id.
    """
    from atlas.geo import paths

    adir = paths.analysis_dir()
    root = adir.parent
    run_id = append_run(
        adir,
        {
            "tool": tool,
            "arguments": _script_arguments(root),
            "outputs": [str(o) for o in outputs],
            "summary": numbers or {},
        },
    )
    print(f"run: {run_id}")
    return run_id
