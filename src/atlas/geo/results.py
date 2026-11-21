"""Typed analysis result, the sole number formatter, CSV spill and text previews."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "GeoResult",
    "grid_with_histogram",
    "histogram_text",
    "json_safe",
    "sig4",
    "spill_table",
    "text_grid",
    "write_csv",
    "write_result_json",
]

_RAMP = "_:-=+*#%@"
_MAX_CELLS = 4_000_000
_MAX_ARRAY_JSON = 100


def sig4(x: object) -> str:
    """Format a number with 4 significant digits; integer types print exactly.

    Args:
        x: A Python or numpy number, or None.

    Returns:
        The text; "n/a" for None/NaN, "inf"/"-inf" for infinities.
    """
    if x is None:
        return "n/a"
    if isinstance(x, (bool, int, np.integer)):
        return str(int(x))
    value = float(x)  # type: ignore[arg-type]
    if math.isnan(value):
        return "n/a"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    text = f"{value:.4g}"
    if value != 0 and not 1e-4 <= abs(value) < 1e9:
        return text
    out = format(Decimal(text), "f")
    return "0" if out in {"-0", "0.0"} else out


def json_safe(obj: Any) -> Any:
    """Make a value JSON-safe: NaN/inf become null and numpy values become Python.

    Raises:
        TypeError: An array has more than 100 elements.
    """
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        if obj.size > _MAX_ARRAY_JSON:
            raise TypeError(f"array of {obj.size} elements is too large for JSON; save a file")
        return json_safe(obj.tolist())
    if isinstance(obj, np.generic):
        return json_safe(obj.item())
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Path):
        return str(obj)
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    return str(obj)


def _cell(value: Any) -> str:
    if isinstance(value, (float, np.floating, np.integer)) and not isinstance(value, bool):
        return sig4(value)
    return "" if value is None else str(value)


@dataclass
class GeoResult:
    """What an analysis tool returns before it becomes a ``ToolResult``."""

    summary: list[str]
    table: list[dict[str, Any]] | None = None
    warnings: list[str] = field(default_factory=list)
    artifacts: list[Path] = field(default_factory=list)
    numbers: dict[str, Any] = field(default_factory=dict)
    run_id: str | None = None
    next: str | None = None
    table_name: str = "table"
    table_limit: int | None = None
    csv_rel: str | None = None

    @property
    def limit(self) -> int:
        """Rows shown inline."""
        if self.table_limit is not None:
            return self.table_limit
        return 12 if "class" in self.table_name else 20

    def _table_lines(self, rows: int) -> list[str]:
        table = self.table or []
        if not table:
            return []
        keys: list[str] = []
        for row in table:
            keys.extend(k for k in row if k not in keys)
        shown = table[: min(rows, self.limit)]
        lines: list[str] = []
        if shown:
            lines.append(" | ".join(keys))
            lines.extend(" | ".join(_cell(r.get(k)) for k in keys) for r in shown)
        hidden = len(table) - len(shown)
        if hidden > 0:
            where = self.csv_rel if self.csv_rel else "(not saved)"
            lines.append(f"+{hidden} more: {where}" if self.csv_rel else f"+{hidden} more {where}")
        return lines

    def _assemble(self, summary: int, rows: int) -> tuple[list[str], list[str], list[str]]:
        head = [f"warning: {w}" for w in self.warnings]
        body = [*self.summary[:summary], *self._table_lines(rows)]
        tail = [f"run: {self.run_id}"] if self.run_id else []
        if self.next:
            tail.append(f"next: {self.next}")
        return head, body, tail

    def render(self, max_chars: int = 1500) -> str:
        """Render for the model: warnings, summary, table, ``run:``, ``next:``.

        Over ``max_chars`` table rows go first, then summary lines; warnings, ``run:`` and
        ``next:`` are kept and only truncated per line as a last resort.
        """

        def size(parts: tuple[list[str], list[str], list[str]]) -> int:
            return len("\n".join([*parts[0], *parts[1], *parts[2]]))

        rows = self.limit
        summary = len(self.summary)
        parts = self._assemble(summary, rows)
        while size(parts) > max_chars and rows > 0 and self.table:
            rows -= 1
            parts = self._assemble(summary, rows)
        while size(parts) > max_chars and summary > 0:
            summary -= 1
            parts = self._assemble(summary, rows)
        head, body, tail = parts
        if size(parts) > max_chars:
            budget = max(max_chars - sum(len(t) + 1 for t in tail), 40)
            each = max(budget // max(len(head) + len(body), 1), 20)
            head = [_clip(line, each) for line in head]
            body = [_clip(line, each) for line in body]
        return "\n".join([*head, *body, *tail])


def _clip(line: str, width: int) -> str:
    return line if len(line) <= width else line[: max(width - 3, 1)] + "..."


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write raw rows as CSV; the header is the union of keys in first-seen order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_csv_bytes(rows))


def _csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    keys: list[str] = []
    for row in rows:
        keys.extend(k for k in row if k not in keys)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=keys, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8")


def spill_table(res: GeoResult, tables_dir: Path, *, base: Path | None = None) -> str | None:
    """Save the full table as a content-addressed CSV when it exceeds the inline limit.

    Args:
        res: The result; ``csv_rel`` is set on it.
        tables_dir: ``<workspace>/.atlas/analysis/tables``.
        base: Directory the returned path is relative to (the store root); default the
            directory above ``analysis/``.

    Returns:
        The CSV path relative to ``base``, or None when nothing was spilled.
    """
    if not res.table or len(res.table) <= res.limit:
        return None
    data = _csv_bytes(res.table)
    path = tables_dir / f"{res.table_name}-{hashlib.sha256(data).hexdigest()[:8]}.csv"
    tables_dir.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        tmp = path.with_name(f".tmp-{hashlib.sha256(data).hexdigest()[:8]}-{id(res)}.csv")
        tmp.write_bytes(data)
        tmp.replace(path)
    root = base if base is not None else tables_dir.parent.parent
    try:
        res.csv_rel = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        res.csv_rel = path.as_posix()
    return res.csv_rel


def write_result_json(res: GeoResult, path: Path, *, root: Path | None = None) -> None:
    """Write the machine-readable result next to the tables.

    Args:
        res: The result.
        path: Destination (``analysis/tables/<run_id>.result.json``).
        root: Base for relative artifact paths; defaults to the artifact root above ``path``.
    """
    base = root if root is not None else path.parents[2] if len(path.parents) > 2 else path.parent

    def rel(p: Path) -> str:
        try:
            return p.resolve().relative_to(base.resolve()).as_posix()
        except ValueError:
            return str(p)

    payload = {
        "summary": res.summary,
        "table": res.table,
        "warnings": res.warnings,
        "artifacts": [rel(Path(a)) for a in res.artifacts],
        "numbers": res.numbers,
        "run_id": res.run_id,
        "next": res.next,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2) + "\n", encoding="utf-8")


def _sample(arr: Any) -> NDArray[np.float64]:
    """Strided float64 view of at most ~4e6 cells (memmap-safe)."""
    a = arr if isinstance(arr, np.ndarray) else np.asarray(arr)
    if a.ndim != 2:
        a = a.reshape(a.shape[-2:]) if a.ndim > 2 else a.reshape(1, -1)
    step = max(1, math.ceil(math.sqrt(a.size / _MAX_CELLS))) if a.size > _MAX_CELLS else 1
    return np.asarray(a[::step, ::step], dtype=np.float64)


def _block_slices(n: int, parts: int) -> list[slice]:
    idx = np.array_split(np.arange(n), parts)
    return [slice(int(i[0]), int(i[-1]) + 1) if len(i) else slice(0, 0) for i in idx]


def _letters(classes: dict[int, str]) -> dict[int, str]:
    used: set[str] = set()
    out: dict[int, str] = {}
    for code in sorted(classes):
        name = classes[code]
        candidates = [c.upper() for c in name if c.isalnum()] + list("0123456789")
        pick = next((c for c in candidates if c not in used), "?")
        used.add(pick)
        out[code] = pick
    return out


def text_grid(
    arr: Any, *, rows: int = 12, cols: int = 24, classes: dict[int, str] | None = None
) -> str:
    """Render a coarse text map of a 2-D array.

    Continuous data uses the ramp ``_:-=+*#%@`` over the global p2-p98 of finite values with
    a final ``_=lo @=hi`` line; ``classes`` switches to modal class letters and a legend.
    ``.`` marks blocks without data.

    Args:
        arr: 2-D array (NaN = no data); sampled by stride so memmaps stay cheap.
        rows: Grid rows.
        cols: Grid columns.
        classes: Optional code-to-name map for categorical data.

    Returns:
        ``rows`` lines of ``cols`` characters plus one legend/ramp line.
    """
    a = _sample(arr)
    rs, cs = _block_slices(a.shape[0], rows), _block_slices(a.shape[1], cols)
    finite = np.isfinite(a)
    if classes is not None:
        letters = _letters(classes)
        lines = []
        for r in rs:
            line = ""
            for c in cs:
                found = a[r, c][finite[r, c]]
                ints = found[found >= 0].astype(np.int64)
                if ints.size == 0:
                    line += "."
                    continue
                code = int(np.bincount(ints).argmax())
                line += letters.get(code, "?")
            lines.append(line)
        legend = " ".join(f"{letters[k]}={classes[k]}" for k in sorted(classes))
        return "\n".join([*lines, legend])
    vals_all = a[finite]
    if vals_all.size:
        lo, hi = (float(v) for v in np.percentile(vals_all, [2, 98]))
    else:
        lo = hi = 0.0
    lines = []
    for r in rs:
        line = ""
        for c in cs:
            block = a[r, c]
            ok = finite[r, c]
            if block.size == 0 or not ok.any():
                line += "."
                continue
            mean = float(block[ok].mean())
            level = 4 if hi <= lo else int(np.clip(round((mean - lo) / (hi - lo) * 8), 0, 8))
            line += _RAMP[level]
        lines.append(line)
    return "\n".join([*lines, f"_={sig4(lo)} @={sig4(hi)}"])


def histogram_text(values: Any, *, bins: int = 10, width: int = 20) -> str:
    """Render a text histogram of finite values, one line per bin: ``lo..hi |#### n%``."""
    a = _sample(values).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return "no finite values"
    counts, edges = np.histogram(a, bins=bins)
    peak = int(counts.max()) or 1
    out = []
    for i, n in enumerate(counts):
        bar = "#" * (round(int(n) / peak * width) if width > 0 else 0)
        pct = round(100 * int(n) / a.size)
        label = f"{sig4(float(edges[i]))}..{sig4(float(edges[i + 1]))}"
        out.append(f"{label} |{bar} {pct}%")
    return "\n".join(out)


def grid_with_histogram(
    arr: Any, *, classes: dict[int, str] | None = None, max_chars: int = 700
) -> str:
    """Grid plus (continuous only) histogram within ``max_chars``.

    Shrinks the histogram width to 12, then drops the legend/ramp line, then the bar glyphs.
    """
    grid_lines = text_grid(arr, classes=classes).split("\n")
    legend = grid_lines[-1]
    grid = grid_lines[:-1]

    def hist(width: int) -> list[str]:
        return [] if classes is not None else histogram_text(arr, width=width).split("\n")

    attempts = [
        ([*grid, legend], hist(20)),
        ([*grid, legend], hist(12)),
        (grid, hist(12)),
        (grid, hist(0)),
    ]
    text = ""
    for head, tail in attempts:
        text = "\n".join([*head, *tail])
        if len(text) <= max_chars:
            break
    return text
