"""Temporal analysis over multi-date stacks: composites, zonal time series and trends.

Pure numpy; no tool code and no optional extra. Invariants: every reduction is NaN-aware, work
proceeds one layer at a time over row strips (never a full T x H x W cube), and only
Stack.valid decides which pixels are usable on a date.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import re
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date as _date
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from atlas.geo._lazy import check_cancel
from atlas.geo.errors import GeoError, GridMismatch, fix

if TYPE_CHECKING:
    from atlas.geo.grid import Grid
    from atlas.geo.stack import Stack

__all__ = [
    "TIME_CAP_S",
    "CompositeResult",
    "Method",
    "SeriesPoint",
    "TrendResult",
    "base_layers",
    "build_composite",
    "choose_trend",
    "date_valid",
    "decimal_year",
    "fit_trend",
    "series_points",
    "zone_mask",
]

Method = Literal["median", "mean", "max", "clearest"]
TIME_CAP_S = 600.0
MAX_LAYERS = 8
MIN_VALID_PCT = 20.0
_STRIP_BYTES = 12_000_000
_MAX_CELLS = 200_000_000
_MAX_FEATURES = 500
_MAX_VERTICES = 500
_REFUSED_KINDS = frozenset({"categorical", "probability", "count"})
_DATE_RE = re.compile(r"@(\d{8})$")
_PCT_SAMPLE = 1_000_000


@dataclass(frozen=True)
class SeriesPoint:
    """One date of a zonal series."""

    date: str
    value: float
    n_valid: int
    n_zone: int
    valid_pct: float
    flagged: bool


@dataclass(frozen=True)
class TrendResult:
    """A fitted trend; excluded counts dates left out of the fit."""

    kind: str
    slope_per_year: float
    n: int
    r2: float
    excluded: int


@dataclass
class CompositeResult:
    """What a composite run measured."""

    method: str
    dates: list[str]
    layers: list[dict[str, Any]]
    count_min: int
    count_median: float
    count_max: int
    valid_px: int
    aoi_px: int
    zero_count_px: int
    warnings: list[str] = field(default_factory=list)

    @property
    def valid_fraction(self) -> float:
        """Share of AOI pixels with a composite value in the first layer."""
        return self.valid_px / self.aoi_px if self.aoi_px else 0.0


# --------------------------------------------------------------------------- helpers


def decimal_year(date: str) -> float:
    """Return a YYYYMMDD (or YYYY-MM-DD) date as ordinal days over 365.2425."""
    digits = date.replace("-", "")[:8]
    day = _date(int(digits[:4]), int(digits[4:6]), int(digits[6:8]))
    return day.toordinal() / 365.2425


def _layer_date(name: str, datetime_: str | None) -> str | None:
    match = _DATE_RE.search(name)
    if match:
        return match.group(1)
    if datetime_:
        digits = re.sub(r"\D", "", datetime_[:10])
        if len(digits) == 8:
            return digits
    return None


def _base_name(name: str) -> str:
    return _DATE_RE.sub("", name)


def base_layers(stack: Stack) -> dict[str, list[str]]:
    """Map each base layer name (red) to its sorted dates (YYYYMMDD)."""
    found: dict[str, set[str]] = defaultdict(set)
    for meta in stack.meta.layers:
        date = _layer_date(meta.name, meta.datetime)
        if date:
            found[_base_name(meta.name)].add(date)
    return {name: sorted(dates) for name, dates in found.items()}


def _check_extra(stack: Stack, extra: Sequence[Stack]) -> None:
    for other in extra:
        if not stack.grid.same_as(other.grid):
            raise GridMismatch(
                fix(
                    f"mask_with stack '{other.id}' is on a different grid than '{stack.id}'.",
                    "prepare_stack both on one grid (same bbox, resolution and crs)",
                )
            )


def date_valid(
    stack: Stack, layer: str, date: str, mask: str, extra: Sequence[Stack]
) -> NDArray[np.bool_]:
    """Return the usable-pixel mask of one layer on one date.

    This is the single adaptation point for per-date QA: it asks Stack.valid about the
    layer's own dated name, so cloud@DATE planes apply only to that date. A single-date
    extra stack applies its planes to every date; a multi-date one matches by date.
    """
    return stack.valid(mask, layers=[_dated_name(stack, layer, date)], extra=extra)


def _dated_name(stack: Stack, layer: str, date: str) -> str:
    return stack.layer(layer, date=date).name


# --------------------------------------------------------------------------- zones


def _polygons(obj: Any) -> list[list[Any]]:
    """Return polygons (each a list of rings) from GeoJSON geometry or features."""
    kind = obj.get("type") if isinstance(obj, dict) else None
    if kind == "FeatureCollection":
        feats = obj.get("features") or []
        if len(feats) > _MAX_FEATURES:
            raise GeoError(
                fix(f"zone has {len(feats)} features (max {_MAX_FEATURES}).", "simplify the zone")
            )
        return [p for f in feats for p in _polygons(f)]
    if kind == "Feature":
        return _polygons(obj.get("geometry") or {})
    if kind == "Polygon":
        return [list(obj.get("coordinates") or [])]
    if kind == "MultiPolygon":
        return [list(p) for p in obj.get("coordinates") or []]
    raise GeoError(
        fix(
            f"zone geometry type {kind!r} is not supported.",
            "pass a Polygon, MultiPolygon, Feature or FeatureCollection in lon/lat",
        )
    )


def _load_zone(zone: dict[str, Any] | str, root: Path | None) -> dict[str, Any]:
    if isinstance(zone, dict):
        return zone
    text = zone.strip()
    try:
        if text.startswith("{"):
            data = json.loads(text)
        else:
            path = Path(text)
            if root is not None and not path.is_absolute():
                path = root / path
            data = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError) as exc:
        raise GeoError(
            fix(f"zone could not be read ({exc}).", "pass GeoJSON or a readable .geojson path")
        ) from exc
    if not isinstance(data, dict):
        raise GeoError(fix("zone file is not a GeoJSON object.", "pass a GeoJSON object"))
    return data


def zone_mask(
    grid: Grid, zone: dict[str, Any] | str | None, *, root: Path | None = None
) -> NDArray[np.bool_]:
    """Rasterise a lon/lat GeoJSON zone onto the grid; None selects every pixel.

    Args:
        grid: Target pixel grid.
        zone: GeoJSON dict, inline JSON text, a workspace-relative path, or None.
        root: Directory that relative zone paths resolve against.

    Returns:
        Bool (H, W); holes are excluded.

    Raises:
        GeoError: Unreadable or oversized zone, non lon/lat coordinates, or no overlap.
    """
    if zone is None:
        return np.ones(grid.shape, dtype=bool)
    mask = np.zeros(grid.shape, dtype=bool)
    for rings in _polygons(_load_zone(zone, root)):
        if not rings:
            continue
        for ring in rings:
            if len(ring) > _MAX_VERTICES:
                raise GeoError(
                    fix(f"zone ring has {len(ring)} vertices (max {_MAX_VERTICES}).", "simplify")
                )
            arr = np.asarray(ring, dtype=float)
            if arr.ndim != 2 or arr.shape[1] < 2 or (np.abs(arr[:, :2]) > 180).any():
                raise GeoError(
                    fix("zone must be lon/lat degrees.", "reproject the zone to EPSG:4326")
                )
        mask |= grid.rasterize(rings)
    if not mask.any():
        raise GeoError(
            fix("zone does not intersect the stack.", "inspect_stack shows bounds_wgs84")
        )
    return mask


# --------------------------------------------------------------------------- composite


def _strip_rows(n_dates: int, width: int) -> int:
    return max(1, _STRIP_BYTES // (n_dates * width * 4))


def _fill_block(
    stack: Stack,
    names: dict[str, str],
    dates: Sequence[str],
    mask: str,
    extra: Sequence[Stack],
    r0: int,
    r1: int,
) -> NDArray[np.float32]:
    """Return the (T, rows, W) block of one layer with unusable pixels set to NaN."""
    view = stack.window(r0, r1, 0, stack.grid.width)
    blk = np.empty((len(dates), r1 - r0, stack.grid.width), dtype=np.float32)
    for i, date in enumerate(dates):
        blk[i] = view.band(names[date])
        blk[i][~view.valid(mask, layers=[names[date]], extra=extra)] = np.nan
    return blk


def _median(blk: NDArray[np.float32], cnt: NDArray[np.int64]) -> NDArray[np.float32]:
    blk.sort(axis=0)
    lo = np.take_along_axis(blk, ((cnt - 1).clip(0) // 2)[None], 0)[0]
    hi = np.take_along_axis(blk, (cnt // 2)[None], 0)[0]
    mid: NDArray[np.float32] = ((lo.astype(np.float64) + hi) / 2).astype(np.float32)
    return mid


def _mean(blk: NDArray[np.float32], cnt: NDArray[np.int64]) -> NDArray[np.float32]:
    total = np.zeros(blk.shape[1:], dtype=np.float64)
    for plane in blk:
        total += np.where(np.isfinite(plane), plane, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (total / cnt).astype(np.float32)


def _clearest(blk: NDArray[np.float32], order: Sequence[int]) -> NDArray[np.float32]:
    out = np.full(blk.shape[1:], np.nan, dtype=np.float32)
    for i in order:
        take = np.isnan(out) & np.isfinite(blk[i])
        out[take] = blk[i][take]
    return out


def _reduce(
    blk: NDArray[np.float32], method: str, order: Sequence[int]
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    cnt = np.isfinite(blk).sum(axis=0).astype(np.int64)
    if method == "median":
        out = _median(blk, cnt)
    elif method == "mean":
        out = _mean(blk, cnt)
    elif method == "max":
        out = np.asarray(np.fmax.reduce(blk, axis=0), dtype=np.float32)
    else:
        out = _clearest(blk, order)
    return out, cnt


def _inside(view: Stack) -> NDArray[np.bool_]:
    """Pixels not flagged outside the AOI polygon."""
    return view.valid(["outside_aoi"], layers=[], extra=())


def _date_fractions(
    stack: Stack,
    names: dict[str, str],
    dates: Sequence[str],
    mask: str,
    extra: Sequence[Stack],
    deadline: float,
) -> dict[str, float]:
    """Per-date share of AOI pixels that are usable (first layer)."""
    height, width = stack.grid.shape
    step = max(1, _STRIP_BYTES // (width * 8))
    good = dict.fromkeys(dates, 0)
    inside = 0
    for r0 in range(0, height, step):
        view = stack.window(r0, min(r0 + step, height), 0, width)
        inside += int(_inside(view).sum())
        for date in dates:
            check_cancel(deadline)
            good[date] += int(view.valid(mask, layers=[names[date]], extra=extra).sum())
    return {d: good[d] / inside if inside else 0.0 for d in dates}


def _pick_layers(stack: Stack, layers: Sequence[str] | None) -> tuple[list[str], list[str]]:
    table = base_layers(stack)
    warnings: list[str] = []
    if layers:
        chosen = list(dict.fromkeys(layers))
        for name in chosen:
            if name not in table:
                raise GeoError(
                    fix(
                        f"layer {name!r} has no dated layers in '{stack.id}'.",
                        f"composite layers=[...] from {sorted(table)}",
                    )
                )
    else:
        chosen = [n for n in table if stack.layer(n, date=table[n][0]).kind not in _REFUSED_KINDS]
        if len(chosen) > MAX_LAYERS:
            warnings.append(f"{len(chosen)} layers; composited the first {MAX_LAYERS}")
            chosen = chosen[:MAX_LAYERS]
    for name in chosen:
        kind = stack.layer(name, date=table[name][0]).kind
        if kind in _REFUSED_KINDS:
            raise GeoError(
                fix(
                    f"layer {name!r} is {kind}; a per-pixel median of codes is meaningless.",
                    "classify the composite instead (composite a continuous layer first)",
                )
            )
    if not chosen:
        raise GeoError(fix("no dated continuous layers to composite.", "prepare_stack bands=[...]"))
    return chosen, warnings


def _sample_stats(arr: NDArray[np.float32]) -> tuple[float, float, float]:
    flat = np.asarray(arr).reshape(-1)
    flat = flat[:: max(1, flat.size // _PCT_SAMPLE)]
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return math.nan, math.nan, math.nan
    p2, med, p98 = np.percentile(flat, [2, 50, 98])
    return float(p2), float(med), float(p98)


def _limit_warnings(items: list[str], limit: int = 5) -> list[str]:
    if len(items) <= limit:
        return list(items)
    return [*items[:limit], f"+{len(items) - limit} more"]


def build_composite(
    stack: Stack,
    layers: Sequence[str] | None,
    method: Method,
    *,
    mask: str = "clear",
    extra: Sequence[Stack] = (),
    min_count: int = 1,
    name: str | None = None,
    overwrite: bool = False,
    analysis_dir: Path | None = None,
) -> tuple[Stack, CompositeResult]:
    """Write a per-pixel composite of a multi-date stack as a new stack.

    Each layer is reduced over its usable dates (NaN-aware) one row strip at a time, so peak
    memory is about one T x rows x W float32 block. Output layers keep the base names and a
    count layer holds the fewest usable dates per pixel across the layers (0 where none,
    never NaN; for clearest it is 1 where a date supplied the value).

    Args:
        stack: Multi-date source stack.
        layers: Base layer names, or None for every continuous layer (at most 8).
        method: median, mean, max or clearest.
        mask: "clear" or "none".
        extra: Same-grid stacks whose QA planes also mask pixels.
        min_count: Pixels with fewer usable dates become NaN.
        name: Output stack id; default derived from the parent.
        overwrite: Replace an existing stack with the same id and different parameters.
        analysis_dir: Directory that receives the stack; default the parent's.

    Returns:
        The committed stack and what was measured.

    Raises:
        GeoError: One date only, refused layer kind, oversize output, GridMismatch, time cap.
    """
    from atlas.geo.stack import LayerMeta, Stack, hash_params

    _check_extra(stack, extra)
    chosen, warnings = _pick_layers(stack, layers)
    table = base_layers(stack)
    dates = sorted(set.intersection(*(set(table[n]) for n in chosen)))
    if len(dates) < 2:
        raise GeoError(
            "composite needs a multi-date stack. Fix: prepare_stack scenes=[...] with >=2 dates"
        )
    height, width = stack.grid.shape
    if (len(chosen) + 1) * height * width > _MAX_CELLS:
        raise GeoError(
            fix("composite output is over 200e6 cells.", "pass layers=[...] or a smaller AOI")
        )
    parent = stack.meta
    digest = hash_params(
        {
            "parent": parent.params_hash,
            "method": method,
            "layers": chosen,
            "mask": mask,
            "extra": [[e.id, e.meta.params_hash] for e in extra],
            "min_count": min_count,
        }
    )
    stack_id = name or f"{stack.id[:40]}-composite-{digest[:6]}"
    adir = analysis_dir or stack.analysis_dir
    scratch_dir = adir / ".scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    scratch = scratch_dir / f"{os.getpid()}-{digest[:6]}.npy"
    deadline = time.monotonic() + TIME_CAP_S
    names = {n: {d: _dated_name(stack, n, d) for d in dates} for n in chosen}
    try:
        fractions = _date_fractions(stack, names[chosen[0]], dates, mask, extra, deadline)
        low = [f"{d} {100 * f:.3g}%" for d, f in fractions.items() if f < MIN_VALID_PCT / 100]
        if low:
            warnings.append(f"{len(low)} of {len(dates)} dates under 20% valid: " + ", ".join(low))
        order = sorted(range(len(dates)), key=lambda i: -fractions[dates[i]])
        if method == "max" and any(
            stack.layer(n, date=dates[0]).kind == "reflectance" for n in chosen
        ):
            warnings.append("max on reflectance favours bright/haze outliers; prefer median")
        out = np.lib.format.open_memmap(
            scratch, mode="w+", dtype=np.float32, shape=(len(chosen) + 1, height, width)
        )
        count = _run_strips(
            stack, chosen, names, dates, method, mask, extra, min_count, order, out, deadline
        )
        result = _summarise(stack, out, count, chosen, dates, method, warnings)
        out.flush()
        span = f"composite:{method} n={len(dates)} first={dates[0]} last={dates[-1]} mask={mask}"
        metas = [
            stack.layer(n, date=dates[0]).model_copy(
                update={
                    "name": n,
                    "datetime": None,
                    "index": 0,
                    "processing": [*stack.layer(n, date=dates[0]).processing, span],
                }
            )
            for n in chosen
        ]
        metas.append(
            LayerMeta(
                name="count",
                kind="count",
                unit="dates",
                sensor=metas[0].sensor,
                processing=[span],
            )
        )
        provenance = {
            "tool": "composite",
            "arguments": {
                "method": method,
                "layers": chosen,
                "mask": mask,
                "min_count": min_count,
                "mask_with": [e.id for e in extra],
            },
            "inputs": [{"stack": stack.id, "sha256": parent.data_sha256}]
            + [{"stack": e.id, "sha256": e.meta.data_sha256} for e in extra],
            "versions": {"numpy": np.__version__},
            "citations": list(parent.provenance.get("citations", [])),
        }
        composite = Stack.create(
            adir,
            id=stack_id,
            grid=stack.grid,
            layers=metas,
            data=out,
            provenance=provenance,
            aoi=parent.aoi,
            parents=[stack.id],
            warnings=_limit_warnings(warnings),
            params_hash=digest,
            overwrite=overwrite,
        )
    finally:
        scratch.unlink(missing_ok=True)
        with contextlib.suppress(OSError):
            scratch_dir.rmdir()
    result.warnings = _limit_warnings(warnings)
    return composite, result


def _run_strips(
    stack: Stack,
    chosen: Sequence[str],
    names: dict[str, dict[str, str]],
    dates: Sequence[str],
    method: str,
    mask: str,
    extra: Sequence[Stack],
    min_count: int,
    order: Sequence[int],
    out: NDArray[np.float32],
    deadline: float,
) -> NDArray[np.uint16]:
    """Fill out[:-1] layer by layer and out[-1] with the clear-date count; return the count."""
    height, width = stack.grid.shape
    rows = _strip_rows(len(dates), width)
    floor = np.full((height, width), np.iinfo(np.uint16).max, dtype=np.uint16)
    for li, layer in enumerate(chosen):
        for r0 in range(0, height, rows):
            check_cancel(deadline)
            r1 = min(r0 + rows, height)
            blk = _fill_block(stack, names[layer], dates, mask, extra, r0, r1)
            value, cnt = _reduce(blk, method, order)
            value[cnt < min_count] = np.nan
            out[li, r0:r1] = value
            shown = np.isfinite(value).astype(np.uint16) if method == "clearest" else cnt
            np.minimum(floor[r0:r1], shown.astype(np.uint16), out=floor[r0:r1])
    out[-1] = floor
    return floor


def _summarise(
    stack: Stack,
    out: NDArray[np.float32],
    count: NDArray[np.uint16],
    chosen: Sequence[str],
    dates: Sequence[str],
    method: str,
    warnings: list[str],
) -> CompositeResult:
    height, width = stack.grid.shape
    inside = np.ones((height, width), dtype=bool)
    step = max(1, _STRIP_BYTES // (width * 8))
    for r0 in range(0, height, step):
        r1 = min(r0 + step, height)
        inside[r0:r1] = _inside(stack.window(r0, r1, 0, width))
    aoi_px = int(inside.sum())
    layer_stats = []
    for li, layer in enumerate(chosen):
        p2, med, p98 = _sample_stats(out[li])
        layer_stats.append({"layer": layer, "p2": p2, "median": med, "p98": p98})
    hist = np.zeros(len(dates) + 1, dtype=np.int64)
    for r0 in range(0, height, step):
        r1 = min(r0 + step, height)
        hist += np.bincount(count[r0:r1][inside[r0:r1]], minlength=len(dates) + 1)
    zero = int(hist[0])
    present = np.nonzero(hist)[0]
    median = float(np.searchsorted(np.cumsum(hist), (hist.sum() + 1) // 2) if hist.sum() else 0)
    result = CompositeResult(
        method=method,
        dates=list(dates),
        layers=layer_stats,
        count_min=int(present[0]) if present.size else 0,
        count_median=median,
        count_max=int(present[-1]) if present.size else 0,
        valid_px=int(np.isfinite(out[0])[inside].sum()),
        aoi_px=aoi_px,
        zero_count_px=zero,
    )
    if zero:
        pct = 100 * zero / aoi_px if aoi_px else 0.0
        warnings.append(f"{zero} of {aoi_px} AOI px ({pct:.3g}%) have no clear date")
    return result


# --------------------------------------------------------------------------- series


def _index_spec(index: str) -> tuple[str, tuple[str, ...]]:
    from atlas.geo.indices import INDEX_CATALOGUE

    spec = INDEX_CATALOGUE.get(index.lower())
    if spec is None:
        names = ", ".join(sorted(INDEX_CATALOGUE))
        raise GeoError(fix(f"unknown index {index!r}; known: {names}.", "time_series layer=<name>"))
    return str(spec["expr"]), tuple(spec["bands"])


def _reduce_stat(values: NDArray[np.float32], stat: str) -> float:
    if values.size == 0:
        return math.nan
    return float(getattr(np, stat)(values))


def series_points(
    stack: Stack,
    layer: str | None,
    index: str | None,
    *,
    zone_px: NDArray[np.bool_],
    stat: str = "mean",
    mask: str = "clear",
    extra: Sequence[Stack] = (),
    deadline: float | None = None,
) -> list[SeriesPoint]:
    """Return one zonal statistic per date for a layer or an index.

    Args:
        stack: Multi-date stack.
        layer: Base layer name (exclusive with index).
        index: Catalogue index name; its bands are bound per date.
        zone_px: Bool (H, W) zone raster.
        stat: mean, median, min or max over usable zone pixels.
        mask: "clear" or "none".
        extra: Same-grid stacks whose QA planes also mask pixels.
        deadline: Absolute time.monotonic() cap.

    Returns:
        Points ordered by date; a date with no usable pixel has NaN and is flagged.

    Raises:
        GeoError: Missing bands or layer, or a grid mismatch.
    """
    from atlas.geo.expr import eval_expr

    _check_extra(stack, extra)
    if (layer is None) == (index is None):
        raise GeoError(fix("pass exactly one of layer or index.", "time_series layer=<name>"))
    table = base_layers(stack)
    expression = ""
    if index is not None:
        expression, bands = _index_spec(index)
        missing = [b for b in bands if b not in table]
        if missing:
            raise GeoError(
                fix(
                    f"index {index} needs {list(bands)}; stack has {sorted(table)}.",
                    f"prepare_stack bands={list(bands)}",
                )
            )
        dates = sorted(set.intersection(*(set(table[b]) for b in bands)))
    else:
        assert layer is not None
        if layer not in table:
            raise GeoError(
                fix(
                    f"layer {layer!r} has no dated layers; stack has {sorted(table)}.",
                    "time_series layer=<one of those>",
                )
            )
        bands, dates = (layer,), table[layer]
    rows, cols = np.nonzero(zone_px)
    r0, r1 = int(rows.min()), int(rows.max()) + 1
    c0, c1 = int(cols.min()), int(cols.max()) + 1
    view = stack.window(r0, r1, c0, c1)
    zone = zone_px[r0:r1, c0:c1]
    n_zone = int(zone.sum())
    points: list[SeriesPoint] = []
    for date in dates:
        check_cancel(deadline)
        used = [_dated_name(view, b, date) for b in bands]
        valid = view.valid(mask, layers=used, extra=extra)
        if index is not None:
            env = {b: view.band(b, date=date) for b in bands}
            with np.errstate(all="ignore"):
                data = np.asarray(eval_expr(expression, env), dtype=np.float32)
        else:
            data = np.asarray(view.band(used[0]))
        picked = data[zone & valid & np.isfinite(data)]
        pct = 100.0 * picked.size / n_zone if n_zone else 0.0
        points.append(
            SeriesPoint(
                date=date,
                value=_reduce_stat(picked, stat),
                n_valid=int(picked.size),
                n_zone=n_zone,
                valid_pct=pct,
                flagged=pct < MIN_VALID_PCT,
            )
        )
    return points


# --------------------------------------------------------------------------- trend


def _usable(points: Sequence[SeriesPoint]) -> list[SeriesPoint]:
    return [p for p in points if not p.flagged and math.isfinite(p.value)]


def choose_trend(points: Sequence[SeriesPoint]) -> Literal["ols", "month_anomaly"]:
    """Pick month_anomaly with >=12 usable dates over >=2 calendar years, else ols."""
    use = _usable(points)
    years = {p.date[:4] for p in use}
    return "month_anomaly" if len(use) >= 12 and len(years) >= 2 else "ols"


def _r2(resid: NDArray[np.float64], centred: NDArray[np.float64]) -> float:
    ss_tot = float((centred**2).sum())
    return 1.0 if ss_tot == 0 else 1.0 - float((resid**2).sum()) / ss_tot


def fit_trend(points: Sequence[SeriesPoint], kind: Literal["ols", "month_anomaly"]) -> TrendResult:
    """Fit a per-year slope to usable (unflagged, finite) points.

    ols is np.polyfit on decimal years. month_anomaly demeans both the values and the decimal
    years within each calendar month (equal to a time term plus 12 month dummies); months
    with one usable sample drop out.

    Raises:
        GeoError: Too few points, or too short a series for month_anomaly.
    """
    use = _usable(points)
    t = np.array([decimal_year(p.date) for p in use], dtype=np.float64)
    v = np.array([p.value for p in use], dtype=np.float64)
    if kind == "ols":
        if len(use) < 3:
            raise GeoError(
                fix(f"ols needs >=3 usable dates; have {len(use)}.", "prepare_stack more scenes")
            )
        slope, intercept = (float(x) for x in np.polyfit(t, v, 1))
        r2 = _r2(v - (slope * t + intercept), v - v.mean())
        return TrendResult("ols", slope, len(use), r2, len(points) - len(use))
    years = {p.date[:4] for p in use}
    if len(use) < 12 or len(years) < 2:
        raise GeoError(
            fix(
                f"month_anomaly needs >=12 usable dates over >=2 calendar years; have "
                f"{len(use)} over {len(years)}.",
                'time_series trend="ols"',
            )
        )
    months = np.array([int(p.date[4:6]) for p in use])
    keep = np.zeros(len(use), dtype=bool)
    for m in set(months.tolist()):
        sel = months == m
        if sel.sum() < 2:
            continue
        t[sel] -= t[sel].mean()
        v[sel] -= v[sel].mean()
        keep |= sel
    t, v = t[keep], v[keep]
    denom = float((t**2).sum())
    if denom == 0:
        raise GeoError(fix("month_anomaly found no repeated months.", 'time_series trend="ols"'))
    slope = float((t * v).sum() / denom)
    n = int(keep.sum())
    return TrendResult("month_anomaly", slope, n, _r2(v - slope * t, v), len(points) - n)
