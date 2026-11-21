"""Class areas, zonal statistics and class-by-zone cross-tabs on a Stack.

Invariants: hectares are ``px * stack.pixel_area_m2 / 1e4`` and nothing else; percentages use
the valid-pixel count as denominator; zone GeoJSON is validated (size, vertex counts, finite
lon/lat) before any rasterising; no file is executed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from atlas.geo.errors import GeoError, GridMismatch, fix
from atlas.geo.stack import QaMeta, Stack

__all__ = [
    "MAX_FEATURES",
    "MAX_FILE_MB",
    "MAX_RING_VERTICES",
    "MAX_VERTICES",
    "Counts",
    "Zone",
    "aoi_qa",
    "class_rows",
    "class_table",
    "counts",
    "crosstab",
    "load_zones",
    "save_csv",
    "table_line",
    "wide_screen_rows",
    "zonal_table",
    "zone_notes",
]

MAX_FILE_MB = 5
MAX_FEATURES = 500
MAX_VERTICES = 50_000
MAX_RING_VERTICES = 20_000
_ACCEPTED = "FeatureCollection, Feature, Polygon, MultiPolygon"
_NAME_KEYS = ("name", "NAME", "id", "ID")
_STATS = ("mean", "median", "min", "max", "sum")
_OVERLAP_PAIRS = 50


@dataclass(frozen=True)
class Counts:
    """Pixel denominators of a stack under one validity mask."""

    aoi_px: int
    valid_px: int
    valid_fraction: float
    px_area_ha: float


@dataclass(frozen=True)
class Zone:
    """A named bool mask on a stack grid."""

    name: str
    mask: NDArray[np.bool_]


def _outside_aoi(stack: Stack) -> NDArray[np.bool_] | None:
    names = [q.name for q in stack.meta.qa if q.semantic == "outside_aoi"]
    if not names:
        return None
    out = np.zeros(stack.grid.shape, dtype=bool)
    for name in names:
        out |= stack.plane(name)
    return out


def counts(stack: Stack, valid: NDArray[np.bool_]) -> Counts:
    """Return AOI and valid pixel counts.

    Args:
        stack: The stack (its ``outside_aoi`` planes shrink the AOI).
        valid: Usable-pixel mask from ``Stack.valid``.

    Returns:
        ``aoi_px``, ``valid_px``, their ratio and the pixel area in hectares.
    """
    outside = _outside_aoi(stack)
    height, width = stack.grid.shape
    aoi = height * width - (int(outside.sum()) if outside is not None else 0)
    n_valid = int(np.count_nonzero(valid))
    return Counts(aoi, n_valid, n_valid / aoi if aoi else 0.0, stack.pixel_area_m2 / 1e4)


def aoi_qa(stack: Stack) -> tuple[list[QaMeta], NDArray[np.uint8] | None]:
    """Return a one-plane ``outside_aoi`` QA pair for a derived stack, or ``([], None)``."""
    outside = _outside_aoi(stack)
    if outside is None:
        return [], None
    return [QaMeta(name="outside_aoi", semantic="outside_aoi")], outside.astype(np.uint8)[None]


def _names(stack: Stack, layer: str, class_names: Mapping[int, str] | None) -> dict[int, str]:
    names = dict(stack.layer(layer).classes or {})
    names.update(class_names or {})
    return {int(k): str(v) for k, v in names.items()}


def class_rows(
    codes: NDArray[Any],
    valid: NDArray[np.bool_],
    names: Mapping[int, str],
    px_area_ha: float,
) -> list[dict[str, Any]]:
    """Count categorical ``codes`` over valid pixels.

    Returns:
        One ``{code, name, px, ha, pct_valid}`` row per code with pixels, by code.
    """
    ok = np.asarray(valid, dtype=bool) & np.isfinite(codes)
    picked = np.asarray(codes)[ok].astype(np.int64)
    total = int(picked.size)
    if total and int(picked.min()) < 0:
        raise GeoError(
            fix("class codes must be non-negative integers.", "classify the layer first")
        )
    tally = np.bincount(picked) if total else np.zeros(0, dtype=np.int64)
    rows: list[dict[str, Any]] = []
    for code in np.flatnonzero(tally):
        px = int(tally[code])
        rows.append(
            {
                "code": int(code),
                "name": names.get(int(code), f"class_{int(code)}"),
                "px": px,
                "ha": px * px_area_ha,
                "pct_valid": 100.0 * px / total,
            }
        )
    return rows


def class_table(
    stack: Stack,
    layer: str,
    *,
    valid: NDArray[np.bool_],
    class_names: Mapping[int, str] | None = None,
) -> list[dict[str, Any]]:
    """Return the class-area table of a categorical layer.

    Args:
        stack: The stack.
        layer: A categorical layer.
        valid: Usable-pixel mask; non-finite codes are also dropped.
        class_names: Names overriding the layer's own.

    Returns:
        Rows ``{code, name, px, ha, pct_valid}``; ``pct_valid`` sums to 100.
    """
    return class_rows(
        stack.band(layer), valid, _names(stack, layer, class_names), stack.pixel_area_m2 / 1e4
    )


# -- zones ---------------------------------------------------------------------------


def _guard(message: str, hint: str) -> GeoError:
    return GeoError(fix(message, hint))


def _read_source(geojson: Mapping[str, Any] | str | Path) -> Mapping[str, Any]:
    if isinstance(geojson, Mapping):
        return geojson
    path = Path(geojson).expanduser()
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise _guard(f"cannot read zones file {path}: {exc.strerror}.", "check the path") from exc
    if size > MAX_FILE_MB * 1_000_000:
        raise _guard(
            f"file is {size / 1e6:.1f} MB (limit {MAX_FILE_MB}).",
            "simplify the polygons or pass fewer features",
        )
    try:
        loaded = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError) as exc:
        raise _guard(f"{path.name} is not valid JSON ({exc}).", "pass a GeoJSON file") from exc
    if not isinstance(loaded, Mapping):
        raise _guard("zones JSON is not an object.", f"pass one of: {_ACCEPTED}")
    return loaded


def _features(doc: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    kind = doc.get("type")
    if kind == "FeatureCollection":
        feats = doc.get("features")
        if not isinstance(feats, list):
            raise _guard("FeatureCollection has no features list.", "pass features=[...]")
        if len(feats) > MAX_FEATURES:
            raise _guard(
                f"zones have {len(feats):,} features (limit {MAX_FEATURES}).",
                "pass fewer features or merge them",
            )
        return [f for f in feats if isinstance(f, Mapping)]
    if kind == "Feature":
        return [doc]
    if kind in ("Polygon", "MultiPolygon"):
        return [{"type": "Feature", "properties": {}, "geometry": doc}]
    raise _guard(f"zones type {kind!r} is not accepted.", f"pass one of: {_ACCEPTED}")


def _polygons(geometry: Any) -> list[list[Any]]:
    if not isinstance(geometry, Mapping):
        raise _guard("a feature has no geometry.", f"use Polygon or MultiPolygon ({_ACCEPTED})")
    kind, coords = geometry.get("type"), geometry.get("coordinates")
    if kind not in ("Polygon", "MultiPolygon") or not isinstance(coords, list):
        raise _guard(
            f"geometry type {kind!r} is not accepted.", "use Polygon or MultiPolygon features"
        )
    return [coords] if kind == "Polygon" else list(coords)


def _check_counts(polys: list[list[list[Any]]]) -> None:
    rings = [ring for poly in polys for ring in poly]
    if not all(isinstance(ring, list) for ring in rings):
        raise _guard("a polygon ring is not a list.", "pass closed [lon, lat] rings")
    total = sum(len(ring) for ring in rings)
    if total > MAX_VERTICES:
        raise _guard(
            f"zones have {total:,} vertices (limit {MAX_VERTICES:,}).",
            "simplify the polygons or pass fewer features",
        )
    longest = max((len(ring) for ring in rings), default=0)
    if longest > MAX_RING_VERTICES:
        raise _guard(
            f"a ring has {longest:,} vertices (limit {MAX_RING_VERTICES:,}).",
            "simplify the polygon",
        )


def _ring(ring: list[Any]) -> list[tuple[float, float]]:
    if len(ring) < 4:
        raise _guard("a ring has fewer than 4 positions.", "pass closed rings")
    out: list[tuple[float, float]] = []
    for pos in ring:
        if not isinstance(pos, (list, tuple)) or len(pos) < 2:
            raise _guard("a position is not [lon, lat].", "pass [lon, lat] positions")
        lon, lat = pos[0], pos[1]
        ok = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (lon, lat))
        if not ok or not (math.isfinite(lon) and math.isfinite(lat)):
            raise _guard("a coordinate is not a finite number.", "pass numeric lon/lat")
        if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
            raise _guard(
                f"coordinate ({lon}, {lat}) is outside lon [-180,180] lat [-90,90].",
                "pass lon/lat in WGS84 degrees, not projected metres",
            )
        out.append((float(lon), float(lat)))
    return out


def _zone_name(feature: Mapping[str, Any], n: int, seen: dict[str, int]) -> str:
    props = feature.get("properties")
    base = f"zone_{n}"
    if isinstance(props, Mapping):
        for key in _NAME_KEYS:
            if props.get(key) not in (None, ""):
                base = str(props[key])[:40]
                break
    seen[base] = seen.get(base, 0) + 1
    return base if seen[base] == 1 else f"{base}_{seen[base]}"


def load_zones(geojson: Mapping[str, Any] | str | Path, grid: Any) -> list[Zone]:
    """Rasterise GeoJSON polygons (lon/lat) onto ``grid``, one zone per feature.

    Guards run before any rasterising: file size, feature, vertex and ring caps, closed
    rings, finite in-range coordinates, accepted types only.

    Args:
        geojson: A GeoJSON mapping, or a path (str or Path) to a file.
        grid: The target ``Grid``.

    Returns:
        Zones in feature order; each mask is a fresh array.

    Raises:
        GeoError: A guard failed or a polygon cannot be rasterised.
    """
    doc = _read_source(geojson)
    feats = _features(doc)
    if not feats:
        raise _guard("zones contain no features.", "pass at least one Polygon")
    shaped = [_polygons(f.get("geometry")) for f in feats]
    _check_counts([p for polys in shaped for p in polys])
    cleaned = [[[_ring(r) for r in poly] for poly in polys] for polys in shaped]
    seen: dict[str, int] = {}
    zones: list[Zone] = []
    for n, (feature, polys) in enumerate(zip(feats, cleaned, strict=True), start=1):
        mask = np.zeros(grid.shape, dtype=bool)
        for rings in polys:
            if not rings:
                continue
            inside = np.array(grid.rasterize([list(rings[0])]), dtype=bool)
            for hole in rings[1:]:
                inside &= ~grid.rasterize([list(hole)])
            mask |= inside
        zones.append(Zone(_zone_name(feature, n, seen), mask))
    return zones


def zone_notes(zones: Sequence[Zone]) -> list[str]:
    """Return warnings for empty (off-grid) zones and overlapping zones."""
    notes = [f"zone {z.name} lies outside the grid (0 px)" for z in zones if not z.mask.any()]
    pairs = 0
    for i, first in enumerate(zones):
        for second in zones[i + 1 :]:
            pairs += 1
            if pairs > _OVERLAP_PAIRS:
                return notes
            if bool(np.any(first.mask & second.mask)):
                notes.append(
                    "zones overlap: a pixel counts in each zone it falls in "
                    f"({first.name}, {second.name})"
                )
                return notes
    return notes


# -- tables --------------------------------------------------------------------------


def _stat(values: NDArray[Any], stat: str) -> float:
    if values.size == 0:
        return math.nan
    fn: Any = {
        "mean": np.mean,
        "median": np.median,
        "min": np.min,
        "max": np.max,
        "sum": np.sum,
    }[stat]
    return float(fn(values))


def zonal_table(
    stack: Stack,
    layer: str,
    zones: Sequence[Zone],
    *,
    valid: NDArray[np.bool_],
    stat: str = "mean",
    class_names: Mapping[int, str] | None = None,
) -> list[dict[str, Any]]:
    """Return one row per zone for a layer.

    Continuous layers give ``{zone, zone_px, valid_px, valid_fraction, ha, <stat>}`` (``ha`` is
    the valid area; an empty zone gives NaN). Categorical layers give the same leading
    columns plus ``ha_<class>`` and ``pct_<class>`` for every class present in any zone.

    Raises:
        GeoError: Unknown ``stat``.
    """
    if stat not in _STATS:
        raise GeoError(fix(f"unknown stat {stat!r}.", f"use one of {list(_STATS)}"))
    band = np.asarray(stack.band(layer))
    ok = np.asarray(valid, dtype=bool) & np.isfinite(band)
    area = stack.pixel_area_m2 / 1e4
    categorical = stack.layer(layer).kind == "categorical"
    names = _names(stack, layer, class_names)
    rows: list[dict[str, Any]] = []
    present: set[int] = set()
    for zone in zones:
        inside = zone.mask & ok
        n_valid = int(inside.sum())
        zone_px = int(zone.mask.sum())
        row: dict[str, Any] = {
            "zone": zone.name,
            "zone_px": zone_px,
            "valid_px": n_valid,
            "valid_fraction": n_valid / zone_px if zone_px else math.nan,
            "ha": n_valid * area,
        }
        if categorical:
            for item in class_rows(band, inside, names, area):
                present.add(int(item["code"]))
                row[f"ha_{item['name']}"] = item["ha"]
                row[f"pct_{item['name']}"] = item["pct_valid"]
        else:
            row[stat] = _stat(band[inside], stat)
        rows.append(row)
    if categorical:
        columns = [
            f"{p}_{names.get(c, f'class_{c}')}" for c in sorted(present) for p in ("ha", "pct")
        ]
        for row in rows:
            for col in columns:
                row.setdefault(col, 0.0)
    return rows


def wide_screen_rows(rows: Sequence[Mapping[str, Any]], keep: int = 4) -> list[dict[str, Any]]:
    """Shrink a wide ``ha_<class>`` table to its ``keep`` largest classes plus ``other``.

    Leading columns and ``ha_total`` are kept; ``pct_`` and ``px_`` columns are dropped. Used for the
    on-screen table only; the CSV keeps every column.
    """
    ha_cols = [k for k in (rows[0] if rows else {}) if k.startswith("ha_") and k != "ha_total"]
    ranked = sorted(ha_cols, key=lambda k: -sum(float(r.get(k, 0.0)) for r in rows))
    top = ranked[:keep]
    out: list[dict[str, Any]] = []
    for r in rows:
        item = {
            k: v
            for k, v in r.items()
            if k == "ha_total" or not k.startswith(("ha_", "pct_", "px_"))
        }
        item.update({k: r.get(k, 0.0) for k in top})
        rest = ranked[keep:]
        if rest:
            item["ha_other"] = sum(float(r.get(k, 0.0)) for k in rest)
        out.append(item)
    return out


def crosstab(
    stack: Stack,
    layer: str,
    zone_stack: Stack,
    zone_layer: str,
    *,
    valid: NDArray[np.bool_],
) -> tuple[list[str], list[str], NDArray[np.int64]]:
    """Cross-tabulate a categorical layer against a categorical zone layer.

    Pixels count when ``valid`` is true and the zone code is finite. The caller reports the
    difference to the plain class table (pixels with no zone code) as "no zone".

    Returns:
        Row labels (classes of ``layer``), column labels (classes of ``zone_layer``) and an
        int64 pixel-count matrix.

    Raises:
        GridMismatch: The stacks are on different grids.
        GeoError: The zone layer is not categorical.
    """
    if not stack.grid.same_as(zone_stack.grid):
        a, b = stack.grid, zone_stack.grid
        raise GridMismatch(
            fix(
                f"zone layer is on a different grid ({b.crs} {b.width}x{b.height} at "
                f"{b.transform[2]:.0f},{b.transform[5]:.0f} vs {a.crs} {a.width}x{a.height} at "
                f"{a.transform[2]:.0f},{a.transform[5]:.0f}).",
                "prepare_stack both with the same crs, resolution and bbox",
            )
        )
    if zone_stack.layer(zone_layer).kind != "categorical":
        raise GeoError(fix(f"zone layer {zone_layer!r} is not categorical.", "classify it first"))
    codes = np.asarray(stack.band(layer))
    zcodes = np.asarray(zone_stack.band(zone_layer))
    ok = np.asarray(valid, dtype=bool) & np.isfinite(codes) & np.isfinite(zcodes)
    a_codes = codes[ok].astype(np.int64)
    z_codes = zcodes[ok].astype(np.int64)
    row_codes = np.unique(a_codes)
    col_codes = np.unique(z_codes)
    ri = np.searchsorted(row_codes, a_codes)
    ci = np.searchsorted(col_codes, z_codes)
    n_rows, n_cols = len(row_codes), len(col_codes)
    flat = np.bincount(ri * n_cols + ci, minlength=n_rows * n_cols)
    row_names, col_names = _names(stack, layer, None), _names(zone_stack, zone_layer, None)
    return (
        [row_names.get(int(c), f"class_{int(c)}") for c in row_codes],
        [col_names.get(int(c), f"class_{int(c)}") for c in col_codes],
        flat.reshape(n_rows, n_cols).astype(np.int64),
    )


def table_line(shown: int, total: int, csv_rel: str) -> str:
    """Return the line that points from an on-screen table to its full CSV."""
    return f"+{total - shown} more: {csv_rel}" if total > shown else f"csv: {csv_rel}"


def save_csv(rows: Sequence[Mapping[str, Any]], tables_dir: Path, stem: str) -> Path:
    """Write rows as a full-precision CSV named ``<stem>-<h6>.csv``; return its path."""
    keys: list[str] = []
    for row in rows:
        keys.extend(k for k in row if k not in keys)
    tables_dir.mkdir(parents=True, exist_ok=True)
    tmp = tables_dir / f".tmp-{stem}.csv"
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    digest = hashlib.sha256(tmp.read_bytes()).hexdigest()[:6]
    path = tables_dir / f"{stem}-{digest}.csv"
    tmp.replace(path)
    return path
