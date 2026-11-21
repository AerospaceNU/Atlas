"""Prepare an analysis-ready Stack from pulled chips and user rasters.

Invariant: the stack grid is computed from sidecars only, so identity (params_hash) and the
reuse fast path never read pixels. No tool, no network, no code execution; scratch files
live under analysis/.scratch-<pid>/ and are removed in all cases.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import fnmatch
import hashlib
import io as _io
import json
import math
import os
import re
import shutil
import uuid
from collections.abc import Callable, Coroutine, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.lib.format import open_memmap
from numpy.typing import NDArray

from atlas.geo import paths, provenance
from atlas.geo._lazy import check_cancel, require_rasterio
from atlas.geo.chipmeta import ChipMeta, read_chip_meta, sidecar_path
from atlas.geo.errors import GeoError, MissingExtraError, StackError, fix
from atlas.geo.grid import (
    Grid,
    crs_to_lonlat,
    k2,
    lonlat_bounds_of,
    lonlat_bounds_to_crs,
    lonlat_to_crs,
    utm_epsg,
)
from atlas.geo.stack import (
    CLEAR_EXCLUDE,
    MAX_CELLS,
    AoiMeta,
    CalibrationMeta,
    LayerMeta,
    OrbitMeta,
    QaMeta,
    SourceMeta,
    Stack,
    exists,
    hash_params,
)

if TYPE_CHECKING:
    from atlas.agent.artifacts import LocalArtifactStore

__all__ = ["MAX_LAYER_PX", "AoiSpec", "PrepareReport", "Source", "build_stack", "discover"]

MAX_LAYER_PX = 36_000_000
_CATEGORICAL = frozenset({"scl", "landcover", "lc", "qa", "qa_pixel", "fmask", "classification"})
_SAR = frozenset({"vv", "vh", "hh", "hv"})
_DEM = frozenset({"dem", "elevation"})
_RUN_ID = re.compile(r"^r\d+$")
_GLOB_CHARS = re.compile(r"[*?\[]")
_DATE_SUFFIX = re.compile(r"@(\d{8})$")
_META = ".meta.json"
_EPS = 1e-6
_PREVIEW_SIDE = 256
_QA_STRIP = 1024
_M_PER_DEG = 111_320.0
Box = tuple[float, float, float, float]


# --------------------------------------------------------------------------- types


@dataclass(frozen=True, eq=False)
class Source:
    """One raster to prepare: a sidecar chip or a user file with an explicit spec."""

    path: Path
    chip: ChipMeta | None
    name: str
    band: str | None
    datetime: str | None
    categorical: bool
    spec: dict[str, Any] | None = None


@dataclass(frozen=True)
class AoiSpec:
    """Area of interest: a lon/lat bbox or a lon/lat polygon ring (never both)."""

    bbox: Box | None = None
    polygon: list[tuple[float, float]] | None = None


@dataclass
class PrepareReport:
    """What the build did, for the result text."""

    combine_notes: list[str] = field(default_factory=list)
    resampling_notes: list[str] = field(default_factory=list)
    qa_fractions: dict[str, float] = field(default_factory=dict)
    clear_pct: float | None = None
    reused: bool = False
    overwrote: bool = False
    aoi_px: int = 0
    warnings: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    layer_bands: list[tuple[str, str | None]] = field(default_factory=list)


@dataclass
class _Layer:
    name: str
    base: str
    date: str | None
    tiles: list[Source]


@dataclass
class _Ctx:
    target: Grid
    grids: dict[Path, Grid]
    sar_db: bool
    despeckle: str
    merge: str
    dilate_px: int
    deadline: float | None


@dataclass
class _QaOut:
    metas: list[QaMeta] = field(default_factory=list)
    files: list[Path] = field(default_factory=list)


# --------------------------------------------------------------------------- discovery


def _chip_name(chip: ChipMeta, path: Path) -> str:
    return (chip.alias or chip.band.common_name or chip.asset_key or path.stem).lower()


def _chip_source(path: Path, chip: ChipMeta) -> Source:
    name = _chip_name(chip, path)
    return Source(
        path=path,
        chip=chip,
        name=name,
        band=chip.band.source_band or chip.asset_key,
        datetime=chip.datetime,
        categorical=bool(chip.qa_preset) or name in _CATEGORICAL,
    )


def _roots(store: LocalArtifactStore) -> list[Path]:
    found: list[Path] = []
    for base in (store.artifact_root, store.root):
        for rel in ("data", ".atlas/data"):
            cand = base / rel
            if cand.is_dir() and cand not in found:
                found.append(cand)
    return found


def _chip_index(store: LocalArtifactStore) -> list[Source]:
    """Return every sidecar chip under data/ and .atlas/data, newest sidecar first."""
    items: list[tuple[float, Source]] = []
    seen: set[Path] = set()
    for root in _roots(store):
        for meta_path in root.rglob("*" + _META):
            chip = Path(str(meta_path)[: -len(_META)])
            if meta_path.name.startswith(".tmp-") or not chip.is_file():
                continue
            resolved = chip.resolve()
            if resolved in seen:
                continue
            try:
                meta = read_chip_meta(chip)
            except GeoError:
                continue
            if meta is not None:
                seen.add(resolved)
                items.append((meta_path.stat().st_mtime, _chip_source(resolved, meta)))
    items.sort(key=lambda it: (it[0], str(it[1].path)), reverse=True)
    return [src for _, src in items]


def _newest_ids(index: Sequence[Source], limit: int = 5) -> list[str]:
    ids: list[str] = []
    for src in index:
        sid = src.chip.scene_id if src.chip else None
        if sid and sid not in ids:
            ids.append(sid)
    return ids[:limit]


def _no_scene(ref: str, index: Sequence[Source]) -> StackError:
    listing = ", ".join(_newest_ids(index)) or "none"
    return StackError(
        fix(
            f"no pulled chips match {ref!r} (newest scene ids: {listing}).",
            "get_satellite_data(..., download=true, assets=[...]) or scenes=<one of those ids>",
        )
    )


def _sidecarless(path: Path) -> StackError:
    return StackError(
        fix(
            f"{path.name} has no chip sidecar, so its units and scaling are unknown.",
            f"files=[{{path:'{path}', name:'red', unit:'reflectance', scale:0.0001, nodata:0}}]",
        )
    )


def _output_paths(outputs: Any) -> list[str]:
    found: list[str] = []
    for item in outputs if isinstance(outputs, list) else []:
        if isinstance(item, str):
            found.append(item)
        elif isinstance(item, Mapping) and isinstance(item.get("path"), str):
            found.append(str(item["path"]))
    return found


def _resolve_output(text: str, store: LocalArtifactStore) -> Path | None:
    raw = Path(text[: -len(_META)] if text.endswith(_META) else text).expanduser()
    candidates = [raw] if raw.is_absolute() else [store.artifact_root / raw, store.root / raw]
    for cand in candidates:
        if cand.is_file():
            return cand.resolve()
    return None


def _run_chips(ref: str, store: LocalArtifactStore, index: Sequence[Source]) -> list[Source]:
    runs = provenance.read_runs(paths.analysis_dir(store))
    if ref == "last_pull":
        pulls = [r for r in runs if r.get("tool") == "get_satellite_data" and r.get("outputs")]
        run = pulls[-1] if pulls else None
    else:
        run = next((r for r in runs if r.get("id") == ref), None)
    if run is None:
        what = "pull" if ref == "last_pull" else "run"
        raise StackError(
            fix(
                f"no {what} {ref!r} in analysis/runs.jsonl.",
                "get_satellite_data(..., download=true, assets=[...]) first",
            )
        )
    by_path = {src.path: src for src in index}
    found: list[Source] = []
    bare: list[Path] = []
    for text in _output_paths(run.get("outputs")):
        resolved = _resolve_output(text, store)
        if resolved is None or resolved.name.endswith(_META):
            continue
        if resolved in by_path:
            found.append(by_path[resolved])
        elif resolved.suffix.lower() in (".tif", ".tiff") and not sidecar_path(resolved).is_file():
            bare.append(resolved)
    if not found and bare:
        raise _sidecarless(bare[0])
    if not found:
        raise _no_scene(ref, index)
    return found


def _glob_hit(path: Path, pattern: str, store: LocalArtifactStore) -> bool:
    rels = {path.name, path.as_posix()}
    patterns = {pattern, pattern.replace("**/", "")}
    for base in (store.artifact_root, store.root):
        try:
            rels.add(path.relative_to(base).as_posix())
        except ValueError:
            continue
    return any(fnmatch.fnmatch(rel, pat) for rel in rels for pat in patterns)


def _ref_chips(ref: str, store: LocalArtifactStore, index: Sequence[Source]) -> list[Source]:
    if ref == "last_pull" or _RUN_ID.match(ref):
        return _run_chips(ref, store, index)
    by_id = [s for s in index if s.chip is not None and s.chip.scene_id == ref]
    if by_id:
        return by_id
    candidate = store.resolve_read(ref)
    if candidate.is_file():
        for src in index:
            if src.path == candidate.resolve():
                return [src]
        raise _sidecarless(candidate)
    if _GLOB_CHARS.search(ref):
        hits = [s for s in index if _glob_hit(s.path, ref, store)]
        if hits:
            return hits
    raise _no_scene(ref, index)


def _user_source(spec: Mapping[str, Any], store: LocalArtifactStore) -> Source:
    path = store.resolve_read(str(spec["path"]))
    if not path.is_file():
        raise StackError(
            fix(f"file {spec['path']!r} does not exist.", "pass an existing raster path")
        )
    name = str(spec.get("name") or "").lower()
    if not name:
        raise StackError(fix(f"file {path.name} needs a name.", "files=[{path:..., name:'dem'}]"))
    return Source(
        path=path,
        chip=None,
        name=name,
        band=spec.get("band") or name,
        datetime=spec.get("datetime"),
        categorical=bool(spec.get("categorical")) or name in _CATEGORICAL,
        spec=dict(spec),
    )


def discover(
    refs: Sequence[str],
    store: LocalArtifactStore,
    files: Sequence[Mapping[str, Any]] | None = None,
) -> list[Source]:
    """Resolve scene references and user files to sources.

    Args:
        refs: Scene ids, "last_pull", a run id (rNNNN), a sidecar glob or a chip path.
        store: Session store; chips are found under data/ and .atlas/data.
        files: User raster specs (path, name, unit, scale, ...).

    Returns:
        Sources in reference order, duplicates removed.

    Raises:
        StackError: Nothing matches, or a referenced chip has no sidecar.
    """
    found: list[Source] = []
    if refs:
        index = _chip_index(store)
        for ref in refs:
            found.extend(_ref_chips(ref, store, index))
    found.extend(_user_source(spec, store) for spec in files or [])
    unique: dict[Path, Source] = {}
    for src in found:
        unique.setdefault(src.path, src)
    if not unique:
        raise StackError(fix("nothing to prepare.", "scenes='last_pull' or files=[{path,...}]"))
    return list(unique.values())


# --------------------------------------------------------------------------- source geometry


def _run_grid(make: Callable[[], Coroutine[Any, Any, Grid]]) -> Grid:
    def run() -> Grid:
        return asyncio.run(make())

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(run).result()


def _probe_rasterio(rio: Any, path: Path) -> Grid:
    with rio.open(path) as ds:
        t = ds.transform
        epsg = ds.crs.to_epsg() if ds.crs is not None else None
        if t.b != 0 or t.d != 0 or t.e >= 0 or t.a <= 0 or epsg is None:
            raise GeoError(
                fix(
                    f"{path.name} is rotated, south-up or has no EPSG CRS.",
                    "gdalwarp -t_srs EPSG:<n> it to a north-up grid first",
                )
            )
        x0, y0 = t.c, t.f
        if str(ds.tags().get("AREA_OR_POINT", "")).lower() == "point":
            x0, y0 = x0 - t.a / 2.0, y0 - t.e / 2.0
        return Grid(f"EPSG:{epsg}", (t.a, 0.0, x0, 0.0, t.e, y0), ds.width, ds.height)


def _probe_cog(path: Path) -> Grid:
    from atlas.data import cog

    async def go() -> Grid:
        image = (await cog.open_cog_file(path))._image
        if image.epsg is None:
            raise GeoError(fix(f"{path.name} has no usable CRS.", "uv sync --extra geo"))
        t = (image.pixel_size_x, 0.0, image.origin_x, 0.0, -image.pixel_size_y, image.origin_y)
        return Grid(f"EPSG:{image.epsg}", t, image.width, image.height)

    try:
        return _run_grid(go)
    except cog.CogError as exc:
        raise MissingExtraError("geo", "rasterio") from exc


def _probe_user(path: Path) -> Grid:
    """Return the PixelIsArea grid of a user raster from its header (no pixel read)."""
    try:
        rio = require_rasterio()
    except MissingExtraError:
        return _probe_cog(path)
    return _probe_rasterio(rio, path)


def _chip_grid(src: Source) -> Grid:
    chip = src.chip
    assert chip is not None
    r = chip.raster
    if (
        r.epsg is None
        or r.transform is None
        or len(r.transform) != 6
        or not r.width
        or not r.height
    ):
        raise StackError(
            fix(
                f"sidecar of {src.path.name} lacks raster epsg/transform/size.",
                "re-pull with get_satellite_data(..., download=true)",
            )
        )
    t = r.transform
    return Grid(f"EPSG:{r.epsg}", (t[0], t[1], t[2], t[3], t[4], t[5]), r.width, r.height)


def _source_grid(src: Source) -> Grid:
    return _chip_grid(src) if src.chip is not None else _probe_user(src.path)


# --------------------------------------------------------------------------- target grid


def _is_geographic(epsg: int) -> bool:
    return 4000 <= epsg < 5000


def _is_qa(src: Source) -> bool:
    return src.chip is not None and bool(src.chip.qa_preset)


def _reference(srcs: Sequence[Source]) -> list[Source]:
    ref = [s for s in srcs if not s.categorical and not _is_qa(s)]
    return ref or list(srcs)


def _aoi_lonlat(aoi: AoiSpec | None, grids: Sequence[Grid]) -> Box:
    if aoi is not None and aoi.bbox is not None:
        return aoi.bbox
    if aoi is not None and aoi.polygon is not None:
        lons = [p[0] for p in aoi.polygon]
        lats = [p[1] for p in aoi.polygon]
        return (min(lons), min(lats), max(lons), max(lats))
    boxes = [lonlat_bounds_of(g) for g in grids]
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _zone(lon: float) -> int:
    return int((lon + 180.0) // 6) % 60 + 1


def _utm_for(box: Box) -> str:
    west, south, east, north = box
    mid = utm_epsg((west + east) / 2, (south + north) / 2)
    if _zone(west) != _zone(east - 1e-9):
        raise StackError(
            fix(
                f"the area spans UTM zones {_zone(west)} and {_zone(east - 1e-9)}.",
                f"crs='EPSG:{mid}' for one zone (edges distort), or a bbox inside one zone",
            )
        )
    return f"EPSG:{mid}"


def _resolve_crs(crs: str, ref_grids: Sequence[Grid], aoi: AoiSpec | None) -> str:
    text = crs.strip()
    if text.lower() == "native":
        projected = sorted({g.crs for g in ref_grids if not _is_geographic(g.epsg)})
        if len(projected) > 1:
            raise StackError(
                fix(
                    f"layers are in different CRSs ({', '.join(projected)}).",
                    "crs='utm' or crs='EPSG:<n>'",
                )
            )
        if projected:
            return projected[0]
    if text.lower() in ("native", "utm"):
        return _utm_for(_aoi_lonlat(aoi, ref_grids))
    return Grid(text, (1.0, 0.0, 0.0, 0.0, -1.0, 0.0), 1, 1).crs


def _res_in_target(grid: Grid, target_crs: str, lat: float) -> float:
    if grid.crs == target_crs or not _is_geographic(grid.epsg):
        return grid.xres
    return grid.xres * _M_PER_DEG * math.cos(math.radians(lat))


def _resolution(spec: str | float, ref_grids: Sequence[Grid], target_crs: str, lat: float) -> float:
    value: float
    if isinstance(spec, str):
        text = spec.strip().lower()
        if text in ("finest", "coarsest"):
            values = [_res_in_target(g, target_crs, lat) for g in ref_grids]
            return round(min(values) if text == "finest" else max(values), 6)
        try:
            value = float(text)
        except ValueError:
            raise StackError(
                fix(f"bad resolution {spec!r}.", "resolution='finest', 'coarsest' or metres")
            ) from None
    else:
        value = float(spec)
    if not math.isfinite(value) or value <= 0:
        raise StackError(fix(f"resolution {value} must be positive.", "resolution=10"))
    return value


def _extent_bounds(crs: str, aoi: AoiSpec | None, ref_grids: Sequence[Grid]) -> Box:
    if aoi is not None and aoi.bbox is not None:
        return lonlat_bounds_to_crs(aoi.bbox, crs)
    if aoi is not None and aoi.polygon is not None:
        xs, ys = lonlat_to_crs(crs, [p[0] for p in aoi.polygon], [p[1] for p in aoi.polygon])
        return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
    boxes = [
        g.bounds if g.crs == crs else lonlat_bounds_to_crs(lonlat_bounds_of(g), crs)
        for g in ref_grids
    ]
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _lattice_grid(crs: str, res: float, bounds: Box, anchor: tuple[float, float]) -> Grid:
    """Snap outward on the pixel lattice anchored at the finest source origin."""
    ax, ay = anchor
    c0 = math.floor((bounds[0] - ax) / res + _EPS)
    c1 = math.ceil((bounds[2] - ax) / res - _EPS)
    r0 = math.floor((ay - bounds[3]) / res + _EPS)
    r1 = math.ceil((ay - bounds[1]) / res - _EPS)
    width, height = max(c1 - c0, 1), max(r1 - r0, 1)
    return Grid(crs, (res, 0.0, ax + c0 * res, 0.0, -res, ay - r0 * res), width, height)


def _guard_size(grid: Grid, n_layers: int, hint: str) -> None:
    px = grid.width * grid.height
    if px > MAX_LAYER_PX:
        coarser = math.ceil(grid.xres * math.sqrt(px / MAX_LAYER_PX) * 102) / 100
        raise StackError(
            fix(
                f"grid is {grid.height}x{grid.width} = {px / 1e6:.1f} MP per layer, limit "
                f"{MAX_LAYER_PX / 1e6:.0f} MP.",
                f"resolution={coarser} or a smaller bbox inside {hint}",
            )
        )
    cells = px * n_layers
    if cells > MAX_CELLS:
        raise StackError(
            fix(
                f"{n_layers} layers x {grid.height}x{grid.width} = {cells / 1e6:.0f}e6 cells, "
                f"limit {MAX_CELLS / 1e6:.0f}e6.",
                "fewer bands, a coarser resolution=N or a smaller bbox=[w,s,e,n]",
            )
        )


def _plan_grid(
    srcs: Sequence[Source],
    grids: Mapping[Path, Grid],
    *,
    crs: str,
    resolution: str | float,
    aoi: AoiSpec | None,
    n_layers: int,
) -> Grid:
    ref_grids = [grids[s.path] for s in _reference(srcs)]
    target_crs = _resolve_crs(crs, ref_grids, aoi)
    box = _aoi_lonlat(aoi, ref_grids)
    res = _resolution(resolution, ref_grids, target_crs, (box[1] + box[3]) / 2)
    bounds = _extent_bounds(target_crs, aoi, ref_grids)
    same = [g for g in ref_grids if g.crs == target_crs]
    anchor = (0.0, 0.0)
    if same:
        finest = min(same, key=lambda g: g.xres)
        anchor = (finest.transform[2], finest.transform[5])
    grid = _lattice_grid(target_crs, res, bounds, anchor)
    _guard_size(grid, n_layers, "[" + ",".join(f"{v:.4f}" for v in box) + "]")
    return grid


# --------------------------------------------------------------------------- layers


def _utc_date(value: str | None) -> str | None:
    if not value:
        return None
    try:
        stamp = datetime.fromisoformat(value)
    except ValueError:
        digits = re.sub(r"\D", "", value[:10])
        return digits if len(digits) == 8 else None
    if stamp.tzinfo is not None:
        stamp = stamp.astimezone(UTC)
    return stamp.strftime("%Y%m%d")


def _match(src: Source, wanted: str) -> bool:
    keys = {src.name, (src.band or "").lower()}
    if src.chip is not None:
        keys.update({(src.chip.asset_key or "").lower(), (src.chip.band.common_name or "").lower()})
    return wanted in keys


def _select(srcs: Sequence[Source], bands: Sequence[str] | None) -> list[Source]:
    if not bands:
        return list(srcs)
    wanted = [b.strip().lower() for b in bands]
    present = sorted({s.name for s in srcs})
    missing = [b for b in wanted if not any(_match(s, b) for s in srcs)]
    if missing:
        raise StackError(
            fix(f"bands {missing} not found (present: {present}).", f"bands={present[:6]}")
        )
    picked = [s for w in wanted for s in srcs if _match(s, w)]
    return list(dict.fromkeys(picked))


def _group(srcs: Sequence[Source], order: Sequence[str]) -> list[_Layer]:
    buckets: dict[tuple[str, str | None], list[Source]] = {}
    for src in srcs:
        buckets.setdefault((src.name, _utc_date(src.datetime)), []).append(src)
    per_base: dict[str, int] = {}
    for base, _ in buckets:
        per_base[base] = per_base.get(base, 0) + 1
    rank = {name.lower(): i for i, name in enumerate(order)}

    def sort_key(item: tuple[tuple[str, str | None], list[Source]]) -> tuple[int, str, str]:
        (base, date), _ = item
        return (rank.get(base, len(rank)), base, date or "")

    layers: list[_Layer] = []
    for (base, date), tiles in sorted(buckets.items(), key=sort_key):
        name = base if per_base[base] == 1 else f"{base}@{date or 'undated'}"
        layers.append(_Layer(name, base, date, sorted(tiles, key=lambda s: str(s.path))))
    return layers


def _unit_kind(src: Source, sar_db: bool) -> tuple[str, str]:
    unit = str(src.spec.get("unit") or "") if src.spec is not None else ""
    if src.categorical:
        return "", "categorical"
    if src.name in _SAR or unit.lower() in ("db", "power", "linear"):
        return unit or ("dB" if sar_db else "power"), "backscatter"
    if unit:
        if unit.lower() == "reflectance":
            return unit, "reflectance"
        return unit, "elevation" if src.name in _DEM else "continuous"
    if src.name in _DEM:
        return "m", "elevation"
    if src.name.startswith(("lwir", "st_")):
        return "K", "continuous"
    return "reflectance", "reflectance"


def _check_units(layers: Sequence[_Layer], sar_db: bool) -> None:
    units: dict[str, set[str]] = {}
    for layer in layers:
        units.setdefault(layer.base, set()).add(_unit_kind(layer.tiles[0], sar_db)[0])
    for base, found in units.items():
        if len(found) > 1:
            raise StackError(
                fix(
                    f"layer {base!r} has mixed units across dates: {sorted(found)}.",
                    f"files=[...] with one unit per name, or bands without the odd {base} file",
                )
            )


def _check_tiles(layers: Sequence[_Layer], merge: str) -> None:
    if merge == "mosaic":
        return
    for layer in layers:
        if len(layer.tiles) > 1:
            ids = ", ".join(
                str(t.chip.scene_id if t.chip else t.path.name) for t in layer.tiles[:3]
            )
            raise StackError(
                fix(
                    f"layer {layer.name} has {len(layer.tiles)} tiles for one date ({ids}).",
                    "merge='mosaic' (first valid pixel wins) or pick one scene id",
                )
            )


def _read_bounds(src_grid: Grid, target: Grid) -> tuple[Box, str]:
    if src_grid.crs == target.crs:
        pad = 2.0 * max(src_grid.xres, target.xres)
        w, s, e, n = target.bounds
        return (w - pad, s - pad, e + pad, n + pad), target.crs
    west, south, east, north = lonlat_bounds_of(target)
    dx, dy = (east - west) * 0.05, (north - south) * 0.05
    return (west - dx, south - dy, east + dx, north + dy), "EPSG:4326"


def _composite(planes: Sequence[NDArray[np.float32]]) -> NDArray[np.float32]:
    out = planes[0].copy()
    for plane in planes[1:]:
        hole = np.isnan(out)
        out[hole] = plane[hole]
    return out


def _user_calibration(src: Source, nodata_file: float | None, dtype: str) -> Any:
    from atlas.geo import calibrate

    spec = src.spec or {}
    scale, offset = spec.get("scale"), spec.get("offset")
    if src.categorical:
        scale = 1.0 if scale is None else scale
        offset = 0.0 if offset is None else offset
    nodata = spec["nodata"] if spec.get("nodata") is not None else nodata_file
    unit = spec.get("unit") or ("class" if src.categorical else None)
    return calibrate.from_user(
        scale=scale, offset=offset, nodata=nodata, unit=unit, band=src.band, dtype=dtype
    )


def _read_tile(src: Source, ctx: _Ctx, *, qa: bool = False) -> tuple[Any, Any, Any, Any]:
    """Read, calibrate (nodata to NaN before scaling), despeckle and warp one tile."""
    from atlas.geo import calibrate, io, warp
    from atlas.geo import sar as sar_mod

    grid = ctx.grids[src.path]
    bounds, bounds_crs = _read_bounds(grid, ctx.target)
    block = io.read_raster(src.path, bounds=bounds, bounds_crs=bounds_crs)
    raw = block.data[0]
    cal: Any = None
    if qa:
        arr = raw.astype(np.float32)
    else:
        if src.chip is not None:
            cal = calibrate.from_sidecar(src.chip, file=src.path.name)
        else:
            cal = _user_calibration(src, block.nodata, str(raw.dtype))
        arr = calibrate.apply(raw, cal)
        if src.name in _SAR and ctx.despeckle != "none":
            arr = sar_mod.despeckle(arr, ctx.despeckle)  # type: ignore[arg-type]
    out, info = warp.resample_to(arr, block.grid, ctx.target, categorical=src.categorical or qa)
    return out, info, cal, block


def _calibration_meta(cal: Any) -> CalibrationMeta | None:
    if cal is None:
        return None
    return CalibrationMeta(
        scale=cal.scale,
        offset=cal.offset,
        offset_applied=cal.offset_applied,
        nodata_in=cal.nodata_in,
        dtype_in=cal.dtype_in,
        valid_range=list(cal.valid_range) if cal.valid_range else None,
        source=cal.source,
    )


def _orbit(chip: ChipMeta | None) -> OrbitMeta | None:
    if chip is None:
        return None
    props = chip.scene_properties
    rel, state = props.get("sat:relative_orbit"), props.get("sat:orbit_state")
    if rel is None and state is None:
        return None
    try:
        relative = int(rel) if rel is not None else None
    except (TypeError, ValueError):
        relative = None
    return OrbitMeta(relative=relative, state=str(state) if state is not None else None)


def _rel(store: LocalArtifactStore, path: Path) -> str:
    try:
        return store.relative(path)
    except (OSError, ValueError):
        return str(path)


def _read_layer_tiles(
    layer: _Layer, ctx: _Ctx
) -> tuple[list[NDArray[np.float32]], tuple[Any, Any, Any], float]:
    from atlas.geo import warp

    planes: list[NDArray[np.float32]] = []
    first: tuple[Any, Any, Any] | None = None
    uncovered: list[float] = []
    covered: NDArray[np.bool_] | None = None
    for tile in layer.tiles:
        check_cancel(ctx.deadline)
        try:
            out, info, cal, block = _read_tile(tile, ctx)
        except GeoError as exc:
            if ctx.merge == "mosaic" and len(layer.tiles) > 1 and "does not overlap" in str(exc):
                continue
            raise
        planes.append(out)
        uncovered.append(info.uncovered_fraction)
        if len(layer.tiles) > 1:
            ones = np.ones(block.data.shape[1:], dtype=np.float32)
            hit = np.isfinite(warp.resample_to(ones, block.grid, ctx.target, categorical=True)[0])
            covered = hit if covered is None else covered | hit
        if first is None:
            first = (info, cal, block)
    if first is None:
        raise StackError(
            fix(f"no tile of {layer.name} overlaps the grid.", "bbox=[...] inside the pulled area")
        )
    gap = min(uncovered) if covered is None else float(1.0 - covered.mean())
    return planes, first, gap


def _build_layer(
    layer: _Layer, ctx: _Ctx, store: LocalArtifactStore, report: PrepareReport
) -> tuple[NDArray[np.float32], LayerMeta]:
    from atlas.geo import sar as sar_mod

    planes, (info, cal, block), gap = _read_layer_tiles(layer, ctx)
    src = layer.tiles[0]
    plane = planes[0] if len(planes) == 1 else _composite(planes)
    processing: list[str] = []
    if len(planes) > 1:
        report.combine_notes.append(
            f"{layer.name}: {len(planes)} tiles mosaicked, first valid pixel in path order wins"
        )
        processing.append(f"mosaic of {len(planes)} tiles")
    if cal is not None and cal.offset_applied:
        processing.append("offset already applied upstream (sidecar offset=0)")
    unit, kind = _unit_kind(src, ctx.sar_db)
    if src.name in _SAR:
        if ctx.despeckle != "none":
            enl = f", ENL {sar_mod.LEE_ENL:g}" if ctx.despeckle == "lee5" else ""
            processing.append(f"despeckle {ctx.despeckle}{enl} on linear power")
        if ctx.sar_db:
            plane = sar_mod.to_db(plane)
            processing.append("dB = 10*log10(power); <=0 -> NaN")
    if info.method != "none":
        processing.append(f"resampled {info.method}" + (f": {info.note}" if info.note else ""))
    if abs(block.grid.xres - ctx.target.xres) > 1e-9:
        unit = "deg" if _is_geographic(block.grid.epsg) else "m"
        report.resampling_notes.append(
            f"{layer.base} {block.grid.xres:g} {unit}->{ctx.target.xres:g} m {info.method}"
        )
    if gap > 1e-4:
        report.warnings.append(
            f"{layer.name}: {gap * 100:.3g}% of the grid lies outside the source footprint (NaN)"
        )
    chip = src.chip
    meta = LayerMeta(
        name=layer.name,
        kind=kind,
        band=src.band,
        unit=unit,
        datetime=src.datetime,
        sensor=chip.source if chip else None,
        wavelength_um=chip.band.wavelength_um if chip else None,
        native_resolution_m=float(block.grid.xres),
        resampling=info.method,
        calibration=_calibration_meta(cal),
        processing=processing,
        source=SourceMeta(
            scene_id=chip.scene_id if chip else None,
            source=chip.source if chip else None,
            collection=chip.collection if chip else None,
            asset_key=chip.asset_key if chip else None,
            file=_rel(store, src.path),
        ),
        orbit=_orbit(chip),
    )
    return plane, meta


# --------------------------------------------------------------------------- QA planes


def _qa_groups(srcs: Sequence[Source], merge: str) -> dict[str | None, list[Source]]:
    groups: dict[str | None, list[Source]] = {}
    for src in srcs:
        if _is_qa(src):
            groups.setdefault(_utc_date(src.datetime), []).append(src)
    for date, tiles in groups.items():
        if len(tiles) > 1 and merge == "none":
            raise StackError(
                fix(
                    f"{len(tiles)} QA tiles for {date or 'one date'} need combining.",
                    "merge='mosaic'",
                )
            )
        tiles.sort(key=lambda s: str(s.path))
    return groups


def _write_plane(path: Path, plane: NDArray[np.bool_]) -> None:
    mm = open_memmap(path, mode="w+", dtype=np.uint8, shape=plane.shape)
    mm[:] = plane
    mm.flush()
    del mm


@dataclass
class _PlaneAcc:
    path: Path
    hits: int = 0
    seen: bool = False


def _decode_strips(
    raw: NDArray[np.float32],
    preset: str,
    ctx: _Ctx,
    scratch: Path,
    aoi_mask: NDArray[np.bool_] | None,
    first_index: int,
) -> dict[str, _PlaneAcc]:
    """Decode QA row strips (with a dilation halo) into per-plane scratch files.

    Bounds peak memory to a strip of planes instead of every plane of the full raster.
    """
    from atlas.geo import qa as qa_mod

    halo, height = ctx.dilate_px, raw.shape[0]
    acc: dict[str, _PlaneAcc] = {}
    for r0 in range(0, height, _QA_STRIP):
        check_cancel(ctx.deadline)
        r1 = min(r0 + _QA_STRIP, height)
        e0, e1 = max(r0 - halo, 0), min(r1 + halo, height)
        planes = qa_mod.planes_from_qa(raw[e0:e1], preset, dilate_px=halo)
        planes.setdefault("nodata", np.zeros((e1 - e0, raw.shape[1]), dtype=bool))
        for semantic, plane in planes.items():
            core = plane[r0 - e0 : r0 - e0 + (r1 - r0)]
            if semantic not in acc:
                path = scratch / f"qa-{first_index + len(acc):03d}.npy"
                open_memmap(path, mode="w+", dtype=np.uint8, shape=raw.shape).flush()
                acc[semantic] = _PlaneAcc(path)
            target = np.load(acc[semantic].path, mmap_mode="r+")
            target[r0:r1] = core
            target.flush()
            del target
            counted = core if aoi_mask is None else core & aoi_mask[r0:r1]
            acc[semantic].hits += int(counted.sum())
            acc[semantic].seen = acc[semantic].seen or bool(core.any())
    return acc


def _qa_for_date(
    date: str | None,
    tiles: Sequence[Source],
    ctx: _Ctx,
    scratch: Path,
    aoi_mask: NDArray[np.bool_] | None,
    multi: bool,
    out: _QaOut,
    report: PrepareReport,
) -> None:
    raws = [_read_tile(t, ctx, qa=True)[0] for t in tiles]
    raw = raws[0] if len(raws) == 1 else _composite(raws)
    del raws
    preset = str(tiles[0].chip.qa_preset) if tiles[0].chip is not None else ""
    acc = _decode_strips(raw, preset, ctx, scratch, aoi_mask, len(out.files))
    total = int(aoi_mask.sum()) if aoi_mask is not None else raw.size
    for semantic, plane in acc.items():
        if semantic not in CLEAR_EXCLUDE and not plane.seen:
            plane.path.unlink(missing_ok=True)
            continue
        name = f"{semantic}@{date}" if multi and date else semantic
        fraction = plane.hits / total if total else 0.0
        target = scratch / f"qa-{len(out.files):03d}.npy"
        if plane.path != target:
            plane.path.replace(target)
        out.files.append(target)
        out.metas.append(
            QaMeta(
                name=name,
                semantic=semantic,
                preset=preset,
                source_layer=tiles[0].name,
                fraction=fraction,
            )
        )
        report.qa_fractions[name] = fraction


def _build_qa(
    srcs: Sequence[Source],
    ctx: _Ctx,
    scratch: Path,
    aoi_mask: NDArray[np.bool_] | None,
    qa_mode: str,
    report: PrepareReport,
) -> _QaOut:
    out = _QaOut()
    groups = _qa_groups(srcs, ctx.merge) if qa_mode == "auto" else {}
    for date, tiles in sorted(groups.items(), key=lambda kv: kv[0] or ""):
        check_cancel(ctx.deadline)
        _qa_for_date(date, tiles, ctx, scratch, aoi_mask, len(groups) > 1, out, report)
    if aoi_mask is not None:
        outside = ~aoi_mask
        path = scratch / f"qa-{len(out.files):03d}.npy"
        _write_plane(path, outside)
        out.files.append(path)
        out.metas.append(
            QaMeta(name="outside_aoi", semantic="outside_aoi", fraction=float(outside.mean()))
        )
    return out


def _plane_iter(files: Sequence[Path]) -> Iterator[Any]:
    for path in files:
        yield np.load(path, mmap_mode="r")


def _layer_date(meta: LayerMeta) -> str | None:
    match = _DATE_SUFFIX.search(meta.name)
    if match:
        return match.group(1)
    digits = re.sub(r"\D", "", (meta.datetime or "")[:10])
    return digits if len(digits) == 8 else None


def _clear_pct(
    files: Sequence[Path],
    metas: Sequence[LayerMeta],
    qa_out: _QaOut,
    aoi_px: int,
    shape: tuple[int, int],
) -> float:
    """Percent of AOI pixels that are finite in every continuous layer and not flagged.

    Mirrors Stack.valid("clear"): dated QA planes only count for layers of the same date.
    """
    ok = np.ones(shape, dtype=bool)
    used = [i for i, m in enumerate(metas) if m.kind != "categorical"]
    for i in used:
        ok &= np.isfinite(np.load(files[i], mmap_mode="r"))
    used_dates = {_layer_date(metas[i]) for i in used} - {None}
    for meta, path in zip(qa_out.metas, qa_out.files, strict=True):
        if meta.semantic not in CLEAR_EXCLUDE:
            continue
        match = _DATE_SUFFIX.search(meta.name)
        if match and match.group(1) not in used_dates:
            continue
        ok &= np.load(path, mmap_mode="r") == 0
    return 100.0 * float(ok.sum()) / max(aoi_px, 1)


# --------------------------------------------------------------------------- identity


def _file_key(src: Source) -> dict[str, Any]:
    stat = src.path.stat()
    if src.chip is not None:
        digest = hashlib.sha256(sidecar_path(src.path).read_bytes()).hexdigest()
        return {"chip": src.path.name, "size": stat.st_size, "sidecar_sha256": digest}
    return {
        "file": str(src.path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "spec": src.spec,
    }


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9_.-]+", "", text.lower().replace(" ", "_")) or "stack"


def _planned_id(srcs: Sequence[Source], phash: str, name: str | None) -> str:
    if name:
        return name
    sensors = [s.chip.source for s in srcs if s.chip is not None and s.chip.source]
    dates = sorted({d for s in srcs if (d := _utc_date(s.datetime))})
    stamp = "nodate" if not dates else dates[0] if len(dates) == 1 else f"{dates[0]}-{dates[-1]}"
    return f"{_slug(sensors[0]) if sensors else 'stack'}_{stamp}_{phash[:6]}"


def _norm_aoi(aoi: AoiSpec | None) -> dict[str, Any] | None:
    if aoi is None:
        return None
    return {
        "bbox": list(aoi.bbox) if aoi.bbox else None,
        "polygon": [list(p) for p in aoi.polygon] if aoi.polygon else None,
    }


def _versions() -> dict[str, str]:
    from importlib import metadata

    found: dict[str, str] = {}
    for pkg in ("atlas", "numpy", "scipy", "rasterio", "pyproj"):
        try:
            found[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            continue
    return found


def _pixel_area_m2(grid: Grid) -> float:
    lon, lat = crs_to_lonlat(
        grid.crs,
        [grid.transform[2] + grid.width * grid.xres / 2],
        [grid.transform[5] - grid.height * grid.yres / 2],
    )
    scale = k2(grid.crs, float(lon[0]), float(lat[0]))
    return grid.xres * grid.yres / (scale[0] if scale is not None else 1.0)


def _citations(srcs: Sequence[Source]) -> list[str]:
    seen: list[str] = []
    for src in srcs:
        if src.chip is None:
            continue
        year = (_utc_date(src.datetime) or "")[:4] or None
        label = src.chip.source or src.chip.collection or ""
        for line in provenance.citations(label, year=year):
            if line not in seen:
                seen.append(line)
    return seen


# --------------------------------------------------------------------------- preview


def _stretch(plane: NDArray[np.float32]) -> NDArray[np.uint8]:
    finite = plane[np.isfinite(plane)]
    if finite.size == 0:
        return np.zeros(plane.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, [2, 98])
    if hi <= lo:
        hi = lo + 1.0
    scaled = np.clip((np.nan_to_num(plane, nan=lo) - lo) / (hi - lo), 0.0, 1.0)
    return np.asarray(scaled * 255.0 + 0.5, dtype=np.uint8)


def _preview_indices(metas: Sequence[LayerMeta]) -> tuple[list[int], bool]:
    by_date: dict[str, dict[str, int]] = {}
    for i, m in enumerate(metas):
        base, _, date = m.name.partition("@")
        by_date.setdefault(date, {})[base] = i
    for _, found in sorted(by_date.items()):
        if all(c in found for c in ("red", "green", "blue")):
            return [found["red"], found["green"], found["blue"]], True
    for i, m in enumerate(metas):
        if m.kind != "categorical":
            return [i], False
    return [0], False


def _preview_png(files: Sequence[Path], metas: Sequence[LayerMeta]) -> bytes | None:
    """Return a tiny 2-98 stretch PNG (RGB when red/green/blue exist), or None on failure."""
    try:
        from PIL import Image

        idx, rgb = _preview_indices(metas)
        chans = []
        for i in idx:
            plane = np.load(files[i], mmap_mode="r")
            step = max(1, math.ceil(max(plane.shape) / _PREVIEW_SIDE))
            chans.append(_stretch(np.asarray(plane[::step, ::step])))
        arr = np.stack(chans, axis=-1) if rgb else chans[0]
        buf = _io.BytesIO()
        Image.fromarray(arr, "RGB" if rgb else "L").save(buf, format="PNG")
        return buf.getvalue()
    except (ImportError, OSError, ValueError):
        return None


# --------------------------------------------------------------------------- build


def _report_from(stack: Stack) -> PrepareReport:
    info = stack.meta.provenance.get("prepare", {})
    return PrepareReport(
        combine_notes=list(info.get("combine_notes", [])),
        resampling_notes=list(info.get("resampling_notes", [])),
        qa_fractions=dict(info.get("qa_fractions", {})),
        clear_pct=info.get("clear_pct"),
        reused=True,
        aoi_px=int(info.get("aoi_px", stack.grid.width * stack.grid.height)),
        warnings=list(stack.meta.warnings),
        citations=list(stack.meta.provenance.get("citations", [])),
        layer_bands=[(m.name, m.band) for m in stack.meta.layers],
    )


def _qa_warnings(
    report: PrepareReport, layers: Sequence[_Layer], all_srcs: Sequence[Source]
) -> None:
    if not any(_unit_kind(layer.tiles[0], True)[1] == "reflectance" for layer in layers):
        return
    qa_dates = {_utc_date(s.datetime) for s in all_srcs if _is_qa(s)}
    if not qa_dates:
        report.warnings.append("no QA chip (scl/qa_pixel) among the sources: no cloud masking")
        return
    missing = sorted({d for layer in layers if (d := layer.date) and d not in qa_dates})
    if missing:
        report.warnings.append(f"no QA chip for dates {missing}: those layers are unmasked")


def build_stack(
    sources: Sequence[Source],
    *,
    bands: Sequence[str] | None,
    resolution: str | float,
    crs: str,
    aoi: AoiSpec | None,
    qa: str,
    dilate_px: int,
    sar: Mapping[str, Any] | None,
    name: str | None,
    merge: str,
    overwrite: bool,
    store: LocalArtifactStore,
    deadline: float | None = None,
    arguments: Mapping[str, Any] | None = None,
) -> tuple[Stack, PrepareReport]:
    """Build (or reuse) an analysis-ready stack from sources.

    The grid and identity come from sidecars only, so an identical repeat call returns the
    existing stack before any pixel is read. One layer per (name, date); dates are never
    combined. Pixels stream into scratch memmaps, never all in RAM.

    Args:
        sources: Output of discover.
        bands: Common names or literal keys to keep; None keeps every layer.
        resolution: "finest", "coarsest" or metres.
        crs: "native", "utm" or "EPSG:n".
        aoi: Bbox or polygon (lon/lat); None uses the footprint of the data.
        qa: "auto" builds QA planes from QA chips; "none" skips them.
        dilate_px: Radius grown around cloud, shadow and cirrus after resampling.
        sar: Options {"db": bool, "despeckle": "none"|"median3"|"lee5"}.
        name: Stack id; default sensor_date_hash6.
        merge: "none" refuses several tiles per (name, date); "mosaic" composites them.
        overwrite: Replace an existing stack of the same id with other parameters.
        store: Session store.
        deadline: Absolute time.monotonic() cap.
        arguments: Tool arguments to record in provenance.

    Returns:
        The stack and a report.

    Raises:
        GeoError: Bad references, mixed units or CRSs, size guards, missing extras.
        GeoCancelled: Cancelled or past the deadline.
    """
    if aoi is not None and aoi.polygon is not None and len(aoi.polygon) > 500:
        raise StackError(fix("polygon has over 500 vertices.", "simplify the ring to <=500"))
    sar_opts = {"db": True, "despeckle": "none", **dict(sar or {})}
    chosen = _select(sources, bands)
    layers = _group(chosen, list(bands or []))
    _check_units(layers, bool(sar_opts["db"]))
    _check_tiles(layers, merge)
    qa_srcs = [s for s in sources if _is_qa(s)] if qa == "auto" else []
    all_srcs = list(dict.fromkeys([*chosen, *qa_srcs]))
    grids = {s.path: _source_grid(s) for s in all_srcs}
    target = _plan_grid(
        chosen, grids, crs=crs, resolution=resolution, aoi=aoi, n_layers=len(layers)
    )
    adir = paths.analysis_dir(store, ensure=True)
    params = {
        "bands": sorted(b.lower() for b in bands) if bands else None,
        "qa": qa,
        "dilate_px": dilate_px,
        "sar": sar_opts,
        "merge": merge,
        "resolution": resolution if isinstance(resolution, str) else float(resolution),
        "crs": crs,
        "aoi": _norm_aoi(aoi),
        "layers": [layer.name for layer in layers],
    }
    keys = sorted((_file_key(s) for s in all_srcs), key=lambda k: json.dumps(k, default=str))
    phash = hash_params({"sources": keys, "grid": target.to_meta(), "params": params})
    stack_id = _planned_id(chosen, phash, name)
    stored = exists(adir, stack_id)
    if stored == phash:
        stack = Stack.open(stack_id, store=store)
        return stack, _report_from(stack)
    if stored is not None and not overwrite:
        raise StackError(
            fix(
                f"stack '{stack_id}' exists with different parameters.",
                f"name='{stack_id}-2' or overwrite=true",
            )
        )
    ctx = _Ctx(
        target, grids, bool(sar_opts["db"]), str(sar_opts["despeckle"]), merge, dilate_px, deadline
    )
    scratch = adir / f".scratch-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    scratch.mkdir(parents=True)
    try:
        return _build(
            ctx, layers, all_srcs, qa, aoi, stack_id, phash, overwrite, stored is not None,
            store, adir, scratch, arguments if arguments is not None else params,
        )  # fmt: skip
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _build(
    ctx: _Ctx,
    layers: Sequence[_Layer],
    all_srcs: Sequence[Source],
    qa_mode: str,
    aoi: AoiSpec | None,
    stack_id: str,
    phash: str,
    overwrite: bool,
    had_old: bool,
    store: LocalArtifactStore,
    adir: Path,
    scratch: Path,
    arguments: Mapping[str, Any],
) -> tuple[Stack, PrepareReport]:
    target = ctx.target
    report = PrepareReport(overwrote=had_old and overwrite)
    files: list[Path] = []
    metas: list[LayerMeta] = []
    for i, layer in enumerate(layers):
        check_cancel(ctx.deadline)
        plane, meta = _build_layer(layer, ctx, store, report)
        files.append(scratch / f"layer-{i:03d}.npy")
        np.save(files[-1], plane)
        metas.append(meta)
        del plane
    polygon = aoi.polygon if aoi is not None else None
    aoi_mask = target.rasterize(polygon) if polygon is not None else None
    aoi_px = int(aoi_mask.sum()) if aoi_mask is not None else target.width * target.height
    qa_out = _build_qa(all_srcs, ctx, scratch, aoi_mask, qa_mode, report)
    if qa_mode == "auto":
        _qa_warnings(report, layers, all_srcs)
    report.aoi_px = aoi_px
    report.layer_bands = [(m.name, m.band) for m in metas]
    report.resampling_notes[:] = list(dict.fromkeys(report.resampling_notes))
    if len({layer.date for layer in layers if layer.date}) > 1:
        report.combine_notes.append(
            "one layer per date (name@YYYYMMDD); dates are never combined or medianed"
        )
    report.citations = _citations(all_srcs)
    report.clear_pct = _clear_pct(files, metas, qa_out, aoi_px, target.shape)
    shas = {s.path: provenance.file_sha256(s.path) for s in all_srcs}
    inputs = [{"file": _rel(store, p), "sha256": sha} for p, sha in shas.items()]
    for meta, layer in zip(metas, layers, strict=True):
        if meta.source is not None:
            meta.source.sha256 = shas[layer.tiles[0].path]
    box = _aoi_lonlat(aoi, [target])
    aoi_meta = AoiMeta(
        bbox=list(box),
        polygon=[list(p) for p in polygon] if polygon else None,
        area_ha=aoi_px * _pixel_area_m2(target) / 1e4,
    )
    prov: dict[str, Any] = {
        "tool": "prepare_stack",
        "arguments": dict(arguments),
        "inputs": inputs,
        "versions": _versions(),
        "citations": report.citations,
        "prepare": {
            "combine_notes": report.combine_notes,
            "resampling_notes": report.resampling_notes,
            "qa_fractions": report.qa_fractions,
            "clear_pct": report.clear_pct,
            "aoi_px": aoi_px,
        },
    }
    stack = Stack.create(
        adir,
        id=stack_id,
        grid=target,
        layers=metas,
        data=_plane_iter(files),
        qa=qa_out.metas,
        qa_data=_plane_iter(qa_out.files) if qa_out.metas else None,
        provenance=prov,
        aoi=aoi_meta,
        warnings=report.warnings,
        params_hash=phash,
        overwrite=overwrite,
        preview=_preview_png(files, metas),
    )
    return stack, report
