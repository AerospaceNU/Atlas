"""Deterministic synthetic scenes for smoke checks: chips, stacks and planted truth.

Invariant: one private generator (_fields) produces the physical rasters, so chips,
stacks and planted_truth always agree. Noise is seeded per (seed, date, alias), so adding
a rectangle never reshuffles the noise elsewhere. Module level imports numpy and the
standard library only; rasterio and pyproj are never used.
"""

from __future__ import annotations

import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from atlas.geo.stack import Stack

__all__ = ["Rect", "SyntheticScenes", "make_scene_chips", "make_stack", "planted_truth"]

Rect = tuple[int, int, int, int]
"""r0, r1, c0, c1, half-open, in finest-grid pixels."""

_DEFAULT_DATES = ("20240501", "20240601")
_S2_BANDS = ("blue", "green", "red", "nir", "swir16", "swir22")
_S2_KEYS = {
    "blue": "B02",
    "green": "B03",
    "red": "B04",
    "nir": "B8A",
    "swir16": "B11",
    "swir22": "B12",
    "scl": "SCL",
}
_S2_WAVELENGTH = {
    "blue": 0.49,
    "green": 0.56,
    "red": 0.665,
    "nir": 0.865,
    "swir16": 1.61,
    "swir22": 2.19,
}
_BACKGROUND = {
    "blue": 0.04,
    "green": 0.08,
    "red": 0.05,
    "nir": 0.35,
    "swir16": 0.18,
    "swir22": 0.10,
}
_BURN = {"nir": 0.12, "swir22": 0.25}
_WATER = {"green": 0.06, "nir": 0.02, "swir16": 0.01}
_CLOUD_REFLECTANCE = 0.60
_SCALE = 1e-4
_OFFSET = -0.1
_SCL_BACKGROUND, _SCL_WATER, _SCL_CLOUD = 4, 6, 9
_REFL_SIGMA = 0.002
_S1_SIGMA_DB = 0.3
_S1_BACKGROUND_DB, _S1_FLOOD_DB, _S1_PERMANENT_DB = -8.0, -22.0, -20.0
_S1_NODATA = -32768.0
_KIND_FEATURES = {"s2": ("burn", "cloud", "water"), "s1": ("flood", "permanent_water")}
_ALL_FEATURES = ("burn", "cloud", "water", "flood", "permanent_water")
_LAST_DATE_ONLY = ("burn", "cloud", "flood")


@dataclass(frozen=True)
class SyntheticScenes:
    """Chips written by make_scene_chips plus what was planted."""

    root: Path
    kind: str
    scene_ids: tuple[str, ...]
    dates: tuple[str, ...]
    chips: dict[tuple[str, str], Path]
    grid: Any
    rects: dict[str, Rect]


# --------------------------------------------------------------------------- validation


def _validate(
    kind: str,
    shape: tuple[int, int],
    dates: Sequence[str],
    features: Mapping[str, Rect | None],
) -> dict[str, Rect]:
    if kind not in _KIND_FEATURES:
        raise ValueError(f"kind must be 's2' or 's1', got {kind!r}")
    if not dates or any(len(d) != 8 or not d.isdigit() for d in dates):
        raise ValueError(f"dates must be non-empty YYYYMMDD strings, got {tuple(dates)!r}")
    height, width = shape
    if height <= 0 or width <= 0:
        raise ValueError(f"size must be positive, got {shape}")
    rects: dict[str, Rect] = {}
    for name, rect in features.items():
        if rect is None:
            continue
        if name not in _KIND_FEATURES[kind]:
            raise ValueError(f"kind {kind!r} accepts only {_KIND_FEATURES[kind]}, got {name!r}")
        r0, r1, c0, c1 = (int(v) for v in rect)
        if not (0 <= r0 < r1 <= height and 0 <= c0 < c1 <= width):
            raise ValueError(f"{name} rect {rect} is outside the {height}x{width} grid")
        rects[name] = (r0, r1, c0, c1)
    names = list(rects)
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            if _overlaps(rects[first], rects[second]):
                raise ValueError(f"{first} and {second} rects overlap")
    return rects


def _overlaps(a: Rect, b: Rect) -> bool:
    return a[0] < b[1] and b[0] < a[1] and a[2] < b[3] and b[2] < a[3]


def _check_coarse(shape: tuple[int, int], rects: Mapping[str, Rect], nodata_rows: int) -> None:
    if shape[0] % 2 or shape[1] % 2:
        raise ValueError(f"coarse_swir needs an even size, got {shape}")
    if nodata_rows % 2:
        raise ValueError(f"coarse_swir needs even nodata_rows, got {nodata_rows}")
    for name, rect in rects.items():
        if any(v % 2 for v in rect):
            raise ValueError(
                f"coarse_swir needs even rect coordinates; {name} {rect} would blur at 2x pixels"
            )


def _check_nodata(rects: Mapping[str, Rect], nodata_rows: int) -> None:
    if nodata_rows < 0:
        raise ValueError(f"nodata_rows must be >= 0, got {nodata_rows}")
    for name, rect in rects.items():
        if nodata_rows > rect[0]:
            raise ValueError(f"nodata_rows={nodata_rows} would cover {name} rect {rect}")


# --------------------------------------------------------------------------- fields


def _noise(seed: int, date: str, alias: str, shape: tuple[int, int], sigma: float) -> Any:
    rng = np.random.default_rng([seed, int(date), zlib.crc32(alias.encode())])
    return (rng.standard_normal(shape) * sigma).astype(np.float32)


def _plant(arr: NDArray[Any], rect: Rect, value: float) -> None:
    arr[rect[0] : rect[1], rect[2] : rect[3]] = value


def _active(rects: Mapping[str, Rect], date: str, dates: Sequence[str]) -> dict[str, Rect]:
    last = date == dates[-1]
    return {n: r for n, r in rects.items() if last or n not in _LAST_DATE_ONLY}


def _s2_fields(
    shape: tuple[int, int], rects: Mapping[str, Rect], date: str, seed: int
) -> dict[str, NDArray[Any]]:
    out: dict[str, NDArray[Any]] = {}
    scl = np.full(shape, _SCL_BACKGROUND, dtype=np.uint8)
    for band in _S2_BANDS:
        arr = np.full(shape, _BACKGROUND[band], dtype=np.float32)
        for name, rect in rects.items():
            if name == "burn" and band in _BURN:
                _plant(arr, rect, _BURN[band])
            elif name == "water" and band in _WATER:
                _plant(arr, rect, _WATER[band])
            elif name == "cloud":
                _plant(arr, rect, _CLOUD_REFLECTANCE)
        out[band] = arr + _noise(seed, date, band, shape, _REFL_SIGMA)
    for name, code in (("water", _SCL_WATER), ("cloud", _SCL_CLOUD)):
        if name in rects:
            _plant(scl, rects[name], code)
    out["scl"] = scl
    return out


def _s1_fields(
    shape: tuple[int, int], rects: Mapping[str, Rect], date: str, seed: int
) -> dict[str, NDArray[Any]]:
    db = np.full(shape, _S1_BACKGROUND_DB, dtype=np.float32)
    if "flood" in rects:
        _plant(db, rects["flood"], _S1_FLOOD_DB)
    if "permanent_water" in rects:
        _plant(db, rects["permanent_water"], _S1_PERMANENT_DB)
    db = db + _noise(seed, date, "vv", shape, _S1_SIGMA_DB)
    vv = np.power(10.0, db / 10.0).astype(np.float32)
    return {"vv": vv, "vh": (vv * 0.25).astype(np.float32)}


def _fields(
    kind: str,
    shape: tuple[int, int],
    rects: Mapping[str, Rect],
    dates: Sequence[str],
    seed: int,
) -> dict[str, dict[str, NDArray[Any]]]:
    """Return date -> alias -> physical array (reflectance, linear gamma0, scl codes)."""
    build = _s2_fields if kind == "s2" else _s1_fields
    return {d: build(shape, _active(rects, d, dates), d, seed) for d in dates}


# --------------------------------------------------------------------------- chips


def _grid(epsg: int, res_m: float, origin: tuple[float, float], shape: tuple[int, int]) -> Any:
    from atlas.geo.grid import Grid

    transform = (res_m, 0.0, origin[0], 0.0, -res_m, origin[1])
    return Grid(f"EPSG:{epsg}", transform, shape[1], shape[0])


def _s2_dn(refl: NDArray[Any], nodata_rows: int) -> NDArray[np.uint16]:
    dn = np.rint((refl.astype(np.float64) - _OFFSET) / _SCALE).astype(np.uint16)
    dn[:nodata_rows] = 0
    return dn


def _block_mean(arr: NDArray[Any]) -> NDArray[Any]:
    h, w = arr.shape
    return arr.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3), dtype=np.float64).astype(np.float32)


def _calibration(kind: str, alias: str) -> dict[str, Any]:
    if kind == "s1":
        return {"scale": 1.0, "offset": 0.0, "nodata_in": _S1_NODATA, "dtype_in": "float32"}
    if alias == "scl":
        return {"scale": 1.0, "offset": 0.0, "nodata_in": 0.0, "dtype_in": "uint8"}
    return {"scale": _SCALE, "offset": _OFFSET, "nodata_in": 0.0, "dtype_in": "uint16"}


def _write_chip(
    path: Path,
    array: NDArray[Any],
    *,
    res_m: float,
    origin: tuple[float, float],
    epsg: int,
    nodata: float,
    meta: dict[str, Any],
) -> None:
    from atlas.data.cog import CogRead, encode_cog
    from atlas.geo.chipmeta import (
        BandInfo,
        Calibration,
        ChipMeta,
        FileInfo,
        RasterInfo,
        write_chip_meta,
    )

    read = CogRead(
        array=array[None],
        origin_x=origin[0],
        origin_y=origin[1],
        pixel_size_x=res_m,
        pixel_size_y=res_m,
        epsg=epsg,
        col_off=0,
        row_off=0,
        bytes_fetched=0,
        nodata=str(int(nodata)),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encode_cog(read, tile=512, compress="deflate"))
    height, width = array.shape
    write_chip_meta(
        path,
        ChipMeta(
            scene_id=meta["scene_id"],
            source=meta["source"],
            collection=meta["collection"],
            platform=meta["platform"],
            datetime=meta["datetime"],
            asset_key=meta["asset_key"],
            alias=meta["alias"],
            band=BandInfo(
                common_name=meta["alias"],
                wavelength_um=meta.get("wavelength_um"),
                gsd_m=res_m,
                source_band=meta["asset_key"],
            ),
            file=FileInfo(bytes=path.stat().st_size, windowed=True),
            raster=RasterInfo(
                dtype=str(array.dtype),
                height=height,
                width=width,
                epsg=epsg,
                transform=[res_m, 0.0, origin[0], 0.0, -res_m, origin[1]],
                nodata=nodata,
                registration="area",
            ),
            calibration=Calibration(source="synthetic", **meta["calibration"]),
            qa_preset=meta.get("qa_preset"),
            scene_properties=meta["scene_properties"],
        ),
    )


def _chip_arrays(
    kind: str,
    fields: Mapping[str, NDArray[Any]],
    nodata_rows: int,
    coarse_swir: bool,
) -> dict[str, tuple[NDArray[Any], int, float]]:
    """Return alias -> (stored array, resolution factor, nodata value)."""
    out: dict[str, tuple[NDArray[Any], int, float]] = {}
    if kind == "s1":
        for alias in ("vv", "vh"):
            arr = fields[alias].copy()
            arr[:nodata_rows] = _S1_NODATA
            out[alias] = (arr, 1, _S1_NODATA)
        return out
    for band in _S2_BANDS:
        coarse = coarse_swir and band in ("swir16", "swir22")
        refl = _block_mean(fields[band]) if coarse else fields[band]
        step = 2 if coarse else 1
        out[band] = (_s2_dn(refl, nodata_rows // step), step, 0.0)
    scl = fields["scl"].copy()
    scl[:nodata_rows] = 0
    out["scl"] = (scl[::2, ::2] if coarse_swir else scl, 2 if coarse_swir else 1, 0.0)
    return out


def make_scene_chips(
    root: Path,
    *,
    kind: str = "s2",
    size: tuple[int, int] = (1200, 1200),
    res_m: float = 10.0,
    dates: Sequence[str] = _DEFAULT_DATES,
    epsg: int = 32610,
    origin: tuple[float, float] = (500000.0, 4100000.0),
    burn: Rect | None = None,
    cloud: Rect | None = None,
    water: Rect | None = None,
    flood: Rect | None = None,
    permanent_water: Rect | None = None,
    coarse_swir: bool = False,
    nodata_rows: int = 0,
    seed: int = 0,
) -> SyntheticScenes:
    """Write raw chips and sidecars for synthetic Sentinel-2 or Sentinel-1 scenes.

    Chips are multi-tile (512) deflate GeoTIFFs under root/data/<source>/. S2 chips hold
    uint16 DN (reflectance = DN*1e-4 - 0.1) plus a uint8 scl chip; S1 chips hold float32
    linear gamma0 vv and vh. burn, cloud and flood apply to the last date only (dates[0]
    is the pre-event date); water and permanent_water apply to every date.

    Args:
        root: Directory that receives data/<source>/.
        kind: "s2" or "s1".
        size: (height, width) in finest-grid pixels.
        res_m: Pixel size in metres.
        dates: YYYYMMDD strings, pre-event first.
        epsg: Projected CRS code.
        origin: Upper-left outer corner (x, y).
        burn: Rect (r0, r1, c0, c1) of burned area (s2).
        cloud: Rect of cloud (s2).
        water: Rect of water (s2).
        flood: Rect of new flood water (s1).
        permanent_water: Rect of permanent water (s1).
        coarse_swir: Write swir16, swir22 and scl at 2 * res_m (block mean and decimation).
        nodata_rows: Leading rows written as nodata; must precede every rect.
        seed: Noise seed.

    Returns:
        The chips, grid and planted rectangles.

    Raises:
        ValueError: Rects outside the grid, overlapping, wrong for the kind, or misaligned
            for coarse_swir.
    """
    shape = (int(size[0]), int(size[1]))
    dates = tuple(dates)
    rects = _validate(
        kind,
        shape,
        dates,
        {
            "burn": burn,
            "cloud": cloud,
            "water": water,
            "flood": flood,
            "permanent_water": permanent_water,
        },
    )
    _check_nodata(rects, nodata_rows)
    if coarse_swir:
        if kind != "s2":
            raise ValueError("coarse_swir applies to kind 's2' only")
        _check_coarse(shape, rects, nodata_rows)
    fields = _fields(kind, shape, rects, dates, seed)
    source, prefix, collection, platform = (
        ("sentinel2", "S2A_SYN_", "sentinel-2-l2a", "sentinel-2a")
        if kind == "s2"
        else ("sentinel1", "S1A_SYN_", "sentinel-1-rtc", "sentinel-1a")
    )
    chips: dict[tuple[str, str], Path] = {}
    for date in dates:
        scene_id = prefix + date
        props: dict[str, Any] = (
            {"eo:cloud_cover": 0.0, "s2:processing_baseline": "05.10"}
            if kind == "s2"
            else {"sat:relative_orbit": "8", "sat:orbit_state": "ascending"}
        )
        for alias, (array, step, nodata) in _chip_arrays(
            kind, fields[date], nodata_rows, coarse_swir
        ).items():
            path = root / "data" / source / f"{scene_id}_{alias}.tif"
            _write_chip(
                path,
                array,
                res_m=res_m * step,
                origin=origin,
                epsg=epsg,
                nodata=nodata,
                meta={
                    "scene_id": scene_id,
                    "source": source,
                    "collection": collection,
                    "platform": platform,
                    "datetime": f"{date[:4]}-{date[4:6]}-{date[6:]}T18:00:00Z",
                    "asset_key": _S2_KEYS[alias] if kind == "s2" else alias,
                    "alias": alias,
                    "wavelength_um": _S2_WAVELENGTH.get(alias),
                    "calibration": _calibration(kind, alias),
                    "qa_preset": "s2-scl" if alias == "scl" else None,
                    "scene_properties": props,
                },
            )
            chips[(date, alias)] = path
    return SyntheticScenes(
        root=root,
        kind=kind,
        scene_ids=tuple(prefix + d for d in dates),
        dates=dates,
        chips=chips,
        grid=_grid(epsg, res_m, origin, shape),
        rects=rects,
    )


# --------------------------------------------------------------------------- stacks


def _stack_layers(kind: str, dates: Sequence[str], res_m: float) -> list[tuple[str, str, str, str]]:
    """Return (layer name, alias, date, iso datetime) in stack order."""
    aliases = _S2_BANDS if kind == "s2" else ("vv", "vh")
    rows = []
    for date in dates:
        iso = f"{date[:4]}-{date[4:6]}-{date[6:]}T18:00:00Z"
        for alias in aliases:
            name = alias if len(dates) == 1 else f"{alias}@{date}"
            rows.append((name, alias, date, iso))
    return rows


def make_stack(
    analysis_dir: Path,
    *,
    kind: str = "s2",
    size: tuple[int, int] = (64, 64),
    res_m: float = 10.0,
    dates: Sequence[str] = _DEFAULT_DATES,
    burn: Rect | None = None,
    cloud: Rect | None = None,
    water: Rect | None = None,
    flood: Rect | None = None,
    permanent_water: Rect | None = None,
    name: str = "synth",
    epsg: int = 32610,
    origin: tuple[float, float] = (500000.0, 4100000.0),
    seed: int = 0,
) -> Stack:
    """Write a synthetic Stack directly in physical units (no chip I/O).

    S2 layers are reflectance; S1 layers are dB (10*log10 of linear gamma0). Layer names
    are plain for one date, else name@date. S2 stacks carry a cloud QA plane for the last
    date (all zeros when no cloud rect is given). The planted rects are recorded in
    provenance.arguments.rects for planted_truth.

    Args:
        analysis_dir: The analysis directory that receives stacks/<name>/.
        kind: "s2" or "s1".
        size: (height, width) in pixels.
        res_m: Pixel size in metres.
        dates: YYYYMMDD strings, pre-event first.
        burn: Burn rect (last date only, s2).
        cloud: Cloud rect (last date only, s2).
        water: Water rect (every date, s2).
        flood: Flood rect (last date only, s1).
        permanent_water: Permanent water rect (every date, s1).
        name: Stack id.
        epsg: Projected CRS code.
        origin: Upper-left outer corner (x, y).
        seed: Noise seed.

    Returns:
        The committed stack.

    Raises:
        ValueError: Invalid rects or kind (see make_scene_chips).
    """
    from atlas.geo.stack import LayerMeta, OrbitMeta, QaMeta, Stack, hash_params

    shape = (int(size[0]), int(size[1]))
    dates = tuple(dates)
    rects = _validate(
        kind,
        shape,
        dates,
        {
            "burn": burn,
            "cloud": cloud,
            "water": water,
            "flood": flood,
            "permanent_water": permanent_water,
        },
    )
    fields = _fields(kind, shape, rects, dates, seed)
    layers: list[LayerMeta] = []
    planes: list[NDArray[Any]] = []
    for lname, alias, date, iso in _stack_layers(kind, dates, res_m):
        raw = fields[date][alias]
        if kind == "s2":
            planes.append(raw.astype(np.float32))
            layers.append(
                LayerMeta(
                    name=lname,
                    kind="reflectance",
                    band=_S2_KEYS[alias],
                    unit="1",
                    datetime=iso,
                    sensor="sentinel-2",
                    wavelength_um=_S2_WAVELENGTH[alias],
                    native_resolution_m=res_m,
                )
            )
        else:
            planes.append((10.0 * np.log10(raw)).astype(np.float32))
            layers.append(
                LayerMeta(
                    name=lname,
                    kind="backscatter",
                    band=alias.upper(),
                    unit="dB",
                    datetime=iso,
                    sensor="sentinel-1",
                    native_resolution_m=res_m,
                    orbit=OrbitMeta(relative=8, state="ascending"),
                )
            )
    qa: list[QaMeta] = []
    qa_data: NDArray[np.uint8] | None = None
    if kind == "s2":
        last = dates[-1]
        qa = [
            QaMeta(
                name="cloud" if len(dates) == 1 else f"cloud@{last}",
                semantic="cloud",
                preset="s2-scl",
            )
        ]
        qa_data = (fields[last]["scl"] == _SCL_CLOUD).astype(np.uint8)[None]
    arguments = {
        "kind": kind,
        "rects": {k: list(v) for k, v in rects.items()},
        "res_m": res_m,
        "size": list(shape),
        "dates": list(dates),
        "seed": seed,
        "epsg": epsg,
        "origin": list(origin),
    }
    return Stack.create(
        analysis_dir,
        id=name,
        grid=_grid(epsg, res_m, origin, shape),
        layers=layers,
        data=np.stack(planes),
        qa=qa,
        qa_data=qa_data,
        provenance={"tool": "synthetic", "arguments": arguments},
        params_hash=hash_params(arguments),
    )


# --------------------------------------------------------------------------- truth


def planted_truth(
    source: SyntheticScenes | Stack | Mapping[str, Rect], *, res_m: float | None = None
) -> dict[str, dict[str, Any]]:
    """Return what was planted, per feature: pixels, hectares and share of the grid.

    Args:
        source: Scenes from make_scene_chips, a Stack from make_stack (rects read from
            provenance.arguments.rects), or a bare name -> rect mapping.
        res_m: Pixel size in metres; required for a bare mapping, ignored otherwise.

    Returns:
        name -> {px, ha, pct_of_grid} (plus dnbr_class "high" for burn). A bare mapping has
        no grid, so its entries omit pct_of_grid.

    Raises:
        ValueError: A bare mapping was given without res_m, or a Stack has no planted rects.
    """
    rects: Mapping[str, Any]
    cells: int | None
    if isinstance(source, SyntheticScenes):
        rects, res, cells = source.rects, source.grid.xres, source.grid.width * source.grid.height
    elif isinstance(source, Mapping):
        if res_m is None:
            raise ValueError("planted_truth(mapping) needs res_m")
        rects, res, cells = source, float(res_m), None
    else:
        args = source.meta.provenance.get("arguments", {})
        if "rects" not in args:
            raise ValueError("stack has no provenance.arguments.rects; not a synthetic stack")
        rects, res = args["rects"], source.grid.xres
        cells = source.grid.width * source.grid.height
    out: dict[str, dict[str, Any]] = {}
    for key, rect in rects.items():
        px = (int(rect[1]) - int(rect[0])) * (int(rect[3]) - int(rect[2]))
        entry: dict[str, Any] = {"px": px, "ha": px * res**2 / 1e4}
        if cells is not None:
            entry["pct_of_grid"] = 100.0 * px / cells
        if key == "burn":
            entry["dnbr_class"] = "high"
        out[key] = entry
    return out
