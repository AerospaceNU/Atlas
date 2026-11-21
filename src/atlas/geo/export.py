"""Export stack layers as GeoTIFF, CSV or GeoJSON, each with a `<file>.json` sidecar.

Invariants: rasterio and pyproj load lazily; without rasterio the GeoTIFF is encoded by
`atlas.data.cog.encode_cog` (no band descriptions); files are written to a temporary
name and renamed into place; an export whose in-memory arrays would exceed 3.5 GiB is refused (a memory guard, not a download limit).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from atlas.geo._lazy import check_cancel, require_pyproj, require_rasterio
from atlas.geo.errors import GeoError, MissingExtraError, fix
from atlas.geo.provenance import file_sha256
from atlas.geo.results import json_safe

if TYPE_CHECKING:
    from atlas.geo.stack import LayerMeta, Stack

__all__ = [
    "default_export_path",
    "write_csv",
    "write_geojson",
    "write_geotiff",
]

MAX_BYTES = int(3.5 * 1024**3)
_EXT = {"geotiff": "tif", "geojson": "geojson", "csv": "csv"}
_STRIP = 256


def _clean(text: str) -> str:
    return "".join(c if c.isalnum() or c in "._+-" else "-" for c in text)


def default_export_path(analysis_dir: Path, stack_id: str, fmt: str, layers: Sequence[str]) -> Path:
    """Return `<analysis_dir>/exports/<stack_id>-<layers>-<h6>.<ext>`.

    Args:
        analysis_dir: The `analysis` directory.
        stack_id: Source stack id.
        fmt: "geotiff", "geojson" or "csv".
        layers: Exported layer names.

    Returns:
        The default path (not created).

    Raises:
        GeoError: Unknown format.
    """
    if fmt not in _EXT:
        raise GeoError(fix(f"unknown format {fmt!r}.", f"use one of {sorted(_EXT)}"))
    digest = hashlib.sha256(json.dumps([stack_id, list(layers), fmt]).encode()).hexdigest()[:6]
    tag = _clean("+".join(layers))[:40]
    return analysis_dir / "exports" / f"{_clean(stack_id)}-{tag}-{digest}.{_EXT[fmt]}"


def _tmp(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_name(f"{path.name}.tmp-{os.getpid()}")


def _metas(stack: Stack, layers: Sequence[str], date: str | None) -> list[LayerMeta]:
    if not layers:
        raise GeoError(fix("no layers given.", f"export_geotiff layers=[...] from {stack.names()}"))
    return [stack.layer(n, date=date) for n in layers]


def _sidecar(
    stack: Stack, path: Path, names: Sequence[str], backend: str, extra: dict[str, Any]
) -> None:
    body = {
        "stack_id": stack.id,
        "layers": list(names),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
        "backend": backend,
        "stack_provenance": {
            "params_hash": stack.meta.params_hash,
            "data_sha256": stack.meta.data_sha256,
            "provenance": stack.meta.provenance,
        },
        **extra,
    }
    side = path.with_name(path.name + ".json")
    tmp = _tmp(side)
    tmp.write_text(json.dumps(json_safe(body), indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, side)


def _dtype_plan(metas: Sequence[LayerMeta], stack: Stack) -> tuple[str, str]:
    """Return `(dtype, nodata)`: float32/nan, or uint8/255 or uint16/65535 for classes."""
    if len(metas) == 1 and metas[0].kind == "categorical":
        band = np.asarray(stack.band(metas[0].name))
        finite = band[np.isfinite(band)]
        top = int(finite.max()) if finite.size else 0
        low = int(finite.min()) if finite.size else 0
        if low < 0 or top > 65534:
            raise GeoError(
                fix(
                    f"class codes {low}..{top} do not fit an unsigned 16-bit raster.",
                    "export_geotiff format=csv layers=[that layer]",
                )
            )
        return ("uint8", "255") if top <= 254 else ("uint16", "65535")
    return "float32", "nan"


def _guard_size(stack: Stack, count: int, itemsize: int) -> None:
    height, width = stack.grid.shape
    if count * height * width * itemsize > MAX_BYTES:
        raise GeoError(
            fix(
                f"{count} layers of {width}x{height} exceed the 3.5 GiB in-memory export guard (memory protection, not a download limit).",
                "export_geotiff layers=[one layer] or shrink with prepare_stack bbox",
            )
        )


def _band_array(stack: Stack, meta: LayerMeta, dtype: str, nodata: str) -> NDArray[Any]:
    band = np.asarray(stack.band(meta.name))
    if dtype == "float32":
        return np.asarray(band, dtype=np.float32)
    fill = int(nodata)
    ok = np.isfinite(band)
    return np.where(ok, np.rint(np.where(ok, band, 0)), fill).astype(dtype)


def _block(n: int) -> int:
    return min(512, max(16, math.ceil(n / 16) * 16))


def _write_rasterio(
    stack: Stack, metas: Sequence[LayerMeta], dtype: str, nodata: str, tmp: Path
) -> None:
    rio = require_rasterio()
    height, width = stack.grid.shape
    with rio.open(
        tmp,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=len(metas),
        dtype=dtype,
        crs=rio.crs.CRS.from_epsg(stack.grid.epsg),
        transform=rio.transform.Affine(*stack.grid.transform),
        nodata=float("nan") if nodata == "nan" else int(nodata),
        tiled=True,
        blockxsize=_block(width),
        blockysize=_block(height),
        compress="deflate",
        predictor=3 if dtype == "float32" else 2,
    ) as dst:
        for i, meta in enumerate(metas, start=1):
            check_cancel()
            dst.write(_band_array(stack, meta, dtype, nodata), i)
            dst.set_band_description(i, meta.name)
        dst.update_tags(atlas_stack=stack.id)


def _write_fallback(
    stack: Stack, metas: Sequence[LayerMeta], dtype: str, nodata: str, tmp: Path
) -> None:
    from atlas.data.cog import CogRead, encode_cog

    grid = stack.grid
    cube = np.stack([_band_array(stack, m, dtype, nodata) for m in metas])
    check_cancel()
    read = CogRead(
        array=cube,
        origin_x=grid.transform[2],
        origin_y=grid.transform[5],
        pixel_size_x=grid.xres,
        pixel_size_y=grid.yres,
        epsg=grid.epsg,
        col_off=0,
        row_off=0,
        bytes_fetched=0,
        nodata=nodata,
    )
    tmp.write_bytes(encode_cog(read, tile=512, compress="deflate"))


def write_geotiff(
    stack: Stack, layers: Sequence[str], path: Path, *, date: str | None = None
) -> dict[str, Any]:
    """Write layers as a tiled, deflate-compressed GeoTIFF with the stack's CRS and transform.

    Continuous or mixed layers become float32 with NaN nodata; a single categorical layer
    becomes uint8 (nodata 255) or uint16 (nodata 65535). Uses rasterio when installed, else
    the in-package COG encoder (no band descriptions).

    Args:
        stack: Source stack.
        layers: Layer names (base names resolve with `date`).
        path: Destination file.
        date: YYYYMMDD resolving `name@date`.

    Returns:
        `{path, bytes, backend, layers, dtype, epsg, nodata}`.

    Raises:
        GeoError: Unknown layer, oversize export or unsupported CRS.
    """
    metas = _metas(stack, layers, date)
    names = [m.name for m in metas]
    dtype, nodata = _dtype_plan(metas, stack)
    _guard_size(stack, len(metas), np.dtype(dtype).itemsize)
    tmp = _tmp(path)
    try:
        try:
            _write_rasterio(stack, metas, dtype, nodata, tmp)
            backend = "rasterio"
        except MissingExtraError:
            _write_fallback(stack, metas, dtype, nodata, tmp)
            backend = "atlas-cog fallback"
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    _sidecar(stack, path, names, backend, {"dtype": dtype, "epsg": stack.grid.epsg})
    return {
        "path": path,
        "bytes": path.stat().st_size,
        "backend": backend,
        "layers": names,
        "dtype": dtype,
        "epsg": stack.grid.epsg,
        "nodata": nodata,
    }


def write_csv(
    stack: Stack,
    layers: Sequence[str],
    path: Path,
    *,
    date: str | None = None,
    max_rows: int = 1_000_000,
) -> dict[str, Any]:
    """Write one row per clear, finite pixel: `x,y,<layers>` with pixel-centre coordinates.

    Coordinates use `%.10g` (metre precision survives); layer values use `%.6g`.

    Args:
        stack: Source stack.
        layers: Layer names.
        path: Destination file.
        date: YYYYMMDD resolving `name@date`.
        max_rows: Refuse more valid pixels than this.

    Returns:
        `{path, bytes, backend, layers, rows, epsg}`.

    Raises:
        GeoError: Unknown layer or too many rows.
    """
    metas = _metas(stack, layers, date)
    names = [m.name for m in metas]
    valid = stack.valid("clear", layers=names)
    rows = int(valid.sum())
    if rows > max_rows:
        raise GeoError(
            fix(
                f"{rows} valid pixels exceed max_rows={max_rows}.",
                "export_geotiff format=geotiff, or shrink with prepare_stack bbox",
            )
        )
    grid = stack.grid
    tmp = _tmp(path)
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as out:
            out.write(",".join(["x", "y", *names]) + "\n")
            cols = np.arange(grid.width)
            for r0 in range(0, grid.height, _STRIP):
                check_cancel()
                r1 = min(r0 + _STRIP, grid.height)
                mask = valid[r0:r1]
                if not mask.any():
                    continue
                rr, cc = np.nonzero(mask)
                xs, ys = grid.xy(rr + r0, cols[cc])
                data = [np.asarray(stack.band(n)[r0:r1])[mask] for n in names]
                for i in range(len(xs)):
                    cells = [f"{xs[i]:.10g}", f"{ys[i]:.10g}", *(f"{d[i]:.6g}" for d in data)]
                    out.write(",".join(cells) + "\n")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    _sidecar(stack, path, names, "csv", {"rows": rows, "epsg": grid.epsg})
    return {
        "path": path,
        "bytes": path.stat().st_size,
        "backend": "csv",
        "layers": names,
        "rows": rows,
        "epsg": grid.epsg,
    }


def _ring_area(ring: Sequence[Sequence[float]]) -> float:
    total = 0.0
    for (x0, y0), (x1, y1) in pairwise(ring):
        total += x0 * y1 - x1 * y0
    return abs(total) / 2.0


def _lonlat(crs: str, rings: list[list[tuple[float, float]]]) -> list[list[list[float]]]:
    if crs == "EPSG:4326":
        return [[[x, y] for x, y in ring] for ring in rings]
    pyproj = require_pyproj()
    tr = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    out = []
    for ring in rings:
        xs, ys = tr.transform([p[0] for p in ring], [p[1] for p in ring])
        out.append([[float(x), float(y)] for x, y in zip(xs, ys, strict=True)])
    return out


def _features(
    stack: Stack, meta: LayerMeta, valid: NDArray[np.bool_], max_features: int
) -> list[dict[str, Any]]:
    rio = require_rasterio()
    from rasterio.features import shapes
    from rasterio.transform import Affine

    codes = np.asarray(stack.band(meta.name))
    ok = valid & np.isfinite(codes)
    ints = np.where(ok, np.rint(np.where(ok, codes, 0)), 0).astype(np.int32)
    names = meta.classes or {}
    grid = stack.grid
    unit = grid.xres * grid.yres
    found: list[dict[str, Any]] = []
    count = 0
    del rio
    for geom, value in shapes(ints, mask=ok, transform=Affine(*grid.transform), connectivity=4):
        count += 1
        if count % 256 == 0:
            check_cancel()
        if count > max_features:
            continue
        rings = [[(float(x), float(y)) for x, y in ring] for ring in geom["coordinates"]]
        area = _ring_area(rings[0]) - sum(_ring_area(r) for r in rings[1:])
        code = int(value)
        found.append(
            {
                "type": "Feature",
                "properties": {
                    "code": code,
                    "name": names.get(code, f"class {code}"),
                    "ha": area / unit * stack.pixel_area_m2 / 1e4,
                },
                "geometry": {"type": "Polygon", "coordinates": _lonlat(grid.crs, rings)},
            }
        )
    if count > max_features:
        raise GeoError(
            fix(
                f"{count} polygons exceed max_features={max_features}.",
                "export_geotiff format=geotiff, or shrink with prepare_stack bbox",
            )
        )
    return found


def write_geojson(
    stack: Stack,
    layer: str,
    path: Path,
    *,
    date: str | None = None,
    max_features: int = 5000,
) -> dict[str, Any]:
    """Vectorise a categorical layer over clear pixels to lon/lat polygons (no simplification).

    Properties are `code`, `name` and `ha` (valid-pixel area of the polygon).

    Args:
        stack: Source stack.
        layer: A categorical layer.
        path: Destination file.
        date: YYYYMMDD resolving `name@date`.
        max_features: Refuse more polygons than this.

    Returns:
        `{path, bytes, backend, layers, features, epsg}`.

    Raises:
        MissingExtraError: rasterio or pyproj is not installed (the `geo` extra).
        GeoError: The layer is not categorical or has too many polygons.
    """
    require_rasterio()
    if stack.grid.crs != "EPSG:4326":
        require_pyproj()
    meta = _metas(stack, [layer], date)[0]
    if meta.kind != "categorical":
        raise GeoError(
            fix(
                f"layer {meta.name!r} is {meta.kind}, not categorical.",
                "export_geotiff format=geotiff, or classify the layer first",
            )
        )
    valid = stack.valid("clear", layers=[meta.name])
    feats = _features(stack, meta, valid, max_features)
    tmp = _tmp(path)
    try:
        tmp.write_text(
            json.dumps(json_safe({"type": "FeatureCollection", "features": feats})) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    _sidecar(stack, path, [meta.name], "rasterio+pyproj", {"features": len(feats)})
    return {
        "path": path,
        "bytes": path.stat().st_size,
        "backend": "rasterio",
        "layers": [meta.name],
        "features": len(feats),
        "epsg": stack.grid.epsg,
    }
