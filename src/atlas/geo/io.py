"""Raster readers: sidecar chips through the in-repo COG reader, other files via rasterio.

Invariant: pixels stay RAW (source dtype, DN); calibration is applied elsewhere. Returned grids
are PixelIsArea with the transform origin at the OUTER corner of pixel (0, 0).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from atlas.geo._lazy import check_cancel, require_rasterio
from atlas.geo.chipmeta import ChipMeta, read_chip_meta
from atlas.geo.errors import GeoError, MissingExtraError, fix
from atlas.geo.grid import Bounds, Grid, lonlat_bounds_to_crs

__all__ = ["MAX_WINDOW_PIXELS", "RasterBlock", "read_raster"]

MAX_WINDOW_PIXELS = 36_000_000
_LONLAT = "EPSG:4326"
_GEO_HINT = "uv sync --extra geo"


@dataclass(frozen=True)
class RasterBlock:
    """A window of a raster.

    Attributes:
        data: (bands, H, W) in the source dtype, raw DN.
        grid: PixelIsArea grid of the window.
        nodata: Declared nodata value, or None.
        colormap: Palette (all reds, then greens, then blues; 16-bit), or None.
        registration: "point" when the file was PixelIsPoint (shifted to area here or in
            the reader).
        backend: Which reader produced the block.
    """

    data: NDArray[Any]
    grid: Grid
    nodata: float | None
    colormap: tuple[int, ...] | None
    registration: Literal["area", "point"]
    backend: Literal["cog", "rasterio"]


def _fmt_bounds(b: tuple[float, float, float, float]) -> str:
    return ",".join(f"{v:.6g}" for v in b)


def _no_overlap(path: Path, file_bounds: tuple[float, float, float, float] | None) -> GeoError:
    where = f" (file bounds {_fmt_bounds(file_bounds)})" if file_bounds else ""
    return GeoError(
        fix(
            f"AOI does not overlap {path.name}{where}.",
            "check bbox order (west,south,east,north)",
        )
    )


def _too_big(pixels: float) -> GeoError:
    return GeoError(
        fix(
            f"window is {pixels / 1e6:.1f} MP, limit {MAX_WINDOW_PIXELS / 1e6:.0f} MP.",
            "shrink bbox",
        )
    )


def _run_sync[T](make: Callable[[], Coroutine[Any, Any, T]]) -> T:
    """Run a coroutine factory to completion, on a worker thread when a loop is running."""

    def run() -> T:
        return asyncio.run(make())

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return run()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(run).result()


def _parse_nodata(text: str | None, path: Path) -> float | None:
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        raise GeoError(
            fix(
                f"{path.name} has an unparsable nodata value {text!r}.", "pass files[{path,nodata}]"
            )
        ) from None


def _cog_file_info(path: Path) -> tuple[int | None, tuple[float, float, float, float]]:
    from atlas.data import cog

    async def go() -> tuple[int | None, tuple[float, float, float, float]]:
        image = (await cog.open_cog_file(path))._image
        west, north = image.origin_x, image.origin_y
        east = west + image.width * image.pixel_size_x
        south = north - image.height * image.pixel_size_y
        return image.epsg, (west, south, east, north)

    return _run_sync(go)


def _cog_bounds(
    path: Path, meta: ChipMeta | None, bounds: Bounds, bounds_crs: str
) -> tuple[Bounds, int | None]:
    epsg = meta.raster.epsg if meta is not None else None
    if epsg is None:
        epsg = _cog_file_info(path)[0]
    if epsg is None:
        raise GeoError(
            fix(
                f"{path.name} has no usable CRS.",
                "pass files[] with a CRS or install geo (uv sync --extra geo)",
            )
        )
    target = f"EPSG:{epsg}"
    if bounds_crs.upper() == target:
        return bounds, epsg
    if bounds_crs.upper() != _LONLAT:
        raise GeoError(
            fix(
                f"cannot convert bounds from {bounds_crs} to {target} for {path.name}.",
                "give bounds in EPSG:4326 or the file CRS",
            )
        )
    try:
        return lonlat_bounds_to_crs(bounds, target), epsg
    except GeoError as exc:
        raise GeoError(fix(f"CRS {target} unsupported without rasterio.", _GEO_HINT)) from exc


def _read_cog(
    path: Path, bounds: Bounds | None, bounds_crs: str, meta: ChipMeta | None
) -> RasterBlock:
    from atlas.data import cog

    native: Bounds | None = None
    if bounds is not None:
        native, _ = _cog_bounds(path, meta, bounds, bounds_crs)
    try:
        read = cog.read_file_window_sync(path, native, max_pixels=MAX_WINDOW_PIXELS)
    except cog.CogMissError:
        raise _no_overlap(path, _cog_file_info(path)[1]) from None
    except cog.CogLimitError as exc:
        if "window" not in str(exc):
            raise
        raise GeoError(
            fix(
                f"window exceeds the {MAX_WINDOW_PIXELS / 1e6:.0f} MP limit ({exc}).", "shrink bbox"
            )
        ) from exc
    if read.epsg is None:
        raise GeoError(
            fix(f"{path.name} has no usable CRS.", "pass files[] with a CRS or install geo")
        )
    nodata = _parse_nodata(read.nodata, path)
    grid = Grid(
        f"EPSG:{read.epsg}",
        (read.pixel_size_x, 0.0, read.origin_x, 0.0, -read.pixel_size_y, read.origin_y),
        int(read.array.shape[2]),
        int(read.array.shape[1]),
    )
    registration = meta.raster.registration if meta is not None else "area"
    check_cancel()
    return RasterBlock(np.asarray(read.array), grid, nodata, read.colormap, registration, "cog")


def _flat_colormap(ds: Any) -> tuple[int, ...] | None:
    try:
        cmap = ds.colormap(1)
    except ValueError:
        return None
    reds, greens, blues = [0] * 256, [0] * 256, [0] * 256
    for idx, rgba in cmap.items():
        if 0 <= idx < 256:
            reds[idx], greens[idx], blues[idx] = (int(v) * 257 for v in rgba[:3])
    return tuple(reds + greens + blues)


def _ds_grid(ds: Any, path: Path) -> tuple[Grid, bool]:
    t = ds.transform
    if t.b != 0 or t.d != 0 or t.e >= 0 or t.a <= 0:
        raise GeoError(
            fix(
                f"{path.name} is rotated, sheared or south-up; only north-up rasters are read.",
                "gdalwarp it to a north-up grid first",
            )
        )
    epsg = ds.crs.to_epsg() if ds.crs is not None else None
    if epsg is None:
        raise GeoError(
            fix(f"{path.name} has no EPSG CRS.", "gdalwarp -t_srs EPSG:<n> it, or pass files[] crs")
        )
    point = str(ds.tags().get("AREA_OR_POINT", "")).lower() == "point"
    x0, y0 = t.c, t.f
    if point:
        x0, y0 = x0 - t.a / 2.0, y0 - t.e / 2.0
    return Grid(f"EPSG:{epsg}", (t.a, 0.0, x0, 0.0, t.e, y0), ds.width, ds.height), point


def _read_rasterio(path: Path, bounds: Bounds | None, bounds_crs: str) -> RasterBlock:
    rio = require_rasterio()
    from rasterio.warp import transform_bounds
    from rasterio.windows import Window

    with rio.open(path) as ds:
        full, point = _ds_grid(ds, path)
        r0, r1, c0, c1 = 0, full.height, 0, full.width
        if bounds is not None:
            native = transform_bounds(bounds_crs, ds.crs, *bounds, densify_pts=21)
            try:
                r0, r1, c0, c1 = full.window(
                    (float(native[0]), float(native[1]), float(native[2]), float(native[3]))
                )
            except GeoError:
                raise _no_overlap(path, full.bounds) from None
        pixels = (r1 - r0) * (c1 - c0)
        if pixels > MAX_WINDOW_PIXELS:
            raise _too_big(pixels)
        data = ds.read(window=Window(c0, r0, c1 - c0, r1 - r0), masked=False)
        nodata = None if ds.nodata is None else float(ds.nodata)
        colormap = _flat_colormap(ds)
    check_cancel()
    grid = full.subgrid(r0, r1, c0, c1)
    return RasterBlock(data, grid, nodata, colormap, "point" if point else "area", "rasterio")


def _not_tiled(path: Path, exc: Exception) -> MissingExtraError:
    err = MissingExtraError("geo", "rasterio")
    err.args = (
        fix(
            f"{path.name} is not a tiled GeoTIFF readable without rasterio ({exc}).",
            _GEO_HINT,
        ),
    )
    return err


def read_raster(
    path: Path | str,
    *,
    bounds: Bounds | None = None,
    bounds_crs: str = _LONLAT,
    backend: Literal["auto", "cog", "rasterio"] = "auto",
) -> RasterBlock:
    """Read a raster window as raw pixels.

    Args:
        path: Raster file. A chip with a sidecar is read by the in-repo COG reader.
        bounds: ``(west, south, east, north)`` in ``bounds_crs``; the window snaps outward
            to whole pixels. None reads the whole file.
        bounds_crs: CRS of ``bounds`` (default EPSG:4326).
        backend: "auto" (sidecar -> cog, else rasterio, else cog), "cog" or "rasterio".

    Returns:
        The raw block.

    Raises:
        GeoError: No overlap, window over 36 MP, rotated raster, unreadable nodata or CRS.
        MissingExtraError: The route needs rasterio and the geo extra is missing.
    """
    from atlas.data import cog

    p = Path(path)
    meta = read_chip_meta(p)
    if backend == "cog" or (backend == "auto" and meta is not None):
        try:
            return _read_cog(p, bounds, bounds_crs, meta)
        except cog.CogError as exc:
            reason = getattr(exc, "reason", None)
            raise GeoError(
                fix(f"{p.name} cannot be read ({exc}{f'; {reason}' if reason else ''}).", _GEO_HINT)
            ) from exc
    if backend == "rasterio":
        return _read_rasterio(p, bounds, bounds_crs)
    try:
        require_rasterio()
    except MissingExtraError:
        try:
            return _read_cog(p, bounds, bounds_crs, meta)
        except (cog.CogError, OSError) as exc:
            raise _not_tiled(p, exc) from exc
    return _read_rasterio(p, bounds, bounds_crs)
