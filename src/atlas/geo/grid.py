"""North-up projected grids, CRS helpers and polygon rasterisation.

Invariant: the transform origin is the OUTER corner of pixel (0, 0). pyproj is
optional; UTM and Web Mercator work without it.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from atlas.geo._lazy import have_pyproj, require_pyproj
from atlas.geo.errors import GeoError, GridMismatch, StackError, fix

__all__ = [
    "Affine6",
    "Grid",
    "crs_to_lonlat",
    "k2",
    "lonlat_bounds_of",
    "lonlat_bounds_to_crs",
    "lonlat_to_crs",
    "polar_epsg",
    "utm_epsg",
]

Affine6 = tuple[float, float, float, float, float, float]
Bounds = tuple[float, float, float, float]
Window = tuple[int, int, int, int]

_EPS_PX = 1e-6
_CRS_RE = re.compile(r"^epsg:(\d+)$", re.IGNORECASE)
_E2_PRIME = 0.00673949674
_K0 = 0.9996
_N = 0.0016792203863837047
_A = 6378137.0
_ROW_CHUNK = 512


def _normalise_crs(crs: object) -> str:
    if isinstance(crs, int) and not isinstance(crs, bool):
        return f"EPSG:{crs}"
    if isinstance(crs, str):
        match = _CRS_RE.match(crs.strip())
        if match:
            return f"EPSG:{int(match.group(1))}"
    raise StackError(
        fix(f"unsupported crs {str(crs)[:60]!r}: only EPSG codes are accepted.", "use crs=EPSG:<n>")
    )


@dataclass(frozen=True)
class Grid:
    """A north-up pixel grid: CRS, affine transform, width and height."""

    crs: str
    transform: Affine6
    width: int
    height: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "crs", _normalise_crs(self.crs))
        transform = tuple(float(v) for v in self.transform)
        object.__setattr__(self, "transform", transform)
        if len(transform) != 6:
            raise StackError(fix("transform needs 6 values.", "prepare_stack crs=utm"))
        xres, b, _, d, e, _ = transform
        if b != 0 or d != 0 or xres <= 0 or e >= 0 or self.width <= 0 or self.height <= 0:
            raise StackError(
                fix(
                    f"grid must be north-up with positive size, got {transform} "
                    f"{self.width}x{self.height}.",
                    "prepare_stack crs=utm",
                )
            )

    @property
    def xres(self) -> float:
        return self.transform[0]

    @property
    def yres(self) -> float:
        return -self.transform[4]

    @property
    def bounds(self) -> Bounds:
        x0, y0 = self.transform[2], self.transform[5]
        return (x0, y0 - self.height * self.yres, x0 + self.width * self.xres, y0)

    @property
    def epsg(self) -> int:
        return int(self.crs.split(":")[1])

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    def window(self, bounds: Bounds, snap: str = "outward") -> Window:
        """Return ``(r0, r1, c0, c1)`` covering ``bounds`` given in the grid CRS.

        Args:
            bounds: ``(minx, miny, maxx, maxy)``.
            snap: "outward", "inward" or "nearest".

        Returns:
            Pixel window clipped to the grid.

        Raises:
            GeoError: The bounds do not overlap the grid.
        """
        minx, miny, maxx, maxy = bounds
        x0, y0 = self.transform[2], self.transform[5]
        lo_c, hi_c = (minx - x0) / self.xres, (maxx - x0) / self.xres
        lo_r, hi_r = (y0 - maxy) / self.yres, (y0 - miny) / self.yres
        if snap == "outward":
            c0, c1 = math.floor(lo_c + _EPS_PX), math.ceil(hi_c - _EPS_PX)
            r0, r1 = math.floor(lo_r + _EPS_PX), math.ceil(hi_r - _EPS_PX)
        elif snap == "inward":
            c0, c1 = math.ceil(lo_c - _EPS_PX), math.floor(hi_c + _EPS_PX)
            r0, r1 = math.ceil(lo_r - _EPS_PX), math.floor(hi_r + _EPS_PX)
        elif snap == "nearest":
            c0, c1, r0, r1 = round(lo_c), round(hi_c), round(lo_r), round(hi_r)
        else:
            raise GeoError(fix(f"unknown snap {snap!r}.", "use outward, inward or nearest"))
        r0, r1 = max(r0, 0), min(r1, self.height)
        c0, c1 = max(c0, 0), min(c1, self.width)
        if r1 <= r0 or c1 <= c0:
            raise GeoError(
                fix(
                    f"bounds do not overlap grid {tuple(round(v, 3) for v in self.bounds)}.",
                    f"use a bbox inside {tuple(round(v, 3) for v in self.bounds)}",
                )
            )
        return (r0, r1, c0, c1)

    def subgrid(self, r0: int, r1: int, c0: int, c1: int) -> Grid:
        """Return the grid of the pixel window ``[r0:r1, c0:c1]``."""
        xres, _, x0, _, e, y0 = self.transform
        transform = (xres, 0.0, x0 + c0 * xres, 0.0, e, y0 + r0 * e)
        return Grid(self.crs, transform, c1 - c0, r1 - r0)

    def xy(
        self, rows: ArrayLike, cols: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return pixel-centre map coordinates for row and column indices."""
        r = np.asarray(rows, dtype=np.float64)
        c = np.asarray(cols, dtype=np.float64)
        return (
            self.transform[2] + (c + 0.5) * self.xres,
            self.transform[5] - (r + 0.5) * self.yres,
        )

    def same_as(self, other: Grid, tol: float = 1e-6) -> bool:
        """Return True when CRS, size and transform agree within ``tol`` pixels."""
        if self.crs != other.crs or self.shape != other.shape:
            return False
        limit = tol * max(self.xres, self.yres)
        return all(
            abs(a - b) <= limit for a, b in zip(self.transform, other.transform, strict=True)
        )

    def require_same(self, other: Grid) -> None:
        """Raise ``GridMismatch`` naming both grids unless they match."""
        if not self.same_as(other):
            raise GridMismatch(
                fix(
                    f"grids differ: {self.crs} {self.transform} {self.width}x{self.height} vs "
                    f"{other.crs} {other.transform} {other.width}x{other.height}.",
                    "prepare_stack with identical crs, resolution and bbox for both",
                )
            )

    def rasterize(self, polygon_lonlat: Sequence[Any]) -> NDArray[np.bool_]:
        """Rasterise a lon/lat polygon to a bool ``(H, W)`` mask (even-odd, pixel centres).

        Args:
            polygon_lonlat: One ring or a list of rings (holes allowed) of ``(lon, lat)``.

        Returns:
            True where a pixel centre lies inside.

        Raises:
            GeoError: Fewer than 3 vertices or a ring spanning more than 180 degrees of longitude.
        """
        rings = _as_rings(polygon_lonlat)
        edges: list[NDArray[np.float64]] = []
        for ring in rings:
            arr = np.asarray(ring, dtype=np.float64)
            if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] < 2:
                raise GeoError(fix("polygon ring needs at least 3 vertices.", "pass a closed ring"))
            if float(arr[:, 0].max() - arr[:, 0].min()) > 180.0:
                raise GeoError(
                    fix(
                        "polygon spans more than 180 degrees of longitude.",
                        "split at the antimeridian",
                    )
                )
            xs, ys = lonlat_to_crs(self.crs, arr[:, 0], arr[:, 1])
            px = (xs - self.transform[2]) / self.xres
            py = (self.transform[5] - ys) / self.yres
            edges.append(np.stack([px, py, np.roll(px, -1), np.roll(py, -1)], axis=1))
        all_edges = np.concatenate(edges, axis=0)
        out = np.zeros((self.height, self.width), dtype=bool)
        for start in range(0, self.height, _ROW_CHUNK):
            stop = min(start + _ROW_CHUNK, self.height)
            out[start:stop] = _scan_rows(all_edges, start, stop, self.width)
        return out

    def to_meta(self) -> dict[str, Any]:
        """Return a JSON-ready dict."""
        return {
            "crs": self.crs,
            "transform": list(self.transform),
            "width": self.width,
            "height": self.height,
        }

    @classmethod
    def from_meta(cls, meta: Mapping[str, Any]) -> Grid:
        """Build a grid from ``to_meta`` output."""
        t = tuple(float(v) for v in meta["transform"])
        if len(t) != 6:
            raise StackError(fix("transform needs 6 values.", "prepare_stack crs=utm"))
        return cls(
            meta["crs"],
            (t[0], t[1], t[2], t[3], t[4], t[5]),
            int(meta["width"]),
            int(meta["height"]),
        )


def _as_rings(polygon: Sequence[Any]) -> list[Sequence[Any]]:
    if len(polygon) == 0:
        raise GeoError(fix("polygon is empty.", "pass a closed ring"))
    first = polygon[0]
    if len(first) > 0 and np.ndim(first[0]) == 0:
        return [polygon]
    return list(polygon)


def _scan_rows(edges: NDArray[np.float64], r0: int, r1: int, width: int) -> NDArray[np.bool_]:
    rows = np.arange(r0, r1, dtype=np.float64) + 0.5
    px1, py1, px2, py2 = (edges[:, i][None, :] for i in range(4))
    y = rows[:, None]
    crosses = (py1 <= y) != (py2 <= y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xs = px1 + (y - py1) * (px2 - px1) / (py2 - py1)
    xs = np.where(crosses, xs, np.inf)
    xs.sort(axis=1)
    if xs.shape[1] % 2:
        xs = np.concatenate([xs, np.full((xs.shape[0], 1), np.inf)], axis=1)
    xa, xb = xs[:, 0::2], xs[:, 1::2]
    live = np.isfinite(xb)
    start = np.where(live, np.ceil(np.where(live, xa, 0.0) - 0.5), 0.0)
    stop = np.where(live, np.ceil(np.where(live, xb, 0.0) - 0.5), 0.0)
    start = np.clip(start, 0, width).astype(np.int64)
    stop = np.clip(stop, 0, width).astype(np.int64)
    diff = np.zeros((r1 - r0, width + 1), dtype=np.int32)
    ridx = np.broadcast_to(np.arange(r1 - r0)[:, None], start.shape)
    np.add.at(diff, (ridx, start), 1)
    np.add.at(diff, (ridx, stop), -1)
    return np.cumsum(diff, axis=1)[:, :width] > 0


def utm_epsg(lon: float, lat: float) -> int:
    """Return the WGS84 UTM EPSG code for a point (``|lat| <= 84``)."""
    if abs(lat) > 84.0:
        raise GeoError(fix(f"latitude {lat} is polar.", "use crs=EPSG:3413/3031"))
    zone = int((lon + 180.0) // 6) % 60 + 1
    return (32600 if lat >= 0 else 32700) + zone


def polar_epsg(lat: float) -> int:
    """Return the polar stereographic EPSG code for a latitude (3413 north, 3031 south)."""
    return 3413 if lat >= 0 else 3031


def _utm_zone(epsg: int) -> tuple[int, bool] | None:
    if 32601 <= epsg <= 32660:
        return epsg - 32600, True
    if 32701 <= epsg <= 32760:
        return epsg - 32700, False
    return None


def _epsg_of(crs: str) -> int:
    return int(_normalise_crs(crs).split(":")[1])


@lru_cache(maxsize=32)
def _transformer(src: str, dst: str) -> Any:
    return require_pyproj().Transformer.from_crs(src, dst, always_xy=True)


def lonlat_to_crs(
    crs: str, lons: ArrayLike, lats: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Project WGS84 lon/lat arrays to ``crs`` (pyproj, else UTM and Web Mercator only)."""
    lon_a = np.atleast_1d(np.asarray(lons, dtype=np.float64))
    lat_a = np.atleast_1d(np.asarray(lats, dtype=np.float64))
    code = _epsg_of(crs)
    if have_pyproj():
        x, y = _transformer("EPSG:4326", f"EPSG:{code}").transform(lon_a, lat_a)
        return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    from atlas.data import cog

    xs = np.empty_like(lon_a)
    ys = np.empty_like(lat_a)
    for i, (lon, lat) in enumerate(zip(lon_a.tolist(), lat_a.tolist(), strict=True)):
        pt = cog.project_lonlat(code, lon, lat)
        if pt is None:
            raise GeoError(fix(f"EPSG:{code} needs pyproj.", "uv sync --extra geo"))
        xs[i], ys[i] = pt
    return xs, ys


def _kruger_inverse(
    zone: int, northern: bool, x: NDArray[np.float64], y: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    n = _N
    big_a = _A / (1 + n) * (1 + n**2 / 4 + n**4 / 64)
    beta = (
        n / 2 - 2 * n**2 / 3 + 37 * n**3 / 96 - n**4 / 360,
        n**2 / 48 + n**3 / 15 - 437 * n**4 / 1440,
        17 * n**3 / 480 - 37 * n**4 / 840,
        4397 * n**4 / 161280,
    )
    delta = (
        2 * n - 2 * n**2 / 3 - 2 * n**3 + 116 * n**4 / 45,
        7 * n**2 / 3 - 8 * n**3 / 5 - 227 * n**4 / 45,
        56 * n**3 / 15 - 136 * n**4 / 35,
        4279 * n**4 / 630,
    )
    xi = (y - (0.0 if northern else 1e7)) / (_K0 * big_a)
    eta = (x - 500000.0) / (_K0 * big_a)
    xi_p, eta_p = xi.copy(), eta.copy()
    for j, b in enumerate(beta, start=1):
        xi_p -= b * np.sin(2 * j * xi) * np.cosh(2 * j * eta)
        eta_p -= b * np.cos(2 * j * xi) * np.sinh(2 * j * eta)
    chi = np.arcsin(np.sin(xi_p) / np.cosh(eta_p))
    phi = chi.copy()
    for j, d in enumerate(delta, start=1):
        phi += d * np.sin(2 * j * chi)
    lam = np.radians(zone * 6.0 - 183.0) + np.arctan2(np.sinh(eta_p), np.cos(xi_p))
    return np.degrees(lam), np.degrees(phi)


def crs_to_lonlat(
    crs: str, xs: ArrayLike, ys: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Inverse of ``lonlat_to_crs`` (pyproj, else a 4-term Kruger series for UTM only)."""
    x_a = np.atleast_1d(np.asarray(xs, dtype=np.float64))
    y_a = np.atleast_1d(np.asarray(ys, dtype=np.float64))
    code = _epsg_of(crs)
    if have_pyproj():
        lon, lat = _transformer(f"EPSG:{code}", "EPSG:4326").transform(x_a, y_a)
        return np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)
    zone = _utm_zone(code)
    if zone is None:
        raise GeoError(fix(f"EPSG:{code} needs pyproj.", "uv sync --extra geo"))
    return _kruger_inverse(zone[0], zone[1], x_a, y_a)


def _perimeter(minx: float, miny: float, maxx: float, maxy: float, n: int) -> tuple[Any, Any]:
    t = np.linspace(0.0, 1.0, max(int(n), 2))
    xs = np.concatenate(
        [
            minx + t * (maxx - minx),
            np.full_like(t, maxx),
            maxx - t * (maxx - minx),
            np.full_like(t, minx),
        ]
    )
    ys = np.concatenate(
        [
            np.full_like(t, miny),
            miny + t * (maxy - miny),
            np.full_like(t, maxy),
            maxy - t * (maxy - miny),
        ]
    )
    return xs, ys


def lonlat_bounds_to_crs(bounds: Bounds, crs: str, densify: int = 21) -> Bounds:
    """Project a lon/lat bbox to ``crs`` by projecting a densified perimeter."""
    xs, ys = _perimeter(*bounds, densify)
    px, py = lonlat_to_crs(crs, xs, ys)
    return (float(px.min()), float(py.min()), float(px.max()), float(py.max()))


def lonlat_bounds_of(grid: Grid, densify: int = 21) -> Bounds:
    """Return ``(west, south, east, north)`` of a grid via its densified perimeter."""
    xs, ys = _perimeter(*grid.bounds, densify)
    lon, lat = crs_to_lonlat(grid.crs, xs, ys)
    return (float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max()))


def k2(crs: str, lon: float, lat: float) -> tuple[float, str] | None:
    """Return ``(areal scale factor, method)`` at a point, or None when unknown.

    Pixel area in m2 is ``xres * yres / k2``.
    """
    code = _epsg_of(crs)
    zone = _utm_zone(code)
    if zone is not None:
        phi = math.radians(lat)
        dlam = math.radians(lon - (zone[0] * 6.0 - 183.0))
        cos_phi = math.cos(phi)
        k = _K0 * (1 + (1 + _E2_PRIME * cos_phi**2) * (dlam * cos_phi) ** 2 / 2)
        return k * k, "utm-analytic"
    if have_pyproj():
        proj = require_pyproj().Proj(f"EPSG:{code}")
        return float(proj.get_factors(lon, lat).areal_scale), "pyproj"
    return None
