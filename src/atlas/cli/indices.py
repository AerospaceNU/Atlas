"""Compute NDVI, McFeeters NDWI, and NBR from surface-reflectance arrays."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from atlas.data.indices import compute_indices

_DESCRIPTION = (
    "Inputs must be surface reflectance arrays stored as .npy files. "
    "Digital numbers can be converted with --scale and --offset "
    "(reflectance = stored * scale + offset). "
    "Sentinel-2 L2A publishes an additive offset of -1000 and a scale of 1e-4, "
    "passed as --scale 1e-4 --offset -0.1. "
    "Landsat Collection 2 uses --scale 2.75e-5 --offset -0.2. "
    "NDWI is the McFeeters index and is written as ndwi_mcfeeters.npy. "
    "A --nodata fill value, or a boolean --mask where True means nodata, "
    "makes that pixel NaN."
)


def add_indices_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Attach the ``indices`` command."""
    parser = sub.add_parser(
        "indices",
        help="Compute NDVI, McFeeters NDWI, and NBR",
        description=_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--red", required=True, type=Path, help="Red band .npy")
    parser.add_argument("--nir", required=True, type=Path, help="NIR band .npy")
    parser.add_argument("--green", type=Path, default=None, help="Green band .npy (McFeeters NDWI)")
    parser.add_argument("--swir22", type=Path, default=None, help="SWIR2.2 band .npy (NBR)")
    parser.add_argument("--out", required=True, type=Path, help="Directory for index .npy files")
    parser.add_argument(
        "--nodata",
        action="append",
        default=None,
        help="Fill value, or band=value. Matching stored pixels become NaN",
    )
    parser.add_argument(
        "--mask",
        type=Path,
        default=None,
        help="Boolean .npy mask; True marks nodata for every band",
    )
    parser.add_argument(
        "--scale",
        action="append",
        default=None,
        help="Scale applied to every band, or band=scale",
    )
    parser.add_argument(
        "--offset",
        action="append",
        default=None,
        help="Offset added after scale, or band=offset",
    )
    parser.set_defaults(_atlas_cmd="indices")


def run_indices(args: argparse.Namespace) -> int:
    """Load band arrays, compute indices, and write one .npy per index.

    Returns:
        Process exit code (0 ok, 2 usage).
    """
    try:
        bands: dict[str, NDArray[np.generic]] = {
            "red": _load_array(args.red, "red"),
            "nir": _load_array(args.nir, "nir"),
        }
        if args.green is not None:
            bands["green"] = _load_array(args.green, "green")
        if args.swir22 is not None:
            bands["swir22"] = _load_array(args.swir22, "swir22")
        mask = _load_array(args.mask, "mask") if args.mask is not None else None
        result = compute_indices(
            bands,
            nodata=_parse_band_floats(args.nodata, "nodata"),
            mask=mask,
            scale=_parse_band_floats(args.scale, "scale"),
            offset=_parse_band_floats(args.offset, "offset"),
        )
    except (OSError, ValueError, TypeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if not result:
        print(
            "red and NIR are required before NDVI, McFeeters NDWI, or NBR can be computed",
            file=sys.stderr,
        )
        return 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, array in result.items():
        path = out / f"{name}.npy"
        np.save(path, array)
        print(f"{name} {path}")
    return 0


def _load_array(path: Path, label: str) -> NDArray[np.generic]:
    """Load one numeric .npy array."""
    if not path.is_file():
        raise OSError(f"{label} file does not exist: {path}")
    try:
        array = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"{label} is not a numeric .npy array") from exc
    if not isinstance(array, np.ndarray):
        raise ValueError(f"{label} is not a numeric .npy array")
    return array


def _parse_band_floats(
    values: list[str] | None,
    label: str,
) -> float | dict[str, float] | None:
    """Parse repeated ``--scale`` style options.

    A bare number applies to every band. ``band=number`` is per band.
    """
    if not values:
        return None
    per_band: dict[str, float] = {}
    global_value: float | None = None
    for raw in values:
        if "=" in raw:
            name, _, number = raw.partition("=")
            key = name.strip().lower()
            if not key:
                raise ValueError(f"{label} band name is empty")
            if key in per_band:
                raise ValueError(f"duplicate {label} for {key!r}")
            per_band[key] = float(number)
        else:
            if global_value is not None:
                raise ValueError(f"{label} was given twice")
            global_value = float(raw)
    if per_band and global_value is not None:
        raise ValueError(f"{label} cannot mix one value for every band with per-band values")
    if per_band:
        return per_band
    return global_value
