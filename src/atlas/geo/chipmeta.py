"""Sidecar metadata for pulled chips (``atlas.chip/1``) and the QA preset table."""

from __future__ import annotations

import json
import os
import secrets
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal, TypedDict

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from atlas.geo.errors import GeoError

__all__ = [
    "CHIP_SCHEMA",
    "QA_PRESETS",
    "BandInfo",
    "Calibration",
    "ChipMeta",
    "FileInfo",
    "QaRule",
    "RasterInfo",
    "apply_rule",
    "read_chip_meta",
    "sidecar_path",
    "validate_presets",
    "write_chip_meta",
]

CHIP_SCHEMA = "atlas.chip/1"


class _Model(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class BandInfo(_Model):
    common_name: str | None = None
    wavelength_um: float | None = None
    gsd_m: float | None = None
    source_band: str | None = None


class FileInfo(_Model):
    bytes: int = 0
    windowed: bool = False
    granule_bytes: int | None = None
    bytes_fetched: int = 0


class RasterInfo(_Model):
    """Raster geometry; ``transform`` origin is the OUTER pixel corner."""

    dtype: str | None = None
    height: int | None = None
    width: int | None = None
    epsg: int | None = None
    transform: list[float] | None = None
    nodata: float | None = None
    registration: Literal["area", "point"] = "area"


class Calibration(_Model):
    scale: float = 1.0
    offset: float = 0.0
    offset_applied: bool = False
    nodata_in: float | None = None
    dtype_in: str | None = None
    valid_range: list[float] | None = None
    source: str = ""


class ChipMeta(_Model):
    """Sidecar describing one raw chip; every field is defaulted so old files load."""

    schema_: str = Field(default=CHIP_SCHEMA, alias="schema")
    scene_id: str | None = None
    source: str | None = None
    collection: str | None = None
    platform: str | None = None
    datetime: str | None = None
    asset_key: str | None = None
    alias: str | None = None
    band: BandInfo = Field(default_factory=BandInfo)
    file: FileInfo = Field(default_factory=FileInfo)
    raster: RasterInfo = Field(default_factory=RasterInfo)
    calibration: Calibration = Field(default_factory=Calibration)
    qa_preset: str | None = None
    aoi: list[float] | dict[str, Any] | None = None
    scene_properties: dict[str, Any] = Field(default_factory=dict)


def sidecar_path(chip: Path) -> Path:
    """Return the sidecar path next to a chip."""
    return chip.with_name(chip.name + ".meta.json")


def write_chip_meta(chip: Path, meta: ChipMeta) -> Path:
    """Write the sidecar atomically (unique temp name, then ``os.replace``).

    Args:
        chip: The chip file the sidecar describes.
        meta: Metadata to store.

    Returns:
        The sidecar path.
    """
    target = sidecar_path(chip)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".tmp-{os.getpid()}-{secrets.token_hex(4)}-{target.name}")
    payload = json.dumps(meta.model_dump(by_alias=True, mode="json"), indent=2, sort_keys=True)
    try:
        tmp.write_text(payload + "\n", encoding="utf-8")
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)
    return target


def _unreadable(path: Path, why: str) -> GeoError:
    return GeoError(
        f"unreadable chip sidecar {path}: {why}. Fix: re-pull with get_satellite_data, "
        "or pass files=[{path,...}] to prepare_stack"
    )


def _major(schema: str) -> int | None:
    head, _, tail = schema.partition("/")
    if head != "atlas.chip":
        return None
    digits = tail.split(".")[0]
    return int(digits) if digits.isdigit() else None


def read_chip_meta(chip: Path) -> ChipMeta | None:
    """Read a chip's sidecar.

    Args:
        chip: The chip file.

    Returns:
        The metadata, or None when no sidecar exists.

    Raises:
        GeoError: The sidecar is corrupt or has an unsupported schema major.
    """
    path = sidecar_path(chip)
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError) as exc:
        raise _unreadable(path, str(exc)) from exc
    if not isinstance(raw, dict):
        raise _unreadable(path, "not a JSON object")
    schema = raw.get("schema", CHIP_SCHEMA)
    if not isinstance(schema, str) or _major(schema) != 1:
        raise _unreadable(path, f"unsupported schema {schema!r}")
    try:
        return ChipMeta.model_validate(raw)
    except ValidationError as exc:
        raise _unreadable(path, exc.errors()[0].get("msg", "invalid")) from exc


class QaRule(TypedDict):
    """One semantic class of a QA band: pixel codes, or bit INDICES (not masks)."""

    kind: Literal["codes", "bits"]
    values: tuple[int, ...]


QA_PRESETS: dict[str, dict[str, QaRule]] = {
    "s2-scl": {
        "nodata": {"kind": "codes", "values": (0,)},
        "saturated": {"kind": "codes", "values": (1,)},
        "topographic_shadow": {"kind": "codes", "values": (2,)},
        "cloud_shadow": {"kind": "codes", "values": (3,)},
        "vegetation": {"kind": "codes", "values": (4,)},
        "bare": {"kind": "codes", "values": (5,)},
        "water": {"kind": "codes", "values": (6,)},
        "unclassified": {"kind": "codes", "values": (7,)},
        "cloud": {"kind": "codes", "values": (8, 9)},
        "cirrus": {"kind": "codes", "values": (10,)},
        "snow": {"kind": "codes", "values": (11,)},
    },
    "landsat-qa-pixel": {
        "nodata": {"kind": "bits", "values": (0,)},
        "cloud": {"kind": "bits", "values": (1, 3)},
        "cirrus": {"kind": "bits", "values": (2,)},
        "cloud_shadow": {"kind": "bits", "values": (4,)},
        "snow": {"kind": "bits", "values": (5,)},
        "water": {"kind": "bits", "values": (7,)},
    },
    "hls-fmask": {
        "nodata": {"kind": "codes", "values": (255,)},
        "cirrus": {"kind": "bits", "values": (0,)},
        "cloud": {"kind": "bits", "values": (1,)},
        "adjacent": {"kind": "bits", "values": (2,)},
        "cloud_shadow": {"kind": "bits", "values": (3,)},
        "snow": {"kind": "bits", "values": (4,)},
        "water": {"kind": "bits", "values": (5,)},
    },
}


def validate_presets(vocab: Iterable[str]) -> None:
    """Raise when a preset uses a semantic key outside the stack vocabulary.

    Args:
        vocab: The stack semantic vocabulary (``atlas.geo.stack.SEMANTICS``).

    Raises:
        GeoError: A preset key is not in ``vocab``.
    """
    known = set(vocab)
    for preset, rules in QA_PRESETS.items():
        for key in rules:
            if key not in known:
                raise GeoError(f"QA preset {preset!r} uses unknown semantic {key!r}")


def apply_rule(rule: Mapping[str, Any], arr: NDArray[Any]) -> NDArray[np.bool_]:
    """Evaluate a QA rule to a boolean mask.

    Non-finite pixels are False. Callers evaluate ``nodata`` first and exclude its pixels
    from bit rules (HLS fill 255 sets every bit).

    Args:
        rule: A ``QaRule``.
        arr: QA plane, integer or float32 stack data.

    Returns:
        Boolean array shaped like ``arr``.
    """
    data = np.asarray(arr)
    finite = np.isfinite(data) if data.dtype.kind in "fc" else np.ones(data.shape, dtype=bool)
    ints = np.where(finite, data, 0).astype(np.int64)
    values = tuple(rule["values"])
    if rule["kind"] == "codes":
        hit = np.isin(ints, values)
    else:
        hit = np.zeros(data.shape, dtype=bool)
        for bit in values:
            hit |= ((ints >> int(bit)) & 1).astype(bool)
    return np.asarray(hit & finite, dtype=np.bool_)
