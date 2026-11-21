"""Per-collection band profiles: alias to asset key, GSD, calibration, citation.

Pure library: no network, no raster I/O. Profiles are fallbacks and rules; the
catalog's own asset facts win where it publishes them. QA bit tables live in
``atlas.geo.chipmeta.QA_PRESETS``; a profile stores only the preset name.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from atlas.data.base import Asset, Scene

BandKind = Literal["reflectance", "backscatter", "thermal", "elevation", "categorical", "qa"]

# Centre wavelengths in micrometres, matching the targets in ``atlas.data.bands``.
_WAVELENGTH: Final[Mapping[str, float]] = {
    "coastal": 0.443,
    "blue": 0.490,
    "green": 0.560,
    "red": 0.665,
    "rededge1": 0.704,
    "rededge2": 0.740,
    "rededge3": 0.783,
    "nir_broad": 0.842,
    "nir": 0.865,
    "swir16": 1.610,
    "swir22": 2.190,
    "lwir": 10.9,
}

_S2_BASELINE_OFFSET: Final = -0.1
_L2_SR_SCALE: Final = 2.75e-5
_L2_SR_OFFSET: Final = -0.2
_L2_SR_RANGE: Final = (7273.0, 43636.0)
_L2_ST_SCALE: Final = 0.00341802
_L2_ST_OFFSET: Final = 149.0
_MAX_LISTED_KEYS: Final = 40
_BASELINE_OFFSET_FROM: Final = 4
_COP_DEM_CITATION: Final = (
    "Copernicus DEM GLO, (c) DLR e.V. 2010-2014 and Airbus Defence and Space GmbH "
    "2014-2018, provided under COPERNICUS by the European Union and ESA"
)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class BandProfile(_Frozen):
    """One alias in a collection: candidate asset keys and band facts."""

    alias: str
    keys: tuple[str, ...]
    source_band: str
    common_name: str | None = None
    wavelength_um: float | None = None
    gsd_m: float | None = None
    kind: BandKind = "reflectance"
    qa_preset: str | None = None


class CollectionProfile(_Frozen):
    """Bands of one STAC collection, keyed by alias."""

    collection: str
    label: str
    bands: tuple[BandProfile, ...]
    extra_aliases: Mapping[str, str] = Field(default_factory=dict)

    def band(self, alias: str) -> BandProfile | None:
        """Return the profile for ``alias`` (extra aliases applied), or None."""
        alias = self.extra_aliases.get(alias, alias)
        return next((b for b in self.bands if b.alias == alias), None)


class CalibrationRule(_Frozen):
    """How raw digital numbers become physical units.

    ``offset_applied`` means the provider already removed the offset, so
    ``offset`` is stored as 0 and must not be applied again. ``valid_range`` is in
    digital numbers, before scaling.
    """

    scale: float
    offset: float = 0.0
    offset_applied: bool = False
    nodata_in: float | None = None
    dtype_in: str | None = None
    valid_range: tuple[float, float] | None = None
    unit: str
    source: str
    warnings: tuple[str, ...] = ()


class ResolvedAsset(_Frozen):
    """An asset picked by alias or literal key."""

    key: str
    asset: Asset
    alias: str | None = None
    band: BandProfile | None = None


class AssetNotFound(ValueError):
    """The requested asset name is neither an alias nor a key on the scene."""


def _b(
    alias: str,
    keys: tuple[str, ...],
    source: str,
    gsd: float | None,
    kind: BandKind = "reflectance",
    preset: str | None = None,
) -> BandProfile:
    return BandProfile(
        alias=alias,
        keys=keys,
        source_band=source,
        common_name=alias if alias in _WAVELENGTH else None,
        wavelength_um=_WAVELENGTH.get(alias),
        gsd_m=gsd,
        kind=kind,
        qa_preset=preset,
    )


def _s2() -> tuple[BandProfile, ...]:
    rows: list[tuple[str, tuple[str, ...], str, float]] = [
        ("coastal", ("B01", "coastal"), "B01", 60.0),
        ("blue", ("B02", "blue"), "B02", 10.0),
        ("green", ("B03", "green"), "B03", 10.0),
        ("red", ("B04", "red"), "B04", 10.0),
        ("rededge1", ("B05", "rededge1"), "B05", 20.0),
        ("rededge2", ("B06", "rededge2"), "B06", 20.0),
        ("rededge3", ("B07", "rededge3"), "B07", 20.0),
        ("nir", ("B8A", "nir08"), "B8A", 20.0),
        ("nir_broad", ("B08", "nir"), "B08", 10.0),
        ("swir16", ("B11", "swir16"), "B11", 20.0),
        ("swir22", ("B12", "swir22"), "B12", 20.0),
    ]
    bands = [_b(a, k, s, g) for a, k, s, g in rows]
    bands.append(_b("scl", ("SCL", "scl"), "SCL", 20.0, "qa", "s2-scl"))
    return tuple(bands)


def _landsat() -> tuple[BandProfile, ...]:
    rows = [
        ("coastal", ("coastal", "SR_B1"), "SR_B1"),
        ("blue", ("blue", "SR_B2"), "SR_B2"),
        ("green", ("green", "SR_B3"), "SR_B3"),
        ("red", ("red", "SR_B4"), "SR_B4"),
        ("nir", ("nir08", "SR_B5"), "SR_B5"),
        ("swir16", ("swir16", "SR_B6"), "SR_B6"),
        ("swir22", ("swir22", "SR_B7"), "SR_B7"),
    ]
    bands = [_b(a, k, s, 30.0) for a, k, s in rows]
    bands.append(_b("lwir", ("lwir11", "ST_B10"), "ST_B10", 100.0, "thermal"))
    bands.append(_b("qa", ("qa_pixel", "QA_PIXEL"), "QA_PIXEL", 30.0, "qa", "landsat-qa-pixel"))
    return tuple(bands)


def _hls(rows: list[tuple[str, str]]) -> tuple[BandProfile, ...]:
    bands = [_b(alias, (band,), band, 30.0) for alias, band in rows]
    bands.append(_b("qa", ("Fmask",), "Fmask", 30.0, "qa", "hls-fmask"))
    return tuple(bands)


_HLS_S30 = [
    ("coastal", "B01"),
    ("blue", "B02"),
    ("green", "B03"),
    ("red", "B04"),
    ("rededge1", "B05"),
    ("rededge2", "B06"),
    ("rededge3", "B07"),
    ("nir_broad", "B08"),
    ("nir", "B8A"),
    ("swir16", "B11"),
    ("swir22", "B12"),
]
_HLS_L30 = [
    ("coastal", "B01"),
    ("blue", "B02"),
    ("green", "B03"),
    ("red", "B04"),
    ("nir", "B05"),
    ("swir16", "B06"),
    ("swir22", "B07"),
]

PROFILES: Final[Mapping[str, CollectionProfile]] = {
    "sentinel-2-l2a": CollectionProfile(
        collection="sentinel-2-l2a",
        label="Sentinel-2 L2A",
        bands=_s2(),
        extra_aliases={"qa": "scl"},
    ),
    "landsat-c2-l2": CollectionProfile(
        collection="landsat-c2-l2", label="Landsat Collection 2 Level-2", bands=_landsat()
    ),
    "hls2-s30": CollectionProfile(
        collection="hls2-s30", label="HLS Sentinel-2 (S30)", bands=_hls(_HLS_S30)
    ),
    "hls2-l30": CollectionProfile(
        collection="hls2-l30", label="HLS Landsat (L30)", bands=_hls(_HLS_L30)
    ),
    "sentinel-1-rtc": CollectionProfile(
        collection="sentinel-1-rtc",
        label="Sentinel-1 RTC",
        bands=(
            _b("vv", ("vv",), "VV", 10.0, "backscatter"),
            _b("vh", ("vh",), "VH", 10.0, "backscatter"),
        ),
    ),
    "cop-dem-glo-30": CollectionProfile(
        collection="cop-dem-glo-30",
        label="Copernicus DEM 30 m",
        bands=(_b("elevation", ("data",), "data", 30.0, "elevation"),),
    ),
    "cop-dem-glo-90": CollectionProfile(
        collection="cop-dem-glo-90",
        label="Copernicus DEM 90 m",
        bands=(_b("elevation", ("data",), "data", 90.0, "elevation"),),
    ),
    "esa-worldcover": CollectionProfile(
        collection="esa-worldcover",
        label="ESA WorldCover",
        bands=(_b("landcover", ("map",), "map", 10.0, "categorical"),),
        extra_aliases={"map": "landcover"},
    ),
}


def profile_for(scene: Scene) -> CollectionProfile | None:
    """Return the profile for the scene's collection, or None."""
    return PROFILES.get(scene.collection)


def _pick_key(scene: Scene, band: BandProfile) -> str | None:
    return next((k for k in band.keys if k in scene.assets), None)


def _literal_key(scene: Scene, name: str) -> str | None:
    if name in scene.assets:
        return name
    folded = name.casefold()
    return next((k for k in sorted(scene.assets) if k.casefold() == folded), None)


def resolve_asset(scene: Scene, name: str) -> ResolvedAsset:
    """Resolve an alias or literal asset key, case-insensitively.

    An alias beats a literal key of the same name. This performs no browse or
    thumbnail check; the caller applies ``download.pick_asset``'s guard.

    Raises:
        AssetNotFound: Nothing matches. The message lists aliases and real keys.
    """
    wanted = name.strip().casefold()
    profile = profile_for(scene)
    if profile is not None:
        band = profile.band(wanted)
        if band is not None:
            key = _pick_key(scene, band)
            if key is not None:
                return ResolvedAsset(key=key, asset=scene.assets[key], alias=band.alias, band=band)
    aliased = scene.band_aliases.get(wanted)
    if profile is None and aliased is not None and aliased in scene.assets:
        return ResolvedAsset(key=aliased, asset=scene.assets[aliased], alias=wanted)
    literal = _literal_key(scene, name.strip())
    if literal is not None:
        bands = profile.bands if profile is not None else ()
        found = next((b for b in bands if literal in b.keys), None)
        alias = found.alias if found is not None else None
        return ResolvedAsset(key=literal, asset=scene.assets[literal], alias=alias, band=found)
    raise AssetNotFound(_not_found_message(scene, name))


def _not_found_message(scene: Scene, name: str) -> str:
    rows = [(a, k, q) for a, k, _g, q in list_aliases(scene)]
    if not rows:
        rows = [(a, k, False) for a, k in sorted(scene.band_aliases.items())]
    alias_text = ", ".join(f"{a}={k}" for a, k, _q in rows) or "none"
    keys = sorted(scene.assets)
    shown = ", ".join(keys[:_MAX_LISTED_KEYS])
    if len(keys) > _MAX_LISTED_KEYS:
        shown += f", +{len(keys) - _MAX_LISTED_KEYS} more"
    example = [a for a, _k, q in rows if not q][:2] or keys[:2]
    fix = ", ".join(f'"{e}"' for e in example)
    return (
        f"asset {name!r} not found in scene {scene.id}. Aliases: {alias_text}. "
        f"Real keys: {shown or 'none'}. Fix: assets=[{fix}]"
    )


def list_aliases(scene: Scene) -> list[tuple[str, str, float | None, bool]]:
    """List ``(alias, key, gsd_m, is_qa)`` for bands present on the scene.

    The GSD is the catalog's ``spatial_resolution`` when published, else the
    profile value. Unprofiled collections give an empty list.
    """
    profile = profile_for(scene)
    if profile is None:
        return []
    rows: list[tuple[str, str, float | None, bool]] = []
    for band in profile.bands:
        key = _pick_key(scene, band)
        if key is None:
            continue
        catalog = scene.assets[key].spatial_resolution
        gsd = catalog if catalog is not None else band.gsd_m
        rows.append((band.alias, key, gsd, band.kind == "qa"))
    return rows


def _baseline(scene: Scene) -> tuple[int, ...] | None:
    raw = scene.properties.get("s2:processing_baseline")
    if isinstance(raw, bool) or not isinstance(raw, int | str):
        return None
    text = str(raw).strip()
    pieces = text.split(".")
    if not text or any(not p.isdigit() for p in pieces):
        return None
    return tuple(int(p) for p in pieces)


def _catalog_offset_warning(asset: Asset, expected: float) -> tuple[str, ...]:
    if asset.offset is None or math.isclose(asset.offset, expected, abs_tol=1e-9):
        return ()
    return (
        f"catalog raster:bands offset {asset.offset:g} differs from profile rule "
        f"{expected:g}; profile used",
    )


def _s2_reflectance(scene: Scene, asset: Asset) -> CalibrationRule:
    baseline = _baseline(scene)
    flagged = scene.properties.get("earthsearch:boa_offset_applied") is True
    offset = 0.0
    applied = False
    warnings: tuple[str, ...] = ()
    if baseline is None:
        warnings = ("processing baseline unknown; BOA offset not applied",)
    elif baseline[0] >= _BASELINE_OFFSET_FROM:
        applied = flagged
        offset = 0.0 if flagged else _S2_BASELINE_OFFSET
    if not applied:
        warnings += _catalog_offset_warning(asset, offset)
    return CalibrationRule(
        scale=1e-4,
        offset=offset,
        offset_applied=applied,
        nodata_in=0.0,
        dtype_in=asset.data_type,
        unit="reflectance",
        source="Sentinel-2 L2A: reflectance = DN * 1e-4 + offset (offset -0.1 from baseline 04.00)",
        warnings=warnings,
    )


def _rule(
    asset: Asset,
    scale: float,
    unit: str,
    source: str,
    *,
    offset: float = 0.0,
    nodata: float | None = None,
    valid_range: tuple[float, float] | None = None,
) -> CalibrationRule:
    return CalibrationRule(
        scale=scale,
        offset=offset,
        nodata_in=nodata,
        dtype_in=asset.data_type,
        valid_range=valid_range,
        unit=unit,
        source=source,
    )


def calibration_for(scene: Scene, key: str) -> CalibrationRule | None:
    """Return the calibration rule for asset ``key``, or None when not profiled."""
    profile = profile_for(scene)
    asset = scene.assets.get(key)
    if profile is None or asset is None:
        return None
    band = next((b for b in profile.bands if key in b.keys), None)
    if band is None:
        return None
    collection = profile.collection
    if collection == "sentinel-2-l2a":
        if band.kind == "qa":
            return _rule(asset, 1.0, "class", "Sentinel-2 SCL class codes", nodata=0.0)
        return _s2_reflectance(scene, asset)
    if collection == "landsat-c2-l2":
        return _landsat_rule(band, asset)
    if collection.startswith("hls2-"):
        if band.kind == "qa":
            return _rule(asset, 1.0, "bitmask", "HLS Fmask bit flags", nodata=255.0)
        return _rule(asset, 1e-4, "reflectance", "HLS surface reflectance", nodata=-9999.0)
    return _other_rule(collection, asset)


def _landsat_rule(band: BandProfile, asset: Asset) -> CalibrationRule:
    if band.kind == "qa":
        return _rule(asset, 1.0, "bitmask", "Landsat QA_PIXEL bit flags", nodata=1.0)
    if band.kind == "thermal":
        return _rule(
            asset, _L2_ST_SCALE, "K", "Landsat C2 L2 ST_B10", offset=_L2_ST_OFFSET, nodata=0.0
        )
    return _rule(
        asset,
        _L2_SR_SCALE,
        "reflectance",
        "Landsat C2 L2 surface reflectance",
        offset=_L2_SR_OFFSET,
        nodata=0.0,
        valid_range=_L2_SR_RANGE,
    )


def _other_rule(collection: str, asset: Asset) -> CalibrationRule | None:
    if collection == "sentinel-1-rtc":
        return _rule(asset, 1.0, "linear", "Sentinel-1 RTC gamma0 power", nodata=-32768.0)
    if collection.startswith("cop-dem-glo-"):
        return _rule(asset, 1.0, "m", "Copernicus DEM elevation")
    if collection == "esa-worldcover":
        return _rule(asset, 1.0, "class", "ESA WorldCover class codes", nodata=0.0)
    return None


def citation_for(scene: Scene) -> str | None:
    """Return the data citation line for the scene's collection, or None."""
    collection = scene.collection
    if collection in {"sentinel-2-l2a", "sentinel-1-rtc"}:
        return f"Contains modified Copernicus Sentinel data {scene.datetime.year}"
    if collection == "landsat-c2-l2":
        return "Landsat Collection 2 Level-2 data courtesy of the U.S. Geological Survey"
    if collection.startswith("hls2-"):
        return "Harmonized Landsat Sentinel-2 (HLS) v2.0, NASA LP DAAC"
    if collection.startswith("cop-dem-glo-"):
        return _COP_DEM_CITATION
    if collection == "esa-worldcover":
        return "ESA WorldCover (c) ESA, CC BY 4.0"
    return None


def epsg_hint(scene: Scene) -> int | None:
    """Return the EPSG code from ``proj:epsg`` or ``proj:code``, or None."""
    props = scene.properties
    raw = props.get("proj:epsg")
    if isinstance(raw, int) and not isinstance(raw, bool):
        return raw
    code = props.get("proj:code")
    if isinstance(code, str) and code.upper().startswith("EPSG:") and code[5:].isdigit():
        return int(code[5:])
    return None
