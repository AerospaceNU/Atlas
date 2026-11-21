"""Input enforcement for model tools: a stack either fits a manifest or is refused.

Invariant: no coercion. Nothing is resampled, reprojected, rescaled or renamed here; every
mismatch is collected into ONE `InputMismatch` that ends with the `prepare_stack` fix.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from atlas.geo.errors import InputMismatch, fix

if TYPE_CHECKING:
    from atlas.geo.stack import LayerMeta, OrbitMeta, Stack
    from atlas.models.manifest import ModelManifest

__all__ = ["Roles", "check_input", "nodata_fraction", "normalise_sensor"]

_DATE_AT = re.compile(r"@(\d{8})$")
_UNIT_KIND = {"reflectance": "reflectance", "dB": "backscatter"}


@dataclass
class Roles:
    """Layers chosen for each manifest band.

    Attributes:
        layers_by_role: Band common name to layer names in ascending date order.
        dates: Distinct dates (YYYYMMDD) of the chosen layers, ascending.
        used: Every chosen layer name, in manifest band order.
        source_bands: Band common name to the real source band recorded on the layer.
        sensors: Distinct sensors of the chosen layers.
        nodata_fraction: Share of pixels not finite in at least one chosen layer.
    """

    layers_by_role: dict[str, tuple[str, ...]]
    dates: list[str] = field(default_factory=list)
    used: list[str] = field(default_factory=list)
    source_bands: dict[str, str] = field(default_factory=dict)
    sensors: list[str] = field(default_factory=list)
    nodata_fraction: float = 0.0


def normalise_sensor(name: str | None) -> str:
    """Return a sensor name reduced to lowercase alphanumerics ("sentinel-2" -> "sentinel2")."""
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def _base(name: str) -> str:
    return _DATE_AT.sub("", name)


def _date_of(meta: LayerMeta) -> str | None:
    match = _DATE_AT.search(meta.name)
    if match:
        return match.group(1)
    digits = re.sub(r"\D", "", (meta.datetime or "")[:10])
    return digits if len(digits) == 8 else None


def _fmt_range(low: float, high: float) -> str:
    def one(x: float) -> str:
        return str(int(x)) if float(x).is_integer() else f"{x:g}"

    return f"{one(low)}-{one(high)}"


def _fix_command(manifest: ModelManifest, stack: Stack) -> str:
    bands = ",".join(manifest.input.bands)
    low, high = manifest.input.resolution_m
    extra = ""
    if manifest.input.mode != "single":
        extra = " scenes=[<pre-scene>,<post-scene>]"
    xres = stack.grid.xres
    if xres < low or xres > high:
        extra += f" resolution={min(max(xres, low), high):g}"
    return f"prepare_stack bands=[{bands}]{extra}"


def _candidates(stack: Stack, band: str) -> list[LayerMeta]:
    return [m for m in stack.meta.layers if _base(m.name) == band and m.kind != "categorical"]


def _band_sources_problem(manifest: ModelManifest, band: str, metas: list[LayerMeta]) -> list[str]:
    problems: list[str] = []
    wanted = {normalise_sensor(s): m for s, m in manifest.input.band_sources.items()}
    for meta in metas:
        mapping = wanted.get(normalise_sensor(meta.sensor))
        expected = mapping.get(band) if mapping else None
        if expected and meta.band and meta.band.upper() != expected.upper():
            problems.append(
                f"layer {meta.name} is {meta.band}, the model was built for {band}={expected}"
            )
    return problems


def _pick(stack: Stack, manifest: ModelManifest, problems: list[str]) -> Roles | None:
    present = {_base(m.name) for m in stack.meta.layers}
    missing = [b for b in manifest.input.bands if not _candidates(stack, b)]
    if missing:
        have = ",".join(sorted(present)) or "none"
        problems.append(f"missing bands {','.join(missing)} (stack has {have})")
        return None
    chosen: dict[str, list[LayerMeta]] = {}
    for band in manifest.input.bands:
        chosen[band] = sorted(_candidates(stack, band), key=lambda m: _date_of(m) or "")
    return Roles(
        layers_by_role={b: tuple(m.name for m in ms) for b, ms in chosen.items()},
        used=[m.name for b in manifest.input.bands for m in chosen[b]],
        source_bands={b: ms[-1].band or "?" for b, ms in chosen.items()},
        sensors=sorted({normalise_sensor(m.sensor) for ms in chosen.values() for m in ms} - {""}),
    )


def _check_dates(
    stack: Stack, manifest: ModelManifest, roles: Roles, problems: list[str]
) -> list[str]:
    spec = manifest.input
    per_band = {
        b: [_date_of(stack.layer(n)) for n in names] for b, names in roles.layers_by_role.items()
    }
    all_dates = sorted({d for ds in per_band.values() for d in ds if d})
    if any(None in ds for ds in per_band.values()) and spec.temporal.max > 1:
        problems.append("some layers carry no date, so pre and post cannot be told apart")
    if len(all_dates) < spec.temporal.min:
        have = ",".join(all_dates) or "none"
        missing = "the earlier (pre) date" if all_dates else "every date"
        problems.append(
            f"needs {spec.temporal.min} dates (pre earlier, post later); stack has "
            f"{len(all_dates)} ({have}), missing {missing}"
        )
    elif len(all_dates) > spec.temporal.max:
        problems.append(
            f"accepts at most {spec.temporal.max} date(s); stack has {len(all_dates)} "
            f"({','.join(all_dates)})"
        )
    elif spec.mode == "pair":
        lacking = [b for b, ds in per_band.items() if set(ds) != set(all_dates)]
        if lacking:
            problems.append(
                f"bands {','.join(lacking)} are not present on both dates {all_dates[0]} and "
                f"{all_dates[-1]}"
            )
    return all_dates


def _orbit_text(orbit: OrbitMeta | None) -> str:
    return "unknown" if orbit is None else f"{orbit.relative}:{orbit.state}"


def _check_pair_orbit(stack: Stack, roles: Roles, problems: list[str]) -> None:
    metas = [stack.layer(n) for names in roles.layers_by_role.values() for n in names]
    ordered = sorted(metas, key=lambda m: _date_of(m) or "")
    pre, post = ordered[0].orbit, ordered[-1].orbit
    if _orbit_text(pre) != _orbit_text(post):
        problems.append(
            "pair needs the same relative orbit and direction; "
            f"pre is {_orbit_text(pre)}, post is {_orbit_text(post)}"
        )


def _check_layers(stack: Stack, manifest: ModelManifest, roles: Roles, problems: list[str]) -> None:
    spec = manifest.input
    metas = {b: [stack.layer(n) for n in names] for b, names in roles.layers_by_role.items()}
    unit_kind = _UNIT_KIND[spec.unit]
    wrong = sorted({m.name for ms in metas.values() for m in ms if m.kind != unit_kind})
    if wrong:
        problems.append(f"layers {','.join(wrong)} are not {spec.unit}")
    sensors = {normalise_sensor(s) for s in spec.sensors}
    foreign = sorted(
        {
            m.sensor
            for ms in metas.values()
            for m in ms
            if m.sensor and normalise_sensor(m.sensor) not in sensors
        }
    )
    if foreign:
        problems.append(f"sensor {','.join(foreign)} is not one of {','.join(spec.sensors)}")
    for band, ms in metas.items():
        problems.extend(_band_sources_problem(manifest, band, ms))


def _check_grid(stack: Stack, manifest: ModelManifest, problems: list[str]) -> None:
    spec = manifest.input
    low, high = spec.resolution_m
    xres, yres = stack.grid.xres, stack.grid.yres
    if not (low <= xres <= high and low <= yres <= high):
        problems.append(f"pixel size is {xres:g} m, the model needs {_fmt_range(low, high)} m")
    height, width = stack.grid.shape
    if min(height, width) < spec.min_size:
        problems.append(f"grid is {width}x{height} px, the model needs at least {spec.min_size}")


def nodata_fraction(stack: Stack, names: list[str]) -> float:
    """Return the share of pixels that are not finite in at least one of `names`."""
    bad: NDArray[np.bool_] | None = None
    for name in names:
        missing = ~np.isfinite(stack.band(name))
        bad = missing if bad is None else (bad | missing)
    return 0.0 if bad is None else float(bad.mean())


def check_input(manifest: ModelManifest, stack: Stack) -> Roles:
    """Return the layers a manifest will read, or raise one combined `InputMismatch`.

    Checks band presence by common name, date roles and temporal depth, unit, sensor,
    `layer.band` against `band_sources`, pixel size, grid size, no-data fraction and, for
    pairs, same relative orbit and direction. Nothing is changed to make the stack fit.

    Args:
        manifest: The model manifest.
        stack: The input stack.

    Returns:
        The chosen layers per band role in ascending date order.

    Raises:
        InputMismatch: One or more checks failed; the message ends with the fix.
    """
    problems: list[str] = []
    roles = _pick(stack, manifest, problems)
    if roles is not None:
        _check_layers(stack, manifest, roles, problems)
        roles.dates = _check_dates(stack, manifest, roles, problems)
        if manifest.input.pair.require_same_orbit and len(roles.dates) >= 2:
            _check_pair_orbit(stack, roles, problems)
    _check_grid(stack, manifest, problems)
    if roles is not None and not problems:
        roles.nodata_fraction = nodata_fraction(stack, roles.used)
        limit = manifest.input.max_nodata_fraction
        if roles.nodata_fraction > limit:
            problems.append(
                f"{roles.nodata_fraction:.1%} of pixels have no data in the input bands; "
                f"the limit is {limit:.1%}"
            )
    if problems or roles is None:
        spec = manifest.input
        res = _fmt_range(*spec.resolution_m)
        need = (
            f"{manifest.name} needs {','.join(spec.bands)} at {res} m {spec.unit} ({spec.mode}); "
            f"stack '{stack.id}' does not fit: "
        )
        raise InputMismatch(fix(need + "; ".join(problems) + ".", _fix_command(manifest, stack)))
    return roles
