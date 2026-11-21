"""The Stack: an immutable, analysis-ready raster cube on one grid.

Layout of ``analysis/stacks/<id>/``: ``data.npy`` float32 (L, H, W), optional ``qa.npy``
uint8 (Q, H, W), optional ``preview.png`` and ``stack.json``, written LAST as the commit
marker. A stack is never edited after commit; derived products are new stacks.
POSIX only: commit relies on atomic directory rename.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.lib.format import open_memmap
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from atlas.geo import paths
from atlas.geo._lazy import check_cancel
from atlas.geo.errors import StackError, fix
from atlas.geo.grid import Grid, crs_to_lonlat, k2, lonlat_bounds_of

if TYPE_CHECKING:
    from atlas.agent.artifacts import LocalArtifactStore

__all__ = [
    "CLEAR_EXCLUDE",
    "ID_RE",
    "LAYER_KINDS",
    "MAX_CELLS",
    "SCHEMA",
    "SEMANTICS",
    "AoiMeta",
    "CalibrationMeta",
    "GridMeta",
    "LayerMeta",
    "OrbitMeta",
    "QaMeta",
    "SourceMeta",
    "Stack",
    "StackMeta",
    "Tile",
    "exists",
    "hash_params",
    "open_stack",
    "validate_name",
]

SCHEMA = "atlas.stack/1"
MAX_CELLS = 200_000_000
ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")
SEMANTICS = (
    "nodata",
    "saturated",
    "topographic_shadow",
    "cloud_shadow",
    "vegetation",
    "bare",
    "water",
    "unclassified",
    "cloud",
    "cirrus",
    "snow",
    "adjacent",
    "outside_aoi",
)
_AOI_EXCLUDE = frozenset({"outside_aoi"})
CLEAR_EXCLUDE = frozenset({"nodata", "saturated", "cloud", "cloud_shadow", "cirrus", "outside_aoi"})
LAYER_KINDS = (
    "reflectance",
    "backscatter",
    "index",
    "elevation",
    "continuous",
    "categorical",
    "probability",
    "count",
)
_SCRATCH_RE = re.compile(r"\.(tmp|old)-")
_DATE_RE = re.compile(r"@(\d{8})$")
_CHUNK = 1 << 20

Win = tuple[int, int, int, int]
PlaneData = NDArray[Any] | Iterable[NDArray[Any]]


def hash_params(obj: object) -> str:
    """Return the sha256 hex of a canonical JSON encoding of ``obj``."""
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_name(name: str) -> str:
    """Return ``name`` when it is a legal stack id, else raise ``StackError``."""
    if not ID_RE.match(name) or name == "latest" or _SCRATCH_RE.search(name):
        raise StackError(
            fix(
                f"invalid stack name {name!r}: use [a-z0-9][a-z0-9_.-]{{0,63}}, not 'latest' "
                "and without '.tmp-' or '.old-'.",
                "pass name=<lowercase-id>",
            )
        )
    return name


# --------------------------------------------------------------------------- metadata


class _Meta(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class GridMeta(_Meta):
    crs: str
    transform: list[float]
    width: int
    height: int
    bounds_wgs84: list[float] | None = None
    pixel_area_m2: float = 1.0
    area_scale: dict[str, Any] = Field(default_factory=dict)


class AoiMeta(_Meta):
    bbox: list[float] | None = None
    polygon: Any | None = None
    area_ha: float | None = None


class CalibrationMeta(_Meta):
    scale: float | None = None
    offset: float | None = None
    offset_applied: bool | None = None
    nodata_in: float | None = None
    dtype_in: str | None = None
    valid_range: list[float] | None = None
    source: str | None = None


class SourceMeta(_Meta):
    scene_id: str | None = None
    source: str | None = None
    collection: str | None = None
    asset_key: str | None = None
    file: str | None = None
    sha256: str | None = None


class OrbitMeta(_Meta):
    relative: int | None = None
    state: str | None = None


class LayerMeta(_Meta):
    index: int = 0
    name: str
    kind: str
    band: str | None = None
    unit: str = ""
    datetime: str | None = None
    sensor: str | None = None
    wavelength_um: float | None = None
    native_resolution_m: float | None = None
    resampling: str | None = None
    calibration: CalibrationMeta | None = None
    classes: dict[int, str] | None = None
    processing: list[str] = Field(default_factory=list)
    source: SourceMeta | None = None
    orbit: OrbitMeta | None = None

    @field_validator("kind")
    @classmethod
    def _kind(cls, value: str) -> str:
        if value not in LAYER_KINDS:
            raise ValueError(f"kind {value!r} not in {LAYER_KINDS}")
        return value


class QaMeta(_Meta):
    index: int = 0
    name: str
    semantic: str
    preset: str | None = None
    source_layer: str | None = None
    fraction: float | None = None

    @field_validator("semantic")
    @classmethod
    def _semantic(cls, value: str) -> str:
        if value not in SEMANTICS:
            raise ValueError(f"semantic {value!r} not in {SEMANTICS}")
        return value


class StackMeta(_Meta):
    schema_: str = Field(default=SCHEMA, alias="schema")
    id: str
    params_hash: str
    created: str = ""
    warnings: list[str] = Field(default_factory=list)
    grid: GridMeta
    aoi: AoiMeta | None = None
    layers: list[LayerMeta]
    qa: list[QaMeta] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)
    parents: list[str] = Field(default_factory=list)
    data_sha256: str = ""
    run_id: str | None = None

    @model_validator(mode="after")
    def _unique(self) -> StackMeta:
        for label, names in (
            ("layer", [x.name for x in self.layers]),
            ("qa", [x.name for x in self.qa]),
        ):
            dup = sorted({n for n in names if names.count(n) > 1})
            if dup:
                raise ValueError(f"duplicate {label} names: {dup}")
        return self


@dataclass(frozen=True)
class Tile:
    """A tile of a stack: ``window`` includes the margin, ``core`` partitions the grid.

    ``inner`` is the core expressed relative to ``window``.
    """

    window: Win
    core: Win
    inner: Win


# --------------------------------------------------------------------------- helpers


def _norm_date(value: str) -> str:
    digits = value.replace("-", "")
    if not re.fullmatch(r"\d{8}", digits):
        raise StackError(fix(f"bad date {value!r}.", "use YYYYMMDD or YYYY-MM-DD"))
    return digits


def _layer_date(name: str, datetime_: str | None) -> str | None:
    match = _DATE_RE.search(name)
    if match:
        return match.group(1)
    if datetime_:
        digits = re.sub(r"\D", "", datetime_[:10])
        if len(digits) == 8:
            return digits
    return None


def _base(name: str) -> str:
    return _DATE_RE.sub("", name)


def _plane_iter(data: PlaneData, count: int, what: str, shape: tuple[int, int]) -> Iterator[Any]:
    if isinstance(data, np.ndarray):
        if data.ndim == 2:
            data = data[None]
        if data.ndim != 3 or data.shape[0] != count or data.shape[1:] != shape:
            raise StackError(
                fix(
                    f"{what} array shape {data.shape} does not match {count} x {shape}.",
                    "pass one (H, W) array per layer",
                )
            )
    seen = 0
    for plane in data:
        seen += 1
        if seen > count:
            break
        arr = np.asarray(plane)
        if arr.shape != shape:
            raise StackError(
                fix(
                    f"{what} plane {seen - 1} has shape {arr.shape}, grid is {shape}.",
                    "regrid first",
                )
            )
        yield arr
    if seen != count:
        raise StackError(
            fix(f"{what} has {seen} planes but {count} were declared.", "match counts")
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _scratch(root: Path, stack_id: str, tag: str) -> Path:
    return root / f"{stack_id}.{tag}-{os.getpid()}-{uuid.uuid4().hex[:8]}"


def _stored_hash(directory: Path) -> str | None:
    marker = directory / "stack.json"
    if not marker.is_file():
        return None
    try:
        value = json.loads(marker.read_text(encoding="utf-8")).get("params_hash")
    except (OSError, ValueError):
        return None
    return value if isinstance(value, str) else None


def exists(analysis_dir: Path, stack_id: str) -> str | None:
    """Return the stored params_hash of a committed stack, else None."""
    return _stored_hash(paths.stacks_dir(analysis_dir) / stack_id)


def _grid_scale(grid: Grid) -> tuple[float, dict[str, Any], list[str]]:
    lon, lat = crs_to_lonlat(
        grid.crs,
        [grid.transform[2] + grid.width * grid.xres / 2],
        [grid.transform[5] - grid.height * grid.yres / 2],
    )
    scale = k2(grid.crs, float(lon[0]), float(lat[0]))
    if scale is None:
        warning = f"pixel area unknown for {grid.crs}: hectares are in map units"
        return grid.xres * grid.yres, {"k2": None, "method": None}, [warning]
    return grid.xres * grid.yres / scale[0], {"k2": scale[0], "method": scale[1]}, []


def _write_preview(preview: bytes | Path | None, target: Path) -> None:
    if preview is None:
        return
    if isinstance(preview, bytes):
        target.write_bytes(preview)
    else:
        shutil.copyfile(preview, target)


def _commit(
    tmp: Path, final: Path, root: Path, stack_id: str, params_hash: str, overwrite: bool
) -> bool:
    """Move ``tmp`` to ``final``. Returns True when an identical stack won the race instead."""
    aside: Path | None = None
    if overwrite and final.exists():
        aside = _scratch(root, stack_id, "old")
        os.rename(final, aside)
    try:
        os.replace(tmp, final)
    except OSError as exc:
        if aside is not None:
            os.rename(aside, final)
            raise StackError(fix(f"could not commit stack '{stack_id}': {exc}.", "retry")) from exc
        if _stored_hash(final) == params_hash:
            return True
        raise StackError(
            fix(
                f"stack '{stack_id}' exists with different parameters.",
                "choose another name or pass overwrite=true",
            )
        ) from exc
    if aside is not None:
        shutil.rmtree(aside, ignore_errors=True)
    return False


# --------------------------------------------------------------------------- Stack


class Stack:
    """A committed stack, or a non-persisted window view of one."""

    def __init__(
        self,
        path: Path,
        meta: StackMeta,
        *,
        win: Win | None = None,
        cache: dict[str, Any] | None = None,
    ) -> None:
        self.path = path
        self.meta = meta
        self._full_grid = Grid.from_meta(meta.grid.model_dump())
        self._win = win
        self._cache: dict[str, Any] = cache if cache is not None else {}
        self.grid = self._full_grid if win is None else self._full_grid.subgrid(*win)

    # -- construction ------------------------------------------------------------

    @classmethod
    def open(cls, ref: str | Path, *, store: LocalArtifactStore | None = None) -> Stack:
        """Open a stack by directory, ``stack.json``, id or "latest"."""
        adir = paths.analysis_dir(store)
        text = str(ref)
        if text == "latest":
            directory = paths.latest_stack_dir(adir)
            if directory is None:
                raise cls._not_found(adir, text)
            return cls._load(directory)
        pathlike = os.sep in text or text.startswith(("~", ".")) or text.endswith(".json")
        candidates: list[Path] = []
        if pathlike:
            candidates.append(Path(text).expanduser())
            if store is not None:
                candidates.append(store.root / Path(text).expanduser())
        else:
            candidates.append(paths.stacks_dir(adir) / text)
        for cand in candidates:
            directory = cand.parent if cand.name == "stack.json" else cand
            if _SCRATCH_RE.search(directory.name):
                break
            if (directory / "stack.json").is_file():
                return cls._load(directory)
        raise cls._not_found(adir, text)

    @staticmethod
    def _not_found(adir: Path, ref: str) -> StackError:
        ids = paths.list_stack_ids(adir)[:10]
        listing = ", ".join(ids) if ids else "none"
        return StackError(
            fix(
                f"no committed stack {ref!r} (available: {listing}).",
                f'use a listed id or "latest"; stacks live in {paths.stacks_dir(adir)}',
            )
        )

    @classmethod
    def _load(cls, directory: Path) -> Stack:
        try:
            meta = StackMeta.model_validate_json((directory / "stack.json").read_text("utf-8"))
        except (OSError, ValueError) as exc:
            raise StackError(
                fix(f"unreadable stack.json in {directory}: {exc}", "re-run prepare_stack")
            ) from exc
        return cls(directory, meta)

    @staticmethod
    def exists(analysis_dir: Path, stack_id: str) -> str | None:
        """Return the stored params_hash of a committed stack, else None."""
        return exists(analysis_dir, stack_id)

    @classmethod
    def create(
        cls,
        analysis_dir: Path,
        *,
        id: str,
        grid: Grid,
        layers: Sequence[LayerMeta | Mapping[str, Any]],
        data: PlaneData,
        qa: Sequence[QaMeta | Mapping[str, Any]] = (),
        qa_data: PlaneData | None = None,
        provenance: Mapping[str, Any] | None = None,
        aoi: AoiMeta | Mapping[str, Any] | None = None,
        parents: Sequence[str] = (),
        warnings: Sequence[str] = (),
        params_hash: str,
        overwrite: bool = False,
        preview: bytes | Path | None = None,
        run_id: str | None = None,
    ) -> Stack:
        """Write a new stack atomically, or reuse an identical one.

        Args:
            analysis_dir: The ``analysis`` directory.
            id: Stack id; validated.
            grid: Pixel grid.
            layers: One entry per data plane (index is set by position).
            data: ``(L, H, W)`` array or an iterable of ``L`` ``(H, W)`` arrays (streamed).
            qa: QA plane metadata.
            qa_data: Bool/uint8 planes matching ``qa``.
            provenance: Free-form provenance dict.
            aoi: Area-of-interest metadata.
            parents: Parent stack ids.
            warnings: Warnings to record.
            params_hash: Identity hash of the inputs.
            overwrite: Replace an existing stack with a different hash.
            preview: PNG bytes or path.
            run_id: Run id recorded in stack.json.

        Returns:
            The committed stack (the existing one when identical).

        Raises:
            StackError: Bad name, size guard, count mismatch or id clash.
        """
        validate_name(id)
        layer_metas = [LayerMeta.model_validate(x) for x in layers]
        qa_metas = [QaMeta.model_validate(x) for x in qa]
        if not layer_metas:
            raise StackError(fix("a stack needs at least one layer.", "pass layers=[...]"))
        n_layers = len(layer_metas)
        cells = n_layers * grid.height * grid.width
        if cells > MAX_CELLS:
            raise StackError(
                fix(
                    f"stack would hold {cells:,} cells ({n_layers} layers x {grid.height} x "
                    f"{grid.width}), above the {MAX_CELLS:,} limit.",
                    "use a narrower bbox, fewer layers or a coarser resolution",
                )
            )
        if qa_metas and qa_data is None:
            raise StackError(fix("qa metadata given without qa_data.", "pass qa_data"))
        root = paths.stacks_dir(analysis_dir)
        final = root / id
        stored = _stored_hash(final)
        if stored == params_hash:
            return cls._load(final)
        if final.exists() and not overwrite:
            detail = "with different parameters" if stored else "incomplete"
            raise StackError(
                fix(
                    f"stack '{id}' exists {detail}.",
                    "choose another name or pass overwrite=true",
                )
            )
        root.mkdir(parents=True, exist_ok=True)
        tmp = _scratch(root, id, "tmp")
        tmp.mkdir()
        try:
            meta = cls._write_tmp(
                tmp, id, grid, layer_metas, data, qa_metas, qa_data, provenance, aoi,
                parents, warnings, params_hash, preview, run_id,
            )  # fmt: skip
            (tmp / "stack.json").write_text(
                json.dumps(meta.model_dump(by_alias=True, mode="json"), indent=1), "utf-8"
            )
            if _commit(tmp, final, root, id, params_hash, overwrite):
                return cls._load(final)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        return cls._load(final)

    @staticmethod
    def _write_tmp(
        tmp: Path,
        stack_id: str,
        grid: Grid,
        layers: list[LayerMeta],
        data: PlaneData,
        qa: list[QaMeta],
        qa_data: PlaneData | None,
        provenance: Mapping[str, Any] | None,
        aoi: AoiMeta | Mapping[str, Any] | None,
        parents: Sequence[str],
        warnings: Sequence[str],
        params_hash: str,
        preview: bytes | Path | None,
        run_id: str | None,
    ) -> StackMeta:
        shape = grid.shape
        mm = open_memmap(tmp / "data.npy", mode="w+", dtype=np.float32, shape=(len(layers), *shape))
        for i, plane in enumerate(_plane_iter(data, len(layers), "data", shape)):
            check_cancel()
            mm[i] = plane.astype(np.float32, copy=False)
        mm.flush()
        del mm
        if qa:
            assert qa_data is not None
            qmm = open_memmap(tmp / "qa.npy", mode="w+", dtype=np.uint8, shape=(len(qa), *shape))
            for i, plane in enumerate(_plane_iter(qa_data, len(qa), "qa_data", shape)):
                flags = np.asarray(plane) != 0
                qmm[i] = flags
                if qa[i].fraction is None:
                    qa[i] = qa[i].model_copy(update={"fraction": float(flags.mean())})
            qmm.flush()
            del qmm
        _write_preview(preview, tmp / "preview.png")
        area, scale_meta, extra_warnings = _grid_scale(grid)
        aoi_meta = AoiMeta.model_validate(aoi) if aoi is not None else None
        return StackMeta(
            id=stack_id,
            params_hash=params_hash,
            created=datetime.now(UTC).isoformat(timespec="seconds"),
            warnings=[*warnings, *extra_warnings],
            grid=GridMeta(
                **grid.to_meta(),
                bounds_wgs84=list(lonlat_bounds_of(grid)),
                pixel_area_m2=area,
                area_scale=scale_meta,
            ),
            aoi=aoi_meta,
            layers=[m.model_copy(update={"index": i}) for i, m in enumerate(layers)],
            qa=[m.model_copy(update={"index": i}) for i, m in enumerate(qa)],
            provenance=dict(provenance or {}),
            parents=list(parents),
            data_sha256=_sha256_file(tmp / "data.npy"),
            run_id=run_id,
        )

    # -- properties --------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, int, int]:
        return (len(self.meta.layers), *self.grid.shape)

    @property
    def id(self) -> str:
        return self.meta.id

    @property
    def analysis_dir(self) -> Path:
        return self.path.parent.parent

    @property
    def pixel_area_m2(self) -> float:
        return self.meta.grid.pixel_area_m2

    # -- reading -----------------------------------------------------------------

    def _mm(self, which: str) -> Any:
        if which not in self._cache:
            self._cache[which] = np.load(self.path / f"{which}.npy", mmap_mode="r")
        return self._cache[which]

    def _slice(self, win: Win | None = None) -> tuple[slice, slice]:
        r0, r1, c0, c1 = (
            win or self._win or (0, *self._full_grid.shape[:1], 0, self._full_grid.width)
        )
        return slice(r0, r1), slice(c0, c1)

    def names(self) -> list[str]:
        """Return layer names in storage order."""
        return [m.name for m in self.meta.layers]

    def dates(self) -> list[str]:
        """Return the sorted distinct layer dates (YYYYMMDD)."""
        found = {_layer_date(m.name, m.datetime) for m in self.meta.layers}
        return sorted(d for d in found if d)

    def layer(self, name: str, *, date: str | None = None) -> LayerMeta:
        """Return the metadata of a layer by exact name, or by base name plus date."""
        by_name = {m.name: m for m in self.meta.layers}
        if date is None and name in by_name:
            return by_name[name]
        wanted = _norm_date(date) if date else None
        hits = [
            m
            for m in self.meta.layers
            if _base(m.name) == _base(name)
            and (wanted is None or _layer_date(m.name, m.datetime) == wanted)
        ]
        if len(hits) == 1:
            return hits[0]
        if not hits:
            raise StackError(
                fix(f"stack '{self.id}' has no layer {name!r}.", f"use one of {self.names()}")
            )
        dates = [_layer_date(m.name, m.datetime) or "?" for m in hits]
        raise StackError(fix(f"layer {name!r} has dates {dates}.", f"use {_base(name)}@YYYYMMDD"))

    def band(self, name: str, *, date: str | None = None) -> NDArray[np.float32]:
        """Return a layer as a read-only float32 ``(H, W)`` memmap view."""
        rows, cols = self._slice()
        view: NDArray[np.float32] = self._mm("data")[self.layer(name, date=date).index, rows, cols]
        return view

    def cube(self, names: Sequence[str], *, date: str | None = None) -> NDArray[np.float32]:
        """Return a float32 ``(N, H, W)`` copy of the named layers."""
        return np.stack([np.array(self.band(n, date=date)) for n in names])

    def series(self, name: str) -> tuple[list[str], NDArray[np.float32]]:
        """Return ``(dates, (T, H, W))`` for every dated layer with base name ``name``."""
        found = sorted(
            (d, m)
            for m in self.meta.layers
            if _base(m.name) == _base(name) and (d := _layer_date(m.name, m.datetime))
        )
        if not found:
            raise StackError(fix(f"no dated layers named {name!r}.", f"use one of {self.names()}"))
        rows, cols = self._slice()
        mm = self._mm("data")
        return [d for d, _ in found], np.stack(
            [np.array(mm[m.index, rows, cols]) for _, m in found]
        )

    def _qa_names(self) -> list[str]:
        return [q.name for q in self.meta.qa]

    def _plane_at(self, index: int, win: Win | None = None) -> NDArray[np.bool_]:
        rows, cols = self._slice(win)
        flags: NDArray[np.bool_] = np.asarray(self._mm("qa")[index, rows, cols]) != 0
        return flags

    def plane(self, name: str) -> NDArray[np.bool_]:
        """Return a QA plane by name, or by semantic when exactly one plane has it."""
        for q in self.meta.qa:
            if q.name == name:
                return self._plane_at(q.index)
        hits = [q for q in self.meta.qa if q.semantic == name]
        if len(hits) == 1:
            return self._plane_at(hits[0].index)
        detail = f"{len(hits)} planes have semantic {name!r}" if hits else f"no plane {name!r}"
        raise StackError(fix(f"stack '{self.id}': {detail}.", f"use one of {self._qa_names()}"))

    def planes(self) -> dict[str, NDArray[np.bool_]]:
        """Return every QA plane keyed by name."""
        return {q.name: self._plane_at(q.index) for q in self.meta.qa}

    # -- validity ----------------------------------------------------------------

    def _extra_window(self, other: Stack) -> Win | None:
        if other.grid.same_as(self.grid):
            return other._win
        if self._win is not None and other._win is None and other.grid.same_as(self._full_grid):
            return self._win
        other.grid.require_same(self.grid)
        return None

    def _excluded(self, mask: str | Sequence[str]) -> frozenset[str]:
        if mask == "clear":
            return CLEAR_EXCLUDE
        if mask == "none":
            return _AOI_EXCLUDE
        if isinstance(mask, str):
            raise StackError(
                fix(f"unknown mask {mask!r}.", 'use "clear", "none" or a list of semantics')
            )
        unknown = [s for s in mask if s not in SEMANTICS]
        if unknown:
            raise StackError(fix(f"unknown semantics {unknown}.", f"use values of {SEMANTICS}"))
        return frozenset(mask) | _AOI_EXCLUDE

    def valid(
        self,
        mask: str | Sequence[str] = "clear",
        *,
        layers: Sequence[str] | None = None,
        extra: Sequence[Stack] = (),
    ) -> NDArray[np.bool_]:
        """Return a bool ``(H, W)`` mask of usable pixels.

        Pixels must be finite in every used layer (default: all non-categorical
        layers) and, per ``mask``, absent from the excluded QA semantics in this
        stack and in ``extra`` stacks. Planes named ``x@DATE`` apply only when a used
        layer carries that date. Snow, water and adjacent stay valid under "clear".
        """
        excluded = self._excluded(mask)
        if layers is None:
            used = [m for m in self.meta.layers if m.kind != "categorical"]
        else:
            used = [self.layer(n) for n in layers]
        rows, cols = self._slice()
        mm = self._mm("data")
        ok = np.ones(self.grid.shape, dtype=bool)
        for m in used:
            ok &= np.isfinite(mm[m.index, rows, cols])
        used_dates = {_layer_date(m.name, m.datetime) for m in used} - {None}
        for stack in (self, *extra):
            win = self._extra_window(stack) if stack is not self else None
            for q in stack.meta.qa:
                if q.semantic not in excluded:
                    continue
                match = _DATE_RE.search(q.name)
                if match and match.group(1) not in used_dates:
                    continue
                ok &= ~stack._plane_at(q.index, win)
        return ok

    # -- windows and tiles -------------------------------------------------------

    def window(self, r0: int, r1: int, c0: int, c1: int) -> Stack:
        """Return a non-persisted view of rows ``r0:r1`` and columns ``c0:c1`` (relative to self)."""
        height, width = self.grid.shape
        if not (0 <= r0 < r1 <= height and 0 <= c0 < c1 <= width):
            raise StackError(
                fix(
                    f"window {(r0, r1, c0, c1)} outside {height}x{width}.",
                    "use a window inside the grid",
                )
            )
        base_r, base_c = (self._win[0], self._win[2]) if self._win else (0, 0)
        win = (base_r + r0, base_r + r1, base_c + c0, base_c + c1)
        return Stack(self.path, self.meta, win=win, cache=self._cache)

    def iter_tiles(self, size: int, margin: int = 0) -> Iterator[Tile]:
        """Yield tiles whose cores partition the grid; windows add ``margin`` pixels."""
        if size <= 0 or margin < 0:
            raise StackError(fix("tile size must be positive.", "pass size>0, margin>=0"))
        height, width = self.grid.shape
        for r in range(0, height, size):
            for c in range(0, width, size):
                core = (r, min(r + size, height), c, min(c + size, width))
                win = (
                    max(core[0] - margin, 0),
                    min(core[1] + margin, height),
                    max(core[2] - margin, 0),
                    min(core[3] + margin, width),
                )
                inner = (core[0] - win[0], core[1] - win[0], core[2] - win[2], core[3] - win[2])
                yield Tile(window=win, core=core, inner=inner)

    # -- derive ------------------------------------------------------------------

    def derive(
        self,
        *,
        op: str,
        layers: Sequence[LayerMeta | Mapping[str, Any]],
        data: PlaneData,
        params: Mapping[str, Any],
        provenance: Mapping[str, Any] | None = None,
        name: str | None = None,
        qa: Sequence[QaMeta | Mapping[str, Any]] = (),
        qa_data: PlaneData | None = None,
        keep_layers: Sequence[str] = (),
        keep_qa: bool = False,
        overwrite: bool = False,
        warnings: Sequence[str] = (),
        preview: bytes | Path | None = None,
    ) -> Stack:
        """Write a new stack on this grid from ``data``, optionally keeping parent layers.

        Kept layers (and QA planes when ``keep_qa``) are copied first. The id is
        ``name`` or ``<parent[:40]>-<op>-<h6>``.
        """
        if self._win is not None:
            raise StackError(fix("cannot derive from a window view.", "derive from the full stack"))
        new_layers = [LayerMeta.model_validate(x) for x in layers]
        digest = hash_params(
            {
                "parent": self.meta.params_hash,
                "op": op,
                "params": dict(params),
                "layers": [m.name for m in new_layers],
                "keep": list(keep_layers),
                "keep_qa": keep_qa,
            }
        )
        slug = re.sub(r"[^a-z0-9_.-]", "_", op.lower())[:16] or "op"
        stack_id = name or f"{self.id[:40]}-{slug}-{digest[:6]}"
        kept = [
            self.layer(n).model_copy(
                update={"processing": [*self.layer(n).processing, f"kept from {self.id}"]}
            )
            for n in keep_layers
        ]
        qa_metas = [QaMeta.model_validate(x) for x in qa]
        if keep_qa:
            qa_metas = [*(q.model_copy() for q in self.meta.qa), *qa_metas]

        def planes() -> Iterator[NDArray[Any]]:
            for n in keep_layers:
                yield self.band(n)
            yield from _plane_iter(data, len(new_layers), "data", self.grid.shape)

        def qa_planes() -> Iterator[NDArray[Any]]:
            if keep_qa:
                for q in self.meta.qa:
                    yield self._plane_at(q.index)
            if qa_data is not None:
                yield from _plane_iter(
                    qa_data,
                    len(qa_metas) - len(self.meta.qa if keep_qa else []),
                    "qa_data",
                    self.grid.shape,
                )

        return Stack.create(
            self.analysis_dir,
            id=stack_id,
            grid=self.grid,
            layers=[*kept, *new_layers],
            data=planes(),
            qa=qa_metas,
            qa_data=qa_planes() if qa_metas else None,
            provenance=provenance,
            aoi=self.meta.aoi,
            parents=[self.id],
            warnings=warnings,
            params_hash=digest,
            overwrite=overwrite,
            preview=preview,
        )


def open_stack(ref: str | Path = "latest", *, store: LocalArtifactStore | None = None) -> Stack:
    """Open a stack by id, path or "latest"."""
    return Stack.open(ref, store=store)
