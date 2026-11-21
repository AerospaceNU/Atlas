"""prepare_stack: pulled chips and user rasters to one analysis-ready Stack."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import Tool, ToolResult

if TYPE_CHECKING:
    from atlas.geo.prepare import PrepareReport
    from atlas.geo.results import GeoResult
    from atlas.geo.stack import Stack

CAP_SECONDS = 600
_MAX_LAYERS_SHOWN = 8
_SAR_KEYS = {"db", "despeckle"}
_DESPECKLE = ("none", "median3", "lee5")
_QA_REPORTED = frozenset(
    {"nodata", "saturated", "cloud", "cloud_shadow", "cirrus", "snow", "outside_aoi"}
)


def _compact(schema: dict[str, Any]) -> None:
    """Shrink the advertised schema: no defaults, no null branches, no open objects."""
    for prop in schema.get("properties", {}).values():
        prop.pop("default", None)
        branches = [b for b in prop.pop("anyOf", []) if b.get("type") != "null"]
        if len(branches) == 1:
            prop.update(branches[0])
        elif branches:
            prop["anyOf"] = branches
        prop.pop("additionalProperties", None)
        prop.pop("pattern", None)
    schema.pop("additionalProperties", None)


class FileSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra=_compact)

    path: str
    name: str
    band: str | None = None
    unit: str | None = None
    scale: float | None = None
    offset: float | None = None
    nodata: float | None = None
    datetime: str | None = None
    categorical: bool = False


class PrepareStackInput(BaseModel):
    model_config = ConfigDict(json_schema_extra=_compact)

    scene: str | None = None
    scenes: list[str] | str | None = Field(
        default=None, description="Ids, 'last_pull' (default), rNNNN or glob."
    )
    files: list[FileSpec] | None = Field(default=None, description="Chips lack sidecar.")
    bands: list[str] | None = Field(default=None, description="Common names; default all.")
    resolution: str | float = Field(default="finest", description="finest|coarsest|metres.")
    crs: str = Field(default="native", description="native|utm|EPSG:n.")
    bbox: list[float] | None = Field(default=None, description="west,south,east,north.")
    polygon: list[list[float]] | None = Field(default=None, description="lon/lat ring.")
    qa: Literal["auto", "none"] = "auto"
    dilate_px: int = Field(default=0, ge=0, le=20)
    sar: dict[str, Any] | None = Field(
        default=None, description="{db:true, despeckle:none|median3|lee5}"
    )
    merge: Literal["none", "mosaic"] = "none"
    name: str | None = Field(default=None, pattern=r"^[a-z0-9][a-z0-9_.-]{0,63}$")
    overwrite: bool = False

    @field_validator("bbox")
    @classmethod
    def _bbox(cls, value: list[float] | None) -> list[float] | None:
        if value is None:
            return None
        if len(value) != 4:
            raise ValueError("bbox needs 4 numbers [west,south,east,north]")
        w, s, e, n = value
        if not (-180 <= w < e <= 180 and -90 <= s < n <= 90):
            raise ValueError("bbox must satisfy west<east, south<north in lon/lat degrees")
        return value

    @field_validator("polygon")
    @classmethod
    def _polygon(cls, value: list[list[float]] | None) -> list[list[float]] | None:
        if value is None:
            return None
        ring = value[:-1] if len(value) > 3 and value[0] == value[-1] else value
        if not 3 <= len(ring) <= 500:
            raise ValueError("polygon needs 3 to 500 lon/lat vertices")
        for point in ring:
            if len(point) != 2 or not (-180 <= point[0] <= 180 and -90 <= point[1] <= 90):
                raise ValueError("polygon vertices must be [lon,lat] degrees")
        return ring

    @field_validator("sar")
    @classmethod
    def _sar(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is None:
            return None
        unknown = set(value) - _SAR_KEYS
        if unknown or value.get("despeckle", "none") not in _DESPECKLE:
            raise ValueError("sar takes db (bool) and despeckle (none|median3|lee5)")
        return value

    @model_validator(mode="after")
    def _exclusive(self) -> PrepareStackInput:
        if self.bbox is not None and self.polygon is not None:
            raise ValueError("give bbox or polygon, not both")
        return self


def _first_error(exc: ValidationError) -> str:
    first = exc.errors()[0]
    where = ".".join(str(p) for p in first.get("loc", ()))
    msg = str(first.get("msg", "invalid")).removeprefix("Value error, ")
    return f"invalid arguments: {where + ': ' if where else ''}{msg}"


class PrepareStackTool(Tool):
    """Build one co-registered physical-unit Stack from pulled chips and user rasters.

    Chips are read with their sidecars (scale, offset, nodata, QA preset); the grid is
    computed from sidecars, so an identical repeat call reuses the stack without reading
    pixels. Dates stay separate layers (name@YYYYMMDD). Writes only under analysis/.
    """

    name = "prepare_stack"
    description = (
        "Build an analysis-ready stack once: reflectance/dB layers on one grid, one layer per "
        "date (red@YYYYMMDD), QA planes from scl/qa chips, optional AOI. Run after "
        "get_satellite_data download. Reuses an identical stack. files=[{path,name:'dem',"
        "unit:'m',scale,offset,nodata,categorical}] adds rasters without a sidecar (integer "
        "rasters need scale)."
    )
    trust: ClassVar[Literal["default", "opt_in"]] = "default"
    domain: ClassVar[str] = "satellite"

    @property
    def input_model(self) -> type[BaseModel]:
        return PrepareStackInput

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        from atlas.agent.tools import _geo
        from atlas.geo import prepare
        from atlas.geo.errors import GeoError

        try:
            req = PrepareStackInput.model_validate(arguments)
        except ValidationError as exc:
            return ToolResult(text=_first_error(exc))
        deadline = time.monotonic() + CAP_SECONDS
        refs = [req.scene] if req.scene else []
        if isinstance(req.scenes, str):
            refs.append(req.scenes)
        elif req.scenes:
            refs.extend(req.scenes)
        files = [f.model_dump(exclude_none=True) for f in req.files or []]
        if not refs and not files:
            refs = ["last_pull"]
        aoi = None
        if req.bbox is not None:
            aoi = prepare.AoiSpec(bbox=(req.bbox[0], req.bbox[1], req.bbox[2], req.bbox[3]))
        elif req.polygon is not None:
            aoi = prepare.AoiSpec(polygon=[(p[0], p[1]) for p in req.polygon])
        recorded = req.model_dump(exclude_none=True)
        try:
            sources = prepare.discover(refs, store, files)
            stack, report = prepare.build_stack(
                sources,
                bands=req.bands,
                resolution=req.resolution,
                crs=req.crs,
                aoi=aoi,
                qa=req.qa,
                dilate_px=req.dilate_px,
                sar=req.sar,
                name=req.name,
                merge=req.merge,
                overwrite=req.overwrite,
                store=store,
                deadline=deadline,
                arguments=recorded,
            )
        except GeoError as exc:
            return _geo.error_result(exc)
        return _geo.result_to_tool_result(
            self._result(stack, report),
            store,
            tool=self.name,
            arguments=recorded,
            stack_ids=[stack.id],
        )

    @staticmethod
    def _layers_line(report: PrepareReport) -> str:
        shown = [f"{n} ({b})" if b else n for n, b in report.layer_bands[:_MAX_LAYERS_SHOWN]]
        extra = len(report.layer_bands) - len(shown)
        return "layers: " + ", ".join(shown) + (f" +{extra} more" if extra > 0 else "")

    @staticmethod
    def _qa_line(report: PrepareReport) -> str | None:
        from atlas.geo.results import sig4

        found = sorted(
            (v, k)
            for k, v in report.qa_fractions.items()
            if v > 0 and k.partition("@")[0] in _QA_REPORTED
        )[::-1]
        parts = [f"{k} {sig4(v * 100)}%" for v, k in found[:6]]
        return "qa: " + ", ".join(parts) if parts else None

    def _result(self, stack: Stack, report: PrepareReport) -> GeoResult:
        from atlas.geo.results import GeoResult, sig4

        grid = stack.grid
        layers, height, width = stack.shape
        hectares = report.aoi_px * stack.pixel_area_m2 / 1e4
        summary = [
            f"stack {stack.id}: {layers}x{height}x{width}, {sig4(grid.xres)} m, {grid.crs}, "
            f"{sig4(hectares)} ha (aoi_px {sig4(report.aoi_px)})",
            self._layers_line(report),
            *report.combine_notes,
        ]
        if report.resampling_notes:
            summary.append("resampling: " + "; ".join(report.resampling_notes))
        qa_line = self._qa_line(report)
        if qa_line:
            summary.append(qa_line)
        if report.clear_pct is not None:
            summary.append(f"clear {sig4(report.clear_pct)}% of AOI")
        summary.extend(report.citations)
        if report.reused:
            summary.append("reused: identical inputs, existing stack returned")
        elif report.overwrote:
            summary.append("overwrote the previous stack of this id")
        preview = stack.path / "preview.png"
        return GeoResult(
            summary=summary,
            warnings=list(report.warnings),
            artifacts=[Path(preview)] if preview.is_file() and not report.reused else [],
            numbers={
                "stack": stack.id,
                "layers": layers,
                "aoi_px": report.aoi_px,
                "clear_pct": report.clear_pct,
            },
            next=f'inspect_stack(stack="{stack.id}")',
        )
