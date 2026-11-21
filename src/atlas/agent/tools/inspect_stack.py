from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import Tool, ToolResult

MAX_ROWS = 24
GRID_MAX_ROWS = 8
MAX_CHARS = 1500
_NAME_CHARS = 20
_CSV_PLACEHOLDER = "analysis/tables/layers-00000000.csv"


class InspectStackInput(BaseModel):
    stack: str = Field(description="Stack id, path or 'latest'.")
    layers: list[str] | None = Field(default=None, description="Layer names; default all.")
    mask: Literal["clear", "none"] = Field(
        default="clear", description="clear drops cloud, shadow, nodata; none keeps all."
    )
    histogram_bins: int = Field(default=10, ge=2, le=50, description="Histogram bins (2-50).")


class InspectStackTool(Tool):
    """Summarise a stack: grid, per-layer statistics on valid pixels, QA fractions.

    Read-only: it never writes a stack and produces no image. Statistics use one valid
    mask per layer, so one layer's no-data does not change another layer's valid percent.
    """

    name = "inspect_stack"
    description = (
        "Inspect a prepared stack: CRS, size, resolution, area, per-layer unit, valid%, min, "
        "median, max, QA fractions, clear %, a coarse text grid and histogram for one layer. "
        "Text only, no image. Run it before band_math or classify."
    )
    domain = "satellite"

    @property
    def input_model(self) -> type[BaseModel]:
        return InspectStackInput

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        request = InspectStackInput.model_validate(arguments)
        from atlas.agent.tools import _geo
        from atlas.geo.errors import GeoError

        try:
            return _inspect(request, store)
        except GeoError as exc:
            return _geo.error_result(exc)


def _resolve_layers(stack: Any, wanted: list[str] | None) -> list[str]:
    from atlas.geo.errors import StackError, fix

    names: list[str] = stack.names()
    if not wanted:
        return names
    out: list[str] = []
    for item in wanted:
        if item in names:
            out.append(item)
            continue
        try:
            out.append(stack.layer(item).name)
        except StackError:
            more = len(names) - 6
            have = ",".join(names[:6]) + (f" (+{more})" if more > 0 else "")
            raise StackError(
                fix(f"no layer '{item[:40]}'. Have: {have}.", f"inspect_stack stack={stack.id}")
            ) from None
    return out


def _grid_line(stack: Any) -> str:
    from atlas.geo.results import sig4

    grid = stack.grid
    area = stack.meta.aoi.area_ha if stack.meta.aoi and stack.meta.aoi.area_ha else None
    if area is None:
        area = grid.width * grid.height * stack.pixel_area_m2 / 1e4
    res = (
        sig4(grid.xres)
        if abs(grid.xres - grid.yres) < 1e-9
        else f"{sig4(grid.xres)}x{sig4(grid.yres)}"
    )
    return (
        f"grid: {grid.crs} {grid.width}x{grid.height} px, {res} m, aoi {sig4(area)} ha, "
        f"px {sig4(stack.pixel_area_m2)} m2"
    )


def _citations(stack: Any) -> str:
    from atlas.geo.provenance import citations

    found: list[str] = []
    stored = stack.meta.provenance.get("citations")
    for text in stored if isinstance(stored, list) else []:
        if str(text) not in found:
            found.append(str(text))
    for meta in stack.meta.layers:
        keys = [meta.sensor, meta.source.collection if meta.source else None]
        for key in keys:
            for text in citations(key, year=(meta.datetime or "")[:4] or None) if key else []:
                if text not in found:
                    found.append(text)
    return "; ".join(found)


def _row_text(row: dict[str, str], with_unit: bool) -> str:
    name = row["layer"][:_NAME_CHARS]
    parts = [f"{name:<{_NAME_CHARS}}"]
    if with_unit:
        parts.append(row["unit"][:6])
    parts += [row["valid%"], row["min"], row["median"], row["max"]]
    return " ".join(parts)


def _extras(arr: Any, valid: Any, bins: int) -> tuple[list[str], list[str]]:
    import numpy as np

    from atlas.geo.results import histogram_text, text_grid

    masked = np.where(valid, arr, np.nan)
    grid = text_grid(masked).split("\n")
    hist = histogram_text(masked, bins=bins, width=12).split("\n")
    return grid, hist


def _inspect(request: InspectStackInput, store: LocalArtifactStore) -> ToolResult:
    import numpy as np

    from atlas.agent.tools import _geo
    from atlas.geo._lazy import check_cancel
    from atlas.geo.results import GeoResult, sig4, spill_table
    from atlas.geo.stats import layer_stats, qa_fractions, stats_row

    stack = _geo.stack_from_args(store, request.stack)
    names = _resolve_layers(stack, request.layers)
    stats = []
    first: tuple[Any, Any] | None = None
    for name in names:
        check_cancel()
        arr = stack.band(name)
        valid = stack.valid(request.mask, layers=[name])
        meta = stack.layer(name)
        stats.append(layer_stats(arr, valid, name=name, unit=meta.unit))
        if first is None:
            first = (np.asarray(arr), valid)
    rows = [stats_row(s) for s in stats]
    mixed = len({s.unit for s in stats}) > 1
    clear = stack.valid(request.mask)
    clear_pct = 100.0 * float(clear.mean())
    qa = qa_fractions(stack)
    cite = _citations(stack)
    warnings = [w[:140] for w in stack.meta.warnings[:3]]
    dates = stack.dates()
    label = "clear" if request.mask == "clear" else "finite"
    unit_note = "" if mixed or not stats else f", unit {stats[0].unit or 'none'}"
    ncell = stack.grid.width * stack.grid.height

    def build(shown: int, extras: str, csv: str) -> GeoResult:
        head = [_grid_line(stack)]
        head.append(f"layers: {len(stats)}, dates: {len(dates)}{unit_note}")
        header = ("layer", *(["unit"] if mixed else []), "valid%", "min", "median", "max")
        lines = [*head, " ".join(header)]
        lines += [_row_text(r, mixed) for r in rows[:shown]]
        if len(rows) > shown:
            lines.append(f"+{len(rows) - shown} more: {csv}")
        if qa:
            lines.append("qa: " + ", ".join(f"{k} {sig4(100 * v)}%" for k, v in qa.items()))
        lines.append(f"{label}: {sig4(clear_pct)}% of {sig4(ncell)} px")
        if extras and first is not None:
            grid, hist = _extras(first[0], first[1], request.histogram_bins)
            lines += grid if "g" in extras else []
            lines += hist if "h" in extras else []
        if cite:
            lines.append(f"cite: {cite}")
        hint = f"inspect_stack layers=[{names[0]}] for the grid" if not extras else "band_math"
        hint = f"{hint} stack={stack.id}" if hint == "band_math" else hint
        return GeoResult(
            summary=lines,
            warnings=warnings,
            numbers={"n_layers": len(stats), "clear_pct": clear_pct},
            run_id="r0000",
            next=hint,
        )

    final_shown, chosen_extras = 0, ""
    for shown in range(min(len(rows), MAX_ROWS), -1, -1):
        options = ["gh", "g", ""] if shown <= GRID_MAX_ROWS and first is not None else [""]
        fit = next(
            (
                o
                for o in options
                if len(build(shown, o, _CSV_PLACEHOLDER).render(MAX_CHARS)) <= MAX_CHARS
            ),
            None,
        )
        if fit is not None:
            final_shown, chosen_extras = shown, fit
            break
    csv = _CSV_PLACEHOLDER
    artifacts = []
    if len(rows) > final_shown:
        full = []
        for s in stats:
            row = stats_row(s, full=True)
            full.append({"layer": row.pop("layer"), "band": stack.layer(s.name).band or "", **row})
        spill = GeoResult(summary=[], table=full, table_name="layers", table_limit=0)
        adir = _geo.analysis_dir(store)
        csv = spill_table(spill, adir / "tables") or csv
        artifacts.append(store.artifact_root / csv)
    result = build(final_shown, chosen_extras, csv)
    result.run_id = None
    result.artifacts = artifacts
    return _geo.result_to_tool_result(
        result, store, tool="inspect_stack", arguments=request.model_dump(), stack_ids=[stack.id]
    )
