"""Agent tool: class areas, zonal statistics and class-by-zone cross-tabs."""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import Tool, ToolResult

_CLASS_ROWS = 12
_ZONE_ROWS = 20


class AreaTableInput(BaseModel):
    stack: str = Field(description="Stack id, path or 'latest'.")
    layer: str = Field(description="Categorical layer (areas) or continuous layer (statistics).")
    zones: dict[str, Any] | None = Field(
        default=None,
        description="One of {geojson: obj}, {path: file}, {stack: id, layer: name} (a "
        "categorical layer on the same grid: class-by-zone table). GeoJSON is lon/lat, at "
        "most 500 features. None: whole AOI.",
    )
    class_names: dict[str, str] | None = Field(
        default=None, description="Code to name, e.g. {'1': 'water'}."
    )
    stat: Literal["mean", "median", "min", "max", "sum"] = Field(
        default="mean", description="Statistic for a continuous layer."
    )
    mask: str | list[str] = Field(default="clear", description="'clear', 'none' or semantics.")
    mask_with: list[str] = Field(default_factory=list, description="Stacks whose QA also applies.")
    chart: bool = Field(default=False, description="Also draw a bar chart.")


class AreaTableTool(Tool):
    """Tabulate class hectares, zonal statistics or a class-by-zone cross-tab.

    Hectares are valid pixels times pixel area; every table states valid-pixel
    denominators. Writes only a CSV under ``analysis/tables``; no stack is written.
    """

    name = "area_table"
    description = (
        "Area (ha, % of valid pixels) per class of a categorical layer, statistics of a "
        "continuous layer, per zone (GeoJSON inline or file) or by another categorical layer "
        "on the same grid (e.g. flood extent by land cover). Reports aoi_px, valid_px, "
        "valid_fraction."
    )
    trust: ClassVar[Literal["default", "opt_in"]] = "default"
    domain: ClassVar[str] = "satellite"

    @property
    def input_model(self) -> type[BaseModel]:
        return AreaTableInput

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        req = AreaTableInput.model_validate(arguments)
        from atlas.agent.tools import _geo
        from atlas.geo.errors import GeoCancelled, GeoError

        try:
            return self._run(req, arguments, store, _geo)
        except GeoCancelled:
            return ToolResult(text="cancelled")
        except GeoError as exc:
            return _geo.error_result(exc)

    def _run(
        self,
        req: AreaTableInput,
        arguments: dict[str, Any],
        store: LocalArtifactStore,
        _geo: Any,
    ) -> ToolResult:
        import numpy as np

        from atlas.geo import zonal
        from atlas.geo._lazy import check_cancel
        from atlas.geo.errors import GeoError, fix
        from atlas.geo.results import GeoResult, sig4

        stack = _geo.stack_from_args(store, req.stack)
        extra = [_geo.stack_from_args(store, ref) for ref in req.mask_with]
        meta = stack.layer(req.layer)
        categorical = meta.kind == "categorical"
        names = {int(k): v for k, v in (req.class_names or {}).items()} or None
        check_cancel()
        valid = stack.valid(req.mask, layers=[req.layer], extra=extra)
        cnt = zonal.counts(stack, valid)
        spec = req.zones or {}
        given = [k for k in ("geojson", "path", "stack") if spec.get(k)]
        if len(given) > 1 or (spec and not given):
            raise GeoError(
                fix(
                    f"zones needs exactly one of geojson, path, stack (got {sorted(spec)}).",
                    "zones={'geojson': {...}} or zones={'stack': id, 'layer': name}",
                )
            )
        warnings: list[str] = []
        stems = f"{stack.id}-{req.layer}-zonal"
        lines: list[str] = []
        if not given:
            if categorical:
                full = [
                    _row(r)
                    for r in zonal.class_table(stack, req.layer, valid=valid, class_names=names)
                ]
                shown, limit, title = full, _CLASS_ROWS, "class_table"
            else:
                whole = zonal.Zone("aoi", np.ones(stack.grid.shape, dtype=bool))
                full = zonal.zonal_table(stack, req.layer, [whole], valid=valid, stat=req.stat)
                shown, limit, title = full, _ZONE_ROWS, "zonal"
        elif given[0] == "stack":
            full, shown, extra_lines = self._cross(req, spec, stack, valid, names, _geo, store)
            lines.extend(extra_lines)
            limit, title = _CLASS_ROWS, "class_table"
        else:
            source: Any = (
                store.resolve_read(str(spec["path"])) if given[0] == "path" else spec["geojson"]
            )
            check_cancel()
            zones = zonal.load_zones(source, stack.grid)
            warnings.extend(zonal.zone_notes(zones))
            check_cancel()
            full = zonal.zonal_table(
                stack, req.layer, zones, valid=valid, stat=req.stat, class_names=names
            )
            shown = zonal.wide_screen_rows(full) if categorical else full
            limit, title = _ZONE_ROWS, "zonal"
        csv_path = zonal.save_csv(full, stack.analysis_dir / "tables", stems)
        csv_rel = store.relative(csv_path)
        artifacts = [csv_path]
        if req.chart:
            note = _chart(shown, req, stack, store, artifacts)
            if note:
                warnings.append(note)
        fraction = sig4(cnt.valid_fraction)
        summary = [
            f"{stack.id}:{req.layer} aoi_px {cnt.aoi_px} valid_px {cnt.valid_px} "
            f"(valid_fraction {fraction}); pixel {sig4(cnt.px_area_ha)} ha",
            *lines,
            zonal.table_line(min(len(shown), limit), len(shown), csv_rel),
        ]
        res = GeoResult(
            summary=summary,
            table=shown[:limit],
            warnings=warnings,
            artifacts=artifacts,
            numbers={"aoi_px": cnt.aoi_px, "valid_px": cnt.valid_px, "rows": len(full)},
            next=f"render_map stack={stack.id} layers=[{req.layer}]",
            table_name=title,
            table_limit=limit,
        )
        result: ToolResult = _geo.result_to_tool_result(
            res, store, tool=self.name, arguments=arguments, stack_ids=[stack.id]
        )
        return result

    def _cross(
        self,
        req: AreaTableInput,
        spec: dict[str, Any],
        stack: Any,
        valid: Any,
        names: Any,
        _geo: Any,
        store: LocalArtifactStore,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:

        from atlas.geo import zonal
        from atlas.geo.errors import GeoError, fix
        from atlas.geo.results import sig4

        zlayer = spec.get("layer")
        if not zlayer:
            raise GeoError(
                fix("zones.stack needs zones.layer.", "zones={'stack': id, 'layer': 'x'}")
            )
        if stack.layer(req.layer).kind != "categorical":
            raise GeoError(
                fix(
                    f"layer {req.layer!r} is continuous; a cross-tab needs classes.",
                    "classify it first",
                )
            )
        zstack = _geo.stack_from_args(store, str(spec["stack"]))
        rows, cols, matrix = zonal.crosstab(stack, req.layer, zstack, str(zlayer), valid=valid)
        area = stack.pixel_area_m2 / 1e4
        full: list[dict[str, Any]] = []
        for i, label in enumerate(rows):
            item: dict[str, Any] = {"class": label}
            for j, col in enumerate(cols):
                item[f"px_{col}"] = int(matrix[i, j])
                item[f"ha_{col}"] = int(matrix[i, j]) * area
            item["ha_total"] = int(matrix[i].sum()) * area
            full.append(item)
        plain = zonal.class_table(stack, req.layer, valid=valid, class_names=names)
        plain_px = sum(int(r["px"]) for r in plain)
        no_zone = plain_px - int(matrix.sum())
        info = [
            f"cross-tab vs {zstack.id}:{zlayer}; total {sig4(int(matrix.sum()) * area)} ha = "
            f"class table {sig4(plain_px * area)} ha minus no-zone {no_zone} px"
        ]
        return full, zonal.wide_screen_rows(full), info


def _row(row: dict[str, Any]) -> dict[str, Any]:
    return {"class": row["name"], "px": row["px"], "ha": row["ha"], "pct_valid": row["pct_valid"]}


def _chart(
    rows: list[dict[str, Any]], req: AreaTableInput, stack: Any, store: Any, artifacts: list[Any]
) -> str | None:
    try:
        from atlas.geo import charts

        key = next((k for k in rows[0] if k.startswith(("ha", "mean", "median"))), "ha")
        label = next(iter(rows[0]))
        out = charts.bar_chart(
            [str(r[label]) for r in rows],
            [float(r.get(key, 0.0)) for r in rows],
            title=f"{stack.id} {req.layer}",
            path=stack.analysis_dir / "figures" / f"{stack.id}-{req.layer}-bar.png",
        )
        artifacts.append(getattr(out, "png", out))
    except (ImportError, AttributeError, TypeError, IndexError, KeyError) as exc:
        return f"chart skipped: {type(exc).__name__}"
    return None
