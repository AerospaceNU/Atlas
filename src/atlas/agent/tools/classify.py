"""Agent tool: classify one continuous layer into categorical classes."""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import Tool, ToolResult

_SHOWN = 12


class ClassifyInput(BaseModel):
    stack: str = Field(description="Stack id or 'latest'.")
    layer: str = Field(description="Continuous layer, e.g. ndvi.")
    rules: list[dict[str, Any]] = Field(
        default_factory=list,
        description="[{name,min,max}], min<=v<max, first match wins.",
    )
    thresholds: list[float] = Field(default_factory=list, description="Ascending cut points.")
    labels: list[str] = Field(default_factory=list, description="len(thresholds)+1 names.")
    preset: Literal["ndvi", "mndwi_water", "dnbr", "sar_db_water"] | None = None
    method: Literal["otsu"] | None = None
    mask: str | list[str] = Field(default="clear", description="'clear', 'none' or semantics.")
    mask_with: list[str] = Field(default_factory=list, description="Stacks whose QA also applies.")
    name: str = Field(default="", description="Output stack id.")
    overwrite: bool = False


class ClassifyTool(Tool):
    """Classify a continuous layer by rules, thresholds, a preset or Otsu.

    Writes a one-layer categorical stack (layer ``class``). Valid pixels come only from
    ``Stack.valid``; nothing is executed.
    """

    name = "classify"
    description = (
        "Classify one continuous stack layer into classes by rules, thresholds+labels, a "
        "preset (ndvi, mndwi_water, dnbr, sar_db_water) or method=otsu. Writes a categorical "
        "stack and returns class pixels, hectares and percent of valid pixels. dnbr expects "
        "pre minus post; change_detect difference is post minus pre, so negate it (rules with "
        "negated bounds) or use burn_severity."
    )
    trust: ClassVar[Literal["default", "opt_in"]] = "default"
    domain: ClassVar[str] = "satellite"

    @property
    def input_model(self) -> type[BaseModel]:
        return ClassifyInput

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        req = ClassifyInput.model_validate(arguments)
        from atlas.agent.tools import _geo
        from atlas.geo.errors import GeoCancelled, GeoError

        try:
            return self._run(req, arguments, store, _geo)
        except GeoCancelled:
            return ToolResult(text="cancelled")
        except GeoError as exc:
            return _geo.error_result(exc)

    def _run(
        self, req: ClassifyInput, arguments: dict[str, Any], store: LocalArtifactStore, _geo: Any
    ) -> ToolResult:
        import numpy as np

        from atlas.geo import zonal
        from atlas.geo._lazy import check_cancel
        from atlas.geo.classify import PRESET_UNITS, classify_layer
        from atlas.geo.errors import GeoError, fix
        from atlas.geo.results import GeoResult, sig4

        stack = _geo.stack_from_args(store, req.stack)
        extra = [_geo.stack_from_args(store, ref) for ref in req.mask_with]
        meta = stack.layer(req.layer)
        if meta.kind == "categorical":
            raise GeoError(
                fix(
                    f"layer {req.layer!r} is already categorical.",
                    f"area_table stack={stack.id} layer={req.layer}, or rules on its codes",
                )
            )
        warnings: list[str] = []
        want_unit = PRESET_UNITS.get(req.preset or "")
        if want_unit and meta.unit != want_unit:
            warnings.append(f"preset {req.preset} expects {want_unit}; layer unit is {meta.unit!r}")
        check_cancel()
        valid = stack.valid(req.mask, layers=[req.layer], extra=extra)
        out = classify_layer(
            stack.band(req.layer),
            valid,
            rules=req.rules or None,
            thresholds=req.thresholds or None,
            labels=req.labels or None,
            preset=req.preset,
            method=req.method,
        )
        warnings.extend(out.notes)
        check_cancel()
        qa, qa_data = zonal.aoi_qa(stack)
        params = {
            "layer": req.layer,
            "rules": req.rules,
            "thresholds": req.thresholds,
            "labels": req.labels,
            "preset": req.preset,
            "method": req.method,
            "mask": req.mask,
            "mask_with": [s.id for s in extra],
        }
        derived = stack.derive(
            op="classify",
            layers=[
                {
                    "name": "class",
                    "kind": "categorical",
                    "classes": out.classes,
                    "processing": [f"classify {req.layer}"],
                }
            ],
            data=out.data[None],
            params=params,
            provenance={"tool": "classify", "arguments": params, "inputs": [{"stack": stack.id}]},
            name=req.name or None,
            qa=qa,
            qa_data=qa_data,
            overwrite=req.overwrite,
            warnings=warnings,
        )
        table = [
            {"class": r["name"], "px": r["px"], "ha": r["ha"], "pct_valid": r["pct_valid"]}
            for r in zonal.class_table(derived, "class", valid=valid)
        ]
        cnt = zonal.counts(derived, valid & np.isfinite(out.data))
        csv_path = zonal.save_csv(table, derived.analysis_dir / "tables", f"{derived.id}-classes")
        csv_rel = store.relative(csv_path)
        how = _how(req, out.threshold)
        res = GeoResult(
            summary=[
                f"classified {stack.id}:{req.layer} ({how}) -> stack {derived.id} layer class",
                f"valid_px {cnt.valid_px}/aoi_px {cnt.aoi_px} (valid_fraction "
                f"{sig4(cnt.valid_fraction)})",
                zonal.table_line(min(len(table), _SHOWN), len(table), csv_rel),
            ],
            table=table[:_SHOWN],
            warnings=warnings,
            artifacts=[csv_path],
            numbers={"valid_px": cnt.valid_px, "aoi_px": cnt.aoi_px, "threshold": out.threshold},
            next=f"area_table stack={derived.id} layer=class",
            table_name="class_table",
        )
        result: ToolResult = _geo.result_to_tool_result(
            res, store, tool=self.name, arguments=arguments, stack_ids=[stack.id, derived.id]
        )
        return result


def _how(req: ClassifyInput, threshold: float | None) -> str:
    from atlas.geo.results import sig4

    if req.preset:
        return f"preset {req.preset}"
    if req.method:
        return f"otsu threshold {sig4(threshold)}"
    if req.thresholds:
        return "thresholds " + ",".join(sig4(t) for t in req.thresholds)
    return f"{len(req.rules or [])} rules"
