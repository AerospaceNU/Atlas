"""Agent tool: change between two dates of one layer."""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import Tool, ToolDefinition, ToolResult
from atlas.agent.tools._schema import slim_definition

_SHOWN = 12


class ChangeDetectInput(BaseModel):
    stack: str = Field(description="Stack id (multi-date, or pre).")
    layer: str = Field(description="Layer name, e.g. nbr.")
    method: Literal["difference", "ratio", "log_ratio_db", "transitions"] = Field(
        description="transitions: categorical layers."
    )
    stack_b: str = Field(default="", description="Post stack on the same grid.")
    delta: float = Field(default=0, description="Size (dB or abs), or ratio factor >1.")
    rules: list[dict[str, Any]] = Field(default_factory=list, description="[{name,min,max}]")
    min_blob_px: int = Field(default=0, ge=0, description="Drop smaller blobs.")
    pre: str = Field(default="", description="YYYYMMDD, default earliest.")
    post: str = Field(default="", description="YYYYMMDD, default latest.")
    mask: str | list[str] = Field(default="clear", description="clear, none or semantics.")
    mask_with: list[str] = Field(default_factory=list, description="QA stacks.")
    name: str = Field(default="", description="Output id.")
    overwrite: bool = False


class ChangeDetectTool(Tool):
    """Detect gain, loss or class transitions between two dates of one layer.

    Only pixels valid in both dates count. The output stack carries ``change`` (continuous)
    and ``change_class`` (categorical).
    """

    name = "change_detect"
    description = (
        "Compare a layer between two dates (multi-date stack, or stack + stack_b): "
        "difference (post minus pre; dNBR is pre minus post), ratio, log_ratio_db (needs delta) "
        "or transitions. Reports gain/loss/stable hectares."
    )
    trust: ClassVar[Literal["default", "opt_in"]] = "default"
    domain: ClassVar[str] = "satellite"

    @property
    def input_model(self) -> type[BaseModel]:
        return ChangeDetectInput

    def definition(self) -> ToolDefinition:
        """Return the advertised schema without null branches or the model docstring."""
        return slim_definition(super().definition())

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        req = ChangeDetectInput.model_validate(arguments)
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
        req: ChangeDetectInput,
        arguments: dict[str, Any],
        store: LocalArtifactStore,
        _geo: Any,
    ) -> ToolResult:
        import numpy as np

        from atlas.geo import zonal
        from atlas.geo._lazy import check_cancel
        from atlas.geo.change import detect_change, pair_valid, resolve_pair
        from atlas.geo.results import GeoResult, sig4, write_csv
        from atlas.geo.stack import Stack, hash_params

        stack = _geo.stack_from_args(store, req.stack)
        stack_b = _geo.stack_from_args(store, req.stack_b) if req.stack_b else None
        extra = [_geo.stack_from_args(store, ref) for ref in req.mask_with]
        pair = resolve_pair(
            stack, req.layer, stack_b=stack_b, pre=req.pre or None, post=req.post or None
        )
        delta = req.delta or None
        check_cancel()
        valid = pair_valid(pair, req.mask, extra)
        names = None
        if req.method == "transitions":
            names = pair.pre_stack.layer(pair.pre_layer).classes
        out = detect_change(
            np.array(pair.pre_stack.band(pair.pre_layer)),
            np.array(pair.post_stack.band(pair.post_layer)),
            valid,
            method=req.method,
            pixel_area_m2=stack.pixel_area_m2,
            delta=delta,
            rules=req.rules or None,
            min_blob_px=req.min_blob_px,
            categorical_names=names,
        )
        check_cancel()
        params = {
            "layer": req.layer,
            "method": req.method,
            "delta": delta,
            "rules": req.rules,
            "min_blob_px": req.min_blob_px,
            "pre": pair.pre_label,
            "post": pair.post_label,
            "mask": req.mask,
            "mask_with": [s.id for s in extra],
        }
        parents = [stack.id] + ([stack_b.id] if stack_b else [])
        digest = hash_params(
            {
                "parents": [stack.meta.params_hash]
                + ([stack_b.meta.params_hash] if stack_b else []),
                "op": "change",
                "params": params,
            }
        )
        layers: list[dict[str, Any]] = []
        planes: list[Any] = []
        if out.value is not None:
            layers.append({"name": "change", "kind": "continuous", "processing": [req.method]})
            planes.append(out.value)
        layers.append(
            {
                "name": "change_class",
                "kind": "categorical",
                "classes": out.class_names,
                "processing": [req.method],
            }
        )
        planes.append(out.classes)
        qa, qa_data = zonal.aoi_qa(stack)
        derived = Stack.create(
            stack.analysis_dir,
            id=req.name or f"{stack.id[:40]}-change-{digest[:6]}",
            grid=stack.grid,
            layers=layers,
            data=np.stack(planes),
            qa=qa,
            qa_data=qa_data,
            provenance={"tool": "change_detect", "arguments": params, "inputs": parents},
            aoi=stack.meta.aoi,
            parents=parents,
            warnings=out.notes,
            params_hash=digest,
            overwrite=req.overwrite,
        )
        tables = derived.analysis_dir / "tables"
        artifacts = []
        warnings = [*pair.notes, *out.notes]
        cnt = zonal.counts(derived, valid & np.isfinite(out.classes))
        if out.matrix is not None and out.labels is not None:
            full = [
                {"from": lab, **{to: int(out.matrix[i, j]) for j, to in enumerate(out.labels)}}
                for i, lab in enumerate(out.labels)
            ]
            path = tables / f"{derived.id}-transitions.csv"
            write_csv(full, path)
            artifacts.append(path)
            rows, name = out.rows, "transitions"
            csv_path = path
        else:
            rows, name = out.rows, "classes"
            csv_path = zonal.save_csv(rows, tables, f"{derived.id}-{name}")
            artifacts.append(csv_path)
        csv_rel = store.relative(csv_path)
        summary = [
            f"change {req.method} {pair.pre_label} -> {pair.post_label} on {req.layer}"
            + (f" delta={sig4(delta)}" if delta is not None else "")
            + f" -> stack {derived.id}",
            f"valid in both dates: {out.valid_px}/aoi_px {cnt.aoi_px} (valid_fraction "
            f"{sig4(out.valid_px / cnt.aoi_px if cnt.aoi_px else 0.0)})",
        ]
        if req.min_blob_px > 1:
            summary.append(
                f"dropped {out.dropped_blobs} blobs ({out.dropped_px} px) under "
                f"min_blob_px={req.min_blob_px}"
            )
        summary.append(zonal.table_line(min(len(rows), _SHOWN), len(rows), csv_rel))
        res = GeoResult(
            summary=summary,
            table=rows[:_SHOWN],
            warnings=warnings,
            artifacts=artifacts,
            numbers={
                "valid_px": out.valid_px,
                "dropped_blobs": out.dropped_blobs,
                "dropped_px": out.dropped_px,
            },
            next=f"area_table stack={derived.id} layer=change_class",
            table_name="class_table",
        )
        result: ToolResult = _geo.result_to_tool_result(
            res, store, tool=self.name, arguments=arguments, stack_ids=[*parents, derived.id]
        )
        return result
