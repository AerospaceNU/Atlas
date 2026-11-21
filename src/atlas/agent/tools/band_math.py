from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import Tool, ToolResult

_ALIAS_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_MAX_BANDS_SHOWN = 8


class BandMathInput(BaseModel):
    stack: str = Field(description="Primary stack id, path or 'latest'.")
    expression: str | None = Field(default=None, description="Band math over layer names.")
    index: str | None = Field(default=None, description="Catalogue index instead of expression.")
    inputs: dict[str, str] | None = Field(
        default=None, description="alias to stack id; use alias.layer in the expression."
    )
    name: str | None = Field(default=None, description="Output stack id [a-z0-9_.-].")
    unit: str | None = Field(default=None, description="Output unit label.")
    mask: Literal["clear", "none"] = Field(default="clear", description="Pixels to keep.")
    mask_with: list[str] | None = Field(
        default=None, description="Stack ids whose QA planes also mask (same grid)."
    )
    overwrite: bool = Field(default=False, description="Replace an existing stack of that name.")

    @model_validator(mode="after")
    def _one_of(self) -> BandMathInput:
        if (self.expression is None) == (self.index is None):
            raise ValueError("pass exactly one of expression or index")
        return self


class BandMathTool(Tool):
    """Compute one derived layer (or one per date) into a NEW stack.

    The expression is parsed and interpreted by ``atlas.geo.expr``; no Python is executed.
    Output is NaN where any used input is non-finite, so comparisons never become silent 0.
    """

    name = "band_math"
    description = (
        "Band math into a new stack: expression like '(nir-red)/(nir+red)' or index=ndvi|ndwi|"
        "mndwi|nbr|ndmi|evi|savi|bsi|ndbi|ndsi|vv_vh_ratio (every date). Grammar: layer names, "
        "alias.layer, numbers, + - * / **, unary -, < <= > >= == !=, & | ~, abs sqrt log log10 "
        "exp min max clip where isnan."
    )
    domain = "satellite"

    @property
    def input_model(self) -> type[BaseModel]:
        return BandMathInput

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        request = BandMathInput.model_validate(arguments)
        from atlas.agent.tools import _geo
        from atlas.geo.errors import GeoError

        try:
            return _band_math(request, store)
        except GeoError as exc:
            return _geo.error_result(exc)


@dataclass
class _Item:
    """One output layer: its expression and the (stack, layer) behind every name."""

    layer: str
    expression: str
    refs: dict[str, tuple[Any, str]]
    datetime: str | None = None
    sensor: str | None = None
    kinds: dict[str, str] = field(default_factory=dict)


def _open_inputs(request: BandMathInput, store: LocalArtifactStore, primary: Any) -> dict[str, Any]:
    from atlas.agent.tools import _geo
    from atlas.geo.errors import ExprError, GridMismatch, fix
    from atlas.geo.expr import FUNCTIONS

    opened: dict[str, Any] = {}
    for alias, ref in (request.inputs or {}).items():
        if not _ALIAS_RE.match(alias) or alias in FUNCTIONS:
            raise ExprError(fix(f"bad alias {alias[:20]!r}.", "use a plain identifier like 'a'"))
        other = _geo.stack_from_args(store, ref)
        if not other.grid.same_as(primary.grid):
            raise GridMismatch(
                fix(
                    f"input '{alias}' ({other.id}) is not on the grid of '{primary.id}'.",
                    "prepare_stack both stacks with the same crs, resolution and bbox",
                )
            )
        opened[alias] = other
    return opened


def _plan_index(request: BandMathInput, primary: Any) -> tuple[list[_Item], list[str]]:
    from atlas.geo.indices import bind_dates, index_spec

    assert request.index is not None
    index = request.index.strip().lower()
    spec = index_spec(index)
    bound, warnings = bind_dates(index, spec, primary)
    items = []
    for date, binding in bound:
        first = primary.layer(next(iter(binding.values())))
        items.append(
            _Item(
                layer=f"{index}@{date}" if date else index,
                expression=spec["expr"],
                refs={band: (primary, layer) for band, layer in binding.items()},
                datetime=first.datetime,
                sensor=first.sensor,
            )
        )
    return items, warnings


def _plan_expression(
    request: BandMathInput, primary: Any, others: dict[str, Any]
) -> tuple[list[_Item], list[str]]:
    from atlas.geo.errors import ExprError, fix
    from atlas.geo.expr import expr_names

    assert request.expression is not None
    refs: dict[str, tuple[Any, str]] = {}
    own = set(primary.names())
    for key in sorted(expr_names(request.expression)):
        alias, _, layer = key.partition(".")
        if key not in own and layer and alias in others:
            refs[key] = (others[alias], others[alias].layer(layer).name)
        elif key in own or not layer:
            refs[key] = (primary, primary.layer(key).name)
        else:
            raise ExprError(
                fix(f"unknown alias '{alias[:20]}' in '{key[:30]}'.", "pass inputs={alias: stack}")
            )
    first = next((m for s, m in refs.values() if s is primary), None)
    meta = primary.layer(first) if first else None
    item = _Item(
        layer=request.name or "result",
        expression=request.expression,
        refs=refs,
        sensor=meta.sensor if meta else None,
    )
    return [item], []


def _hash(request: BandMathInput, primary: Any, others: dict[str, Any], masks: list[Any]) -> str:
    from atlas.geo.stack import hash_params

    return hash_params(
        {
            "tool": "band_math",
            "primary": [primary.id, primary.meta.params_hash],
            "expression": request.expression,
            "index": request.index.strip().lower() if request.index else None,
            "inputs": {a: [s.id, s.meta.params_hash] for a, s in sorted(others.items())},
            "mask": request.mask,
            "mask_with": [[s.id, s.meta.params_hash] for s in masks],
            "unit": request.unit,
        }
    )


def _compute(
    item: _Item, request: BandMathInput, primary: Any, masks: list[Any]
) -> tuple[Any, int, bool, int]:
    """Return (plane, undefined px, boolean result, valid px before the expression)."""
    import numpy as np

    from atlas.geo.errors import ExprError, fix
    from atlas.geo.expr import eval_expr

    env = {key: stack.band(layer) for key, (stack, layer) in item.refs.items()}
    with np.errstate(all="ignore"):
        raw = eval_expr(item.expression, env)
    if raw.ndim == 0:
        raise ExprError(fix("expression reads no layer.", "use layer names like nir and red"))
    is_bool = bool(raw.dtype == np.bool_)
    out = np.asarray(raw, dtype=np.float32).copy()
    own = [layer for stack, layer in item.refs.values() if stack is primary]
    ok = primary.valid(request.mask, layers=own, extra=masks)
    for stack, layer in item.refs.values():
        if stack is not primary:
            ok &= np.isfinite(stack.band(layer))
    finite = np.isfinite(out)
    undefined = int((ok & ~finite).sum())
    out[~(ok & finite)] = np.nan
    return out, undefined, is_bool, int(ok.sum())


def _sources(items: list[_Item]) -> str:
    seen: dict[str, str] = {}
    for key, (stack, layer) in items[0].refs.items():
        band = stack.layer(layer).band
        seen[key] = f"{key}={band}" if band else key
    shown = list(seen.values())[:_MAX_BANDS_SHOWN]
    extra = len(seen) - len(shown)
    return "bands: " + " ".join(shown) + (f" +{extra}" if extra > 0 else "")


def _citation_text(items: list[_Item], request: BandMathInput) -> str:
    from atlas.geo.indices import INDEX_CATALOGUE
    from atlas.geo.provenance import citations

    found: list[str] = []
    for item in items:
        for text in citations(item.sensor) if item.sensor else []:
            if text not in found:
                found.append(text)
    if request.index:
        spec = INDEX_CATALOGUE.get(request.index.strip().lower())
        if spec:
            found.append(spec["citation"])
    return "; ".join(found)


def _band_math(request: BandMathInput, store: LocalArtifactStore) -> ToolResult:
    import numpy as np

    from atlas.agent.tools import _geo
    from atlas.geo._lazy import check_cancel
    from atlas.geo.errors import GeoError, StackError, fix
    from atlas.geo.indices import INDEX_CATALOGUE
    from atlas.geo.results import GeoResult, sig4
    from atlas.geo.stack import LayerMeta, Stack, validate_name
    from atlas.geo.stats import layer_stats, stats_row

    primary = _geo.stack_from_args(store, request.stack)
    others = _open_inputs(request, store, primary)
    masks = [_geo.stack_from_args(store, ref) for ref in request.mask_with or []]
    if request.name:
        validate_name(request.name)
    if request.index:
        items, warnings = _plan_index(request, primary)
    else:
        items, warnings = _plan_expression(request, primary, others)
    spec = INDEX_CATALOGUE.get(request.index.strip().lower()) if request.index else None
    params_hash = _hash(request, primary, others, masks)
    op = request.index.strip().lower() if request.index else "bm"
    stack_id = request.name or f"{primary.id[:40]}-{op}-{params_hash[:6]}"
    adir = _geo.analysis_dir(store)
    stored = Stack.exists(adir, stack_id)
    if stored is not None and stored != params_hash and not request.overwrite:
        raise StackError(
            fix(
                f"stack '{stack_id}' exists with different parameters.",
                "choose another name or pass overwrite=true",
            )
        )
    reused = stored == params_hash
    planes: list[Any] = []
    metas: list[Any] = []
    undefined = 0
    if reused:
        out_stack = Stack.open(stack_id, store=store)
        warnings = [*out_stack.meta.warnings]
        for meta in out_stack.meta.layers:
            planes.append(out_stack.band(meta.name))
            metas.append(meta)
    else:
        for item in items:
            check_cancel()
            plane, bad, is_bool, ok_px = _compute(item, request, primary, masks)
            if int(np.isfinite(plane).sum()) == 0:
                if len(items) == 1:
                    why = (
                        f"no valid pixels after mask={request.mask} ({ok_px} of {plane.size})"
                        if ok_px == 0
                        else f"expression is undefined at all {ok_px} valid pixels"
                    )
                    cmd = (
                        "use mask=none or inspect_stack to see which layer is empty"
                        if ok_px == 0
                        else "check the denominator with inspect_stack"
                    )
                    raise GeoError(fix(why + ".", cmd))
                warnings.append(f"{item.layer}: no valid pixels, skipped")
                continue
            undefined += bad
            unit = "bool 0/1" if is_bool else request.unit or (spec["unit"] if spec else "")
            planes.append(plane)
            metas.append(
                LayerMeta(
                    name=item.layer,
                    kind="index" if spec else "continuous",
                    unit=unit,
                    datetime=item.datetime,
                    sensor=item.sensor,
                    processing=[item.expression],
                )
            )
        if not planes:
            raise GeoError(
                fix(f"no valid pixels after mask={request.mask} in any date.", "use mask=none")
            )
        if undefined:
            warnings.append(f"{sig4(undefined)} px undefined (divide by zero) -> no data")
        cite = _citation_text(items, request)
        out_stack = Stack.create(
            adir,
            id=stack_id,
            grid=primary.grid,
            layers=metas,
            data=planes,
            provenance={
                "tool": "band_math",
                "arguments": request.model_dump(),
                "inputs": [
                    {"stack": s.id, "sha256": s.meta.data_sha256}
                    for s in [primary, *others.values()]
                ],
                "citations": [cite] if cite else [],
            },
            aoi=primary.meta.aoi,
            parents=list(dict.fromkeys([primary.id, *(s.id for s in others.values())])),
            warnings=warnings,
            params_hash=params_hash,
            overwrite=request.overwrite,
        )
    all_stats = [
        layer_stats(p, None, name=m.name, unit=m.unit) for p, m in zip(planes, metas, strict=True)
    ]
    total = all_stats[0].n_total
    head = (
        f"stack: {out_stack.id}{' (reused, unchanged)' if reused else ''}, "
        f"{len(all_stats)} layer(s), unit {metas[0].unit or 'none'}"
    )
    lines = [head, "expr: " + (items[0].expression if items else str(metas[0].processing[:1]))]
    if items:
        lines.append(_sources(items))
    if len(all_stats) == 1:
        lines.append(f"valid px: {sig4(all_stats[0].n_valid)} of {sig4(total)}")
    else:
        lines.append(f"px per layer: {sig4(total)}; valid% per layer in table")
    cite = _citation_text(items, request) if items else ""
    if cite:
        lines.append(f"cite: {cite}")
    result = GeoResult(
        summary=lines,
        table=[stats_row(s) for s in all_stats],
        table_name="layers",
        table_limit=12,
        warnings=warnings[:4],
        numbers={
            "n_layers": len(all_stats),
            "valid_px": sum(s.n_valid for s in all_stats),
            "valid_pct": float(np.mean([s.valid_pct for s in all_stats])),
        },
        next=f"classify stack={out_stack.id} layer={metas[0].name} method=otsu",
    )
    return _geo.result_to_tool_result(
        result,
        store,
        tool="band_math",
        arguments=request.model_dump(),
        stack_ids=[out_stack.id],
    )
