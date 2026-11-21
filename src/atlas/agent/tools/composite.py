"""Agent tool that builds a per-pixel composite of a multi-date analysis stack."""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field, ValidationError

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import Tool, ToolResult


class CompositeInput(BaseModel):
    stack: str = Field(description="Multi-date stack id, path or 'latest'.")
    method: Literal["median", "mean", "max", "clearest"] = Field(
        default="median", description="Per-pixel reduction over clear dates."
    )
    layers: list[str] | None = Field(
        default=None, description="Base layer names, e.g. red. Default: all continuous, max 8."
    )
    mask: Literal["clear", "none"] = Field(
        default="clear", description="clear drops cloud, shadow, nodata dates per pixel."
    )
    mask_with: list[str] | None = Field(
        default=None, description="Same-grid stack ids whose QA planes also mask pixels."
    )
    min_count: int = Field(
        default=1, ge=1, le=50, description="Pixels with fewer clear dates become NaN."
    )
    name: str | None = Field(default=None, description="Output stack id, [a-z0-9_.-].")
    overwrite: bool = Field(default=False, description="Replace a stack with the same name.")


class CompositeTool(Tool):
    """Median, mean, max or clearest composite of a multi-date stack, with a count layer."""

    name = "composite"
    description = (
        "Composite a multi-date stack per pixel over clear dates into a new stack with a "
        "count layer. Needs prepare_stack with 2+ scenes."
    )
    domain: ClassVar[str] = "satellite"
    trust: ClassVar[Literal["default", "opt_in"]] = "default"

    @property
    def input_model(self) -> type[BaseModel]:
        return CompositeInput

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        from atlas.agent.tools import _geo
        from atlas.geo import temporal
        from atlas.geo.errors import GeoCancelled, GeoError

        try:
            request = CompositeInput.model_validate(arguments)
        except ValidationError as exc:
            err = exc.errors()[0]
            where = ".".join(str(x) for x in err["loc"])
            return ToolResult(text=f"invalid arguments: {where}: {err['msg']}")
        try:
            stack = _geo.stack_from_args(store, request.stack)
            extra = [_geo.stack_from_args(store, ref) for ref in request.mask_with or []]
            product, result = temporal.build_composite(
                stack,
                request.layers,
                request.method,
                mask=request.mask,
                extra=extra,
                min_count=request.min_count,
                name=request.name,
                overwrite=request.overwrite,
                analysis_dir=_geo.analysis_dir(store),
            )
            geo = _summary(request, stack, product, result, extra)
            return _geo.result_to_tool_result(
                geo,
                store,
                tool=self.name,
                arguments=request.model_dump(),
                stack_ids=[stack.id, product.id, *[e.id for e in extra]],
            )
        except GeoCancelled as exc:
            cap = int(temporal.TIME_CAP_S)
            return ToolResult(text=f"time cap {cap} s" if "deadline" in str(exc) else "cancelled")
        except GeoError as exc:
            return _geo.error_result(exc)


def _summary(
    request: CompositeInput, stack: Any, product: Any, result: Any, extra: list[Any]
) -> Any:
    from atlas.geo.results import GeoResult, sig4

    dates = result.dates
    area_ha = stack.pixel_area_m2 / 1e4
    grid = product.grid
    summary = [
        f"composite {result.method} of {len(dates)} dates {dates[0]}..{dates[-1]}, "
        f"mask={request.mask}",
        f"stack {product.id}: {len(result.layers)} layer(s) + count, {grid.height}x{grid.width}, "
        f"{sig4(grid.xres)} m, {grid.crs}",
        f"clear dates per px min/median/max: {result.count_min}/{sig4(result.count_median)}/"
        f"{result.count_max}",
        f"valid output {sig4(100 * result.valid_fraction)}%: {result.valid_px} of "
        f"{result.aoi_px} px = {sig4(result.valid_px * area_ha)} of "
        f"{sig4(result.aoi_px * area_ha)} ha",
    ]
    if extra:
        summary.append("mask_with: " + ", ".join(f"{e.id} ({_scope(e)})" for e in extra))
    return GeoResult(
        summary=summary,
        table=[dict(row) for row in result.layers],
        table_name="layers",
        table_limit=12,
        warnings=list(result.warnings),
        numbers={
            "dates": len(dates),
            "valid_fraction": result.valid_fraction,
            "zero_count_px": result.zero_count_px,
            "stack": product.id,
        },
        next=f"inspect_stack stack={product.id}; band_math stack={product.id} index=ndvi",
    )


def _scope(stack: Any) -> str:
    """Say whether an extra stack's planes match by date or apply to every date."""
    dated = any(q.name[-9:-8] == "@" and q.name[-8:].isdigit() for q in stack.meta.qa)
    return "planes matched by date" if dated else "planes apply to every date"
