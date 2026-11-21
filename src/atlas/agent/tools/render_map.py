"""render_map: draw a stack layer, composite, class overlay or comparison as a PNG figure."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, Field, ValidationError, field_validator

from atlas.agent.contracts import Tool, ToolResult

if TYPE_CHECKING:
    from atlas.agent.artifacts import LocalArtifactStore


class RenderMapInput(BaseModel):
    stack: str = Field(description="Stack id or 'latest'.")
    layers: list[str] = Field(default_factory=list, description="One layer, or three for RGB.")
    composite: str = Field(
        default="", description="truecolor, falsecolor or swir; replaces layers."
    )
    stretch: str = Field(default="p2-98", description="p2-98, minmax or 'lo,hi'.")
    colormap: str = Field(default="", description="sequential, diverging or categorical.")
    overlay: str = Field(
        default="", description="Class layer outlined on top; 'fill:<layer>' fills."
    )
    compare: str = Field(default="", description="Second stack id on the same grid.")
    date: str = Field(default="", description="YYYYMMDD for name@date layers.")
    legend: bool = True
    scale_bar: bool = True
    max_side: int = Field(default=800, ge=64, le=1200, description="Longest side in px.")
    attach: bool = Field(default=True, description="Show the PNG to you; false keeps it on disk.")
    path: str = Field(default="", description="Explicit PNG path (needs approval).")

    @field_validator("composite")
    @classmethod
    def _composite(cls, value: str) -> str:
        if value not in {"", "truecolor", "falsecolor", "swir"}:
            raise ValueError("composite must be truecolor, falsecolor or swir")
        return value

    @field_validator("colormap")
    @classmethod
    def _colormap(cls, value: str) -> str:
        if value not in {"", "sequential", "diverging", "categorical"}:
            raise ValueError("colormap must be sequential, diverging or categorical")
        return value


class RenderMapTool(Tool):
    """Render a figure; the PNG is always saved, and numbers come from the text, not the image."""

    name = "render_map"
    description = (
        "Render a stack layer, 3-layer or preset composite, class overlay or two-stack "
        "comparison to PNG with legend, scale bar and a text grid. Quote areas from the "
        "text, never from the picture."
    )
    domain: ClassVar[str] = "satellite"

    @property
    def input_model(self) -> type[BaseModel]:
        return RenderMapInput

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        from atlas.agent.tools import _geo
        from atlas.geo import render
        from atlas.geo.errors import GeoCancelled, GeoError
        from atlas.geo.results import GeoResult

        try:
            req = RenderMapInput.model_validate(arguments)
        except ValidationError as exc:
            first = exc.errors()[0]
            where = ".".join(str(p) for p in first["loc"])
            return ToolResult(text=f"invalid argument {where}: {first['msg']}"[:400])
        try:
            stack = _geo.stack_from_args(store, req.stack)
            other = _geo.stack_from_args(store, req.compare) if req.compare else None
            res = render.render_map(
                stack,
                req.layers,
                composite=req.composite or None,
                stretch=req.stretch,
                colormap=req.colormap or None,
                overlay=req.overlay or None,
                compare=other,
                legend=req.legend,
                scale_bar=req.scale_bar,
                max_side=req.max_side,
                attach=req.attach,
                path=store.resolve(req.path) if req.path else None,
                date=req.date or None,
            )
        except GeoCancelled:
            return ToolResult(text="cancelled")
        except GeoError as exc:
            return _geo.error_result(exc)
        figure = res.figure
        summary = [res.text]
        if figure is not None:
            summary.append(f"figure: {store.relative(figure)}")
        numbers: dict[str, Any] = {"valid_fraction": res.valid_fraction}
        if res.lo_hi is not None:
            numbers.update(lo=res.lo_hi[0], hi=res.lo_hi[1])
        result = GeoResult(
            summary=summary,
            numbers=numbers,
            artifacts=[p for p in (res.png, res.sidecar) if p is not None],
            next="cite the figure path; quote areas from numbers, not pixels",
        )
        return _geo.result_to_tool_result(
            result,
            store,
            tool=self.name,
            arguments=arguments,
            stack_ids=[stack.id],
            attach=req.attach and render.attach_enabled(),
        )
