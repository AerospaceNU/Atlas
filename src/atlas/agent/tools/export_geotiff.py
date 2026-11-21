"""export_geotiff: write stack layers as GeoTIFF, CSV or GeoJSON under analysis/exports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import BaseModel, Field, ValidationError

from atlas.agent.contracts import Tool, ToolResult

if TYPE_CHECKING:
    from atlas.agent.artifacts import LocalArtifactStore


class ExportGeotiffInput(BaseModel):
    stack: str = Field(description="Stack id or 'latest'.")
    layers: list[str] = Field(
        min_length=1, description="Layer names; geojson takes one class layer."
    )
    format: Literal["geotiff", "geojson", "csv"] = "geotiff"
    date: str | None = Field(default=None, description="YYYYMMDD for name@date layers.")
    path: str | None = Field(default=None, description="Explicit output path (needs approval).")


class ExportGeotiffTool(Tool):
    """Export layers for GIS use; the file keeps the stack CRS and transform."""

    name = "export_geotiff"
    description = (
        "Export stack layers as a GeoTIFF (default), CSV of clear pixels, or GeoJSON polygons "
        "of one class layer. Saved under analysis/exports unless path is given."
    )
    domain: ClassVar[str] = "satellite"

    @property
    def input_model(self) -> type[BaseModel]:
        return ExportGeotiffInput

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        from atlas.agent.tools import _geo
        from atlas.geo import export
        from atlas.geo.errors import GeoCancelled, GeoError
        from atlas.geo.results import GeoResult, sig4

        try:
            req = ExportGeotiffInput.model_validate(arguments)
        except ValidationError as exc:
            first = exc.errors()[0]
            where = ".".join(str(p) for p in first["loc"])
            return ToolResult(text=f"invalid argument {where}: {first['msg']}"[:400])
        try:
            stack = _geo.stack_from_args(store, req.stack)
            adir = _geo.analysis_dir(store)
            target = (
                store.resolve(req.path)
                if req.path
                else export.default_export_path(adir, stack.id, req.format, req.layers)
            )
            if req.format == "geotiff":
                info = export.write_geotiff(stack, req.layers, target, date=req.date)
            elif req.format == "csv":
                info = export.write_csv(stack, req.layers, target, date=req.date)
            else:
                info = export.write_geojson(stack, req.layers[0], target, date=req.date)
        except GeoCancelled:
            return ToolResult(text="cancelled")
        except GeoError as exc:
            return _geo.error_result(exc)
        lines = [
            f"exported {req.format} via {info['backend']}: {store.relative(info['path'])} "
            f"({sig4(info['bytes'])} bytes)",
            f"layers {', '.join(info['layers'])}; EPSG:{info['epsg']}"
            + (f"; {info['dtype']} nodata {info['nodata']}" if "dtype" in info else "")
            + (f"; {sig4(info['rows'])} rows" if "rows" in info else "")
            + (f"; {sig4(info['features'])} polygons" if "features" in info else ""),
        ]
        if info["backend"] == "atlas-cog fallback":
            lines.append("no band descriptions (rasterio not installed; uv sync --extra geo)")
        sidecar = info["path"].with_name(info["path"].name + ".json")
        result = GeoResult(
            summary=lines,
            numbers={"bytes": info["bytes"]},
            artifacts=[info["path"], sidecar],
            next="cite the exported path; open it in a GIS or read it with rasterio",
        )
        return _geo.result_to_tool_result(
            result, store, tool=self.name, arguments=arguments, stack_ids=[stack.id]
        )
