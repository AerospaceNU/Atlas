"""Agent tool for a zonal time series (and trend) over a multi-date analysis stack."""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field, ValidationError

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import Tool, ToolResult

OLS_CAVEAT = (
    "caveat: plain OLS on seasonal data is biased by which months are sampled; "
    "use trend=month_anomaly for >=2 years."
)
_SKETCH_ROWS = 6


class TimeSeriesInput(BaseModel):
    stack: str = Field(description="Multi-date stack id, path or 'latest'.")
    layer: str | None = Field(
        default=None, description="Base layer name, e.g. ndvi. One of layer/index."
    )
    index: str | None = Field(
        default=None, description="Index name, e.g. ndvi. One of layer/index."
    )
    zone: Any = Field(
        default=None,
        description="Lon/lat GeoJSON object or workspace .geojson path. Default: whole stack.",
    )
    stat: Literal["mean", "median", "min", "max"] = Field(
        default="mean", description="Statistic over clear zone pixels per date."
    )
    trend: Literal["auto", "ols", "month_anomaly", "none"] = Field(
        default="auto", description="auto: month_anomaly with 12+ dates over 2+ years, else ols."
    )
    mask: Literal["clear", "none"] = Field(default="clear", description="Per-date cloud mask.")
    mask_with: list[str] | None = Field(
        default=None, description="Same-grid stack ids whose QA planes also mask pixels."
    )
    chart: bool = Field(default=True, description="Also write a line chart PNG.")


class TimeSeriesTool(Tool):
    """Zonal statistic per date, CSV, chart and trend per year."""

    name = "time_series"
    description = (
        "Per-date zonal statistic of a layer or index over a multi-date stack, with CSV, "
        "chart and trend per year. Dates under 20% valid are flagged."
    )
    domain: ClassVar[str] = "satellite"
    trust: ClassVar[Literal["default", "opt_in"]] = "default"

    @property
    def input_model(self) -> type[BaseModel]:
        return TimeSeriesInput

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        from atlas.agent.tools import _geo
        from atlas.geo import temporal
        from atlas.geo.errors import GeoCancelled, GeoError, fix

        try:
            request = TimeSeriesInput.model_validate(arguments)
        except ValidationError as exc:
            err = exc.errors()[0]
            where = ".".join(str(x) for x in err["loc"])
            return ToolResult(text=f"invalid arguments: {where}: {err['msg']}")
        try:
            if (request.layer is None) == (request.index is None):
                raise GeoError(fix("pass exactly one of layer or index.", "time_series layer=ndvi"))
            stack = _geo.stack_from_args(store, request.stack)
            extra = [_geo.stack_from_args(store, ref) for ref in request.mask_with or []]
            zone_px = temporal.zone_mask(stack.grid, request.zone, root=store.root)
            points = temporal.series_points(
                stack,
                request.layer,
                request.index,
                zone_px=zone_px,
                stat=request.stat,
                mask=request.mask,
                extra=extra,
                deadline=_deadline(temporal),
            )
            geo = _analyse(request, stack, points, zone_px, store)
            return _geo.result_to_tool_result(
                geo,
                store,
                tool=self.name,
                arguments=request.model_dump(),
                stack_ids=[stack.id, *[e.id for e in extra]],
            )
        except GeoCancelled as exc:
            cap = int(temporal.TIME_CAP_S)
            return ToolResult(text=f"time cap {cap} s" if "deadline" in str(exc) else "cancelled")
        except GeoError as exc:
            return _geo.error_result(exc)


def _deadline(temporal: Any) -> float:
    import time

    return time.monotonic() + float(temporal.TIME_CAP_S)


def _resolve_trend(request: TimeSeriesInput, points: list[Any], temporal: Any) -> Any:
    """Return the TrendResult, or None when none was asked for or auto found too little."""
    from atlas.geo.errors import GeoError

    if request.trend == "none":
        return None
    kind = temporal.choose_trend(points) if request.trend == "auto" else request.trend
    try:
        return temporal.fit_trend(points, kind)
    except GeoError:
        if request.trend == "auto":
            return None
        raise


def _write_csv(points: list[Any], path: Any) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "value", "n_valid", "valid_pct", "flag"])
        for p in points:
            writer.writerow(
                [p.date, p.value, p.n_valid, p.valid_pct, "low_valid" if p.flagged else ""]
            )


def _chart(points: list[Any], path: Any, what: str, trend: Any) -> Any:
    """Draw with S-06 charts.line_chart; fall back to the minimal Pillow chart."""
    from atlas.geo import temporal

    title = f"{what} by date"
    try:
        from atlas.geo import charts

        xs = [temporal.decimal_year(p.date) for p in points]
        linear = "linear" if trend is not None and trend.kind == "ols" else None
        return charts.line_chart(
            xs, {what: [p.value for p in points]}, title=title, path=path, trend=linear
        )
    except (ImportError, AttributeError, TypeError, ValueError):
        return temporal.draw_series_png(points, path, title=title)


def _sketch(points: list[Any]) -> list[dict[str, Any]]:
    if len(points) <= _SKETCH_ROWS:
        picks = list(range(len(points)))
    else:
        step = (len(points) - 1) / (_SKETCH_ROWS - 1)
        picks = sorted({round(i * step) for i in range(_SKETCH_ROWS)})
    return [
        {"date": points[i].date, "value": points[i].value, "valid_pct": points[i].valid_pct}
        for i in picks
    ]


def _extremes(points: list[Any]) -> list[str]:
    import math

    from atlas.geo.results import sig4

    finite = [p for p in points if math.isfinite(p.value)]
    if not finite:
        return ["no date has a usable value"]
    lo, hi = min(finite, key=lambda p: p.value), max(finite, key=lambda p: p.value)
    return [
        f"first {finite[0].date} {sig4(finite[0].value)}; last {finite[-1].date} "
        f"{sig4(finite[-1].value)}",
        f"min {lo.date} {sig4(lo.value)}; max {hi.date} {sig4(hi.value)}",
    ]


def _analyse(
    request: TimeSeriesInput, stack: Any, points: list[Any], zone_px: Any, store: LocalArtifactStore
) -> Any:
    from atlas.agent.tools._geo import _rel
    from atlas.geo import paths, temporal
    from atlas.geo.results import GeoResult, sig4
    from atlas.geo.stack import hash_params

    what = request.layer or request.index or "value"
    trend = _resolve_trend(request, points, temporal)
    adir = paths.analysis_dir(store, ensure=True)
    digest = hash_params(
        [stack.meta.params_hash, what, request.stat, request.mask, request.mask_with,
         str(request.zone), request.trend]
    )[:6]  # fmt: skip
    stem = f"{stack.id[:40]}-{what}-{request.stat}-{digest}"
    csv_path = adir / "tables" / f"{stem}.csv"
    _write_csv(points, csv_path)
    artifacts = [csv_path]
    if request.chart:
        png = _chart(points, adir / "figures" / f"{stem}.png", what, trend)
        _sidecar(png, stack, what, request, points)
        artifacts.append(png)
    flagged = [p for p in points if p.flagged]
    n_zone = points[0].n_zone if points else 0
    ha = n_zone * stack.pixel_area_m2 / 1e4
    summary = [
        f"{request.stat} of {what}: {len(points)} dates {points[0].date}..{points[-1].date}; "
        f"zone {n_zone} px = {sig4(ha)} ha",
        *_extremes(points),
    ]
    warnings: list[str] = []
    if flagged:
        warnings.append(
            f"{len(flagged)} of {len(points)} dates under 20% valid are flagged and excluded "
            "from the trend"
        )
    summary.append(_trend_line(request, trend, len(points)))
    if trend is not None and trend.kind == "ols":
        summary.append(OLS_CAVEAT)
    summary.append(f"csv: {_rel(store, csv_path)} ({len(points)} rows)")
    return GeoResult(
        summary=summary,
        table=_sketch(points),
        table_name="series",
        table_limit=_SKETCH_ROWS,
        warnings=warnings,
        artifacts=artifacts,
        numbers={
            "dates": len(points),
            "flagged": len(flagged),
            "trend_kind": trend.kind if trend else None,
            "slope_per_year": trend.slope_per_year if trend else None,
        },
        next="change_detect or composite; the full series is in the csv",
    )


def _trend_line(request: TimeSeriesInput, trend: Any, n_dates: int) -> str:
    from atlas.geo.results import sig4

    if trend is None:
        why = "none requested" if request.trend == "none" else "needs >=3 usable dates"
        return f"trend: n/a ({why})"
    used = f"auto -> {trend.kind}" if request.trend == "auto" else trend.kind
    return (
        f"trend ({used}): {sig4(trend.slope_per_year)} per year, n={trend.n}, "
        f"r2={sig4(trend.r2)}, excluded {trend.excluded}"
    )


def _sidecar(png: Any, stack: Any, what: str, request: TimeSeriesInput, points: list[Any]) -> None:
    import json
    from pathlib import Path

    path = Path(png)
    side = path.with_suffix(".json")
    side.write_text(
        json.dumps(
            {
                "schema": "atlas.figure/1",
                "stack_id": stack.id,
                "layers": [what],
                "stat": request.stat,
                "dates": [points[0].date, points[-1].date],
                "png_bytes": path.stat().st_size,
            }
        ),
        "utf-8",
    )
