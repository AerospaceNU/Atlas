from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Sequence
from dataclasses import replace
from datetime import date
from math import cos, radians
from pathlib import Path
from typing import Any, ClassVar

import httpx  # to download the data files
from pydantic import BaseModel, Field, field_validator, model_validator

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import Tool, ToolDefinition, ToolResult
from atlas.agent.layout import project_data_dir
from atlas.cli.download import DownloadOutcome, DownloadReport, download_assets
from atlas.cli.estimate import estimate_pull, render_pull_estimate
from atlas.compile import SceneCatalog, SourceScenes, aggregate
from atlas.compile.select import (
    Period,
    SelectMode,
    filter_orbit,
    orbit_of,
    parse_orbit,
    select_scenes,
    sort_clearest,
)
from atlas.data.base import BBox, Polygon, PullRequest, Scene, SceneKind
from atlas.data.profiles import AssetNotFound, list_aliases, resolve_asset
from atlas.data.registry import SOURCES, sources

SEARCH_CAP = 200  # scenes asked of each source when a select mode picks among them
MAX_WINDOW_PIXELS = 36_000_000  # memory guard for one decoded window, not a download limit
CHUNK_SCENES = 8  # scenes per download_assets call, so cancel and the clock are checked between
WALL_CLOCK_S = 600.0  # cap for one download call of the tool
PER_PERIOD_CAP = 120  # total scenes a per_month / per_period selection may return
LISTED_PER_SOURCE = 12  # scene lines shown per source before the per-year digest
TEXT_CAP = 12_000
_SEARCH_ONLY_PREFIXES = ("firms_", "gibs_")
_ITEMSIZE = {"uint8": 1, "int8": 1, "int16": 2, "uint16": 2, "int32": 4, "uint32": 4}
_BOUND_KEYS = ("maximum", "minimum", "exclusiveMinimum", "maxItems", "minItems", "format")
_ORBIT_STATE = {"ascending": "asc", "descending": "desc"}


# pydantic model for the inputs to the tool
class GetDataToolInput(BaseModel):
    data_type: SceneKind
    start_date: date = Field(description="Optical needs a week of margin.")
    end_date: date
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    radius_km: float = Field(default=2.0, gt=0.0, le=100.0, description="Half-width, km.")
    bbox: BBox | None = Field(default=None, description="{west,south,east,north}")
    polygon: list[list[float]] | None = Field(
        default=None, min_length=4, max_length=500, description="[lon, lat] ring."
    )
    sources: list[str] | None = Field(default=None, description="Registry keys; default all.")
    max_cloud_cover: float | None = Field(default=None, ge=0.0, le=100.0)
    limit: int = Field(default=3, ge=1, le=30)
    select: SelectMode | None = Field(default=None, description="Choose among up to 200.")
    period: Period = Field(default="month")
    orbit: str | None = Field(default=None, description="SAR: 8, ascending, 8:descending.")
    assets: list[str] | None = Field(
        default=None, max_length=12, description="Aliases to cut, e.g. red, nir."
    )
    include_qa: bool = True
    download: bool = Field(default=False, description="True pulls.")

    @field_validator("polygon")
    @classmethod
    def _check_vertices(cls, ring: list[list[float]] | None) -> list[list[float]] | None:
        if ring is not None and any(len(vertex) != 2 for vertex in ring):
            raise ValueError("polygon vertices must be [lon, lat] pairs")
        return ring

    @model_validator(mode="after")
    def check_location(self) -> GetDataToolInput:
        """Require one location: lat/long, bbox, or polygon (bbox may accompany a polygon)."""
        has_point = self.latitude is not None and self.longitude is not None
        if self.polygon is not None and (self.latitude is not None or self.longitude is not None):
            raise ValueError(
                "Cannot include a polygon together with latitude and longitude. "
                "Give the polygon, optionally with a bbox that contains it."
            )
        if has_point and self.bbox:
            raise ValueError(
                "Cannot include both latitude and longitude coordinates and bounding box coordinates. Please include one or the other, not both."
            )
        if not has_point and not self.bbox and self.polygon is None:
            raise ValueError(
                "Must include values for either a latitude and longitude corrdinates, or bounding box coordinates. Cannot include nothing."
            )
        if self.orbit is not None:
            if self.data_type is not SceneKind.sar:
                raise ValueError("orbit applies to data_type sar only. Fix: drop orbit")
            parse_orbit(self.orbit)
        if self.polygon is not None:
            self.build_polygon()
        return self

    def build_polygon(self) -> Polygon | None:
        """Return the validated polygon, or None when none was given."""
        if self.polygon is None:
            return None
        return Polygon(ring=[(v[0], v[1]) for v in self.polygon])

    def pull_request(self, *, limit: int) -> PullRequest:
        """Build the catalog request; a polygon alone gives the envelope as its bbox."""
        body: dict[str, Any] = {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "max_cloud_cover": self.max_cloud_cover,
            "limit": limit,
        }
        polygon = self.build_polygon()
        if polygon is not None:
            body["polygon"] = polygon
        if self.bbox is not None or polygon is None:
            body["bbox"] = self.lat_long_to_bbox()
        return PullRequest.model_validate(body)

    def lat_long_to_bbox(self) -> BBox:
        """Converts a latitude + longitude point + radius_km into bounding box coordinates."""
        if self.bbox:
            return self.bbox  # we already are using the bounding box, so return it

        # check_location guarantees both are set when there is no bbox; this tells mypy
        assert self.latitude is not None and self.longitude is not None

        lat_padding = (
            self.radius_km / 111.32
        )  # one degree of latitude is approximately 111.32 km, so we can use that to convert the radius in km to degrees of latitude
        south = max(-90.0, self.latitude - lat_padding)
        north = min(90.0, self.latitude + lat_padding)

        # A radius that reaches a pole covers every longitude. Near a pole, the
        # approximate longitude padding can also span the entire globe.
        reaches_pole = self.latitude - lat_padding <= -90.0 or self.latitude + lat_padding >= 90.0
        if reaches_pole:
            west, east = -180.0, 180.0
        else:
            long_padding = lat_padding / cos(radians(self.latitude))
            if long_padding >= 180.0:
                west, east = -180.0, 180.0
            else:
                west_edge = self.longitude - long_padding
                east_edge = self.longitude + long_padding

                # Preserve exact ±180° edges and wrap only values beyond them.
                west = (
                    (west_edge + 180.0) % 360.0 - 180.0
                    if west_edge < -180.0 or west_edge > 180.0
                    else west_edge
                )
                east = (
                    (east_edge + 180.0) % 360.0 - 180.0
                    if east_edge < -180.0 or east_edge > 180.0
                    else east_edge
                )

        return BBox(west=west, south=south, east=east, north=north)


# helpers


def _asset_rows(scene: Scene, names: Sequence[str]) -> list[tuple[str, str, float | None, bool]]:
    """Return (name, key, gsd_m, is_qa) for each name that resolves on the scene."""
    rows: list[tuple[str, str, float | None, bool]] = []
    for name in names:
        try:
            found = resolve_asset(scene, name)
        except AssetNotFound:
            continue
        gsd = found.asset.spatial_resolution or (found.band.gsd_m if found.band else None)
        is_qa = found.band is not None and found.band.kind == "qa"
        rows.append((found.alias or name, found.key, gsd, is_qa))
    return rows


def _effective_assets(scene: Scene, assets: Sequence[str], include_qa: bool) -> list[str]:
    """Return the user's assets plus the profile quality layer when it is not already there."""
    names = list(assets)
    if not include_qa:
        return names
    wanted = {name.strip().casefold() for name in names}
    keys = {key for _alias, key, _gsd, _qa in _asset_rows(scene, names)}
    for alias, key, _gsd, is_qa in list_aliases(scene):
        if is_qa:
            if alias not in wanted and key not in keys:
                names.append(alias)
            break
    return names


def _gsd_text(gsd: float | None) -> str:
    return f" ({gsd:g} m)" if gsd is not None else ""


def _assets_line(scene: Scene, assets: Sequence[str] | None, include_qa: bool) -> str:
    """Describe the assets a pull would cut, judged on the source's first scene."""
    if not assets:
        rows = list_aliases(scene)
        shown = [f"{a}={k}{_gsd_text(g)}{' qa' if q else ''}" for a, k, g, q in rows]
        listed = ", ".join(shown[:12]) + (f", +{len(shown) - 12} more" if len(shown) > 12 else "")
        base = "no assets given: legacy default asset, preview-grade, not for analysis"
        return f"{base}. Aliases: {listed}" if listed else base
    items: list[str] = []
    for name in _effective_assets(scene, assets, include_qa):
        rows = _asset_rows(scene, [name])
        if rows:
            alias, key, gsd, is_qa = rows[0]
            items.append(f"{alias}={key}{_gsd_text(gsd)}{' qa' if is_qa else ''}")
        else:
            keys = ", ".join(sorted(scene.assets)[:12])
            items.append(f"{name}: not found; keys: {keys}")
    return "assets: " + ", ".join(items)


def _window_pixels(bbox: BBox, gsd: float) -> tuple[int, int]:
    mid_lat = (bbox.south + bbox.north) / 2
    width_m = 111_320.0 * cos(radians(mid_lat)) * bbox.lon_span
    height_m = 110_574.0 * (bbox.north - bbox.south)
    return max(1, math.ceil(width_m / gsd)), max(1, math.ceil(height_m / gsd))


def _windows(scene: Scene, names: Sequence[str], bbox: BBox) -> list[tuple[str, int, int, int]]:
    """Return (alias, width px, height px, bytes) of each resolvable window on the scene."""
    out: list[tuple[str, int, int, int]] = []
    for name in names:
        try:
            found = resolve_asset(scene, name)
        except AssetNotFound:
            continue
        gsd = found.asset.spatial_resolution or (found.band.gsd_m if found.band else None)
        if gsd is None:
            continue
        width, height = _window_pixels(bbox, gsd)
        size = _ITEMSIZE.get(found.asset.data_type or "", 2)
        out.append((found.alias or name, width, height, width * height * size))
    return out


def _window_line(scene: Scene, names: Sequence[str], bbox: BBox, n_scenes: int) -> str:
    wins = _windows(scene, names, bbox)
    if not wins:
        return ""
    alias, width, height, _ = max(wins, key=lambda w: w[1] * w[2])
    total = sum(w[3] for w in wins) * n_scenes
    line = f"window ~{width}x{height} px per asset (~{total / 1e6:.3g} MB uncompressed)"
    if width * height > MAX_WINDOW_PIXELS:
        line += (
            f"; {alias} is {width * height / 1e6:.3g} MP, over the {MAX_WINDOW_PIXELS // 10**6} "
            "MP limit, so a download will be refused: shrink the AOI"
        )
    return line


def _oversized(scene: Scene, names: Sequence[str], bbox: BBox) -> str | None:
    for alias, width, height, _size in _windows(scene, names, bbox):
        if width * height > MAX_WINDOW_PIXELS:
            return (
                f"refused: window for {alias} is ~{width}x{height} px "
                f"({width * height / 1e6:.3g} MP), over the {MAX_WINDOW_PIXELS // 10**6} MP memory guard for one window. "
                "Fix: shrink bbox/polygon (or radius_km) and call again"
            )
    return None


def _cloud_text(scene: Scene) -> str:
    return "n/a" if scene.cloud_cover is None else f"{scene.cloud_cover:.0f}%"


def _scene_line(source: str, scene: Scene) -> str:
    line = f"  {source} {scene.id} {scene.datetime:%Y-%m-%d} cloud {_cloud_text(scene)}"
    rel, state = orbit_of(scene)
    if rel is not None or state is not None:
        line += f" orbit {rel or '?'} {_ORBIT_STATE.get(state or '', state or '?')}"
    return line


def _digest(source: str, scenes: Sequence[Scene]) -> str:
    years: dict[int, list[Scene]] = {}
    for scene in scenes:
        years.setdefault(scene.datetime.year, []).append(scene)
    parts = []
    for year in sorted(years):
        clouds = [s.cloud_cover for s in years[year] if s.cloud_cover is not None]
        best = f", best cloud {min(clouds):.0f}%" if clouds else ""
        parts.append(f"{year}: {len(years[year])}{best}")
    return f"  {source} by year: " + "; ".join(parts)


def _scene_lines(src: SourceScenes) -> list[str]:
    if len(src.scenes) <= LISTED_PER_SOURCE:
        return [_scene_line(src.source, s) for s in src.scenes]
    clearest = sort_clearest(src.scenes)[:LISTED_PER_SOURCE]
    return [
        _digest(src.source, src.scenes),
        f"  {src.source} {LISTED_PER_SOURCE} clearest of {len(src.scenes)}:",
        *[_scene_line(src.source, s) for s in clearest],
    ]


def _is_search_only(name: str) -> bool:
    return name.startswith(_SEARCH_ONLY_PREFIXES) or SOURCES[name].scene_kind is SceneKind.browse


def _bytes_of(outcome: DownloadOutcome) -> int:
    if outcome.bytes_fetched:
        return outcome.bytes_fetched
    return outcome.bytes_read or 0 if outcome.status == "saved" else 0


def _saved_line(store: LocalArtifactStore, outcome: DownloadOutcome) -> str:
    assert outcome.path is not None
    line = f"saved: {store.relative(outcome.path)}"
    if outcome.windowed and outcome.bytes_read:
        size = _size(outcome.bytes_read)
        line += (
            f" (windowed {size} of {_size(outcome.granule_size)})"
            if outcome.granule_size
            else (f" (windowed {size})")
        )
    return line


def _size(n: int) -> str:
    return f"{n / 1e6:.3g} MB" if n >= 100_000 else f"{n / 1e3:.3g} KB"


def _scene_status(outcomes: Sequence[DownloadOutcome], scene_id: str) -> bool:
    mine = [o for o in outcomes if o.scene_id == scene_id]
    return bool(mine) and all(o.status in {"saved", "skipped"} for o in mine)


def _data_dir(store: LocalArtifactStore) -> Path:
    """Return the workspace ``.atlas/data`` directory: pulls persist across sessions."""
    return project_data_dir(store.root)


class _Stop(Exception):
    """Internal: ends the chunk loop with a reason for the result text."""


class GetDataTool(Tool):
    name = "get_satellite_data"
    description = (
        "Search satellite catalogs for a place and dates. download=true cuts assets as chips "
        "into .atlas/data (kept across sessions); analysis needs assets."
    )
    trust = "default"
    domain: ClassVar[str] = "satellite"

    @property
    def input_model(self) -> type[BaseModel]:
        return GetDataToolInput

    def definition(self) -> ToolDefinition:
        """Return the advertised schema, compacted: no null branches, no shared definitions."""
        base = super().definition()
        return ToolDefinition(
            name=base.name, description=base.description, parameters=_compact(base.parameters)
        )

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        request = GetDataToolInput.model_validate(arguments)
        names = self._source_names(request)
        if request.download or request.assets:
            for name in names:
                if _is_search_only(name):
                    raise ValueError(f"{name} is search-only here: use `atlas data`")
        pull_request = request.pull_request(limit=SEARCH_CAP if request.select else request.limit)
        return asyncio.run(self._pull(request, pull_request, names, store))

    def _source_names(self, request: GetDataToolInput) -> list[str]:
        if request.sources:
            unknown = [name for name in request.sources if name not in SOURCES]
            if unknown:
                raise ValueError(f"Unknown source(s): {unknown}. Known: {sorted(SOURCES)}")
            mismatched = [
                name
                for name in request.sources
                if SOURCES[name].scene_kind is not request.data_type
            ]
            if mismatched:
                raise ValueError(
                    f"Source(s) {mismatched} do not provide data type {request.data_type.value}"
                )
            names = list(request.sources)
        else:
            names = [name for name, cls in SOURCES.items() if cls.scene_kind == request.data_type]
        if not names:
            raise ValueError(f"No sources for data type {request.data_type.value}")
        if "fldas" in names:
            raise ValueError(
                "fldas builds a CSV table, not files. Use `atlas data fldas landsurface` instead."
            )
        return names

    def _narrow(
        self, request: GetDataToolInput, catalog: SceneCatalog, warnings: list[str]
    ) -> SceneCatalog:
        """Apply orbit and select to copies of each source's scenes."""
        out: list[SourceScenes] = []
        total = 0
        for src in catalog.sources:
            scenes = list(src.scenes)
            if src.error is None and request.select and len(scenes) >= SEARCH_CAP:
                warnings.append(
                    f"{src.source}: search returned {SEARCH_CAP} (cap); narrow the dates"
                )
            if request.orbit and src.error is None:
                scenes, dropped = filter_orbit(scenes, request.orbit)
                missing = sum(1 for s in dropped if orbit_of(s) == (None, None))
                warnings.append(
                    f"{src.source}: orbit {request.orbit} kept {len(scenes)} of "
                    f"{len(scenes) + len(dropped)} scenes ({len(dropped)} dropped, "
                    f"{missing} without orbit data)"
                )
            if request.select and src.error is None:
                scenes = select_scenes(
                    scenes,
                    request.select,
                    limit=request.limit,
                    period=request.period,
                    max_cloud=request.max_cloud_cover,
                )
                if request.select in {"per_month", "per_period"}:
                    room = max(0, PER_PERIOD_CAP - total)
                    if len(scenes) > room:
                        warnings.append(
                            f"{src.source}: {len(scenes)} scenes selected, kept {room} "
                            f"(cap {PER_PERIOD_CAP} in total); narrow the dates"
                        )
                        scenes = scenes[:room]
                    total += len(scenes)
            out.append(src.model_copy(update={"scenes": scenes}))
        return catalog.model_copy(update={"sources": out})

    async def _pull(
        self,
        request: GetDataToolInput,
        pull_request: PullRequest,
        names: list[str],
        store: LocalArtifactStore,
    ) -> ToolResult:
        """Search, narrow, then report the estimate or download chips.

        Clients are always closed. A download has no size limit; it is bounded by a
        per-window memory guard and a wall clock, and is never approval-gated.
        """
        clients = sources(*names)
        try:
            catalog = await aggregate(pull_request, clients)
            warnings: list[str] = []
            catalog = self._narrow(request, catalog, warnings)
            if not request.download:
                return self._search_result(request, pull_request, catalog, store, warnings)
            return await self._download(request, pull_request, catalog, clients, store, warnings)
        finally:
            for client in clients.values():
                close = getattr(client, "aclose", None)
                if close is not None:
                    await close()

    def _search_result(
        self,
        request: GetDataToolInput,
        pull_request: PullRequest,
        catalog: SceneCatalog,
        store: LocalArtifactStore,
        warnings: list[str],
    ) -> ToolResult:
        estimate = estimate_pull(catalog, out_dir=_data_dir(store), bbox=pull_request.bbox)
        # the per-source assets line below replaces the estimate's alias listing
        estimate = replace(
            estimate, sources=tuple(replace(s, assets_text="") for s in estimate.sources)
        )
        lines = [*warnings, render_pull_estimate(estimate)]
        for src in catalog.sources:
            if src.error is None and src.scenes:
                first = src.scenes[0]
                lines.append(
                    f"{src.source} {_assets_line(first, request.assets, request.include_qa)}"
                )
                if request.assets:
                    effective = _effective_assets(first, request.assets, request.include_qa)
                    window = _window_line(first, effective, pull_request.bbox, len(src.scenes))
                    if window:
                        lines.append(f"{src.source} {window}")
        if request.select in {"per_month", "per_period"}:
            period = "month" if request.select == "per_month" else request.period
            lines.append(
                f"picked the clearest acquisition per {period} (newest on ties); seasons are "
                "DJF Dec-Feb, MAM Mar-May, JJA Jun-Aug, SON Sep-Nov"
            )
        lines.append("scenes:")
        for src in catalog.sources:
            lines.extend(_scene_lines(src))
        lines.append(self._next_hint(request, download=False))
        return ToolResult(text=_cap_text("\n".join(lines)))

    def _next_hint(self, request: GetDataToolInput, *, download: bool) -> str:
        if not download:
            default = ["vv", "vh"] if request.data_type is SceneKind.sar else ["red", "nir"]
            assets = request.assets or default
            return f"next: call again with download=true and assets={assets!r}".replace("'", '"')
        if request.assets:
            bands = list(request.assets)
            return f'next: prepare_stack scenes="last_pull" bands={bands!r}'.replace("'", '"')
        return "next: repeat with assets=[...] before any analysis; this pull is preview-grade"

    async def _download(
        self,
        request: GetDataToolInput,
        pull_request: PullRequest,
        catalog: SceneCatalog,
        clients: Any,
        store: LocalArtifactStore,
        warnings: list[str],
    ) -> ToolResult:
        from atlas.agent.tools import _geo
        from atlas.geo._lazy import check_cancel
        from atlas.geo.errors import GeoCancelled

        bbox = pull_request.bbox
        plan: list[tuple[SourceScenes, list[str] | None]] = []
        for src in catalog.sources:
            if src.error is not None or not src.scenes:
                continue
            first = src.scenes[0]
            effective = (
                _effective_assets(first, request.assets, request.include_qa)
                if request.assets
                else None
            )
            if effective:
                refusal = _oversized(first, effective, bbox)
                if refusal:
                    return ToolResult(text=refusal)
            plan.append((src, effective))
        adir = _geo.analysis_dir(store)

        ring = [list(v) for v in pull_request.polygon.ring] if pull_request.polygon else None
        total_scenes = sum(len(src.scenes) for src, _ in plan)
        out_dir = _data_dir(store)
        deadline = time.monotonic() + WALL_CLOCK_S
        outcomes: list[DownloadOutcome] = []
        pulled = 0
        done = 0
        stop: str | None = None
        run_id = ""
        try:
            async with httpx.AsyncClient(timeout=300.0, follow_redirects=True) as http:
                for src, effective in plan:
                    if stop:
                        break
                    for start in range(0, len(src.scenes), CHUNK_SCENES):
                        stop = self._stop_reason(check_cancel, GeoCancelled, deadline, done)
                        if stop:
                            break
                        chunk = src.scenes[start : start + CHUNK_SCENES]
                        report = await download_assets(
                            chunk,
                            out_dir,
                            http=http,
                            clients=clients,
                            bbox=bbox,
                            asset_names=effective,
                            max_pixels=MAX_WINDOW_PIXELS,
                            deadline_seconds=max(1.0, deadline - time.monotonic()),
                            polygon=ring,
                        )
                        outcomes.extend(report.outcomes)
                        pulled += sum(_bytes_of(o) for o in report.outcomes)
                        done += len(chunk)
                        statuses = {o.status for o in report.outcomes}
                        if "cancelled" in statuses:
                            stop = f"cancelled after {done} of {total_scenes} scenes"
                        elif "timeout" in statuses:
                            stop = f"stopped: 600 s cap, pulled {done} of {total_scenes} scenes"
                        if stop:
                            break
        finally:
            report = DownloadReport(outcomes=outcomes)
            run_id = self._record(
                request, pull_request, report, store, adir, warnings, pulled, stop
            )
        return ToolResult(
            text=_cap_text(
                self._download_text(request, catalog, report, store, warnings, stop, run_id)
            ),
            artifacts=[
                store.relative(path) for paths in report.saved_paths().values() for path in paths
            ],
        )

    def _stop_reason(
        self, check_cancel: Any, cancelled: type[Exception], deadline: float, done: int
    ) -> str | None:
        try:
            check_cancel()
        except cancelled:
            return f"cancelled after {done} scenes"
        if time.monotonic() >= deadline:
            return f"stopped: 600 s cap, pulled {done} scenes"
        return None

    def _record(
        self,
        request: GetDataToolInput,
        pull_request: PullRequest,
        report: DownloadReport,
        store: LocalArtifactStore,
        adir: Path,
        warnings: list[str],
        pulled: int,
        stop: str | None,
    ) -> str:
        """Append the pull to the run ledger."""
        from atlas.geo import provenance

        outputs: list[str] = []
        for paths in report.saved_paths().values():
            for path in paths:
                outputs.append(store.relative(path))
                sidecar = path.with_name(path.name + ".meta.json")
                if sidecar.is_file():
                    outputs.append(store.relative(sidecar))
        arguments = request.model_dump(mode="json", exclude_none=True)
        if request.polygon is not None:
            arguments["polygon"] = [list(v) for v in request.polygon]
        saved = sum(1 for o in report.outcomes if o.status == "saved")
        failed = sum(1 for o in report.outcomes if o.status in {"failed", "no_asset"})
        return provenance.append_run(
            adir,
            {
                "tool": self.name,
                "arguments": arguments,
                "outputs": outputs,
                "summary": {
                    "saved": saved,
                    "skipped": sum(1 for o in report.outcomes if o.status == "skipped"),
                    "failed": failed,
                    "bbox": pull_request.bbox.model_dump(),
                },
                "warnings": [*warnings, *([stop] if stop else [])],
                "bytes_fetched": pulled,
            },
        )

    def _download_text(
        self,
        request: GetDataToolInput,
        catalog: SceneCatalog,
        report: DownloadReport,
        store: LocalArtifactStore,
        warnings: list[str],
        stop: str | None,
        run_id: str,
    ) -> str:
        lines = [*warnings, *([stop] if stop else [])]
        for src in catalog.sources:
            if src.error is not None:
                lines.append(f"{src.source}: search failed: {src.error}")
                continue
            outcomes = report.for_source(src.source)
            saved = sum(1 for s in src.scenes if _scene_status(outcomes, s.id))
            lines.append(f"{src.source}: {saved} of {len(src.scenes)} scenes saved")
            failed = [o for o in outcomes if o.error]
            for o in failed[:8]:
                label = " ".join(x for x in (o.scene_id, o.alias or o.asset_key) if x)
                lines.append(f"  {label}: {o.error}")
            if len(failed) > 8:
                lines.append(f"  +{len(failed) - 8} more errors: see run {run_id}")
        saved_outcomes = [
            o for o in report.outcomes if o.status in {"saved", "skipped"} and o.path is not None
        ]
        shown = saved_outcomes if len(saved_outcomes) <= 20 else saved_outcomes[:10]
        lines.extend(_saved_line(store, o) for o in shown)
        if len(shown) < len(saved_outcomes):
            lines.append(f"+{len(saved_outcomes) - len(shown)} more: see run {run_id}")
        lines.append(f"run: {run_id}")
        lines.append(self._next_hint(request, download=True))
        return "\n".join(lines)


def _compact(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline shared definitions and drop null branches and null defaults (token cost)."""
    defs = schema.pop("$defs", {})
    for name, prop in schema["properties"].items():
        branches = prop.pop("anyOf", None)
        if branches is not None:
            kept = [b for b in branches if b.get("type") != "null"]
            prop.update(kept[0])
        if prop.get("default") is None:
            prop.pop("default", None)
        for bound in _BOUND_KEYS:
            prop.pop(bound, None)
        if prop.get("$ref", "").endswith("/BBox") or name == "bbox":
            prop.pop("$ref", None)
            prop["type"] = "object"
        elif prop.get("$ref", "").endswith("/SceneKind"):
            prop.pop("$ref")
            prop.update(defs["SceneKind"])
    return schema


def _cap_text(text: str) -> str:
    if len(text) <= TEXT_CAP:
        return text
    return text[: TEXT_CAP - 40].rstrip() + "\n... (truncated; narrow the request)"
