from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import httpx
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from atlas.agent.layout import (
    EARTHDATA_TOKEN,
    MANIFEST_TEMP_GLOB,
    MANIFEST_TEMP_PREFIX,
    MANIFEST_TEMP_SUFFIX,
    project_data_dir,
    pull_manifest_path,
    resolve_credential,
)
from atlas.cli.catalog import (
    CatalogError,
    only_kind,
    render_catalog,
    render_satellite,
    render_satellite_tree,
    resolve_sources,
)
from atlas.cli.download import (
    BrowseAssetError,
    DownloadOutcome,
    DownloadReport,
    download_assets,
    inside_output,
    is_forbidden_asset_name,
    plain_source_name,
    retire_product,
)
from atlas.cli.estimate import confirm_download, estimate_pull, render_pull_estimate
from atlas.cli.progress import ProgressReporter, pick_reporter
from atlas.cli.source_outputs import write_source_outputs, writes_own_output
from atlas.compile.aggregate import aggregate
from atlas.compile.product import SceneCatalog, SourceScenes
from atlas.data.base import BBox, DataPullClient, Polygon, PullRequest, Scene
from atlas.data.registry import sources
from atlas.data.scene_identity import (
    SourceResolution,
    StoredScene,
    project_records,
    reconcile,
    resolution_lines,
    stored_scenes,
)
from atlas.data.sources.fldas import (
    MISSING_TOKEN_MESSAGE,
    FldasAuthError,
    FldasClient,
    FldasError,
    FldasRequestError,
    _skip_note,
    load_repo_env,
    parse_points_csv,
    resolve_variables,
    split_variables,
)
from atlas.data.stac import encode_page_cursor

_EXAMPLE = (
    "atlas data sentinel2 optical --date 2024-07-01:2024-07-07 "
    "--coords -71.12,42.32,-71.02,42.40\n"
    "  atlas data sentinel2 optical --date 2024-07-01:2024-07-07 "
    "--polygon -71.12,42.32;-71.02,42.32;-71.02,42.40;-71.12,42.40\n"
    "  atlas data fldas landsurface --date 2020-01-01:2020-01-31 "
    "--coords -71.2,42.2,-70.9,42.5 --out .atlas/data/fldas"
)
_EXTRACT_HINT = "FLDAS table extraction runs alone; pass only the fldas source"
_TABLE_NAME = "fldas.csv"
_EMPTY_WINDOW = "requested window is empty"


def add_data_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Attach the ``data`` command to the root parser."""
    data = sub.add_parser(
        "data",
        help="Search and pull registered satellite sources",
        description=(
            "List discovered satellites, or search a satellite and download native assets. "
            "A geographic, Web Mercator, or WGS84 UTM COG is read as the window "
            "overlapping --coords. fldas landsurface writes a CSV training table. "
            "A signed STAC next link is not stored, so later pages can stay unfetched. "
            "That pull prints a warning and still exits 0. "
            "A download prints a size estimate from the search. Files already on disk "
            "or listed in a resume manifest are left out of that estimate. "
            "The download starts only after confirmation, or when --yes is passed. "
            "Without a terminal, --yes is required. Remaining quota is included "
            "when a provider publishes it; otherwise the estimate says quota is not readable. "
            "A saved 16-bit, signed, float, or multiband GeoTIFF that viewers show dark gets "
            "a contrast-stretched <stem>.preview.png beside it; the data file is unchanged."
        ),
        epilog=f"example:\n  {_EXAMPLE}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    data.add_argument(
        "satellite",
        nargs="?",
        help="Satellite family, registry key, shorthand (s2, l8), or 'all'",
    )
    data.add_argument(
        "kind",
        nargs="?",
        help="Data kind (optical, sar, thermal, ...) or 'all'",
    )
    data.add_argument(
        "--list",
        action="store_true",
        dest="list_catalog",
        help="List kinds and sources (one satellite if given, otherwise the full catalog)",
    )
    data.add_argument(
        "--date",
        help="Date window as START:END (also START/END or START,END), ISO dates",
    )
    data.add_argument(
        "--coords",
        help="Bounding box as west,south,east,north (west can be negative: --coords -71.12,42.32,-71.02,42.40)",
    )
    data.add_argument(
        "--polygon",
        help=(
            "AOI ring as lon,lat pairs separated by semicolons, or a GeoJSON Polygon. "
            "An open ring is closed. Without --coords the search box is the ring envelope."
        ),
    )
    data.add_argument(
        "--out",
        type=Path,
        default=project_data_dir(),
        help="Output directory (default: .atlas/data)",
    )
    data.add_argument("--limit", type=int, default=100, help="Max scenes per source (1-1000)")
    data.add_argument(
        "--min-cloud-cover",
        type=float,
        default=None,
        help="Minimum scene cloud cover percentage (0-100)",
    )
    data.add_argument(
        "--max-cloud-cover",
        type=float,
        default=None,
        help="Maximum scene cloud cover percentage (0-100)",
    )
    data.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Max parallel preview downloads",
    )
    data.add_argument(
        "--search-only",
        action="store_true",
        help="Search and write a manifest; do not download assets",
    )
    data.add_argument(
        "--yes",
        action="store_true",
        help=(
            "Start the download without a confirmation prompt. "
            "Required when stdin is not a terminal."
        ),
    )
    data.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Continue from next-page cursors in the output manifest "
            "instead of requesting page one again. A signed next link leaves "
            "later pages unfetched and prints a warning; the exit status stays 0."
        ),
    )
    data.add_argument(
        "--asset",
        default=None,
        help=(
            "Band name (red, nir) or asset key to download; band names win over a "
            "literal key of the same name. Default: visual COG, then the red band, then "
            "an imagery band; QA, mask, and angle layers are skipped. "
            "Previews/thumbnails are rejected."
        ),
    )
    data.add_argument(
        "--table",
        type=Path,
        default=None,
        help="Training-table CSV path for extract sources (default: <out>/fldas.csv)",
    )
    data.add_argument(
        "--variables",
        default=None,
        help=(
            "Comma-separated FLDAS variables for the training table "
            "(default: the ten soil/snow/radiation columns)"
        ),
    )
    data.add_argument(
        "--points",
        type=Path,
        default=None,
        help="CSV of points with a header (longitude/latitude, lon/lat, or x/y)",
    )
    data.set_defaults(_atlas_cmd="data")


def run_data(
    args: argparse.Namespace,
    *,
    clients: Mapping[str, DataPullClient] | None = None,
    http: httpx.AsyncClient | None = None,
) -> int:
    """Dispatch ``atlas data``: list catalog, list one satellite, or pull.

    Returns:
        Process exit code (0 ok, 1 all sources failed, 2 usage).
    """
    load_repo_env()
    console = Console(highlight=False)
    if args.satellite is None:
        render_catalog(console)
        return 0
    if args.list_catalog:
        try:
            render_satellite_tree(console, args.satellite, kind=args.kind)
        except CatalogError as exc:
            Console(stderr=True, highlight=False).print(str(exc), style="red")
            return 2
        return 0
    if args.kind is None and args.date and args.coords:
        try:
            args.kind = only_kind(args.satellite)
        except CatalogError as exc:
            Console(stderr=True, highlight=False).print(str(exc), style="red")
            return 2
    if args.kind is None:
        try:
            render_satellite(console, args.satellite)
        except CatalogError as exc:
            Console(stderr=True, highlight=False).print(str(exc), style="red")
            return 2
        return 0
    polygon_text = getattr(args, "polygon", None)
    if not args.date or not (args.coords or polygon_text):
        print(
            "data pull requires --date START:END and --coords W,S,E,N or --polygon",
            file=sys.stderr,
        )
        return 2
    try:
        names = resolve_sources(args.satellite, args.kind)
        start, end = parse_date_range(args.date)
        polygon = parse_polygon(polygon_text) if polygon_text else None
        request_body: dict[str, object] = {
            "start_date": start,
            "end_date": end,
            "min_cloud_cover": args.min_cloud_cover,
            "max_cloud_cover": args.max_cloud_cover,
            "limit": args.limit,
            "polygon": polygon,
        }
        if args.coords:
            request_body["bbox"] = parse_coords(args.coords)
        request = PullRequest.model_validate(request_body)
    except (CatalogError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.asset and is_forbidden_asset_name(args.asset):
        print(
            f"{args.asset!r} is a preview/thumbnail and cannot be downloaded",
            file=sys.stderr,
        )
        return 2

    # Keyed off the resolved registry names so family tokens (fldas, all) work too.
    # A mixed catalog pull still extracts FLDAS and downloads the other sources.
    # --table/--variables/--points describe only the FLDAS table, so they are
    # rejected unless FLDAS is the whole pull.
    extracting = FldasClient.satellite in names
    fldas_only = extracting and len(names) == 1
    variables = None
    points = None
    if (
        extracting
        and not fldas_only
        and any(getattr(args, flag, None) is not None for flag in ("table", "variables", "points"))
    ):
        print(_EXTRACT_HINT, file=sys.stderr)
        return 2
    if fldas_only:
        if args.asset is not None:
            print(
                "FLDAS writes a training table; --asset does not download anything",
                file=sys.stderr,
            )
            return 2
        if request.bbox.west > request.bbox.east:
            print(
                f"FLDAS needs a west<=east bounding box; got west={request.bbox.west} "
                f"east={request.bbox.east}",
                file=sys.stderr,
            )
            return 2
        try:
            raw_variables = getattr(args, "variables", None)
            raw_points = getattr(args, "points", None)
            if raw_variables is not None:
                variables = resolve_variables(split_variables(raw_variables))
            if raw_points is not None:
                points = parse_points_csv(raw_points)
        except FldasRequestError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    if (
        fldas_only
        and not args.search_only
        and clients is None
        and not (resolve_credential(EARTHDATA_TOKEN) or "").strip()
    ):
        print(MISSING_TOKEN_MESSAGE, file=sys.stderr)
        return 1

    reporter = pick_reporter(stream=sys.stderr)
    page_cursors: Mapping[str, str | None] | None = None
    prior_manifest: Mapping[str, Any] | None = None
    if getattr(args, "resume", False):
        try:
            stored = _load_pull_manifest(pull_manifest_path(args.out))
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
        run_sources = set(clients) if clients is not None else set(names)
        prior_manifest = _resume_view(stored, run_sources, request)
        if prior_manifest is None:
            print("resume query does not match the pull manifest", file=sys.stderr)
            return 2
        page_cursors = _page_cursors(prior_manifest)
    return asyncio.run(
        execute_pull(
            source_names=names,
            request=request,
            out_dir=args.out,
            search_only=bool(args.search_only),
            concurrency=int(args.concurrency),
            asset_name=args.asset,
            reporter=reporter,
            clients=clients,
            http=http,
            table_path=getattr(args, "table", None),
            variables=variables,
            points=points,
            extract_tables=extracting,
            page_cursors=page_cursors,
            prior_manifest=prior_manifest,
            assume_yes=bool(getattr(args, "yes", False)),
        )
    )


def parse_date_range(text: str) -> tuple[date, date]:
    """Parse ``START:END`` (also ``/`` or ``,``) into ordered ISO dates."""
    raw = text.strip()
    sep = next((item for item in (":", "/", ",") if item in raw), None)
    if sep is None:
        raise ValueError("date must be START:END (ISO dates)")
    left, right = raw.split(sep, 1)
    start = date.fromisoformat(left.strip())
    end = date.fromisoformat(right.strip())
    if end < start:
        raise ValueError("end date must be on or after start date")
    return start, end


def parse_coords(text: str) -> BBox:
    """Parse ``west,south,east,north`` into a ``BBox``."""
    parts = [item.strip() for item in text.split(",")]
    if len(parts) != 4:
        raise ValueError("coords must be west,south,east,north")
    west, south, east, north = (float(item) for item in parts)
    return BBox(west=west, south=south, east=east, north=north)


def parse_polygon(text: str) -> Polygon:
    """Parse a polygon AOI.

    A GeoJSON Polygon (or a Feature whose geometry is one), a JSON ring of
    ``[lon, lat]`` positions, or ``lon,lat`` pairs separated by semicolons.
    An open ring is closed.

    Args:
        text: The ``--polygon`` value.

    Returns:
        The exterior ring.

    Raises:
        ValueError: If the text is not a usable polygon.
    """
    raw = text.strip()
    if not raw:
        raise ValueError("polygon is empty")
    if raw.startswith("{") or raw.startswith("["):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("polygon JSON is invalid") from exc
        if isinstance(payload, dict) and payload.get("type") == "Feature":
            payload = payload.get("geometry")
        try:
            if isinstance(payload, dict):
                return Polygon.from_geojson(payload)
            if isinstance(payload, list):
                return Polygon(ring=payload)
        except ValidationError as exc:
            raise ValueError(_polygon_error(exc)) from exc
        raise ValueError("polygon JSON must be a Polygon or a ring")
    parts = [item.strip() for item in raw.split(";") if item.strip()]
    ring: list[tuple[float, float]] = []
    for part in parts:
        bits = [item.strip() for item in part.split(",")]
        if len(bits) != 2:
            raise ValueError("polygon pairs must be lon,lat separated by semicolons")
        try:
            ring.append((float(bits[0]), float(bits[1])))
        except ValueError as exc:
            raise ValueError("polygon pairs must be lon,lat separated by semicolons") from exc
    if len(ring) >= 1 and (len(ring) == 1 or ring[0] != ring[-1]):
        ring.append(ring[0])
    try:
        return Polygon(ring=ring)
    except ValidationError as exc:
        raise ValueError(_polygon_error(exc)) from exc


def _polygon_error(exc: ValidationError) -> str:
    messages = [error["msg"].removeprefix("Value error, ") for error in exc.errors()]
    return "; ".join(messages) if messages else "polygon is invalid"


async def execute_pull(
    *,
    source_names: list[str],
    request: PullRequest,
    out_dir: Path,
    search_only: bool,
    concurrency: int,
    asset_name: str | None,
    reporter: ProgressReporter,
    clients: Mapping[str, DataPullClient] | None = None,
    http: httpx.AsyncClient | None = None,
    table_path: Path | None = None,
    variables: Sequence[str] | None = None,
    points: Sequence[tuple[float, float]] | None = None,
    extract_tables: bool = False,
    page_cursors: Mapping[str, str | None] | None = None,
    prior_manifest: Mapping[str, Any] | None = None,
    assume_yes: bool = False,
) -> int:
    """Fan out search, then extract training tables or download native assets.

    A source that derives a table instead of assets (FLDAS) writes its CSV here;
    its failures are recorded so a table that was never written returns 1.
    After the search, a download prints a size estimate and waits for
    confirmation unless ``assume_yes`` is set or nothing remains to download.
    Files already on disk or listed in the resume manifest are left out of
    the estimate. Search-only does not prompt. A cancelled or refused download
    writes nothing and still resumes the progress display.

    Args:
        source_names: Registry names to search.
        request: Date window and area.
        out_dir: Directory for assets, tables, and the manifest.
        search_only: Write the manifest and do not download.
        concurrency: Maximum parallel asset downloads.
        asset_name: Band or asset key to download.
        reporter: Progress display.
        clients: Pull clients. The registry is used when omitted.
        http: Client for asset downloads.
        table_path: Training-table path for an extract source.
        variables: Extract columns.
        points: Extract sample points.
        extract_tables: Run training-table extraction for FLDAS.
        page_cursors: Per-source resume cursors.
        prior_manifest: Manifest a resume continues.
        assume_yes: Start the download without a prompt (``--yes``).
    """
    owns_clients = clients is None
    live_clients = dict(clients) if clients is not None else sources(*source_names)
    owns_http = http is None
    live_http = (
        http if http is not None else httpx.AsyncClient(timeout=300.0, follow_redirects=True)
    )
    try:
        if prior_manifest is not None:
            mismatch = _resume_mismatch(prior_manifest, live_clients)
            if mismatch is not None:
                print(mismatch, file=sys.stderr)
                return 2
        unavailable = _unavailable_cursors(prior_manifest) if prior_manifest is not None else None
        if unavailable:
            for name in sorted(unavailable):
                if name in live_clients:
                    print(
                        _unavailable_warning(name, unavailable[name]),
                        file=sys.stderr,
                    )
        for name in live_clients:
            reporter.source_start(name)
        catalog = await aggregate(
            request,
            live_clients,
            on_source=lambda src: reporter.source_search_done(
                src.source, scenes=len(src.scenes), error=src.error
            ),
            page_cursors=page_cursors,
            unavailable_cursors=unavailable,
        )
        identity = (
            prior_manifest if prior_manifest is not None else _matching_manifest(out_dir, request)
        )
        resolutions = await _resolve_catalog(identity, catalog, live_clients)
        _inject_replacements(catalog, resolutions)
        if not search_only:
            estimate = estimate_pull(
                catalog,
                asset_name=asset_name,
                unknown_byte_sources=_unknown_byte_sources(live_clients, extract_tables),
                out_dir=out_dir,
                prior_manifest=prior_manifest,
                bbox=request.bbox,
            )
            reporter.suspend()
            try:
                print(render_pull_estimate(estimate), file=sys.stderr, flush=True)
                if estimate.scene_count > 0:
                    blocked = confirm_download(assume_yes=assume_yes)
                    if blocked is not None:
                        return blocked
            finally:
                reporter.resume()
        report = DownloadReport()
        table_failed = False
        table_usage = False
        if not search_only:
            to_pull = [
                scene
                for src in catalog.sources
                if src.error is None and not isinstance(live_clients[src.source], FldasClient)
                for scene in src.scenes
            ]
            # GIBS images and FIRMS detection tables are written by their own
            # module; browse assets stay forbidden in download_assets.
            own_output = [s for s in to_pull if writes_own_output(live_clients.get(s.source))]
            to_pull = [s for s in to_pull if not writes_own_output(live_clients.get(s.source))]
            for src in catalog.sources:
                if src.error is None:
                    reporter.source_pull(src.source, saved=0, total=len(src.scenes))
            try:
                report = await download_assets(
                    to_pull,
                    out_dir,
                    http=live_http,
                    clients=live_clients,
                    concurrency=concurrency,
                    asset_name=asset_name,
                    bbox=request.bbox,
                    on_progress=lambda name, n_saved, total: reporter.source_pull(
                        name, saved=n_saved, total=total
                    ),
                )
            except BrowseAssetError as exc:
                print(str(exc), file=sys.stderr)
                return 2
            report.outcomes.extend(
                await write_source_outputs(
                    own_output,
                    out_dir,
                    http=live_http,
                    clients=live_clients,
                    concurrency=concurrency,
                    on_progress=lambda name, n_saved, total: reporter.source_pull(
                        name, saved=n_saved, total=total
                    ),
                )
            )
            if extract_tables:
                for src in catalog.sources:
                    client = live_clients.get(src.source)
                    if src.error is not None or not isinstance(client, FldasClient):
                        continue
                    source_name = src.source

                    def report_extract(done: int, total: int, name: str = source_name) -> None:
                        reporter.source_pull(name, saved=done, total=total, label="extracting")

                    outcome, usage_error = await _extract_table(
                        client,
                        source_name,
                        request,
                        src.scenes,
                        out_dir=out_dir,
                        table_path=table_path,
                        variables=variables,
                        points=points,
                        on_progress=report_extract,
                    )
                    report.outcomes.append(outcome)
                    table_failed = table_failed or outcome.status != "saved"
                    table_usage = table_usage or usage_error
        _record_empty_windows(
            catalog,
            report,
            live_clients,
            out_dir,
            search_only=search_only,
            prior=prior_manifest,
            held=_held_sources(resolutions),
        )
        for src in catalog.sources:
            reporter.source_done(src.source)
        _write_manifest(
            out_dir,
            catalog,
            report,
            prior=prior_manifest,
            identity=identity,
            resolutions=resolutions,
        )
        reporter.close()
        _print_run_summary(catalog, report, search_only=search_only)
        for src in catalog.sources:
            for line in resolution_lines(
                src.source, resolutions.get(src.source, SourceResolution.unchecked())
            ):
                print(line, file=sys.stderr)
        if catalog.sources and all(src.error for src in catalog.sources):
            return 1
        # Usage and auth failures change the exit code only for a FLDAS-only pull.
        # A mixed catalog pull keeps the existing rule: search success exits 0,
        # and the table failure is still listed in the summary.
        fldas_only = bool(live_clients) and all(
            isinstance(client, FldasClient) for client in live_clients.values()
        )
        if fldas_only and table_usage:
            return 2
        if fldas_only and table_failed:
            return 1
        return 0
    finally:
        reporter.close()
        if owns_http:
            await live_http.aclose()
        if owns_clients:
            await _aclose_all(live_clients)


def _unknown_byte_sources(
    clients: Mapping[str, DataPullClient],
    extract_tables: bool,
) -> frozenset[str]:
    """Return sources whose download size is not a catalog asset size.

    A training-table extract reads a subset of each granule. The asset's
    ``file:size``, when one exists, is not the table, so the estimate leaves
    those bytes unknown.
    """
    if not extract_tables:
        return frozenset()
    return frozenset(name for name, client in clients.items() if isinstance(client, FldasClient))


async def _extract_table(
    client: FldasClient,
    source: str,
    request: PullRequest,
    scenes: Sequence[Scene],
    *,
    out_dir: Path,
    table_path: Path | None,
    variables: Sequence[str] | None,
    points: Sequence[tuple[float, float]] | None,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[DownloadOutcome, bool]:
    """Run one extract-source pull; report the outcome and whether it was usage."""
    path = table_path or (out_dir / _TABLE_NAME)
    previous_path = client.table_path
    try:
        client.table_path = path
        table = await client.extract_training_table(
            request,
            scenes,
            points=points,
            variables=variables,
            on_progress=on_progress,
        )
        table.write_csv(path)
    except FldasRequestError as exc:
        if _empty_fldas_window(exc):
            _retire_inside(out_dir, path)
        return (
            DownloadOutcome(
                source=source,
                scene_id=f"{source}-table",
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            ),
            True,
        )
    except FldasAuthError as exc:
        return (
            DownloadOutcome(
                source=source,
                scene_id=f"{source}-table",
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            ),
            False,
        )
    except (FldasError, httpx.HTTPError, OSError) as exc:
        if _empty_fldas_window(exc):
            _retire_inside(out_dir, path)
        return (
            DownloadOutcome(
                source=source,
                scene_id=f"{source}-table",
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            ),
            False,
        )
    except ValueError as exc:
        return (
            DownloadOutcome(
                source=source,
                scene_id=f"{source}-table",
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            ),
            False,
        )
    finally:
        client.table_path = previous_path
    note = _skip_note(table.skipped) if table.skipped else None
    return (
        DownloadOutcome(
            source=source,
            scene_id=f"{source}-table",
            status="saved",
            path=path,
            error=note,
        ),
        False,
    )


def _record_empty_windows(
    catalog: SceneCatalog,
    report: DownloadReport,
    clients: Mapping[str, DataPullClient],
    out_dir: Path,
    *,
    search_only: bool,
    prior: Mapping[str, Any] | None = None,
    held: set[str] | None = None,
) -> None:
    """Record an empty date window without presenting an older product.

    Search-only records the failure and leaves files alone. A search that
    failed, including an outage, rate limit, or timeout, also leaves files
    alone. A window that really is empty moves aside only the files the
    previous manifest listed for this source and this request. FLDAS pulls
    record the failure from the extract itself. A resume keeps the earlier
    page's files; an empty next page is not an empty date window. A source
    whose stored scene could not be matched is not an empty region.
    """
    if prior is not None:
        return
    kept = held or set()
    for src in catalog.sources:
        client = clients.get(src.source)
        fldas = isinstance(client, FldasClient)
        if src.error is not None or src.scenes or src.source in kept:
            continue
        if not search_only and fldas:
            continue
        report.outcomes.append(
            DownloadOutcome(
                source=src.source,
                scene_id=f"{src.source}-window",
                status="failed",
                error=_EMPTY_WINDOW,
            )
        )
        if search_only or not plain_source_name(src.source):
            continue
        for product in _products_for_request(out_dir, src.source, catalog.request):
            retire_product(product)


def _empty_fldas_window(exc: BaseException) -> bool:
    """True when FLDAS found no granules or no grid cell in the window."""
    text = str(exc)
    if isinstance(exc, FldasRequestError) and "grid-cell" in text:
        return True
    return (
        isinstance(exc, FldasError)
        and not isinstance(exc, FldasAuthError)
        and ("no FLDAS granules" in text)
    )


def _retire_inside(out_dir: Path, path: Path) -> None:
    """Move one table aside when it resolves inside ``out_dir``."""
    if inside_output(out_dir, path):
        retire_product(path)


def _products_for_request(out_dir: Path, source: str, request: PullRequest) -> list[Path]:
    """Return manifest-listed products for this source and this request.

    Only the row for ``source`` is read, and only when that row's own request
    is the same window. Files from another date or area, files listed for
    another source, and files the manifest did not list are not included. A
    listed path that leaves ``out_dir`` is skipped.
    """
    payload = _read_manifest(out_dir)
    if payload is None:
        return []
    products: list[Path] = []
    for item in _source_rows(payload):
        if item["source"] != source or not _same_window(item.get("request"), request):
            continue
        for key in ("saved", "skipped"):
            listed = item.get(key)
            if not isinstance(listed, list):
                continue
            for rel in listed:
                candidate = _listed_product(out_dir, source, rel)
                if candidate is not None:
                    products.append(candidate)
    return products


def _same_window(previous: object, request: PullRequest) -> bool:
    """True when a stored request is the same date window and area."""
    if not isinstance(previous, dict):
        return False
    if previous.get("start_date") != request.start_date.isoformat():
        return False
    if previous.get("end_date") != request.end_date.isoformat():
        return False
    bbox = previous.get("bbox")
    if not isinstance(bbox, dict):
        return False
    current = request.bbox.model_dump(mode="json")
    if any(bbox.get(key) != current[key] for key in ("west", "south", "east", "north")):
        return False
    current_polygon = None if request.polygon is None else request.polygon.model_dump(mode="json")
    return previous.get("polygon") == current_polygon


def _listed_product(out_dir: Path, source: str, rel: object) -> Path | None:
    """Resolve one manifest path that belongs to ``source`` and stays in ``out_dir``."""
    if not isinstance(rel, str) or not rel or rel.startswith(("/", "\\")) or "\\" in rel:
        return None
    parts = Path(rel).parts
    if not parts or parts[0] != source or any(part in {".", ".."} for part in parts):
        return None
    candidate = out_dir.joinpath(*parts)
    if not inside_output(out_dir, candidate):
        return None
    return candidate


def _catalog_bbox_note(request: PullRequest) -> str:
    if request.polygon is None:
        return "Every catalog was searched with this bounding box."
    return (
        "Catalogs that only accept a bounding box were searched with this box. "
        "The polygon is the area of interest. Scenes from every source are filtered "
        "to it by footprint bounding-box overlap; point scenes must fall inside the ring. "
        "A scene stores that box, not the full footprint outline."
    )


def _load_pull_manifest(path: Path) -> dict[str, Any]:
    """Read a pull manifest that resume can continue.

    Args:
        path: ``manifest.json`` written by an earlier pull.

    Returns:
        The parsed manifest object.

    Raises:
        ValueError: The file is missing or has no source list.
    """
    if not path.is_file():
        raise ValueError(f"no pull manifest at {path}; cannot resume")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("sources"), list):
        raise ValueError(f"pull manifest at {path} has no sources")
    return payload


def _same_pull(stored: object, request: PullRequest) -> bool:
    """True when ``stored`` is the same search window as ``request``.

    The page cursor is only valid for that window. ``limit`` may change.
    """
    if not isinstance(stored, dict):
        return False
    try:
        previous = PullRequest.model_validate(stored)
    except ValueError:
        return False
    return (
        previous.start_date == request.start_date
        and previous.end_date == request.end_date
        and previous.bbox == request.bbox
        and previous.polygon == request.polygon
        and previous.min_cloud_cover == request.min_cloud_cover
        and previous.max_cloud_cover == request.max_cloud_cover
    )


def _page_cursors(payload: Mapping[str, Any]) -> dict[str, str | None]:
    """Return each source's stored next-page cursor, including explicit nulls."""
    cursors: dict[str, str | None] = {}
    sources = payload.get("sources")
    if not isinstance(sources, list):
        return cursors
    for item in sources:
        if not isinstance(item, dict):
            continue
        name = item.get("source")
        if not isinstance(name, str) or not name:
            continue
        cursors[name] = _wire_cursor(item.get("next_cursor"))
    return cursors


def _write_manifest(
    out_dir: Path,
    catalog: SceneCatalog,
    report: DownloadReport,
    *,
    prior: Mapping[str, Any] | None = None,
    identity: Mapping[str, Any] | None = None,
    resolutions: Mapping[str, SourceResolution] | None = None,
) -> None:
    """Write this run's sources into ``<out_dir>/manifest.json``.

    The manifest is shared by every pull into ``out_dir``. Each row in
    ``sources`` carries the ``request`` that produced it, so one directory can
    hold sources pulled for different windows. A row for a source this run did
    not search is kept as it is on disk; a row for a source this run searched
    is replaced (or, on resume, extended from ``prior``). The top-level
    ``request``, ``polygon``, and ``catalog_bbox`` describe the most recent run.
    A row without its own ``request`` was written before rows carried one and
    belongs to the top-level request; it is stamped with it when kept.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    request = catalog.request
    request_json = request.model_dump(mode="json")
    bbox = request.bbox.as_list()
    prior_sources = _prior_sources(prior)
    identity_sources = _prior_sources(identity)
    checks = resolutions or {}
    sources_payload = [
        _stamp_request(
            _source_entry(
                src,
                report,
                out_dir,
                bbox,
                prior_row=prior_sources.get(src.source),
                identity_row=identity_sources.get(src.source),
                resolution=checks.get(src.source, SourceResolution.unchecked()),
            ),
            request_json,
        )
        for src in catalog.sources
    ]
    existing = _read_manifest(out_dir)
    payload = {
        "request": request_json,
        "polygon": None if request.polygon is None else request.polygon.as_geojson(),
        "catalog_bbox": bbox,
        "catalog_bbox_note": _catalog_bbox_note(request),
        "sources": _merge_rows([] if existing is None else _source_rows(existing), sources_payload),
    }
    _write_json_atomic(pull_manifest_path(out_dir), json.dumps(payload, indent=2) + "\n")


def _stamp_request(row: dict[str, Any], request_json: object) -> dict[str, Any]:
    """Return ``row`` with ``request`` placed right after ``source``."""
    stamped: dict[str, Any] = {"source": row.get("source"), "request": request_json}
    stamped.update((key, value) for key, value in row.items() if key not in stamped)
    return stamped


def _merge_rows(
    existing: Sequence[dict[str, Any]], current: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Replace rows for this run's sources and keep every other source's row.

    Order on disk is kept; a source new to the manifest is appended.
    """
    fresh = {row["source"]: row for row in current}
    merged: list[dict[str, Any]] = []
    placed: set[str] = set()
    for row in existing:
        name = row["source"]
        if name in placed:
            continue
        placed.add(name)
        merged.append(fresh.get(name, row))
    merged.extend(row for name, row in fresh.items() if name not in placed)
    return merged


def _prior_sources(prior: Mapping[str, Any] | None) -> dict[str, Mapping[str, Any]]:
    if prior is None:
        return {}
    rows = prior.get("sources")
    if not isinstance(rows, list):
        return {}
    found: dict[str, Mapping[str, Any]] = {}
    for item in rows:
        if isinstance(item, dict) and isinstance(item.get("source"), str):
            found[item["source"]] = item
    return found


def _read_manifest(out_dir: Path) -> dict[str, Any] | None:
    """Return the readable manifest in ``out_dir``, or None.

    A missing, symlinked, unreadable, or malformed file is treated as absent
    so it never fails a pull.
    """
    path = pull_manifest_path(out_dir)
    if path.is_symlink() or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("sources"), list):
        return None
    return payload


def _source_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return copies of the source rows, each with its effective ``request``.

    A row that has no ``request`` of its own (the format before rows carried
    one) inherits the manifest's top-level ``request``. Rows without a string
    ``source`` are dropped.
    """
    rows: list[dict[str, Any]] = []
    raw = payload.get("sources")
    if not isinstance(raw, list):
        return rows
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = item.get("source")
        if not isinstance(name, str) or not name:
            continue
        own = item.get("request")
        rows.append(_stamp_request(item, own if isinstance(own, dict) else payload.get("request")))
    return rows


def _resume_view(
    payload: Mapping[str, Any], names: set[str], request: PullRequest
) -> dict[str, Any] | None:
    """Return the part of a shared manifest that a resume of ``names`` continues.

    Only rows for ``names`` are kept, so other sources in the same directory
    neither block the resume nor feed it cursors. Returns None when any kept
    row was written for a different search than ``request``. A requested
    source with no row is left for ``_resume_mismatch`` to reject.
    """
    rows = [row for row in _source_rows(payload) if row["source"] in names]
    if not rows and not _same_pull(payload.get("request"), request):
        return None
    if any(not _same_pull(row.get("request"), request) for row in rows):
        return None
    return {"request": request.model_dump(mode="json"), "sources": rows}


def _matching_manifest(out_dir: Path, request: PullRequest) -> dict[str, Any] | None:
    """Return the rows of an existing manifest written for this same search.

    Rows written for a different date, box, or polygon are left out. A
    missing or unreadable file does not fail the pull.
    """
    payload = _read_manifest(out_dir)
    if payload is None:
        return None
    rows = [row for row in _source_rows(payload) if _same_pull(row.get("request"), request)]
    if not rows:
        return None
    return {"request": request.model_dump(mode="json"), "sources": rows}


def _held_sources(resolutions: Mapping[str, SourceResolution]) -> set[str]:
    """Sources whose stored scenes are gone and were not uniquely replaced."""
    return {name for name, item in resolutions.items() if item.unresolved}


async def _resolve_catalog(
    identity: Mapping[str, Any] | None,
    catalog: SceneCatalog,
    clients: Mapping[str, DataPullClient],
) -> dict[str, SourceResolution]:
    """Check stored ids and look up a reprocessed scene by place and time.

    Only a STAC source resolves replacements. A client that cannot check ids
    returns an unchecked resolution, and that source keeps its ids as they
    are. A source whose search failed is skipped so a later retry still has
    the stored id.
    """
    if identity is None:
        return {}
    rows = _prior_sources(identity)
    found: dict[str, SourceResolution] = {}
    for src in catalog.sources:
        if src.error is not None:
            continue
        stored = stored_scenes(rows.get(src.source))
        if not stored:
            continue
        client = clients.get(src.source)
        if client is None:
            continue
        found[src.source] = await _resolve_source(client, stored, collection=src.collection)
    return found


async def _resolve_source(
    client: DataPullClient,
    stored: Sequence[StoredScene],
    *,
    collection: str,
) -> SourceResolution:
    try:
        present = await client.resolve_scene_ids([item.id for item in stored])
    except (OSError, TimeoutError, TypeError, ValueError, httpx.HTTPError):
        return SourceResolution.unchecked()
    if present is None:
        return SourceResolution.unchecked()
    usable = {scene_id: scene for scene_id, scene in present.items() if isinstance(scene, Scene)}
    bundles: dict[str, list[Scene]] = {}
    for item in stored:
        if item.id in usable or item.observed_at is None or item.footprint is None:
            bundles[item.id] = []
            continue
        try:
            hits = await client.find_scenes_by_place_time(
                collection=item.collection or collection,
                when=item.observed_at,
                footprint=item.footprint,
            )
        except (OSError, TimeoutError, TypeError, ValueError, httpx.HTTPError):
            return SourceResolution.unchecked()
        if hits is None:
            bundles[item.id] = []
        else:
            bundles[item.id] = [scene for scene in hits if isinstance(scene, Scene)]
    return reconcile(stored, usable, bundles, collection=collection)


def _inject_replacements(
    catalog: SceneCatalog,
    resolutions: Mapping[str, SourceResolution],
) -> None:
    """Add a uniquely matched replacement so the region is not left empty."""
    for src in catalog.sources:
        resolution = resolutions.get(src.source)
        if resolution is None or src.error is not None:
            continue
        known = {scene.id for scene in src.scenes}
        for item in resolution.replacements:
            if item.new_id in known:
                continue
            updates: dict[str, str] = {}
            if item.scene.source != src.source:
                updates["source"] = src.source
            if not item.scene.collection and src.collection:
                updates["collection"] = src.collection
            scene = item.scene.model_copy(update=updates) if updates else item.scene
            src.scenes.append(scene)
            known.add(scene.id)


def _source_entry(
    src: SourceScenes,
    report: DownloadReport,
    out_dir: Path,
    bbox: list[float],
    *,
    prior_row: Mapping[str, Any] | None,
    identity_row: Mapping[str, Any] | None,
    resolution: SourceResolution,
) -> dict[str, Any]:
    merged = _merge_source(
        {
            "source": src.source,
            "collection": src.collection,
            "bbox": bbox,
            "search_url": src.search_url,
            "next_cursor": _manifest_cursor(src.next_cursor),
            "next_cursor_unavailable": src.next_cursor_unavailable,
            "note": _unavailable_note(src.next_cursor_unavailable)
            if src.next_cursor_unavailable
            else None,
            "error": src.error,
            "_cursor_rejected": src.cursor_rejected,
            "n_scenes": len(src.scenes),
            "scene_ids": [scene.id for scene in src.scenes],
            "saved": [
                _rel(out_dir, item.path)
                for item in report.for_source(src.source)
                if item.status == "saved" and item.path is not None
            ],
            "skipped": [
                _rel(out_dir, item.path)
                for item in report.for_source(src.source)
                if item.status == "skipped" and item.path is not None
            ],
            "failed": [
                {"id": item.scene_id, "error": item.error}
                for item in report.for_source(src.source)
                if item.status in {"failed", "no_asset"}
            ],
            "previews": [
                _rel(out_dir, item.preview)
                for item in report.for_source(src.source)
                if item.preview is not None
            ],
            "windows": [
                {
                    "id": item.scene_id,
                    "bytes_read": item.bytes_read,
                    "granule_size": item.granule_size,
                }
                for item in report.for_source(src.source)
                if item.bytes_read is not None and item.granule_size is not None
            ],
        },
        prior_row,
    )
    records = project_records(
        src.scenes,
        stored_scenes(identity_row),
        resolution,
        collection=src.collection,
        keep_stored=prior_row is not None,
    )
    merged["scenes"] = [record.as_manifest() for record in records]
    merged["scene_ids"] = [record.id for record in records]
    merged["n_scenes"] = len(records)
    merged["replacements"] = _replacement_entries(identity_row, resolution)
    merged["unresolved_scenes"] = _unresolved_entries(identity_row, resolution)
    mapping = {item.old_id: item.new_id for item in resolution.replacements}
    if mapping:
        merged["windows"] = _rewrite_ids(merged.get("windows"), mapping)
        merged["failed"] = _rewrite_ids(merged.get("failed"), mapping)
    return merged


def _replacement_entries(
    row: Mapping[str, Any] | None,
    resolution: SourceResolution,
) -> list[dict[str, str]]:
    carried: list[dict[str, str]] = []
    seen: set[str] = set()
    raw = None if row is None else row.get("replacements")
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            old_id = item.get("old_id")
            new_id = item.get("new_id")
            if (
                isinstance(old_id, str)
                and old_id
                and isinstance(new_id, str)
                and new_id
                and old_id not in seen
            ):
                seen.add(old_id)
                carried.append({"old_id": old_id, "new_id": new_id})
    if not resolution.checked:
        return carried
    replaced = {item.old_id for item in resolution.replacements}
    fresh = [item for item in carried if item["old_id"] not in replaced]
    fresh.extend({"old_id": item.old_id, "new_id": item.new_id} for item in resolution.replacements)
    return fresh


def _unresolved_entries(
    row: Mapping[str, Any] | None,
    resolution: SourceResolution,
) -> list[dict[str, Any]]:
    if resolution.checked:
        return [
            {"id": item.scene_id, "reason": item.reason, "candidates": list(item.candidates)}
            for item in resolution.unresolved
        ]
    raw = None if row is None else row.get("unresolved_scenes")
    if not isinstance(raw, list):
        return []
    entries: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        scene_id = item.get("id")
        reason = item.get("reason")
        if not isinstance(scene_id, str) or not scene_id:
            continue
        if reason not in {"no match", "ambiguous"}:
            continue
        names = item.get("candidates")
        candidates = (
            [name for name in names if isinstance(name, str)] if isinstance(names, list) else []
        )
        entries.append({"id": scene_id, "reason": reason, "candidates": candidates})
    return entries


def _rewrite_ids(rows: object, mapping: Mapping[str, str]) -> list[dict[str, Any]]:
    rewritten: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in _as_dicts(rows):
        scene_id = item.get("id")
        if not isinstance(scene_id, str):
            continue
        updated = dict(item)
        updated["id"] = mapping.get(scene_id, scene_id)
        if updated["id"] in seen:
            continue
        seen.add(updated["id"])
        rewritten.append(updated)
    return rewritten


def _resume_mismatch(prior: Mapping[str, Any], clients: Mapping[str, DataPullClient]) -> str | None:
    """Return why this resume cannot use ``prior``, or None when it can.

    The cursor is only valid for the sources, collections, and search
    endpoints that produced it.
    """
    recorded = _prior_sources(prior)
    if set(clients) != set(recorded):
        return "resume sources do not match the pull manifest"
    for name in sorted(clients):
        row = recorded[name]
        client = clients[name]
        collection = getattr(client, "collection", "")
        if not isinstance(collection, str):
            collection = ""
        if row.get("collection") != collection:
            return f"resume collection for {name} does not match the pull manifest"
        search_url = getattr(client, "search_url", "")
        if not isinstance(search_url, str):
            search_url = ""
        stored_url = row.get("search_url")
        if not isinstance(stored_url, str):
            stored_url = ""
        if stored_url != search_url:
            return f"resume search endpoint for {name} does not match the pull manifest"
    return None


def _manifest_cursor(value: object) -> str | dict[str, Any] | None:
    """Return the cursor as stored in ``manifest.json``.

    A JSON cursor is an object. A legacy token or URL stays a string.
    """
    if isinstance(value, dict):
        text = encode_page_cursor(value)
        if not text:
            return None
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.startswith("{"):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return text
            if isinstance(parsed, dict):
                return _manifest_cursor(parsed)
        return text
    return None


def _wire_cursor(value: object) -> str | None:
    """Return the cursor string ``search_from_cursor`` accepts."""
    if isinstance(value, dict):
        text = encode_page_cursor(value)
        return text or None
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _cursor_present(value: object) -> bool:
    return _wire_cursor(value) is not None


def _unavailable_note(reason: str) -> str:
    """Return the source note for a cursor that was not stored."""
    if reason == "signed":
        return "The next page link was signed and was not stored. Later pages were not fetched."
    return "The next page cursor is unavailable. Later pages were not fetched."


def _unavailable_warning(name: str, reason: str) -> str:
    """Return the stderr warning for pages left unfetched. The pull still exits 0."""
    return f"warning: {name}: {_unavailable_note(reason)}"


def _unavailable_cursors(payload: Mapping[str, Any]) -> dict[str, str]:
    """Return sources whose next page exists but was not stored."""
    found: dict[str, str] = {}
    sources = payload.get("sources")
    if not isinstance(sources, list):
        return found
    for item in sources:
        if not isinstance(item, dict):
            continue
        name = item.get("source")
        if not isinstance(name, str) or not name or _wire_cursor(item.get("next_cursor")):
            continue
        reason = item.get("next_cursor_unavailable")
        if isinstance(reason, str) and reason.strip():
            found[name] = reason.strip()
    return found


def _fsync_directory(directory: Path) -> None:
    """Flush the directory entry written by ``os.replace``."""
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


# A temp newer than this may belong to a pull that is still writing it.
_MANIFEST_TEMP_MAX_AGE_SECONDS = 15 * 60


def _manifest_file_mode() -> int:
    """Return ``0o666`` with the process umask cleared."""
    umask = os.umask(0)
    try:
        return 0o666 & ~umask
    finally:
        os.umask(umask)


def _sweep_manifest_temps(directory: Path) -> None:
    """Remove manifest temp files left by a crash more than 15 minutes ago.

    A newer ``.manifest.json.*.tmp`` is left alone so a concurrent pull can
    finish its own replace.
    """
    if not directory.is_dir():
        return
    now = time.time()
    for leftover in directory.glob(MANIFEST_TEMP_GLOB):
        if leftover.is_dir() and not leftover.is_symlink():
            continue
        try:
            modified = leftover.stat(follow_symlinks=False).st_mtime
        except OSError:
            continue
        if now - modified <= _MANIFEST_TEMP_MAX_AGE_SECONDS:
            continue
        with contextlib.suppress(OSError):
            leftover.unlink()


def _write_json_atomic(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` via an exclusive temp file, fsync, and ``os.replace``.

    The temp name is random. ``.manifest.json.*.tmp`` files in the same
    directory that are older than 15 minutes are removed first. The file
    mode is ``0o666`` with the process umask cleared. The directory is
    fsynced after the replace.
    A failed replace leaves the previous file in place and removes the temp
    file this write created.
    """
    directory = path.parent
    _sweep_manifest_temps(directory)
    fd, name = tempfile.mkstemp(
        dir=directory, prefix=MANIFEST_TEMP_PREFIX, suffix=MANIFEST_TEMP_SUFFIX
    )
    tmp = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp, _manifest_file_mode())
        os.replace(tmp, path)
        _fsync_directory(directory)
    except Exception:
        if tmp.exists():
            with contextlib.suppress(OSError):
                tmp.unlink()
        raise


def _merge_source(current: dict[str, Any], prior: Mapping[str, Any] | None) -> dict[str, Any]:
    """Append this page to the previous source record.

    A failed resume keeps the previous scenes, windows, and cursor so the
    next resume can try the same page again. A rejected cursor is cleared
    instead of being copied back.
    """
    rejected = bool(current.pop("_cursor_rejected", False))
    if prior is None:
        return current
    if current.get("error"):
        kept = dict(current)
        kept["scene_ids"] = _extend_str(prior.get("scene_ids"), [])
        kept["n_scenes"] = len(kept["scene_ids"])
        kept["saved"] = _extend_str(prior.get("saved"), [])
        kept["skipped"] = _extend_str(prior.get("skipped"), [])
        kept["failed"] = _merge_failed(prior.get("failed"), [])
        kept["windows"] = _merge_windows(prior.get("windows"), [])
        kept["previews"] = _extend_str(prior.get("previews"), [])
        if rejected:
            kept["next_cursor"] = None
        elif not _cursor_present(current.get("next_cursor")):
            kept["next_cursor"] = _manifest_cursor(prior.get("next_cursor"))
        return kept
    scene_ids = _extend_str(prior.get("scene_ids"), current.get("scene_ids"))
    merged = dict(current)
    merged["scene_ids"] = scene_ids
    merged["n_scenes"] = len(scene_ids)
    merged["saved"] = _extend_str(prior.get("saved"), current.get("saved"))
    merged["skipped"] = _extend_str(prior.get("skipped"), current.get("skipped"))
    merged["failed"] = _merge_failed(prior.get("failed"), current.get("failed"))
    merged["windows"] = _merge_windows(prior.get("windows"), current.get("windows"))
    merged["previews"] = _extend_str(prior.get("previews"), current.get("previews"))
    return merged


def _extend_str(prior: object, current: object) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for item in _as_list(prior) + _as_list(current):
        if isinstance(item, str) and item not in seen:
            seen.add(item)
            values.append(item)
    return values


def _merge_failed(prior: object, current: object) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    index: dict[str, int] = {}
    for item in _as_dicts(prior) + _as_dicts(current):
        scene_id = item.get("id")
        if not isinstance(scene_id, str):
            continue
        if scene_id in index:
            rows[index[scene_id]] = dict(item)
        else:
            index[scene_id] = len(rows)
            rows.append(dict(item))
    return rows


def _merge_windows(prior: object, current: object) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in _as_dicts(prior) + _as_dicts(current):
        scene_id = item.get("id")
        if not isinstance(scene_id, str) or scene_id in seen:
            continue
        seen.add(scene_id)
        rows.append(dict(item))
    return rows


def _as_list(value: object) -> list[object]:
    if isinstance(value, list):
        return list(value)
    return []


def _as_dicts(value: object) -> list[dict[str, Any]]:
    return [item for item in _as_list(value) if isinstance(item, dict)]


def _print_run_summary(
    catalog: SceneCatalog,
    report: DownloadReport,
    *,
    search_only: bool,
) -> None:
    """Print per-source search/pull totals after the progress bar stops."""
    console = Console(stderr=True, highlight=False)
    table = Table(title="atlas data", box=None, pad_edge=False, header_style="bold dim")
    table.add_column("SOURCE", style="cyan")
    table.add_column("SEARCH")
    table.add_column("SCENES", justify="right")
    table.add_column("SAVED", justify="right")
    table.add_column("SKIPPED", justify="right")
    table.add_column("FAILED", justify="right")
    failure_lines: list[str] = []
    preview_lines: list[str] = []
    for src in catalog.sources:
        items = report.for_source(src.source)
        n_saved = sum(1 for item in items if item.status == "saved")
        n_skipped = sum(1 for item in items if item.status == "skipped")
        n_failed = sum(1 for item in items if item.status in {"failed", "no_asset"})
        search_status = "error" if src.error else "ok"
        table.add_row(
            src.source,
            search_status,
            str(len(src.scenes)),
            "—" if search_only else str(n_saved),
            "—" if search_only else str(n_skipped),
            "—" if search_only else str(n_failed),
        )
        if src.error:
            failure_lines.append(f"{src.source}: search {src.error}")
        elif src.next_cursor_unavailable:
            failure_lines.append(_unavailable_warning(src.source, src.next_cursor_unavailable))
        for item in items:
            # A failure is always listed, even one recorded without text.
            if item.error or item.status in {"failed", "no_asset"}:
                detail = item.error or f"{item.status} (no error detail recorded)"
                failure_lines.append(f"{src.source} {item.scene_id}: {detail}")
            if item.preview_error:
                preview_lines.append(f"{src.source} {item.scene_id}: {item.preview_error}")
    console.print()
    console.print(table)
    for line in failure_lines:
        console.print(f"[red]{line}[/red]")
    for line in preview_lines:
        console.print(f"[yellow]{line}[/yellow]")


def _rel(out_dir: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(out_dir.resolve()).as_posix()
    except ValueError:
        return path.name


async def _aclose_all(clients: Mapping[str, DataPullClient]) -> None:
    for client in clients.values():
        close = getattr(client, "aclose", None)
        if close is not None:
            await close()
