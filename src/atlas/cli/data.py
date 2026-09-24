from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import date
from pathlib import Path

import httpx
from rich.console import Console
from rich.table import Table

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
    is_forbidden_asset_name,
)
from atlas.cli.progress import ProgressReporter, pick_reporter
from atlas.compile.aggregate import aggregate
from atlas.compile.product import SceneCatalog
from atlas.data.base import BBox, DataPullClient, PullRequest, Scene
from atlas.data.registry import sources
from atlas.data.sources.fldas import (
    MISSING_TOKEN_MESSAGE,
    FldasClient,
    FldasError,
    FldasRequestError,
    _skip_note,
    load_repo_env,
    parse_points_csv,
    resolve_variables,
    split_variables,
)

_EXAMPLE = (
    "atlas data sentinel2 optical --date 2024-07-01:2024-07-07 "
    "--coords -71.12,42.32,-71.02,42.40\n"
    "  atlas data fldas landsurface --date 2020-01-01:2020-01-31 "
    "--coords -71.2,42.2,-70.9,42.5 --out .atlas/data/fldas"
)
_EXTRACT_HINT = "FLDAS table extraction runs alone; pass only the fldas source"
_TABLE_NAME = "fldas.csv"


def add_data_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Attach the ``data`` command to the root parser."""
    data = sub.add_parser(
        "data",
        help="Search and pull registered satellite sources",
        description=(
            "List discovered satellites, or search a satellite and download native assets. "
            "fldas landsurface writes a CSV training table."
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
        "--out",
        type=Path,
        default=Path(".atlas/data"),
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
        "--asset",
        default=None,
        help="Asset key to download (default: visual COG, then data). Previews/thumbnails are rejected.",
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
    if not args.date or not args.coords:
        print("data pull requires --date START:END and --coords W,S,E,N", file=sys.stderr)
        return 2
    try:
        names = resolve_sources(args.satellite, args.kind)
        start, end = parse_date_range(args.date)
        request = PullRequest(
            start_date=start,
            end_date=end,
            bbox=parse_coords(args.coords),
            min_cloud_cover=args.min_cloud_cover,
            max_cloud_cover=args.max_cloud_cover,
            limit=args.limit,
        )
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
        and not (os.environ.get("EARTHDATA_TOKEN") or "").strip()
    ):
        print(MISSING_TOKEN_MESSAGE, file=sys.stderr)
        return 1

    reporter = pick_reporter(stream=sys.stderr)
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
) -> int:
    """Fan out search, then extract training tables or download native assets.

    A source that derives a table instead of assets (FLDAS) writes its CSV here;
    its failures are recorded so a table that was never written returns 1.
    """
    owns_clients = clients is None
    live_clients = dict(clients) if clients is not None else sources(*source_names)
    owns_http = http is None
    live_http = (
        http if http is not None else httpx.AsyncClient(timeout=300.0, follow_redirects=True)
    )
    try:
        for name in live_clients:
            reporter.source_start(name)
        catalog = await aggregate(
            request,
            live_clients,
            on_source=lambda src: reporter.source_search_done(
                src.source, scenes=len(src.scenes), error=src.error
            ),
        )
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
                    on_progress=lambda name, n_saved, total: reporter.source_pull(
                        name, saved=n_saved, total=total
                    ),
                )
            except BrowseAssetError as exc:
                print(str(exc), file=sys.stderr)
                return 2
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
        for src in catalog.sources:
            reporter.source_done(src.source)
        _write_manifest(out_dir, catalog, report)
        reporter.close()
        _print_run_summary(catalog, report, search_only=search_only)
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
        return (
            DownloadOutcome(
                source=source,
                scene_id=f"{source}-table",
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            ),
            True,
        )
    except (FldasError, httpx.HTTPError, OSError, ValueError) as exc:
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


def _write_manifest(
    out_dir: Path,
    catalog: SceneCatalog,
    report: DownloadReport,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "request": catalog.request.model_dump(mode="json"),
        "sources": [
            {
                "source": src.source,
                "collection": src.collection,
                "error": src.error,
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
            }
            for src in catalog.sources
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


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
        for item in items:
            if item.error:
                failure_lines.append(f"{src.source} {item.scene_id}: {item.error}")
    console.print()
    console.print(table)
    for line in failure_lines:
        console.print(f"[red]{line}[/red]")


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
