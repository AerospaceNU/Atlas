"""``atlas data``: list discovered satellites or search and pull previews."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Mapping
from datetime import date
from pathlib import Path

import httpx
from rich.console import Console
from rich.table import Table

from atlas.cli.catalog import (
    CatalogError,
    render_catalog,
    render_satellite,
    render_satellite_tree,
    resolve_sources,
)
from atlas.cli.download import (
    BrowseAssetError,
    DownloadReport,
    download_assets,
    is_forbidden_asset_name,
)
from atlas.cli.progress import ProgressReporter, pick_reporter
from atlas.compile.aggregate import aggregate
from atlas.compile.product import SceneCatalog
from atlas.data.base import BBox, DataPullClient, PullRequest
from atlas.data.registry import sources

_EXAMPLE = (
    "atlas data sentinel2 optical --date 2024-07-01:2024-07-07 --coords -71.12,42.32,-71.02,42.40"
)


def add_data_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Attach the ``data`` command to the root parser."""
    data = sub.add_parser(
        "data",
        help="Search and pull registered satellite sources",
        description=(
            "List discovered satellites, or search a satellite/kind and download "
            "preview assets in parallel."
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
    data.add_argument("--max-cloud-cover", type=float, default=None)
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
) -> int:
    """Fan out search, optionally download native-resolution assets, write a manifest."""
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
        if not search_only:
            to_pull = [
                scene for src in catalog.sources if src.error is None for scene in src.scenes
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
        for src in catalog.sources:
            reporter.source_done(src.source)
        _write_manifest(out_dir, catalog, report)
        reporter.close()
        _print_run_summary(catalog, report, search_only=search_only)
        if catalog.sources and all(src.error for src in catalog.sources):
            return 1
        return 0
    finally:
        reporter.close()
        if owns_http:
            await live_http.aclose()
        if owns_clients:
            await _aclose_all(live_clients)


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
            if item.status in {"failed", "no_asset"}:
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
