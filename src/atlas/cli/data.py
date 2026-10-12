from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import httpx
from pydantic import ValidationError
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
from atlas.data.base import BBox, DataPullClient, Polygon, PullRequest, Scene
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


def add_data_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Attach the ``data`` command to the root parser."""
    data = sub.add_parser(
        "data",
        help="Search and pull registered satellite sources",
        description=(
            "List discovered satellites, or search a satellite and download native assets. "
            "A geographic, Web Mercator, or WGS84 UTM COG is read as the window "
            "overlapping --coords. fldas landsurface writes a CSV training table."
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
        "--resume",
        action="store_true",
        help=(
            "Continue from next-page cursors in the output manifest "
            "instead of requesting page one again"
        ),
    )
    data.add_argument(
        "--asset",
        default=None,
        help=(
            "Band name (red, nir) or asset key to download; band names win over a "
            "literal key of the same name. Default: visual COG, then data. "
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
        and not (os.environ.get("EARTHDATA_TOKEN") or "").strip()
    ):
        print(MISSING_TOKEN_MESSAGE, file=sys.stderr)
        return 1

    reporter = pick_reporter(stream=sys.stderr)
    page_cursors: Mapping[str, str | None] | None = None
    prior_manifest: Mapping[str, Any] | None = None
    if getattr(args, "resume", False):
        try:
            prior_manifest = _load_pull_manifest(args.out / "manifest.json")
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
        if not _same_pull(prior_manifest.get("request"), request):
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
        if prior_manifest is not None:
            mismatch = _resume_mismatch(prior_manifest, live_clients)
            if mismatch is not None:
                print(mismatch, file=sys.stderr)
                return 2
        unavailable = _unavailable_cursors(prior_manifest) if prior_manifest is not None else None
        if unavailable:
            for name in sorted(unavailable):
                if name in live_clients:
                    print(f"{name}: {_unavailable_note(unavailable[name])}", file=sys.stderr)
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
                    bbox=request.bbox,
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
        _write_manifest(out_dir, catalog, report, prior=prior_manifest)
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
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    request = catalog.request
    bbox = request.bbox.as_list()
    prior_sources = _prior_sources(prior)
    sources_payload = [
        _merge_source(
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
            prior_sources.get(src.source),
        )
        for src in catalog.sources
    ]
    payload = {
        "request": request.model_dump(mode="json"),
        "polygon": None if request.polygon is None else request.polygon.as_geojson(),
        "catalog_bbox": bbox,
        "catalog_bbox_note": _catalog_bbox_note(request),
        "sources": sources_payload,
    }
    _write_json_atomic(out_dir / "manifest.json", json.dumps(payload, indent=2) + "\n")


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


def _write_json_atomic(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` via an exclusive temp file, fsync, and ``os.replace``.

    The directory is fsynced after the replace. A failed replace leaves the
    previous file in place and removes the temp file this write created.
    """
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd: int | None = None
    created = False
    try:
        fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        created = True
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = None
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        _fsync_directory(path.parent)
    except Exception:
        if fd is not None:
            os.close(fd)
        if created and tmp.exists():
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
            failure_lines.append(f"{src.source}: {_unavailable_note(src.next_cursor_unavailable)}")
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
