"""``atlas scenes``: answer "what imagery do we have here?" from the local index."""

from __future__ import annotations

import argparse
import sqlite3
from datetime import date
from pathlib import Path

from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from atlas.archive import (
    INDEX_FILENAME,
    Availability,
    CloudStatus,
    IndexVersionError,
    QueryGroup,
    QueryResult,
    SceneCard,
    SceneIndex,
    SceneOrder,
    SceneQuery,
)
from atlas.cli.data import parse_coords, parse_date_range

_EXAMPLE = (
    "atlas scenes --coords -71.12,42.32,-71.02,42.40 --date 2024-06-01:2024-06-30\n"
    "  atlas scenes --coords -71.12,42.32,-71.02,42.40 --on 2024-06-03 --on 2024-09-01"
)


def add_scenes_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    scenes = sub.add_parser(
        "scenes",
        help="Query the local scene index without contacting providers",
        description=(
            "List indexed scenes over an area and time. Every `atlas data` pull, and every "
            "get_satellite_data call, adds the scenes it found or saved to "
            "<out>/index.sqlite. Dates are UTC days. Paths are shown relative to the "
            "current directory."
        ),
        epilog=f"examples:\n  {_EXAMPLE}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    scenes.add_argument("--coords", help="Area of interest as west,south,east,north")
    scenes.add_argument("--date", help="Date window as START:END (inclusive)")
    scenes.add_argument(
        "--on",
        action="append",
        type=date.fromisoformat,
        help="One date, answered on its own; repeat for before/after (gaps are reported)",
    )
    scenes.add_argument("--source", action="append", help="Registry source (repeatable)")
    scenes.add_argument(
        "--satellite", action="append", help="Satellite family, e.g. sentinel2 (repeatable)"
    )
    scenes.add_argument("--platform", action="append", help="Platform, e.g. landsat-9 (repeatable)")
    scenes.add_argument(
        "--max-cloud-cover",
        type=float,
        default=None,
        help=(
            "Maximum cloud cover. Sensors that do not see cloud (radar, lidar, elevation) "
            "pass; optical scenes without cloud data are excluded and counted"
        ),
    )
    scenes.add_argument(
        "--order",
        choices=[order.value for order in SceneOrder],
        default=SceneOrder.time.value,
        help="time: oldest first (default); cloud: clearest first, applied before --limit",
    )
    scenes.add_argument("--limit", type=int, default=50, help="Max scenes per date or range")
    scenes.add_argument(
        "--out",
        type=Path,
        default=Path(".atlas/data"),
        help="Data directory holding index.sqlite (default: .atlas/data)",
    )
    scenes.add_argument("--json", action="store_true", help="Print the query result as JSON")
    scenes.add_argument(
        "--rebuild",
        action="store_true",
        help="Re-derive every card from stored scene metadata (offline)",
    )
    scenes.set_defaults(_atlas_cmd="scenes")


def run_scenes(args: argparse.Namespace) -> int:
    """Run ``atlas scenes``.

    Returns:
        Process exit code: 0 when the query ran (even with no scenes), 1 when
        there is no readable index, 2 for usage errors.
    """
    err = Console(stderr=True, highlight=False)
    path: Path = args.out / INDEX_FILENAME
    if not path.exists():
        err.print(
            f"no scene index at {path}; search or pull first, e.g.\n"
            "  atlas sentinel2 optical --date 2024-06-01:2024-06-10 "
            "--coords -71.12,42.32,-71.02,42.40 --search-only",
            style="red",
        )
        return 1
    if not args.rebuild and not args.coords:
        err.print("atlas scenes needs --coords, plus --date START:END or --on DATE", style="red")
        return 2
    query: SceneQuery | None = None
    if args.coords:
        try:
            start, end = parse_date_range(args.date) if args.date else (None, None)
            query = SceneQuery(
                bbox=parse_coords(args.coords),
                start=start,
                end=end,
                on=args.on or [],
                sources=args.source or [],
                satellites=args.satellite or [],
                platforms=args.platform or [],
                max_cloud_cover=args.max_cloud_cover,
                order=args.order,
                limit=args.limit,
            )
        except (ValidationError, ValueError) as exc:
            err.print(_usage_message(exc), style="red")
            return 2
    try:
        with SceneIndex(path) as index:
            if args.rebuild:
                err.print(f"rebuilt {index.rebuild()} scene cards in {path}", style="dim")
            if query is None:
                return 0
            result = index.query(query, workspace=Path.cwd())
    except (IndexVersionError, sqlite3.Error) as exc:
        err.print(f"cannot read {path}: {exc}", style="red")
        return 1
    if args.json:
        print(result.to_json(indent=2))
        return 0
    console = Console(highlight=False)
    for group in result.groups:
        _render_group(console, group, result)
    return 0


def _usage_message(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        return "; ".join(error["msg"].removeprefix("Value error, ") for error in exc.errors())
    return str(exc)


def _render_group(console: Console, group: QueryGroup, result: QueryResult) -> None:
    window = (
        group.start.isoformat()
        if group.start == group.end
        else f"{group.start.isoformat()} to {group.end.isoformat()}"
    )
    notes: list[str] = []
    if group.excluded_unknown_cloud:
        notes.append(f"{group.excluded_unknown_cloud} without cloud data excluded")
    if group.gap:
        suffix = f" ({'; '.join(notes)})" if notes else ""
        console.print(f"[bold]{window}[/bold]: [yellow]no scenes found[/yellow]{suffix}")
        return
    if group.matched > len(group.cards):
        notes.append(f"showing {len(group.cards)} of {group.matched}; raise --limit for more")
    table = Table(
        title=f"{window}: {group.matched} scene{'s' if group.matched != 1 else ''}",
        title_justify="left",
        box=None,
        pad_edge=False,
        header_style="bold dim",
    )
    # Index ids are citations: never wrap or elide them, shrink other columns instead.
    table.add_column("INDEX ID", no_wrap=True, min_width=len(group.cards[0].index_id))
    table.add_column("TIME (UTC)", no_wrap=True)
    paths = [_local_path(card, result) for card in group.cards]
    columns = ("SATELLITE", "COLLECTION", "CLOUD", "LOCAL", *(["PATH"] if any(paths) else []))
    for column in columns:
        table.add_column(column, overflow="fold")
    for card, path in zip(group.cards, paths, strict=True):
        table.add_row(
            f"[cyan]{card.index_id}[/cyan]",
            card.datetime.strftime("%Y-%m-%d %H:%M"),
            card.satellite,
            card.collection,
            _cloud(card),
            _local_state(card),
            *([path or ""] if any(paths) else []),
        )
    console.print(table)
    for note in notes:
        console.print(f"[dim]{note}[/dim]")
    console.print()


def _cloud(card: SceneCard) -> str:
    if card.cloud_status is CloudStatus.not_applicable:
        return "n/a"
    return "—" if card.cloud_cover is None else f"{card.cloud_cover:.0f}%"


def _local_state(card: SceneCard) -> str:
    """The most complete local state of any layer: full, windowed, missing, or —."""
    states = {layer.availability for layer in card.layers.values()}
    for state in (Availability.full, Availability.windowed, Availability.missing):
        if state in states:
            return state.value
    return "—"


def _local_path(card: SceneCard, result: QueryResult) -> str | None:
    for layer in card.layers.values():
        if layer.locator is not None:
            if layer.locator.scheme == "workspace":
                return layer.locator.key
            return (Path(result.index_dirs[0]) / layer.locator.key).as_posix()
    return None
