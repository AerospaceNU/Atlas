"""Discover satellites and kinds from registered ``DataPullClient`` classes.

The CLI never hardcodes the source list. It walks ``SOURCES`` (or a caller-supplied
registry) and groups clients by their ``satellite`` and ``scene_kind`` ClassVars.
A new client that sets those the same way as an existing sibling is picked up
without CLI changes.
"""

from __future__ import annotations

import io
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from atlas.data.base import DataPullClient, SceneKind
from atlas.data.registry import SOURCES

ALL = "all"

# Shorthands only — not a source list. Kind values come from SceneKind.
_KIND_ALIASES: dict[str, str] = {
    "l2a": "optical",
    "rgb": "optical",
    "visual": "optical",
    "truecolor": "optical",
    "rtc": "sar",
    "grd": "sar",
    "fire": "detection",
    "hotspot": "detection",
    "lst": "thermal",
    "frp": "thermal",
    "elevation": "dem",
    "dsm": "dem",
    "dtm": "dem",
}

_SAT_ALIASES: dict[str, str] = {
    "s1": "sentinel1",
    "s2": "sentinel2",
    "s3": "sentinel3",
    "s5p": "sentinel5p",
    "tropomi": "sentinel5p",
    "l8": "landsat8",
    "l9": "landsat9",
}

_KIND_STYLE: dict[SceneKind, str] = {
    SceneKind.optical: "green",
    SceneKind.sar: "cyan",
    SceneKind.thermal: "red",
    SceneKind.lidar: "magenta",
    SceneKind.dem: "yellow",
    SceneKind.atmosphere: "blue",
    SceneKind.browse: "bright_black",
    SceneKind.detection: "bright_red",
    SceneKind.altimetry: "bright_blue",
    SceneKind.precipitation: "bright_cyan",
    SceneKind.landcover: "bright_green",
}

_PULL_HINT = "atlas data <satellite> <kind> --date START:END --coords W,S,E,N"


class CatalogError(ValueError):
    """Unknown satellite or kind, or no matching registered sources."""


@dataclass(frozen=True)
class SatelliteInfo:
    """One discovered satellite and the kinds/sources that implement it."""

    name: str
    kinds: tuple[SceneKind, ...]
    sources: tuple[str, ...]
    sources_by_kind: dict[SceneKind, tuple[str, ...]]


def normalize(name: str) -> str:
    """Lowercase a token and strip hyphen/underscore so aliases can match."""
    return name.strip().lower().replace("-", "").replace("_", "")


def _registry(
    registry: Mapping[str, type[DataPullClient]] | None,
) -> dict[str, type[DataPullClient]]:
    return dict(SOURCES if registry is None else registry)


def client_satellite(cls: type[DataPullClient]) -> str:
    """Return the CLI satellite name declared on ``cls``.

    Raises:
        CatalogError: if the ClassVar is missing or blank.
    """
    sat = getattr(cls, "satellite", "")
    if not isinstance(sat, str) or not sat.strip():
        raise CatalogError(f"{cls.__name__} has no satellite ClassVar")
    return sat.strip().lower()


def client_kind(cls: type[DataPullClient]) -> SceneKind:
    """Return the scene kind declared on ``cls``.

    Raises:
        CatalogError: if ``scene_kind`` is not a ``SceneKind``.
    """
    kind = getattr(cls, "scene_kind", None)
    if not isinstance(kind, SceneKind):
        raise CatalogError(f"{cls.__name__} scene_kind is not SceneKind")
    return kind


def list_satellites(
    registry: Mapping[str, type[DataPullClient]] | None = None,
) -> list[SatelliteInfo]:
    """Group registered clients by satellite, then by kind.

    Args:
        registry: Source map to walk. Defaults to ``SOURCES``.

    Returns:
        Sorted satellite records derived only from ClassVars on the registry.
    """
    grouped: dict[str, dict[SceneKind, list[str]]] = defaultdict(lambda: defaultdict(list))
    for name, cls in _registry(registry).items():
        grouped[client_satellite(cls)][client_kind(cls)].append(name)

    infos: list[SatelliteInfo] = []
    for sat in sorted(grouped):
        by_kind = {kind: tuple(sorted(names)) for kind, names in grouped[sat].items()}
        kinds = tuple(sorted(by_kind, key=lambda item: item.value))
        sources = tuple(sorted({name for names in by_kind.values() for name in names}))
        infos.append(SatelliteInfo(name=sat, kinds=kinds, sources=sources, sources_by_kind=by_kind))
    return infos


def _kind_label(kind: SceneKind) -> Text:
    return Text(kind.value, style=_KIND_STYLE.get(kind, "white"))


def _capture(render: Callable[[Console], None]) -> str:
    buffer = io.StringIO()
    console = Console(
        file=buffer,
        force_terminal=False,
        color_system=None,
        width=120,
        highlight=False,
        soft_wrap=True,
    )
    render(console)
    return buffer.getvalue()


def _kind_source_line(info: SatelliteInfo, kind: SceneKind) -> Text:
    sources = ", ".join(info.sources_by_kind[kind])
    return Text.assemble(_kind_label(kind), "  ", Text(sources, style="dim"))


def _add_kinds(tree: Tree, info: SatelliteInfo, kinds: tuple[SceneKind, ...]) -> None:
    for kind in kinds:
        tree.add(_kind_source_line(info, kind))


def _kinds_text(info: SatelliteInfo) -> Text:
    line = Text()
    for i, kind in enumerate(info.kinds):
        if i:
            line.append("  ")
        line.append_text(_kind_label(kind))
    return line


def _kinds_width(info: SatelliteInfo) -> int:
    return len("  ".join(kind.value for kind in info.kinds))


def _catalog_pair_row(
    left: SatelliteInfo,
    right: SatelliteInfo | None,
    sat_w: tuple[int, int],
    kind_w: tuple[int, int],
) -> Text:
    row = Text()
    row.append(f"{left.name:<{sat_w[0]}}  ", style="bold cyan")
    kinds = _kinds_text(left)
    kinds.pad_right(max(0, kind_w[0] - _kinds_width(left)))
    row.append_text(kinds)
    row.append("    ")
    if right is not None:
        row.append(f"{right.name:<{sat_w[1]}}  ", style="bold cyan")
        row.append_text(_kinds_text(right))
    return row


def render_catalog(
    console: Console,
    registry: Mapping[str, type[DataPullClient]] | None = None,
) -> None:
    """Print a compact satellite → kinds overview (sources live on ``atlas SAT --list``)."""
    infos = list_satellites(registry)
    if not infos:
        console.print("No satellites registered.")
        return

    mid = (len(infos) + 1) // 2
    left_rows, right_rows = infos[:mid], infos[mid:]
    sat_w = (
        max(len("SATELLITE"), max(len(item.name) for item in left_rows)),
        max(len("SATELLITE"), max((len(item.name) for item in right_rows), default=0)),
    )
    kind_w = (
        max(len("KINDS"), max(_kinds_width(item) for item in left_rows)),
        max(len("KINDS"), max((_kinds_width(item) for item in right_rows), default=0)),
    )

    header = Text()
    header.append(f"{'SATELLITE':<{sat_w[0]}}  ", style="bold dim")
    header.append(f"{'KINDS':<{kind_w[0]}}", style="bold dim")
    header.append("    ")
    header.append(f"{'SATELLITE':<{sat_w[1]}}  ", style="bold dim")
    header.append("KINDS", style="bold dim")
    rule = Text(
        f"{'─' * sat_w[0]}  {'─' * kind_w[0]}    {'─' * sat_w[1]}  {'─' * kind_w[1]}",
        style="dim",
    )

    console.print(Text(f"satellites  {len(infos)}", style="bold"))
    console.print(header, overflow="ignore", crop=False, soft_wrap=False)
    console.print(rule, overflow="ignore", crop=False, soft_wrap=False)
    for index, left in enumerate(left_rows):
        right = right_rows[index] if index < len(right_rows) else None
        console.print(
            _catalog_pair_row(left, right, sat_w, kind_w),
            overflow="ignore",
            crop=False,
            soft_wrap=False,
        )
    console.print()
    console.print(Text("atlas <satellite> --list   kinds and sources", style="dim"))
    console.print(Text(_PULL_HINT, style="dim"))


def format_catalog(registry: Mapping[str, type[DataPullClient]] | None = None) -> str:
    """Render the satellite catalog as text (used by tests and non-console callers)."""
    return _capture(lambda console: render_catalog(console, registry))


def render_satellite_tree(
    console: Console,
    name: str,
    registry: Mapping[str, type[DataPullClient]] | None = None,
    *,
    kind: str | None = None,
) -> None:
    """Print one satellite's kinds as a tree (``atlas alos --list``)."""
    info = _find_satellite(name, registry)
    kinds = info.kinds
    if kind:
        want = _parse_kind(kind)
        if want is not None:
            if want not in info.sources_by_kind:
                available = ", ".join(k.value for k in info.kinds)
                raise CatalogError(
                    f"No sources for satellite {info.name!r} with kind {kind!r}. "
                    f"Available kinds: {available}"
                )
            kinds = (want,)
    tree = Tree(Text(info.name, style="bold cyan"))
    _add_kinds(tree, info, kinds)
    console.print(tree)
    console.print()
    console.print(
        Text(f"atlas data {info.name} <kind> --date START:END --coords W,S,E,N", style="dim")
    )


def render_satellite(
    console: Console,
    name: str,
    registry: Mapping[str, type[DataPullClient]] | None = None,
) -> None:
    """Print kinds and sources for one satellite."""
    info = _find_satellite(name, registry)
    grid = Table.grid(padding=(0, 2))
    grid.add_column(no_wrap=True)
    grid.add_column()
    for kind in info.kinds:
        grid.add_row(
            _kind_label(kind),
            Text(", ".join(info.sources_by_kind[kind]), style="dim"),
        )
    n_kinds = len(info.kinds)
    subtitle = f"{n_kinds} kind" if n_kinds == 1 else f"{n_kinds} kinds"
    console.print(
        Panel(
            grid,
            title=Text(info.name, style="bold cyan"),
            subtitle=Text(subtitle, style="dim"),
            border_style="cyan",
            padding=(0, 1),
            expand=False,
        )
    )
    console.print(
        Text(f"atlas data {info.name} <kind> --date START:END --coords W,S,E,N", style="dim")
    )


def is_known_target(
    token: str,
    registry: Mapping[str, type[DataPullClient]] | None = None,
) -> bool:
    """True if ``token`` is ``all``, a satellite family, shorthand, or registry key."""
    if not token or token.startswith("-"):
        return False
    sat_norm = normalize(_SAT_ALIASES.get(normalize(token), token))
    if sat_norm == ALL:
        return True
    reg = _registry(registry)
    if _pin_source(token.strip(), reg) is not None:
        return True
    return any(normalize(info.name) == sat_norm for info in list_satellites(reg))


def format_satellite(
    name: str,
    registry: Mapping[str, type[DataPullClient]] | None = None,
) -> str:
    """Render kinds and sources for one satellite (or a pinned registry key)."""
    return _capture(lambda console: render_satellite(console, name, registry))


def resolve_sources(
    satellite: str,
    kind: str,
    registry: Mapping[str, type[DataPullClient]] | None = None,
) -> list[str]:
    """Map a user satellite + kind onto registry keys.

    ``all`` selects every registered source (then filtered by kind). A registry
    key (``sentinel2``, ``hls_sentinel``) pins that one client. Shorthands like
    ``s2`` resolve to the same key. A family name that is not itself a registry
    key (``alos``, ``modis``, ``landsat``) selects every client with that
    ``satellite`` ClassVar.

    Args:
        satellite: Family name, registry key, shorthand, or ``all``.
        kind: ``SceneKind`` value, shorthand, or ``all``.
        registry: Source map to walk. Defaults to ``SOURCES``.

    Returns:
        Sorted registry keys to instantiate.

    Raises:
        CatalogError: unknown satellite/kind, or no client matches the pair.
    """
    reg = _registry(registry)
    sat_token = satellite.strip()
    sat_norm = normalize(_SAT_ALIASES.get(normalize(sat_token), sat_token))
    want_kind = _parse_kind(kind)

    if sat_norm == ALL:
        candidates = list(reg)
    else:
        families = {normalize(info.name): info.name for info in list_satellites(reg)}
        pinned = _pin_source(sat_token, reg)
        if pinned is None:
            pinned = next((key for key in reg if normalize(key) == sat_norm), None)
        if pinned is not None:
            candidates = [pinned]
        elif sat_norm in families:
            family = families[sat_norm]
            candidates = [name for name, cls in reg.items() if client_satellite(cls) == family]
        else:
            known = ", ".join(sorted(families.values()))
            raise CatalogError(f"Unknown satellite {satellite!r}. Known: {known}, all")

    if want_kind is None:
        chosen = list(candidates)
    else:
        chosen = [name for name in candidates if client_kind(reg[name]) is want_kind]

    if not chosen:
        available = ", ".join(sorted({client_kind(reg[n]).value for n in candidates})) or "(none)"
        raise CatalogError(
            f"No sources for satellite {satellite!r} with kind {kind!r}. Available kinds: {available}"
        )
    return sorted(chosen)


def _parse_kind(kind: str) -> SceneKind | None:
    token = kind.strip()
    norm = normalize(token)
    if norm == ALL:
        return None
    mapped = _KIND_ALIASES.get(norm, token.strip().lower())
    try:
        return SceneKind(mapped)
    except ValueError:
        known = ", ".join(item.value for item in SceneKind)
        raise CatalogError(f"Unknown kind {kind!r}. Known: {known}, all") from None


def _pin_source(token: str, reg: Mapping[str, type[DataPullClient]]) -> str | None:
    if token in reg:
        return token
    lower = token.lower()
    if lower in reg:
        return lower
    return None


def _find_satellite(
    name: str,
    registry: Mapping[str, type[DataPullClient]] | None,
) -> SatelliteInfo:
    reg = _registry(registry)
    sat_norm = normalize(_SAT_ALIASES.get(normalize(name), name))
    for info in list_satellites(reg):
        if normalize(info.name) == sat_norm:
            return info
    pinned = _pin_source(name.strip(), reg)
    if pinned is not None:
        cls = reg[pinned]
        kind = client_kind(cls)
        return SatelliteInfo(
            name=pinned,
            kinds=(kind,),
            sources=(pinned,),
            sources_by_kind={kind: (pinned,)},
        )
    known = ", ".join(info.name for info in list_satellites(reg))
    raise CatalogError(f"Unknown satellite {name!r}. Known: {known}, all")
