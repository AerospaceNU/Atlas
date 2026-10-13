"""Size estimate for a pull, computed from search metadata.

The figures describe the download that would follow. They do not start it,
so a caller can show the estimate without prompting.
"""

from __future__ import annotations

import sys
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import TextIO

from atlas.cli.download import BrowseAssetError, pick_asset
from atlas.compile.product import SceneCatalog
from atlas.data.base import Scene

# A successful search that returned nothing will not be downloaded.
NO_COVERAGE = "no coverage"
CONFIRM_PROMPT = "Start the download? [y/N] "
DOWNLOAD_CANCELLED = "download cancelled"
CONFIRMATION_REQUIRED = (
    "refusing to download without confirmation because stdin is not a terminal; "
    "pass --yes to start the download"
)
EXIT_CANCELLED = 1
EXIT_REFUSED = 2
_YES = frozenset({"y", "yes"})


@dataclass(frozen=True)
class SkippedSource:
    """One source that will not be downloaded, and why.

    Attributes:
        source: Registry name.
        reason: Search error, or :data:`NO_COVERAGE` when the search was empty.
    """

    source: str
    reason: str


@dataclass(frozen=True)
class SourceSize:
    """Catalog size for one source that will be downloaded.

    Attributes:
        source: Registry name.
        scene_count: Scenes the search returned.
        known_bytes: Sum of reported asset sizes. Scenes without a size
            contribute nothing here.
        unknown_scenes: Scenes whose size the catalog did not report.
    """

    source: str
    scene_count: int
    known_bytes: int
    unknown_scenes: int

    @property
    def bytes_text(self) -> str:
        """Byte total, or ``unknown`` when this source did not report sizes."""
        return format_byte_estimate(
            scene_count=self.scene_count,
            known_bytes=self.known_bytes,
            unknown_scenes=self.unknown_scenes,
        )


@dataclass(frozen=True)
class PullEstimate:
    """Scene count, known bytes, and sources a pull would skip.

    Attributes:
        sources: Sources that have scenes and no search error, in catalog order.
        skipped: Sources that will not be downloaded, in catalog order.
    """

    sources: tuple[SourceSize, ...]
    skipped: tuple[SkippedSource, ...]

    @property
    def scene_count(self) -> int:
        """Scenes that would be downloaded."""
        return sum(item.scene_count for item in self.sources)

    @property
    def known_bytes(self) -> int:
        """Sum of asset sizes the catalogs reported.

        Scenes that did not report a size are not included. Check
        :attr:`unknown_scenes` before treating this as the whole pull.
        """
        return sum(item.known_bytes for item in self.sources)

    @property
    def unknown_scenes(self) -> int:
        """Scenes whose byte size the catalog did not report."""
        return sum(item.unknown_scenes for item in self.sources)

    @property
    def bytes_text(self) -> str:
        """Byte total for the pull, or ``unknown`` when none was reported."""
        return format_byte_estimate(
            scene_count=self.scene_count,
            known_bytes=self.known_bytes,
            unknown_scenes=self.unknown_scenes,
        )


def format_byte_estimate(*, scene_count: int, known_bytes: int, unknown_scenes: int) -> str:
    """Format a byte total without filling in sizes the catalog omitted.

    Args:
        scene_count: Scenes included in this total.
        known_bytes: Sum of reported sizes.
        unknown_scenes: Scenes with no reported size.

    Returns:
        ``0`` when there are no scenes, ``unknown`` when every scene lacks a
        size, the integer total when every scene reported one, or the known
        total plus an unknown count when the catalog was only partly sized.
    """
    if scene_count <= 0:
        return "0"
    if unknown_scenes <= 0:
        return str(known_bytes)
    if unknown_scenes >= scene_count:
        return "unknown"
    return f"{known_bytes} ({unknown_scenes} unknown)"


def estimate_pull(
    catalog: SceneCatalog,
    *,
    asset_name: str | None = None,
    unknown_byte_sources: Collection[str] = (),
) -> PullEstimate:
    """Estimate a pull from search results, before any granule is downloaded.

    Scene counts come from the catalog. Byte totals sum ``file:size`` on the
    asset the download would select. A scene whose source did not report that
    size stays unknown. Names in ``unknown_byte_sources`` stay unknown even
    when an asset carries a size, because that size is not what the pull
    writes (a training-table extract, for example). Nothing is guessed. A
    source with a search error is skipped with that error. A source that
    returned no scenes is skipped as no coverage.

    Args:
        catalog: Search results for one pull.
        asset_name: Band or asset key the download would select.
        unknown_byte_sources: Registry names whose byte size this catalog
            cannot report. Their scenes still count.

    Returns:
        Scene counts, known bytes, and skipped sources.
    """
    unknown_names = set(unknown_byte_sources)
    included: list[SourceSize] = []
    skipped: list[SkippedSource] = []
    for src in catalog.sources:
        if src.error is not None:
            skipped.append(
                SkippedSource(source=src.source, reason=src.error.strip() or "search failed")
            )
            continue
        if not src.scenes:
            skipped.append(SkippedSource(source=src.source, reason=NO_COVERAGE))
            continue
        known_bytes, unknown_scenes = _bytes_for_scenes(
            src.scenes,
            asset_name=asset_name,
            force_unknown=src.source in unknown_names,
        )
        included.append(
            SourceSize(
                source=src.source,
                scene_count=len(src.scenes),
                known_bytes=known_bytes,
                unknown_scenes=unknown_scenes,
            )
        )
    return PullEstimate(sources=tuple(included), skipped=tuple(skipped))


def render_pull_estimate(estimate: PullEstimate) -> str:
    """Return the estimate as plain text for the CLI.

    Args:
        estimate: Figures from :func:`estimate_pull`.

    Returns:
        A short block: scene count, bytes, per-source lines, and skipped sources.
    """
    lines = [
        "Pull estimate",
        f"scenes: {estimate.scene_count}",
        f"bytes: {estimate.bytes_text}",
    ]
    if estimate.sources:
        lines.append("sources:")
        for source in estimate.sources:
            lines.append(f"  {source.source}: {source.scene_count} scenes, {source.bytes_text}")
    if estimate.skipped:
        lines.append("skipped:")
        for skipped in estimate.skipped:
            lines.append(f"  {skipped.source}: {skipped.reason}")
    else:
        lines.append("skipped: none")
    return "\n".join(lines)


def confirm_download(
    *,
    assume_yes: bool,
    stdin: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int | None:
    """Decide whether a download may start.

    ``assume_yes`` skips the prompt. A terminal reads one line; ``y`` and
    ``yes`` proceed. Any other answer cancels. When stdin is not a terminal,
    the download is refused without reading stdin, so a pipe cannot hang.

    Args:
        assume_yes: True when ``--yes`` was passed.
        stdin: Stream to read. Defaults to ``sys.stdin``.
        stderr: Stream for the prompt and the refusal. Defaults to ``sys.stderr``.

    Returns:
        ``None`` when the download may start. :data:`EXIT_CANCELLED` when the
        person declined. :data:`EXIT_REFUSED` when there is no terminal and
        ``assume_yes`` is false.
    """
    if assume_yes:
        return None
    incoming = stdin if stdin is not None else sys.stdin
    outgoing = stderr if stderr is not None else sys.stderr
    if not _is_tty(incoming):
        print(CONFIRMATION_REQUIRED, file=outgoing, flush=True)
        return EXIT_REFUSED
    print(CONFIRM_PROMPT, file=outgoing, flush=True)
    try:
        line = incoming.readline()
    except EOFError:
        print(DOWNLOAD_CANCELLED, file=outgoing, flush=True)
        return EXIT_CANCELLED
    if line.strip().casefold() in _YES:
        return None
    print(DOWNLOAD_CANCELLED, file=outgoing, flush=True)
    return EXIT_CANCELLED


def _bytes_for_scenes(
    scenes: Sequence[Scene],
    *,
    asset_name: str | None,
    force_unknown: bool,
) -> tuple[int, int]:
    """Return ``(known_bytes, unknown_scenes)`` for ``scenes``.

    A scene with no downloadable asset contributes zero known bytes. That is
    a reported empty transfer, not an unknown size.
    """
    if force_unknown:
        return 0, len(scenes)
    known_bytes = 0
    unknown_scenes = 0
    for scene in scenes:
        size = _selected_size(scene, asset_name)
        if size is None:
            unknown_scenes += 1
        else:
            known_bytes += size
    return known_bytes, unknown_scenes


def _selected_size(scene: Scene, asset_name: str | None) -> int | None:
    """Return the selected asset's reported size.

    ``0`` when nothing would be downloaded. ``None`` when the asset has no
    catalog size.
    """
    try:
        picked = pick_asset(scene, asset_name)
    except BrowseAssetError:
        return 0
    if picked is None:
        return 0
    return picked[1].size_bytes


def _is_tty(stream: TextIO) -> bool:
    """True when ``stream`` is an interactive terminal."""
    check = getattr(stream, "isatty", None)
    if not callable(check):
        return False
    try:
        return bool(check())
    except (OSError, ValueError):
        return False
