"""Size estimate for a pull, computed from search metadata.

The figures describe the download that would follow. They do not start it,
so a caller can show the estimate without prompting.
"""

from __future__ import annotations

import sys
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from atlas.cli.download import (
    BrowseAssetError,
    _is_geotiff_asset,
    asset_suffix,
    existing_product,
    output_source_dir,
    pick_asset,
    plain_source_name,
    same_aoi,
    scene_filename,
)
from atlas.compile.product import SceneCatalog
from atlas.data.base import Asset, BBox, Scene
from atlas.data.registry import SOURCES

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
# Published policy limits are not an account balance. These providers do not
# return the signed-in account's remaining transfer, so the estimate says so
# instead of filling in a number.
QUOTA_NOT_READABLE = "not readable"
_BYTE_UNITS = ("bytes", "KB", "MB", "GB", "TB", "PB")
_EARTHDATA_MODULES = frozenset({"atlas.data.modis_base", "atlas.data.sources.fldas"})


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
        scene_count: Scenes that would still be downloaded. Files already on
            disk or listed in a resume manifest are not included.
        known_bytes: Sum of reported asset sizes. Scenes without a size
            contribute nothing here.
        unknown_scenes: Scenes whose size the catalog did not report.
        windowed: Scenes read as a COG window. Their catalog size is an upper
            bound; the bytes transferred are usually much smaller.
    """

    source: str
    scene_count: int
    known_bytes: int
    unknown_scenes: int
    windowed: int = 0

    @property
    def bytes_text(self) -> str:
        """Byte total, or ``unknown`` when this source did not report sizes."""
        return format_byte_estimate(
            scene_count=self.scene_count,
            known_bytes=self.known_bytes,
            unknown_scenes=self.unknown_scenes,
        )


@dataclass(frozen=True)
class ProviderQuota:
    """Remaining quota for one provider, when that provider returned a number.

    ``remaining_bytes`` is set only from a quota response. A published policy
    limit is not a remaining balance, so it stays ``None`` and the text says
    the quota is not readable.

    Attributes:
        provider: ``CDSE``, ``Earthdata``, or ``""`` when no named provider
            in this pull publishes a quota.
        remaining_bytes: Remaining transfer in bytes, or ``None`` when the
            provider does not expose one.
    """

    provider: str
    remaining_bytes: int | None


@dataclass(frozen=True)
class PullEstimate:
    """Scene count, known bytes, and sources a pull would skip.

    Attributes:
        sources: Sources that still have something to download, in catalog order.
        skipped: Sources that will not be downloaded, in catalog order.
        already_present: Scenes left out because the file is on disk or listed
            in the resume manifest.
        quotas: Remaining quota for providers in this pull. Empty means the
            estimate has not attached a quota note.
    """

    sources: tuple[SourceSize, ...]
    skipped: tuple[SkippedSource, ...]
    already_present: int = 0
    quotas: tuple[ProviderQuota, ...] = ()

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
        """Scenes whose byte count the catalog does not provide."""
        return sum(item.unknown_scenes for item in self.sources)

    @property
    def windowed_scenes(self) -> int:
        """Scenes read as a window, so the catalog size is only an upper bound."""
        return sum(item.windowed for item in self.sources)

    @property
    def bytes_text(self) -> str:
        """Byte total for the pull, or ``unknown`` when none was reported."""
        return format_byte_estimate(
            scene_count=self.scene_count,
            known_bytes=self.known_bytes,
            unknown_scenes=self.unknown_scenes,
        )

    @property
    def quota_text(self) -> str:
        """Remaining quota, or a note that none of these providers expose one."""
        if not self.quotas:
            return QUOTA_NOT_READABLE
        parts: list[str] = []
        for item in self.quotas:
            if item.remaining_bytes is None:
                if item.provider:
                    parts.append(f"{item.provider}: {QUOTA_NOT_READABLE}")
                else:
                    parts.append(QUOTA_NOT_READABLE)
                continue
            label = item.provider or "quota"
            parts.append(f"{label}: {format_bytes(item.remaining_bytes)} remaining")
        return "; ".join(parts)


def format_bytes(n: int) -> str:
    """Return ``n`` as ``700 MB (734003200 bytes)``.

    Units step by 1024. The number in parentheses is the exact byte count.
    A whole number of units is printed without a decimal.

    Args:
        n: A byte count of zero or more.

    Returns:
        The human-readable size and the exact byte count.
    """
    if n < 0:
        raise ValueError("byte count must be >= 0")
    return f"{_human_bytes(n)} ({n} bytes)"


def format_byte_estimate(*, scene_count: int, known_bytes: int, unknown_scenes: int) -> str:
    """Format a byte total without filling in sizes the catalog omitted.

    Args:
        scene_count: Scenes included in this total.
        known_bytes: Sum of reported sizes.
        unknown_scenes: Scenes with no reported size.

    Returns:
        ``0 bytes (0 bytes)`` when there are no scenes, ``unknown`` when every
        scene lacks a size, a human-readable total when every scene reported
        one, or that total plus an unknown count when the catalog was only
        partly sized.
    """
    if scene_count <= 0:
        return format_bytes(0)
    if unknown_scenes <= 0:
        return format_bytes(known_bytes)
    if unknown_scenes >= scene_count:
        return "unknown"
    return f"{_human_bytes(known_bytes)} ({known_bytes} bytes, {unknown_scenes} unknown)"


def estimate_pull(
    catalog: SceneCatalog,
    *,
    asset_name: str | None = None,
    unknown_byte_sources: Collection[str] = (),
    out_dir: Path | None = None,
    prior_manifest: Mapping[str, Any] | None = None,
    bbox: BBox | None = None,
) -> PullEstimate:
    """Estimate a pull from search results, before any granule is downloaded.

    Scene counts come from the catalog. Byte totals sum ``file:size`` on the
    asset the download would select. A scene whose source did not report that
    size stays unknown. Names in ``unknown_byte_sources`` stay unknown even
    when an asset carries a size, because that size is not what the pull
    writes (a training-table extract, for example). Nothing is guessed. A
    source with a search error is skipped with that error. A source that
    returned no scenes is skipped as no coverage.

    A scene is left out of the counts when its file is already a non-empty
    file on disk, or when that path is listed in ``saved`` or ``skipped`` on
    ``prior_manifest``. A GeoTIFF that would be read again as a window stays
    in the estimate, because that read still transfers bytes. Quota is a
    provider figure: CDSE and Earthdata do not expose the account's remaining
    transfer, so those lines say so. A published policy limit is not used.

    Args:
        catalog: Search results for one pull.
        asset_name: Band or asset key the download would select.
        unknown_byte_sources: Registry names whose byte size this catalog
            cannot report. Their scenes still count.
        out_dir: Directory the download would write. ``None`` skips the disk check.
        prior_manifest: Resume manifest. Paths under ``saved`` and ``skipped``
            are already present.
        bbox: Area passed to the download. A GeoTIFF in this area is a window
            read and is not treated as already present.

    Returns:
        Scene counts, known bytes, files already present, quota, and skipped sources.
    """
    unknown_names = set(unknown_byte_sources)
    recorded = _recorded_paths(prior_manifest)
    included: list[SourceSize] = []
    skipped: list[SkippedSource] = []
    already_present = 0
    for src in catalog.sources:
        if src.error is not None:
            skipped.append(
                SkippedSource(source=src.source, reason=src.error.strip() or "search failed")
            )
            continue
        if not src.scenes:
            skipped.append(SkippedSource(source=src.source, reason=NO_COVERAGE))
            continue
        remaining: list[Scene] = []
        for scene in src.scenes:
            if _is_present(
                scene,
                asset_name=asset_name,
                out_dir=out_dir,
                recorded=recorded,
                bbox=bbox,
                force_unknown=src.source in unknown_names,
            ):
                already_present += 1
            else:
                remaining.append(scene)
        if not remaining:
            continue
        known_bytes, unknown_scenes, windowed = _bytes_for_scenes(
            remaining,
            asset_name=asset_name,
            force_unknown=src.source in unknown_names,
            bbox=bbox,
        )
        included.append(
            SourceSize(
                source=src.source,
                scene_count=len(remaining),
                known_bytes=known_bytes,
                unknown_scenes=unknown_scenes,
                windowed=windowed,
            )
        )
    return PullEstimate(
        sources=tuple(included),
        skipped=tuple(skipped),
        already_present=already_present,
        quotas=_quotas_for([src.source for src in catalog.sources]),
    )


def render_pull_estimate(estimate: PullEstimate) -> str:
    """Return the estimate as plain text for the CLI.

    Args:
        estimate: Figures from :func:`estimate_pull`.

    Returns:
        A short block: scene count, bytes, files already present, quota,
        per-source lines, and skipped sources.
    """
    lines = [
        "Pull estimate",
        f"scenes: {estimate.scene_count}",
        f"bytes: {estimate.bytes_text}",
        f"already present: {estimate.already_present}",
        f"quota: {estimate.quota_text}",
    ]
    if estimate.windowed_scenes:
        lines.append(
            f"windowed: {estimate.windowed_scenes} scenes read a smaller window than "
            "their catalog size"
        )
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
    bbox: BBox | None = None,
) -> tuple[int, int, int]:
    """Return ``(known_bytes, unknown_scenes, windowed)`` for ``scenes``.

    A scene with no downloadable asset contributes zero known bytes. That is
    a reported empty transfer, not an unknown size. A windowed GeoTIFF is
    counted in ``windowed`` because its catalog size covers the whole granule.
    """
    if force_unknown:
        return 0, len(scenes), 0
    known_bytes = 0
    unknown_scenes = 0
    windowed = 0
    for scene in scenes:
        size, is_window = _selected_size(scene, asset_name, bbox)
        if size is None:
            unknown_scenes += 1
        else:
            known_bytes += size
        windowed += int(is_window)
    return known_bytes, unknown_scenes, windowed


def _selected_size(
    scene: Scene,
    asset_name: str | None,
    bbox: BBox | None = None,
) -> tuple[int | None, bool]:
    """Return the selected asset's reported size and whether it is windowed.

    ``0`` when nothing would be downloaded. ``None`` when the asset has no
    catalog size.
    """
    try:
        picked = pick_asset(scene, asset_name)
    except BrowseAssetError:
        return 0, False
    if picked is None:
        return 0, False
    _key, asset = picked
    windowed = bbox is not None and _is_geotiff_asset(asset)
    return asset.size_bytes, windowed


def _is_tty(stream: TextIO) -> bool:
    """True when ``stream`` is an interactive terminal."""
    check = getattr(stream, "isatty", None)
    if not callable(check):
        return False
    try:
        return bool(check())
    except (OSError, ValueError):
        return False


def _human_bytes(n: int) -> str:
    """Return ``n`` in the largest 1024-based unit that is at least 1."""
    value = float(n)
    unit_index = 0
    while value >= 1024 and unit_index < len(_BYTE_UNITS) - 1:
        value /= 1024
        unit_index += 1
    unit = _BYTE_UNITS[unit_index]
    if unit == "bytes":
        return f"{n} bytes"
    if value.is_integer():
        return f"{int(value)} {unit}"
    return f"{value:.1f} {unit}"


def _recorded_paths(prior: Mapping[str, Any] | None) -> frozenset[str]:
    """Return ``saved`` and ``skipped`` paths from a resume manifest."""
    if prior is None:
        return frozenset()
    sources = prior.get("sources")
    if not isinstance(sources, list):
        return frozenset()
    found: set[str] = set()
    for item in sources:
        if not isinstance(item, dict):
            continue
        for key in ("saved", "skipped"):
            listed = item.get(key)
            if not isinstance(listed, list):
                continue
            for rel in listed:
                if isinstance(rel, str) and rel and not rel.startswith(("/", "\\")):
                    found.add(rel.replace("\\", "/"))
    return frozenset(found)


def _is_present(
    scene: Scene,
    *,
    asset_name: str | None,
    out_dir: Path | None,
    recorded: frozenset[str],
    bbox: BBox | None,
    force_unknown: bool,
) -> bool:
    """True when this scene would not transfer.

    A training-table extract does not write the catalog asset, so those scenes
    stay in the estimate. A windowed GeoTIFF counts as present only when the
    chip and its AOI marker match this request, so a chip from another area is
    not reported as this one.
    """
    if force_unknown:
        return False
    planned = _planned_asset(scene, asset_name)
    if planned is None:
        return False
    relative, asset, key, suffix = planned
    if relative in recorded:
        return True
    if out_dir is None:
        return False
    dest = output_source_dir(out_dir, scene.source)
    if dest is None:
        return False
    found = existing_product(dest, scene.id, key, suffix)
    if found is None:
        return False
    if bbox is not None and _is_geotiff_asset(asset):
        return same_aoi(found, bbox)
    return True


def _planned_asset(scene: Scene, asset_name: str | None) -> tuple[str, Asset, str, str] | None:
    """Return the planned relative path, asset, key, and suffix.

    ``None`` when nothing would be written. The path matches the name
    :func:`atlas.cli.download.download_assets` checks before it fetches.
    """
    try:
        picked = pick_asset(scene, asset_name)
    except BrowseAssetError:
        return None
    if picked is None or not plain_source_name(scene.source):
        return None
    key, asset = picked
    suffix = asset_suffix(asset)
    name = scene_filename(scene.id, key, suffix)
    return f"{scene.source}/{name}", asset, key, suffix


def _quotas_for(source_names: Sequence[str]) -> tuple[ProviderQuota, ...]:
    """Return quota notes for providers in ``source_names``.

    CDSE documents a rolling transfer limit and a Sentinel Hub processing-unit
    account. Neither is the remaining volume of an S3, OData, or STAC download
    for the username and password this program stores, and the processing-unit
    endpoint needs an OAuth client that is not stored. Earthdata Login lists
    tokens. CMR and the DAACs do not publish a remaining download balance for
    ``EARTHDATA_TOKEN``. No other registered provider publishes one either.
    The published limits are not copied in as a remaining figure.
    """
    saw_cdse = False
    saw_earthdata = False
    for name in source_names:
        provider = _quota_provider(name)
        if provider == "CDSE":
            saw_cdse = True
        elif provider == "Earthdata":
            saw_earthdata = True
    notes: list[ProviderQuota] = []
    if saw_cdse:
        notes.append(ProviderQuota("CDSE", None))
    if saw_earthdata:
        notes.append(ProviderQuota("Earthdata", None))
    if not notes:
        notes.append(ProviderQuota("", None))
    return tuple(notes)


def _quota_provider(name: str) -> str | None:
    """Return ``CDSE`` or ``Earthdata`` when ``name`` downloads from that provider."""
    cls = SOURCES.get(name)
    if cls is None:
        return None
    required = getattr(cls, "required_credentials", ())
    if isinstance(required, tuple):
        if "CDSE_USERNAME" in required:
            return "CDSE"
        if "EARTHDATA_TOKEN" in required:
            return "Earthdata"
    for base in cls.__mro__:
        if base.__module__ in _EARTHDATA_MODULES:
            return "Earthdata"
    return None
