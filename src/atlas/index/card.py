"""Scene cards: the compact, citable record the index keeps for one observation."""

from __future__ import annotations

import hashlib
from datetime import UTC
from datetime import datetime as DateTime
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, Field

from atlas.data.base import Asset, Scene, SceneKind, is_browse_asset

# Bump when card fields or how they are derived from a Scene change; the store
# then asks for `atlas scenes --rebuild`, which re-derives every card from the
# saved scene snapshots without contacting a provider.
CARD_VERSION = 1

# Keeps a card a few hundred bytes so dozens fit in a model context. Sentinel-2
# and Landsat list every band; a scene with more lists its common bands first.
MAX_CARD_BANDS = 24

_NON_BAND_ROLES = frozenset({"metadata", "tiles"})
_NON_BAND_SUFFIXES = frozenset({".xml", ".json", ".html", ".txt"})


class Availability(StrEnum):
    found = "found"  # seen in a catalog search; no local file
    downloaded = "downloaded"  # a native asset is saved locally
    missing = "missing"  # recorded as downloaded, but the file is gone


class SceneCard(BaseModel):
    """One observation, short enough to hand to a model or cite in a report.

    ``index_id`` is the citation handle. It hashes the catalog host, collection
    and item id, so it survives re-pulls, rebuilds and registry renames.
    """

    index_id: str
    catalog: str
    collection: str
    item_id: str
    found_via: list[str] = Field(default_factory=list)
    datetime: DateTime
    platform: str
    instrument: str | None = None
    kind: SceneKind
    gsd_m: float | None = None
    bbox: list[float]
    footprint_exact: bool
    cloud_cover: float | None = None
    bands: list[str] = Field(default_factory=list, max_length=MAX_CARD_BANDS)
    band_count: int = 0  # data bands on the scene; > len(bands) means the list was cut
    band_aliases: dict[str, str] = Field(default_factory=dict)  # e.g. {"red": "B04"}
    availability: Availability
    local_path: str | None = None


def make_index_id(catalog: str, collection: str, item_id: str) -> str:
    digest = hashlib.sha256(f"{catalog}|{collection}|{item_id}".encode()).hexdigest()
    return f"sc_{digest[:16]}"


def scene_index_id(scene: Scene) -> str:
    return make_index_id(scene.catalog or scene.source, scene.collection, scene.id)


def utc(value: DateTime) -> DateTime:
    """Read naive datetimes as UTC, the STAC convention."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def is_band_asset(key: str, asset: Asset) -> bool:
    """True for imagery assets; false for previews, metadata files and tile endpoints."""
    if is_browse_asset(key, asset):
        return False
    if {role.lower() for role in asset.roles} & _NON_BAND_ROLES:
        return False
    media = (asset.media_type or "").lower()
    if media.startswith("text/") or any(word in media for word in ("xml", "json", "html")):
        return False
    return PurePosixPath(urlsplit(asset.href).path).suffix.lower() not in _NON_BAND_SUFFIXES


def scene_bands(scene: Scene) -> list[str]:
    """Every band key, common bands (red, green, nir, swir22) first, then the rest sorted."""
    keys = {key for key, asset in scene.assets.items() if is_band_asset(key, asset)}
    common = [key for key in dict.fromkeys(scene.band_aliases.values()) if key in keys]
    return common + sorted(keys - set(common))


def scene_band_aliases(scene: Scene) -> dict[str, str]:
    """Common band name to band key, for the bands the card lists."""
    return {
        name: key
        for name, key in scene.band_aliases.items()
        if is_band_asset(key, scene.assets[key])
    }


def scene_snapshot(scene: Scene) -> dict[str, Any]:
    """Scene as JSON for rebuilds, with query strings (signatures, tokens) cut from hrefs."""
    data = scene.model_dump(mode="json")
    for asset in data.get("assets", {}).values():
        parts = urlsplit(asset["href"])
        asset["href"] = urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
    return data
