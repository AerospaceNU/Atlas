"""Compact scene card: one short, citable record per observed scene.

A card says what a scene is, where it was observed, and which of its layers
exist locally. It never carries pixels, raw STAC metadata, signed URLs, or
secrets. The index can be rebuilt from source metadata, so ``CARD_VERSION``
changes whenever a field is added, removed, or reinterpreted.

Availability is recorded per layer, not per scene. Files are stored per asset,
and a file on disk can be a bbox window of the scene instead of the whole
asset, so one scene-level flag would misreport both cases.
"""

from __future__ import annotations

import hashlib
import re
from datetime import UTC
from datetime import datetime as DateTime
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from atlas.data.base import BBox, GeometryKind, Scene, SceneKind

CARD_VERSION = 1

# Kinds that do not measure cloud. Cloud is a real gap, not a non-question, for
# the rest (optical, thermal, browse), so a missing value there stays unknown.
_CLOUD_NOT_APPLICABLE = frozenset(
    {
        SceneKind.sar,
        SceneKind.lidar,
        SceneKind.altimetry,
        SceneKind.atmosphere,
        SceneKind.dem,
        SceneKind.precipitation,
        SceneKind.landcover,
        SceneKind.landsurface,
        SceneKind.detection,
    }
)

_INDEX_ID_PREFIX = "sc_"
_INDEX_ID_HEX = 20
_SCHEME = re.compile(r"^[a-z][a-z0-9+.-]*$")
_SEPARATOR = "\x1f"


class CloudStatus(StrEnum):
    """Why ``cloud_cover`` is, or is not, a number.

    ``unknown`` is a gap in the catalog. ``not_applicable`` means the sensor
    does not measure cloud (SAR, lidar, atmosphere). Queries must not treat
    either as 0 %.
    """

    value = "value"
    unknown = "unknown"
    not_applicable = "not_applicable"


class Availability(StrEnum):
    """Local state of one layer: not stored, a bbox chip, or the whole asset."""

    absent = "absent"
    windowed = "windowed"
    full = "full"


class Locator(BaseModel):
    """Storage-neutral place for a layer's data.

    ``scheme`` names the backend that resolves the key, for example
    ``workspace`` for a path relative to the workspace. The key is relative and
    never a URL, so a signed href cannot be stored here.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    scheme: str
    key: str

    @field_validator("scheme")
    @classmethod
    def _scheme_is_lowercase_word(cls, value: str) -> str:
        if not _SCHEME.fullmatch(value):
            raise ValueError("scheme must be lowercase letters, digits, '+', '-' or '.'")
        return value

    @field_validator("key")
    @classmethod
    def _key_is_relative_and_plain(cls, value: str) -> str:
        if not value:
            raise ValueError("key must not be empty")
        if value.startswith(("/", "~")) or "\\" in value or re.match(r"^[A-Za-z]:", value):
            raise ValueError("key must be a relative POSIX path")
        if "://" in value or "?" in value or "#" in value:
            raise ValueError("key must be a plain path, not a URL or query")
        if ".." in value.split("/"):
            raise ValueError("key must not leave its root")
        return value


class LayerRef(BaseModel):
    """One asset of a scene: what it is, how big, and whether it is local.

    Attributes:
        common_name: Band alias such as ``nir``, when the catalog maps one.
        size_bytes: Full asset size reported by the catalog, when known.
        availability: ``absent``, ``windowed`` (a bbox chip), or ``full``.
        locator: Where the local file is. None exactly when ``absent``.
        window: Footprint of the chip. Set exactly when ``windowed``.
    """

    model_config = ConfigDict(extra="forbid")

    common_name: str | None = None
    size_bytes: int | None = Field(default=None, ge=0)
    availability: Availability = Availability.absent
    locator: Locator | None = None
    window: BBox | None = None

    @model_validator(mode="after")
    def _availability_matches_fields(self) -> Self:
        if self.availability is Availability.absent:
            if self.locator is not None or self.window is not None:
                raise ValueError("an absent layer has no locator or window")
        elif self.locator is None:
            raise ValueError(f"a {self.availability} layer needs a locator")
        elif self.availability is Availability.windowed and self.window is None:
            raise ValueError("a windowed layer needs its window")
        elif self.availability is Availability.full and self.window is not None:
            raise ValueError("a full layer has no window")
        return self


def make_index_id(source: str, collection: str, provider_id: str) -> str:
    """Derive the stable index id for one catalog entry.

    The id depends only on the source key, the collection, and the provider's
    scene id, so it survives rebuilds and does not depend on catalog order.
    Merging several entries into one physical granule is deduplication's job
    and is recorded in ``SceneCard.found_in``, not in this id.
    """
    digest = hashlib.sha256(
        _SEPARATOR.join((source, collection, provider_id)).encode("utf-8")
    ).hexdigest()
    return _INDEX_ID_PREFIX + digest[:_INDEX_ID_HEX]


def _registered_satellite(source: str) -> str:
    # Imported here: the registry pulls in every client module.
    from atlas.data.registry import SOURCES

    client = SOURCES.get(source)
    family = getattr(client, "satellite", "") if client is not None else ""
    if not isinstance(family, str) or not family.strip():
        raise ValueError(f"source {source!r} has no registered satellite; pass satellite=")
    return family.strip().lower()


def _as_utc(value: DateTime) -> DateTime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime must be timezone-aware")
    return value.astimezone(UTC)


class SceneCard(BaseModel):
    """Compact, versioned record of one scene.

    Attributes:
        index_id: ``make_index_id`` of source, collection, and provider id.
        source: Registry key of the client that found the scene.
        satellite: Platform family, such as ``landsat8`` or ``sentinel3``.
        platform: Catalog platform string.
        instrument: Instrument, when the platform carries several.
        collection: Catalog collection or product.
        provider_id: Raw provider scene id, kept so the card links to its source.
        kind: Sensing modality.
        datetime: Representative time, UTC.
        start_datetime: Start of the observation window, UTC, when it is a range.
        end_datetime: End of the observation window, UTC, when it is a range.
        bbox: Footprint envelope.
        geometry_kind: Whether the footprint is a point, a bbox, or a polygon.
        gsd_m: Ground sample distance in metres, when known.
        cloud_cover: Percent cloud, only when ``cloud_status`` is ``value``.
        cloud_status: Why ``cloud_cover`` is or is not a number.
        layers: Assets by catalog asset key.
        found_in: Every source that returned this observation.
    """

    model_config = ConfigDict(extra="forbid")

    card_version: int = CARD_VERSION
    index_id: str
    source: str = Field(min_length=1)
    satellite: str = Field(min_length=1)
    platform: str = Field(min_length=1)
    instrument: str | None = None
    collection: str = Field(min_length=1)
    provider_id: str = Field(min_length=1)
    kind: SceneKind
    datetime: DateTime
    start_datetime: DateTime | None = None
    end_datetime: DateTime | None = None
    bbox: BBox
    geometry_kind: GeometryKind = GeometryKind.bbox
    gsd_m: float | None = Field(default=None, gt=0)
    cloud_cover: float | None = Field(default=None, ge=0, le=100)
    cloud_status: CloudStatus = CloudStatus.unknown
    layers: dict[str, LayerRef] = Field(default_factory=dict)
    found_in: list[str] = Field(default_factory=list)

    @field_validator("datetime", "start_datetime", "end_datetime")
    @classmethod
    def _utc(cls, value: DateTime | None) -> DateTime | None:
        return None if value is None else _as_utc(value)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.card_version != CARD_VERSION:
            raise ValueError(f"unsupported card_version {self.card_version}")
        expected = make_index_id(self.source, self.collection, self.provider_id)
        if self.index_id != expected:
            raise ValueError("index_id does not match source, collection, and provider_id")
        if (
            self.start_datetime is not None
            and self.end_datetime is not None
            and self.end_datetime < self.start_datetime
        ):
            raise ValueError("end_datetime must be on or after start_datetime")
        has_cloud = self.cloud_cover is not None
        if (self.cloud_status is CloudStatus.value) != has_cloud:
            raise ValueError("cloud_cover is set exactly when cloud_status is 'value'")
        others = dict.fromkeys(name for name in self.found_in if name != self.source)
        self.found_in = [self.source, *others]
        return self

    @classmethod
    def from_scene(cls, scene: Scene, *, satellite: str | None = None) -> Self:
        """Build the card for a scene a client just returned.

        Every layer starts ``absent``: a scene found in search has no local
        data until a download records one.

        Args:
            scene: Scene from a registered client. ``source`` and ``collection``
                must be set, as ``item_to_scene`` does for registered sources.
            satellite: Platform family. Defaults to the ``satellite`` class
                attribute of the client registered under ``scene.source``.

        Raises:
            ValueError: If ``scene.source`` is empty, or ``satellite`` is not
                given and the source has none registered.
        """
        if not scene.source:
            raise ValueError("scene has no source, so its card cannot be indexed")
        family = satellite if satellite is not None else _registered_satellite(scene.source)

        if scene.cloud_cover is not None:
            cloud_status = CloudStatus.value
        elif scene.kind in _CLOUD_NOT_APPLICABLE:
            cloud_status = CloudStatus.not_applicable
        else:
            cloud_status = CloudStatus.unknown

        # Several aliases can name one asset; the sorted first keeps this stable.
        common_names: dict[str, str] = {}
        for alias, key in sorted(scene.band_aliases.items(), reverse=True):
            common_names[key] = alias
        layers = {
            key: LayerRef(common_name=common_names.get(key), size_bytes=asset.size_bytes)
            for key, asset in scene.assets.items()
        }

        return cls(
            index_id=make_index_id(scene.source, scene.collection, scene.id),
            source=scene.source,
            satellite=family,
            platform=scene.platform,
            instrument=scene.instrument,
            collection=scene.collection,
            provider_id=scene.id,
            kind=scene.kind,
            datetime=scene.datetime,
            start_datetime=scene.start_datetime,
            end_datetime=scene.end_datetime,
            bbox=scene.bbox,
            geometry_kind=scene.geometry_kind,
            gsd_m=scene.gsd_m,
            cloud_cover=scene.cloud_cover,
            cloud_status=cloud_status,
            layers=layers,
        )

    @property
    def has_local_data(self) -> bool:
        """Whether any layer exists locally, as a chip or in full."""
        # dict() keeps Pylint from reading the Field() default as the attribute.
        layers = dict(self.layers)
        return any(layer.availability is not Availability.absent for layer in layers.values())
