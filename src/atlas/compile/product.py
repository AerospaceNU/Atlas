from __future__ import annotations

from pydantic import BaseModel, Field

from atlas.data.base import PullRequest, Scene


class SourceScenes(BaseModel):
    """Scenes returned by a single source for one request."""

    source: str
    collection: str
    scenes: list[Scene] = Field(default_factory=list)
    error: str | None = None  # populated if this source's search failed
    # Next STAC page for this source. None means the search has no further page,
    # or the next page could not be recorded. See next_cursor_unavailable.
    next_cursor: str | None = None
    # Set when a next page exists but its cursor was not stored, for example
    # "signed". None means the source is exhausted or the cursor is present.
    next_cursor_unavailable: str | None = None
    # Item-search URL. Empty when the client does not search a STAC API.
    search_url: str = ""
    # The resume cursor was rejected and must not be copied back from the manifest.
    cursor_rejected: bool = False


class SceneCatalog(BaseModel):
    """Unified result of fanning a single request out across all sources."""

    request: PullRequest
    sources: list[SourceScenes]

    @property
    def total(self) -> int:
        return sum(len(s.scenes) for s in self.sources)

    @property
    def ok_sources(self) -> list[SourceScenes]:
        """Sources that returned at least one scene without error."""
        return [s for s in self.sources if s.error is None and s.scenes]


class MosaicResult(BaseModel):
    """A rendered, gap-filled composite for one source over the AOI."""

    source: str
    collection: str
    search_id: str
    n_tiles: int
    width: int
    height: int
    path: str  # where the PNG was written


class MosaicSkip(BaseModel):
    """Why one source contributed scenes but no rendered mosaic.

    Attributes:
        source: Registry name of the source that was not rendered.
        collection: Collection the source searched.
        reason: Why the render did not happen, either an unsupported backend
            or the error that ended the attempt.
    """

    source: str
    collection: str
    reason: str


class CompiledProduct(BaseModel):
    """Consolidated output for one AOI + time window across sources."""

    request: PullRequest
    sources: list[str]
    scene_ids: dict[str, list[str]] = Field(default_factory=dict)  # provenance
    n_scenes: dict[str, int] = Field(default_factory=dict)
    cloud_cover: dict[str, float | None] = Field(default_factory=dict)
    coverage: dict[str, float] = Field(default_factory=dict)  # AOI fraction, 0..1
    mosaics: list[MosaicResult] = Field(default_factory=list)
    skipped_mosaics: list[MosaicSkip] = Field(default_factory=list)
