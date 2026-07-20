from __future__ import annotations

from atlas.data.base import DataPullClient
from atlas.data.landsat8 import Landsat8Client
from atlas.data.landsat9 import Landsat9Client
from atlas.data.sentinel1 import Sentinel1Client
from atlas.data.sentinel2 import Sentinel2Client

# Registry of known satellite data sources by stable name. The consolidation
# layer iterates this to fan out a search across every source.
SOURCES: dict[str, type[DataPullClient]] = {
    "sentinel1": Sentinel1Client,
    "sentinel2": Sentinel2Client,
    "landsat8": Landsat8Client,
    "landsat9": Landsat9Client,
}


def all_sources() -> dict[str, DataPullClient]:
    """Instantiate one client per registered source."""
    return {name: cls() for name, cls in SOURCES.items()}


def sources(*names: str) -> dict[str, DataPullClient]:
    """Instantiate clients for the named sources (defaults to all)."""
    if not names:
        return all_sources()
    unknown = [n for n in names if n not in SOURCES]
    if unknown:
        raise KeyError(f"Unknown source(s): {unknown}. Known: {sorted(SOURCES)}")
    return {name: SOURCES[name]() for name in names}
