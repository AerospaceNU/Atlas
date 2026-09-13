from __future__ import annotations

from atlas.data.base import DataPullClient
from atlas.data.cdse import CdseSentinel3OlciClient, CdseSentinel5PNo2Client
from atlas.data.cop_dem import CopDemGlo30Client
from atlas.data.earth_search import EarthSearchSentinel1GrdClient
from atlas.data.firms import FirmsClient
from atlas.data.gibs import GibsModisTrueColorClient, GibsViirsTrueColorClient
from atlas.data.goes import GoesClient
from atlas.data.hls_landsat import HLSLandsatClient
from atlas.data.hls_sentinel import HLSSentinelClient
from atlas.data.landsat8 import Landsat8Client
from atlas.data.landsat9 import Landsat9Client
from atlas.data.modis_active_fire import MODISActiveFireClient
from atlas.data.modis_land_cover import MODISLandCoverClient
from atlas.data.modis_surface import MODISSurfaceReflClient
from atlas.data.modis_vegetation import MODISVegetationClient
from atlas.data.naip import NaipClient
from atlas.data.sentinel1 import Sentinel1Client
from atlas.data.sentinel2 import Sentinel2Client

# Registry of known satellite data sources by stable name. The consolidation
# layer iterates this to fan out a search across every source.
SOURCES: dict[str, type[DataPullClient]] = {
    "sentinel1": Sentinel1Client,
    "sentinel2": Sentinel2Client,
    "landsat8": Landsat8Client,
    "landsat9": Landsat9Client,
    "modis_active_fire": MODISActiveFireClient,
    "modis_land_cover": MODISLandCoverClient,
    "modis_surface": MODISSurfaceReflClient,
    "modis_vegetation": MODISVegetationClient,
    "hls_landsat": HLSLandsatClient,
    "hls_sentinel": HLSSentinelClient,
    "naip": NaipClient,
    "cop_dem": CopDemGlo30Client,
    "earthsearch_sentinel1_grd": EarthSearchSentinel1GrdClient,
    "cdse_sentinel3_olci": CdseSentinel3OlciClient,
    "cdse_sentinel5p_no2": CdseSentinel5PNo2Client,
    "firms_viirs": FirmsClient,
    "gibs_modis_truecolor": GibsModisTrueColorClient,
    "gibs_viirs_truecolor": GibsViirsTrueColorClient,
    "goes": GoesClient,
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
