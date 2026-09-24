from __future__ import annotations

from atlas.data.base import DataPullClient
from atlas.data.cdse import (
    CdseClmsBurntAreaClient,
    CdseSentinel3OlciClient,
    CdseSentinel3SlstrFrpClient,
    CdseSentinel3SlstrLstClient,
    CdseSentinel3SralWatClient,
    CdseSentinel5PCh4Client,
    CdseSentinel5PCoClient,
    CdseSentinel5PNo2Client,
)
from atlas.data.cmr_stac import (
    GediL2AClient,
    IceSat2Atl03Client,
    SmapL3PassiveClient,
    ViirsVnp09gaClient,
)
from atlas.data.cop_dem import CopDemGlo30Client, CopDemGlo90Client
from atlas.data.digitalearth import DeAfricaS2Client, DeaS2ArdClient
from atlas.data.earth_search import EarthSearchSentinel1GrdClient
from atlas.data.firms import (
    FirmsClient,
    FirmsLandsatClient,
    FirmsModisClient,
    FirmsViirsNoaa20Client,
    FirmsViirsNoaa21Client,
)
from atlas.data.gibs import (
    GibsAquaTrueColorClient,
    GibsBlackMarbleClient,
    GibsFlood3DayClient,
    GibsModisTrueColorClient,
    GibsNightLightsClient,
    GibsSnowCoverClient,
    GibsThermalAnomaliesClient,
    GibsViirsTrueColorClient,
)
from atlas.data.goes import GoesClient
from atlas.data.himawari import HimawariClient
from atlas.data.hls_landsat import HLSLandsatClient
from atlas.data.hls_sentinel import HLSSentinelClient
from atlas.data.inpe import Amazonia1WfiClient, Cbers4MuxClient
from atlas.data.landsat8 import Landsat8Client
from atlas.data.landsat9 import Landsat9Client
from atlas.data.maxar_opendata import MaxarOpenDataClient
from atlas.data.modis_active_fire import MODISActiveFireClient
from atlas.data.modis_land_cover import MODISLandCoverClient
from atlas.data.modis_surface import MODISSurfaceReflClient
from atlas.data.modis_vegetation import MODISVegetationClient
from atlas.data.naip import NaipClient
from atlas.data.pc_more import (
    AlosFnfMosaicClient,
    AlosPalsarMosaicClient,
    AsterL1tClient,
    EsaWorldCoverClient,
    GoesCmiClient,
    IoLulcAnnualClient,
    LandsatC2L1Client,
    NasaDemClient,
    NoaaMrmsQpe24hClient,
    PcModis09A1Client,
    PcModis13Q1Client,
    PcModis14A1Client,
)
from atlas.data.sentinel1 import Sentinel1Client
from atlas.data.sentinel2 import Sentinel2Client
from atlas.data.sources.fldas import FldasClient
from atlas.data.usgs import UsgsLandsatC2L1Client, UsgsLandsatC2L2SrClient

# Registry of known satellite data sources by stable name. The consolidation
# layer iterates this to fan out a search across every source.
SOURCES: dict[str, type[DataPullClient]] = {
    "sentinel1": Sentinel1Client,
    "sentinel2": Sentinel2Client,
    "landsat8": Landsat8Client,
    "landsat9": Landsat9Client,
    "landsat_c2_l1": LandsatC2L1Client,
    "modis_active_fire": MODISActiveFireClient,
    "modis_land_cover": MODISLandCoverClient,
    "modis_surface": MODISSurfaceReflClient,
    "modis_vegetation": MODISVegetationClient,
    "hls_landsat": HLSLandsatClient,
    "hls_sentinel": HLSSentinelClient,
    "naip": NaipClient,
    "cop_dem": CopDemGlo30Client,
    "cop_dem_90": CopDemGlo90Client,
    "nasadem": NasaDemClient,
    "aster": AsterL1tClient,
    "alos_palsar": AlosPalsarMosaicClient,
    "alos_fnf": AlosFnfMosaicClient,
    "pc_modis_14a1": PcModis14A1Client,
    "pc_modis_13q1": PcModis13Q1Client,
    "pc_modis_09a1": PcModis09A1Client,
    "goes_cmi": GoesCmiClient,
    "esa_worldcover": EsaWorldCoverClient,
    "io_lulc_annual": IoLulcAnnualClient,
    "mrms_qpe_24h": NoaaMrmsQpe24hClient,
    "earthsearch_sentinel1_grd": EarthSearchSentinel1GrdClient,
    "usgs_landsat_l1": UsgsLandsatC2L1Client,
    "usgs_landsat_l2_sr": UsgsLandsatC2L2SrClient,
    "cdse_sentinel3_olci": CdseSentinel3OlciClient,
    "cdse_sentinel3_slstr_lst": CdseSentinel3SlstrLstClient,
    "cdse_sentinel3_slstr_frp": CdseSentinel3SlstrFrpClient,
    "cdse_sentinel3_sral": CdseSentinel3SralWatClient,
    "cdse_sentinel5p_no2": CdseSentinel5PNo2Client,
    "cdse_sentinel5p_ch4": CdseSentinel5PCh4Client,
    "cdse_sentinel5p_co": CdseSentinel5PCoClient,
    "cdse_clms_burnt_area": CdseClmsBurntAreaClient,
    "gedi": GediL2AClient,
    "viirs_vnp09ga": ViirsVnp09gaClient,
    "icesat2_atl03": IceSat2Atl03Client,
    "smap_l3": SmapL3PassiveClient,
    "dea_s2_ard": DeaS2ArdClient,
    "deafrica_s2": DeAfricaS2Client,
    "cbers4_mux": Cbers4MuxClient,
    "amazonia1_wfi": Amazonia1WfiClient,
    "maxar_opendata": MaxarOpenDataClient,
    "firms_viirs": FirmsClient,
    "firms_viirs_noaa20": FirmsViirsNoaa20Client,
    "firms_viirs_noaa21": FirmsViirsNoaa21Client,
    "firms_modis": FirmsModisClient,
    "firms_landsat": FirmsLandsatClient,
    "fldas": FldasClient,
    "gibs_modis_truecolor": GibsModisTrueColorClient,
    "gibs_viirs_truecolor": GibsViirsTrueColorClient,
    "gibs_aqua_truecolor": GibsAquaTrueColorClient,
    "gibs_night_lights": GibsNightLightsClient,
    "gibs_black_marble": GibsBlackMarbleClient,
    "gibs_flood_3day": GibsFlood3DayClient,
    "gibs_snow": GibsSnowCoverClient,
    "gibs_thermal": GibsThermalAnomaliesClient,
    "goes": GoesClient,
    "himawari": HimawariClient,
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
