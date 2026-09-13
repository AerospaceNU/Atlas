"""Additional Planetary Computer collections (new sensors/products, same STAC API)."""

from __future__ import annotations

from typing import Any, ClassVar

from atlas.data.base import PullRequest
from atlas.data.planetary_computer import PlanetaryComputerClient


class _CloudFilterPC(PlanetaryComputerClient):
    def _build_query_filter(self, request: PullRequest) -> dict[str, Any] | None:
        if request.max_cloud_cover is None:
            return None
        return {"eo:cloud_cover": {"lte": request.max_cloud_cover}}


class LandsatC2L1Client(_CloudFilterPC):
    """USGS Landsat Collection 2 Level-1 (includes older missions)."""

    default_collection: ClassVar[str] = "landsat-c2-l1"


class AsterL1tClient(PlanetaryComputerClient):
    default_collection: ClassVar[str] = "aster-l1t"


class AlosPalsarMosaicClient(PlanetaryComputerClient):
    default_collection: ClassVar[str] = "alos-palsar-mosaic"
    include_datetime: ClassVar[bool] = False


class AlosFnfMosaicClient(PlanetaryComputerClient):
    default_collection: ClassVar[str] = "alos-fnf-mosaic"
    include_datetime: ClassVar[bool] = False


class PcModis14A1Client(PlanetaryComputerClient):
    """MODIS thermal anomalies / fire, 8-day, COG on Planetary Computer."""

    default_collection: ClassVar[str] = "modis-14A1-061"


class PcModis13Q1Client(PlanetaryComputerClient):
    """MODIS vegetation indices, 16-day 250 m, COG on Planetary Computer."""

    default_collection: ClassVar[str] = "modis-13Q1-061"


class PcModis09A1Client(PlanetaryComputerClient):
    """MODIS surface reflectance, 8-day 500 m, COG on Planetary Computer."""

    default_collection: ClassVar[str] = "modis-09A1-061"


class GoesCmiClient(PlanetaryComputerClient):
    """GOES-R Cloud and Moisture Imagery on Planetary Computer STAC."""

    default_collection: ClassVar[str] = "goes-cmi"


class EsaWorldCoverClient(PlanetaryComputerClient):
    default_collection: ClassVar[str] = "esa-worldcover"
    include_datetime: ClassVar[bool] = False


class IoLulcAnnualClient(PlanetaryComputerClient):
    default_collection: ClassVar[str] = "io-lulc-annual-v02"
    include_datetime: ClassVar[bool] = False


class NasaDemClient(PlanetaryComputerClient):
    default_collection: ClassVar[str] = "nasadem"
    include_datetime: ClassVar[bool] = False


class NoaaMrmsQpe24hClient(PlanetaryComputerClient):
    """NOAA MRMS 24-hour quantitative precipitation estimate."""

    default_collection: ClassVar[str] = "noaa-mrms-qpe-24h-pass2"
