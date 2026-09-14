"""Additional Planetary Computer collections (new sensors/products, same STAC API)."""

from __future__ import annotations

from typing import ClassVar

from atlas.data.base import SceneKind
from atlas.data.planetary_computer import PlanetaryComputerClient


class LandsatC2L1Client(PlanetaryComputerClient):
    """USGS Landsat Collection 2 Level-1 (includes older missions)."""

    default_collection: ClassVar[str] = "landsat-c2-l1"
    scene_kind: ClassVar[SceneKind] = SceneKind.optical
    nominal_gsd_m: ClassVar[float | None] = 30.0


class AsterL1tClient(PlanetaryComputerClient):
    default_collection: ClassVar[str] = "aster-l1t"
    scene_kind: ClassVar[SceneKind] = SceneKind.optical
    nominal_gsd_m: ClassVar[float | None] = 15.0


class AlosPalsarMosaicClient(PlanetaryComputerClient):
    default_collection: ClassVar[str] = "alos-palsar-mosaic"
    include_datetime: ClassVar[bool] = False
    scene_kind: ClassVar[SceneKind] = SceneKind.sar
    nominal_gsd_m: ClassVar[float | None] = 25.0


class AlosFnfMosaicClient(PlanetaryComputerClient):
    default_collection: ClassVar[str] = "alos-fnf-mosaic"
    include_datetime: ClassVar[bool] = False
    scene_kind: ClassVar[SceneKind] = SceneKind.landcover
    nominal_gsd_m: ClassVar[float | None] = 25.0


class PcModis14A1Client(PlanetaryComputerClient):
    """MODIS thermal anomalies / fire, 8-day, COG on Planetary Computer."""

    default_collection: ClassVar[str] = "modis-14A1-061"
    scene_kind: ClassVar[SceneKind] = SceneKind.thermal
    nominal_gsd_m: ClassVar[float | None] = 1000.0


class PcModis13Q1Client(PlanetaryComputerClient):
    """MODIS vegetation indices, 16-day 250 m, COG on Planetary Computer."""

    default_collection: ClassVar[str] = "modis-13Q1-061"
    scene_kind: ClassVar[SceneKind] = SceneKind.optical
    nominal_gsd_m: ClassVar[float | None] = 250.0


class PcModis09A1Client(PlanetaryComputerClient):
    """MODIS surface reflectance, 8-day 500 m, COG on Planetary Computer."""

    default_collection: ClassVar[str] = "modis-09A1-061"
    scene_kind: ClassVar[SceneKind] = SceneKind.optical
    nominal_gsd_m: ClassVar[float | None] = 500.0


class GoesCmiClient(PlanetaryComputerClient):
    """GOES-R Cloud and Moisture Imagery on Planetary Computer STAC."""

    default_collection: ClassVar[str] = "goes-cmi"
    scene_kind: ClassVar[SceneKind] = SceneKind.optical
    nominal_gsd_m: ClassVar[float | None] = 2000.0


class EsaWorldCoverClient(PlanetaryComputerClient):
    default_collection: ClassVar[str] = "esa-worldcover"
    include_datetime: ClassVar[bool] = False
    scene_kind: ClassVar[SceneKind] = SceneKind.landcover
    nominal_gsd_m: ClassVar[float | None] = 10.0


class IoLulcAnnualClient(PlanetaryComputerClient):
    default_collection: ClassVar[str] = "io-lulc-annual-v02"
    include_datetime: ClassVar[bool] = False
    scene_kind: ClassVar[SceneKind] = SceneKind.landcover
    nominal_gsd_m: ClassVar[float | None] = 10.0


class NasaDemClient(PlanetaryComputerClient):
    default_collection: ClassVar[str] = "nasadem"
    include_datetime: ClassVar[bool] = False
    scene_kind: ClassVar[SceneKind] = SceneKind.dem
    nominal_gsd_m: ClassVar[float | None] = 30.0


class NoaaMrmsQpe24hClient(PlanetaryComputerClient):
    """NOAA MRMS 24-hour quantitative precipitation estimate."""

    default_collection: ClassVar[str] = "noaa-mrms-qpe-24h-pass2"
    scene_kind: ClassVar[SceneKind] = SceneKind.precipitation
    nominal_gsd_m: ClassVar[float | None] = 1000.0
