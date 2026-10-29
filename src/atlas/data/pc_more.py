from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from atlas.data.base import PullRequest, PullResult, Scene, SceneKind
from atlas.data.planetary_computer import PlanetaryComputerClient


class LandsatC2L1Client(PlanetaryComputerClient):
    """USGS Landsat Collection 2 Level-1.

    No ``platform`` filter by design, unlike the L8/L9 L2 clients: this
    collection spans Landsat 1-5 and 7, and pinning one mission would hide the
    archive. Filter on ``Scene.platform`` downstream.
    """

    default_collection: ClassVar[str] = "landsat-c2-l1"
    satellite: ClassVar[str] = "landsat"
    scene_kind: ClassVar[SceneKind] = SceneKind.optical
    nominal_gsd_m: ClassVar[float | None] = 30.0


class AsterL1tClient(PlanetaryComputerClient):
    default_collection: ClassVar[str] = "aster-l1t"
    satellite: ClassVar[str] = "aster"
    scene_kind: ClassVar[SceneKind] = SceneKind.optical
    nominal_gsd_m: ClassVar[float | None] = 15.0


class AlosPalsarMosaicClient(PlanetaryComputerClient):
    default_collection: ClassVar[str] = "alos-palsar-mosaic"
    include_datetime: ClassVar[bool] = False
    satellite: ClassVar[str] = "alos"
    scene_kind: ClassVar[SceneKind] = SceneKind.sar
    nominal_gsd_m: ClassVar[float | None] = 25.0


class AlosFnfMosaicClient(PlanetaryComputerClient):
    default_collection: ClassVar[str] = "alos-fnf-mosaic"
    include_datetime: ClassVar[bool] = False
    satellite: ClassVar[str] = "alos"
    scene_kind: ClassVar[SceneKind] = SceneKind.landcover
    nominal_gsd_m: ClassVar[float | None] = 25.0


class PcModis14A1Client(PlanetaryComputerClient):
    """MODIS thermal anomalies / fire, 8-day, COG on Planetary Computer."""

    default_collection: ClassVar[str] = "modis-14A1-061"
    satellite: ClassVar[str] = "modis"
    scene_kind: ClassVar[SceneKind] = SceneKind.thermal
    nominal_gsd_m: ClassVar[float | None] = 1000.0


class PcModis13Q1Client(PlanetaryComputerClient):
    """MODIS vegetation indices, 16-day 250 m, COG on Planetary Computer."""

    default_collection: ClassVar[str] = "modis-13Q1-061"
    satellite: ClassVar[str] = "modis"
    scene_kind: ClassVar[SceneKind] = SceneKind.optical
    nominal_gsd_m: ClassVar[float | None] = 250.0


class PcModis09A1Client(PlanetaryComputerClient):
    """MODIS surface reflectance, 8-day 500 m, COG on Planetary Computer."""

    default_collection: ClassVar[str] = "modis-09A1-061"
    satellite: ClassVar[str] = "modis"
    scene_kind: ClassVar[SceneKind] = SceneKind.optical
    nominal_gsd_m: ClassVar[float | None] = 500.0


# Preferred asset per alias, first present wins. Wavelength matching alone
# picks ``C02_0.5km``: a 0.5 km full-disk band near 600 MB in the GEOS
# projection, which the windowed reader cannot clip. The ``_wm`` 2 km assets
# are Web Mercator (EPSG:3857) resamples, so the download reads only the AOI
# window. A full-disk C02_2km_wm is about 15 MB whole with about 6.3 km
# Mercator pixels. ``C02_2km`` (GEOS, about 38 MB full disk) is the fallback
# when an item has no ``_wm`` asset. Sizes are from 2026-09-24 GOES-18 items.
_GOES_BAND_ALIASES: Mapping[str, tuple[str, ...]] = {
    "red": ("C02_2km_wm", "C02_2km"),
    "nir": ("C03_2km_wm", "C03_2km"),
    "blue": ("C01_2km_wm", "C01_2km"),
}


class GoesCmiClient(PlanetaryComputerClient):
    """GOES-R Cloud and Moisture Imagery on Planetary Computer STAC.

    ``red``, ``nir``, and ``blue`` name the 2 km Web Mercator band 2, 3, and
    1 COGs, falling back to the 2 km GEOS COGs. The default download is
    therefore a clippable 2 km red band, never the 0.5 km full-disk band.
    ``C02_0.5km`` is still downloadable by name.

    Search does not filter ``goes:image-type``. CONUS and mesoscale items are
    smaller, but Planetary Computer has none over much of the AOIs and dates
    that full disk covers, and the ``_wm`` default makes any image type
    clippable.
    """

    default_collection: ClassVar[str] = "goes-cmi"
    satellite: ClassVar[str] = "goes"
    scene_kind: ClassVar[SceneKind] = SceneKind.optical
    nominal_gsd_m: ClassVar[float | None] = 2000.0

    def _scenes_in(self, payload: Mapping[str, Any]) -> list[Scene]:
        return [_with_goes_aliases(scene) for scene in super()._scenes_in(payload)]

    def _result_from_payload(self, request: PullRequest, payload: Any) -> PullResult:
        result = super()._result_from_payload(request, payload)
        scenes = [_with_goes_aliases(scene) for scene in result.scenes]
        return result.model_copy(update={"scenes": scenes})


def _with_goes_aliases(scene: Scene) -> Scene:
    pinned: dict[str, str] = {}
    for alias, keys in _GOES_BAND_ALIASES.items():
        key = next((key for key in keys if key in scene.assets), None)
        if key is not None:
            pinned[alias] = key
    if not pinned:
        return scene
    return scene.model_copy(update={"band_aliases": {**scene.band_aliases, **pinned}})


class EsaWorldCoverClient(PlanetaryComputerClient):
    default_collection: ClassVar[str] = "esa-worldcover"
    include_datetime: ClassVar[bool] = False
    satellite: ClassVar[str] = "worldcover"
    scene_kind: ClassVar[SceneKind] = SceneKind.landcover
    nominal_gsd_m: ClassVar[float | None] = 10.0


class IoLulcAnnualClient(PlanetaryComputerClient):
    default_collection: ClassVar[str] = "io-lulc-annual-v02"
    include_datetime: ClassVar[bool] = False
    satellite: ClassVar[str] = "lulc"
    scene_kind: ClassVar[SceneKind] = SceneKind.landcover
    nominal_gsd_m: ClassVar[float | None] = 10.0


class NasaDemClient(PlanetaryComputerClient):
    default_collection: ClassVar[str] = "nasadem"
    include_datetime: ClassVar[bool] = False
    satellite: ClassVar[str] = "nasa"
    scene_kind: ClassVar[SceneKind] = SceneKind.dem
    nominal_gsd_m: ClassVar[float | None] = 30.0


class NoaaMrmsQpe24hClient(PlanetaryComputerClient):
    """NOAA MRMS 24-hour quantitative precipitation estimate."""

    default_collection: ClassVar[str] = "noaa-mrms-qpe-24h-pass2"
    satellite: ClassVar[str] = "mrms"
    scene_kind: ClassVar[SceneKind] = SceneKind.precipitation
    nominal_gsd_m: ClassVar[float | None] = 1000.0
