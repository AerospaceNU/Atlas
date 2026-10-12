from __future__ import annotations

from typing import ClassVar

from atlas.data.base import Scene, SceneKind
from atlas.data.indices import Calibration
from atlas.data.stac import StacApiClient

EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1/search"
# Sentinel-2 L2A quantification value. Reflectance is stored * scale + offset.
_L2A_SCALE = 1e-4


class EarthSearchClient(StacApiClient):
    search_url: ClassVar[str] = EARTH_SEARCH_URL


class EarthSearchSentinel1GrdClient(EarthSearchClient):
    """Sentinel-1 GRD on AWS Open Data (not the PC RTC collection)."""

    default_collection: ClassVar[str] = "sentinel-1-grd"
    satellite: ClassVar[str] = "sentinel1"
    scene_kind: ClassVar[SceneKind] = SceneKind.sar
    nominal_gsd_m: ClassVar[float | None] = 10.0


class EarthSearchSentinel2L2aClient(EarthSearchClient):
    """Sentinel-2 L2A on AWS Open Data (not the Planetary Computer collection).

    v1 items often set ``earthsearch:boa_offset_applied`` after the COG has
    already had the baseline offset removed, while ``raster:bands`` still
    publishes offset ``-0.1``. That published offset is the ``s2-l2a`` preset.
    ``l2a_reflectance_calibration`` keeps the asset scale and uses offset 0
    when the flag is set, so the preset is not applied a second time.
    """

    default_collection: ClassVar[str] = "sentinel-2-l2a"
    satellite: ClassVar[str] = "sentinel2"
    scene_kind: ClassVar[SceneKind] = SceneKind.optical
    nominal_gsd_m: ClassVar[float | None] = 10.0


def l2a_reflectance_calibration(scene: Scene, asset_key: str) -> Calibration:
    """Return scale and offset for one Earth Search Sentinel-2 L2A asset.

    Reflectance is ``stored * scale + offset``. When
    ``earthsearch:boa_offset_applied`` is true, the returned offset is 0 even
    if ``raster:bands`` still says ``-0.1``. Otherwise the asset's published
    offset is used. A missing scale falls back to ``1e-4``.

    Args:
        scene: Scene from ``EarthSearchSentinel2L2aClient.search``.
        asset_key: Key in ``scene.assets``, such as ``red`` or ``swir22``.

    Returns:
        Scale and offset for that asset.

    Raises:
        KeyError: ``asset_key`` is not on the scene.
    """
    asset = scene.assets.get(asset_key)
    if asset is None:
        raise KeyError(asset_key)
    scale = _L2A_SCALE if asset.scale is None else asset.scale
    published = 0.0 if asset.offset is None else asset.offset
    applied = scene.properties.get("earthsearch:boa_offset_applied") is True
    return Calibration(scale, 0.0 if applied else published)
