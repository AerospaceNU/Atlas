from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest
from pydantic import ValidationError

from atlas.archive import (
    CARD_VERSION,
    MAX_CARD_LAYERS,
    Availability,
    CloudStatus,
    LayerRef,
    Locator,
    SceneCard,
    make_index_id,
)
from atlas.data.base import Asset, BBox, GeometryKind, Scene, SceneKind

BOX = BBox(west=-71.2, south=42.2, east=-70.9, north=42.5)
WHEN = datetime(2024, 7, 1, 15, 30, tzinfo=UTC)


def card(**overrides: Any) -> SceneCard:
    data: dict[str, Any] = {
        "index_id": make_index_id("landsat8", "landsat-c2-l2", "LC08_X"),
        "source": "landsat8",
        "satellite": "landsat8",
        "platform": "landsat-8",
        "collection": "landsat-c2-l2",
        "provider_id": "LC08_X",
        "kind": SceneKind.optical,
        "datetime": WHEN,
        "bbox": BOX,
        "cloud_cover": 12.5,
        "cloud_status": CloudStatus.value,
    }
    data.update(overrides)
    return SceneCard(**data)


def test_index_id_is_deterministic_and_field_sensitive() -> None:
    a = make_index_id("landsat8", "c", "id")
    assert a == make_index_id("landsat8", "c", "id")
    assert a.startswith("sc_") and len(a) == 23
    assert a != make_index_id("landsat9", "c", "id")
    assert a != make_index_id("landsat8", "c2", "id")
    # Field boundaries matter: ("ab","c") must not collide with ("a","bc").
    assert make_index_id("ab", "c", "d") != make_index_id("a", "bc", "d")


def test_index_id_must_match_its_inputs() -> None:
    with pytest.raises(ValidationError, match="index_id"):
        card(index_id="sc_00000000000000000000")


def test_found_in_always_contains_source_without_duplicates() -> None:
    assert card().found_in == ["landsat8"]
    assert card(found_in=["usgs_landsat_l2_sr", "landsat8", "landsat8"]).found_in == [
        "landsat8",
        "usgs_landsat_l2_sr",
    ]


def test_datetimes_are_normalized_to_utc_and_must_be_aware() -> None:
    est = timezone(timedelta(hours=-5))
    c = card(datetime=datetime(2024, 7, 1, 10, 30, tzinfo=est))
    assert c.datetime == WHEN and c.datetime.utcoffset() == timedelta(0)
    with pytest.raises(ValidationError, match="timezone-aware"):
        card(datetime=datetime(2024, 7, 1, 15, 30))


def test_end_before_start_is_rejected() -> None:
    with pytest.raises(ValidationError, match="end_datetime"):
        card(start_datetime=WHEN, end_datetime=WHEN - timedelta(minutes=1))


@pytest.mark.parametrize(
    ("cloud_cover", "status", "ok"),
    [
        (0.0, CloudStatus.value, True),
        (None, CloudStatus.unknown, True),
        (None, CloudStatus.not_applicable, True),
        (None, CloudStatus.value, False),
        (0.0, CloudStatus.unknown, False),
        (5.0, CloudStatus.not_applicable, False),
    ],
)
def test_cloud_status_matches_cloud_cover(
    cloud_cover: float | None, status: CloudStatus, ok: bool
) -> None:
    if ok:
        assert card(cloud_cover=cloud_cover, cloud_status=status).cloud_status is status
    else:
        with pytest.raises(ValidationError, match="cloud_cover"):
            card(cloud_cover=cloud_cover, cloud_status=status)


def test_unsupported_card_version_is_rejected() -> None:
    with pytest.raises(ValidationError, match="card_version"):
        card(card_version=CARD_VERSION + 1)


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        card(signed_url="https://example.com/x?sig=abc")


@pytest.mark.parametrize(
    "key",
    [
        "",
        "/abs/path.tif",
        "~/x.tif",
        "C:/x.tif",
        "a\\b.tif",
        "../x.tif",
        "a/../../x.tif",
        "https://host/x.tif",
        "x.tif?sig=abc",
        "x.tif#frag",
    ],
)
def test_locator_rejects_absolute_escaping_and_url_keys(key: str) -> None:
    with pytest.raises(ValidationError):
        Locator(scheme="workspace", key=key)


def test_locator_accepts_relative_path_and_validates_scheme() -> None:
    assert Locator(scheme="workspace", key="landsat8/LC08_X_red.tif").key.endswith("red.tif")
    with pytest.raises(ValidationError, match="scheme"):
        Locator(scheme="Workspace", key="x.tif")


def test_layer_availability_rules() -> None:
    loc = Locator(scheme="workspace", key="landsat8/LC08_X_red.tif")
    assert LayerRef().availability is Availability.absent
    LayerRef(availability=Availability.full, locator=loc)
    LayerRef(availability=Availability.windowed, locator=loc, window=BOX)
    with pytest.raises(ValidationError, match="absent"):
        LayerRef(locator=loc)
    with pytest.raises(ValidationError, match="absent"):
        LayerRef(window=BOX)
    with pytest.raises(ValidationError, match="locator"):
        LayerRef(availability=Availability.full)
    with pytest.raises(ValidationError, match="window"):
        LayerRef(availability=Availability.windowed, locator=loc)
    with pytest.raises(ValidationError, match="no window"):
        LayerRef(availability=Availability.full, locator=loc, window=BOX)


def test_has_local_data_distinguishes_found_from_downloaded() -> None:
    loc = Locator(scheme="workspace", key="landsat8/LC08_X_red.tif")
    found = card(layers={"red": LayerRef(common_name="red", size_bytes=10)})
    chip = card(
        layers={
            "red": LayerRef(availability=Availability.windowed, locator=loc, window=BOX),
            "nir": LayerRef(),
        }
    )
    assert not found.has_local_data
    assert chip.has_local_data


def test_card_round_trips_through_json() -> None:
    loc = Locator(scheme="workspace", key="landsat8/LC08_X_red.tif")
    original = card(
        instrument="oli",
        gsd_m=30,
        layers={"red": LayerRef(availability=Availability.full, locator=loc, size_bytes=5)},
        found_in=["usgs_landsat_l2_sr"],
    )
    assert SceneCard.model_validate_json(original.model_dump_json()) == original


def test_non_optical_card_with_not_applicable_cloud() -> None:
    sar = card(
        index_id=make_index_id("sentinel1", "s1-grd", "S1_X"),
        source="sentinel1",
        satellite="sentinel1",
        platform="sentinel-1a",
        collection="s1-grd",
        provider_id="S1_X",
        kind=SceneKind.sar,
        cloud_cover=None,
        cloud_status=CloudStatus.not_applicable,
        start_datetime=WHEN,
        end_datetime=WHEN + timedelta(seconds=25),
    )
    assert sar.kind is SceneKind.sar and sar.cloud_cover is None


# --- SceneCard.from_scene -------------------------------------------------


def scene(**overrides: Any) -> Scene:
    data: dict[str, Any] = {
        "id": "LC08_X",
        "source": "landsat8",
        "collection": "landsat-c2-l2",
        "platform": "landsat-8",
        "kind": SceneKind.optical,
        "datetime": WHEN,
        "bbox": BOX,
        "gsd_m": 30,
        "cloud_cover": 12.5,
        "assets": {
            "red": Asset(href="https://example.com/red.tif", size_bytes=100),
            "B05": Asset(href="https://example.com/nir.tif"),
        },
        "band_aliases": {"nir": "B05"},
    }
    data.update(overrides)
    return Scene(**data)


def test_from_scene_copies_identity_time_and_footprint() -> None:
    c = SceneCard.from_scene(scene(instrument="oli"))
    assert c.index_id == make_index_id("landsat8", "landsat-c2-l2", "LC08_X")
    assert (c.source, c.satellite, c.provider_id) == ("landsat8", "landsat8", "LC08_X")
    assert (c.platform, c.instrument, c.collection) == ("landsat-8", "oli", "landsat-c2-l2")
    assert (c.datetime, c.bbox, c.gsd_m, c.cloud_cover) == (WHEN, BOX, 30, 12.5)
    assert c.cloud_status is CloudStatus.value
    assert c.found_in == ["landsat8"]


def test_from_scene_layers_start_absent_with_common_names_and_sizes() -> None:
    c = SceneCard.from_scene(scene())
    assert set(c.layers) == {"red", "B05"}
    assert c.layers["B05"].common_name == "nir"
    assert c.layers["red"].common_name is None
    assert c.layers["red"].size_bytes == 100
    assert c.layers["B05"].size_bytes is None
    layers = dict(c.layers)
    assert all(layer.availability is Availability.absent for layer in layers.values())
    assert not c.has_local_data


def test_from_scene_cloud_status_follows_kind() -> None:
    assert SceneCard.from_scene(scene(cloud_cover=None)).cloud_status is CloudStatus.unknown
    for kind in (SceneKind.sar, SceneKind.lidar, SceneKind.atmosphere, SceneKind.altimetry):
        c = SceneCard.from_scene(scene(kind=kind, cloud_cover=None))
        assert c.cloud_status is CloudStatus.not_applicable
    # A reported value wins even on a kind that usually has none.
    assert SceneCard.from_scene(scene(kind=SceneKind.sar, cloud_cover=0)).cloud_cover == 0


def test_from_scene_sensor_fixtures_keep_kind_geometry_and_range() -> None:
    sar = SceneCard.from_scene(
        scene(
            id="S1_X",
            source="sentinel1",
            collection="sentinel-1-grd",
            platform="sentinel-1a",
            kind=SceneKind.sar,
            cloud_cover=None,
            start_datetime=WHEN,
            end_datetime=WHEN + timedelta(seconds=25),
            assets={},
            band_aliases={},
        )
    )
    assert sar.satellite == "sentinel1" and sar.kind is SceneKind.sar
    assert sar.end_datetime == WHEN + timedelta(seconds=25) and sar.layers == {}

    lidar = SceneCard.from_scene(
        scene(
            id="GEDI_X",
            source="gedi",
            collection="GEDI02_A",
            platform="iss",
            kind=SceneKind.lidar,
            geometry_kind=GeometryKind.point,
            lon=-71.0,
            lat=42.3,
            cloud_cover=None,
            gsd_m=None,
            assets={},
            band_aliases={},
        )
    )
    assert lidar.satellite == "gedi" and lidar.geometry_kind is GeometryKind.point
    assert lidar.cloud_status is CloudStatus.not_applicable

    thermal = SceneCard.from_scene(
        scene(
            id="S3_X",
            source="cdse_sentinel3_slstr_lst",
            collection="SL_2_LST___",
            platform="sentinel-3a",
            instrument="slstr",
            kind=SceneKind.thermal,
            cloud_cover=None,
            assets={},
            band_aliases={},
        )
    )
    assert thermal.satellite == "sentinel3" and thermal.instrument == "slstr"
    assert thermal.cloud_status is CloudStatus.unknown


def test_from_scene_satellite_can_be_given_explicitly() -> None:
    c = SceneCard.from_scene(scene(source="custom_source"), satellite="Landsat8")
    assert c.satellite == "landsat8"


def test_satellite_has_one_spelling_whether_given_or_registered() -> None:
    registered = SceneCard.from_scene(scene())
    for spelling in ("landsat8", "Landsat8", "  LANDSAT8 "):
        given = SceneCard.from_scene(scene(), satellite=spelling)
        assert given.satellite == registered.satellite == "landsat8"
    assert card(satellite=" Landsat8 ").satellite == "landsat8"


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_satellite_is_rejected(blank: str) -> None:
    with pytest.raises(ValueError, match="satellite"):
        SceneCard.from_scene(scene(), satellite=blank)
    with pytest.raises(ValidationError, match="satellite"):
        card(satellite=blank)


def test_from_scene_rejects_missing_source_and_unregistered_satellite() -> None:
    with pytest.raises(ValueError, match="no source"):
        SceneCard.from_scene(scene(source=""))
    with pytest.raises(ValueError, match="no registered satellite"):
        SceneCard.from_scene(scene(source="not_a_source"))


def test_from_scene_card_round_trips_and_matches_across_rebuilds() -> None:
    first = SceneCard.from_scene(scene())
    assert SceneCard.from_scene(scene()) == first
    assert SceneCard.model_validate_json(first.model_dump_json()) == first


def test_from_scene_reads_processing_version_and_level() -> None:
    s2 = SceneCard.from_scene(
        scene(
            source="sentinel2",
            properties={"s2:processing_baseline": "05.11", "processing:level": "L2A"},
        )
    )
    assert (s2.processing_version, s2.processing_level) == ("05.11", "L2A")
    landsat = SceneCard.from_scene(scene(properties={"landsat:collection_number": "02"}))
    assert (landsat.processing_version, landsat.processing_level) == ("02", None)


def test_processing_version_falls_through_blank_and_non_scalar_values() -> None:
    props = {"s2:processing_baseline": "  ", "landsat:collection_number": ["02"]}
    c = SceneCard.from_scene(scene(properties={**props, "processing:version": 3}))
    assert c.processing_version == "3"
    assert SceneCard.from_scene(scene(properties=props)).processing_version is None
    assert SceneCard.from_scene(scene()).processing_version is None


def test_from_scene_copies_layer_roles_scale_and_offset() -> None:
    assets = {
        "B04": Asset(href="https://example.com/b04.tif", roles=["data"], scale=0.0001, offset=-0.1),
        "SCL": Asset(href="https://example.com/scl.tif", roles=["data", "cloud"]),
    }
    c = SceneCard.from_scene(scene(assets=assets, band_aliases={"red": "B04"}))
    assert c.layers["B04"].roles == ["data"]
    assert (c.layers["B04"].scale, c.layers["B04"].offset) == (0.0001, -0.1)
    assert c.layers["SCL"].roles == ["data", "cloud"]
    assert (c.layers["SCL"].scale, c.layers["SCL"].offset) == (None, None)


def test_layer_scale_and_offset_must_be_finite() -> None:
    with pytest.raises(ValidationError):
        LayerRef(scale=float("nan"))
    with pytest.raises(ValidationError):
        LayerRef(offset=float("inf"))


# --- image layers and the size cap ------------------------------------------


def test_from_scene_lists_only_image_layers_with_common_bands_first() -> None:
    assets = {
        "AOT": Asset(href="https://example.com/aot.tif", roles=["data"]),
        "B08": Asset(href="https://example.com/b08.tif", roles=["data"]),
        "B04": Asset(href="https://example.com/b04.tif", roles=["data"]),
        "SCL": Asset(href="https://example.com/scl.tif", roles=["data", "cloud"]),
        "granule-metadata": Asset(href="https://example.com/MTD.xml", roles=["metadata"]),
        "tilejson": Asset(
            href="https://example.com/tilejson", media_type="application/json", roles=["tiles"]
        ),
        "notes": Asset(href="https://example.com/readme.txt"),
        "rendered_preview": Asset(href="https://example.com/p.png", roles=["overview"]),
    }
    c = SceneCard.from_scene(scene(assets=assets, band_aliases={"red": "B04", "nir": "B08"}))
    assert list(c.layers) == ["B04", "B08", "AOT", "SCL"]
    assert c.layers_omitted == 0


def test_from_scene_caps_layers_and_keeps_requested_ones() -> None:
    assets = {
        f"band_{n:03d}": Asset(href=f"https://example.com/{n}.tif", roles=["data"])
        for n in range(MAX_CARD_LAYERS + 16)
    }
    c = SceneCard.from_scene(
        scene(assets=assets, band_aliases={"red": "band_030"}), keep=["band_039"]
    )
    assert len(c.layers) == MAX_CARD_LAYERS
    assert c.layers_omitted == 16
    assert list(c.layers)[:2] == ["band_030", "band_039"]
    assert len(c.model_dump_json(exclude_none=True)) < 2048


def test_a_card_cannot_hold_more_than_the_cap() -> None:
    with pytest.raises(ValidationError):
        card(layers={f"b{n}": LayerRef() for n in range(MAX_CARD_LAYERS + 1)})


def test_missing_layer_keeps_its_locator_and_is_not_local() -> None:
    loc = Locator(scheme="index", key="sentinel2/x.tif")
    with pytest.raises(ValidationError, match="needs a locator"):
        LayerRef(availability=Availability.missing)
    gone = LayerRef(availability=Availability.missing, locator=loc, window=BOX)
    assert not card(layers={"red": gone}).has_local_data
