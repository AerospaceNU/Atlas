"""NDVI, NDWI, and NBR compute only once red and NIR are both available."""

from __future__ import annotations

import numpy as np
import pytest

from atlas.data.indices import (
    CALIBRATION_PRESETS,
    MAX_BAND_BYTES,
    MAX_BAND_ELEMENTS,
    available_indices,
    compute_indices,
    normalized_difference,
)
from atlas.data.stac import item_to_scene


def test_no_index_until_red_and_nir_are_both_available() -> None:
    assert available_indices([]) == ()
    assert available_indices(["red"]) == ()
    assert available_indices(["nir"]) == ()
    assert available_indices(["green", "swir22"]) == ()
    assert available_indices(["red", "green", "swir22"]) == ()
    assert available_indices(["nir", "green", "swir22"]) == ()
    assert compute_indices({"nir": np.array([0.4]), "green": np.array([0.2])}) == {}
    # Bands that cannot form an index are not inspected until the gate opens.
    assert compute_indices({"green": "not-an-array"}) == {}  # type: ignore[dict-item]


def test_ndvi_is_available_from_red_and_nir_alone() -> None:
    assert available_indices(["red", "nir"]) == ("ndvi",)
    result = compute_indices({"red": np.array([0.1]), "nir": np.array([0.5])})
    assert tuple(result) == ("ndvi",)


def test_ndwi_needs_green_in_addition_to_red_and_nir() -> None:
    assert available_indices(["red", "nir", "green"]) == ("ndvi", "ndwi_mcfeeters")
    assert "ndwi_mcfeeters" not in available_indices(["red", "nir", "swir22"])
    assert "ndwi" not in available_indices(["red", "nir", "green"])


def test_nbr_needs_swir22_in_addition_to_red_and_nir() -> None:
    assert available_indices(["red", "nir", "swir22"]) == ("ndvi", "nbr")
    # swir16 is a different measurement and does not satisfy NBR.
    assert available_indices(["red", "nir", "swir", "swir16"]) == ("ndvi",)


def test_all_three_compute_together_in_order() -> None:
    assert available_indices(["swir22", "GREEN", "nir", "red"]) == ("ndvi", "ndwi_mcfeeters", "nbr")
    result = compute_indices(
        {
            "swir22": np.array([0.2]),
            "GREEN": np.array([0.4]),
            "nir": np.array([0.5]),
            "red": np.array([0.1]),
        }
    )
    assert tuple(result) == ("ndvi", "ndwi_mcfeeters", "nbr")


def test_ndvi_is_normalized_difference_of_nir_and_red() -> None:
    result = compute_indices(
        {
            "red": np.array([[0.10, 0.20], [0.00, 0.40]]),
            "nir": np.array([[0.50, 0.20], [0.00, 0.20]]),
        }
    )
    expected = np.array([[2.0 / 3.0, 0.0], [np.nan, -1.0 / 3.0]])
    np.testing.assert_allclose(result["ndvi"], expected, equal_nan=True)
    assert result["ndvi"].dtype == np.float64
    assert result["ndvi"].shape == (2, 2)


def test_ndwi_is_mcfeeters_green_and_nir() -> None:
    result = compute_indices(
        {
            "red": np.array([0.10, 0.20, 0.00, 0.40]),
            "nir": np.array([0.50, 0.20, 0.00, 0.20]),
            "green": np.array([0.40, 0.10, 0.30, 0.10]),
        }
    )
    expected = np.array([-0.10 / 0.90, -0.10 / 0.30, 1.0, -0.10 / 0.30])
    np.testing.assert_allclose(result["ndwi_mcfeeters"], expected, equal_nan=True)


def test_nbr_is_normalized_difference_of_nir_and_swir22() -> None:
    result = compute_indices(
        {
            "red": np.array([0.10, 0.20, 0.00, 0.40]),
            "nir": np.array([0.50, 0.20, 0.00, 0.20]),
            "swir22": np.array([0.20, 0.30, 0.10, 0.20]),
        }
    )
    expected = np.array([0.30 / 0.70, -0.2, -1.0, 0.0])
    np.testing.assert_allclose(result["nbr"], expected, equal_nan=True)


def test_zero_denominator_is_nan_not_inf() -> None:
    result = compute_indices(
        {
            "red": np.array([0.0, 1.0]),
            "nir": np.array([0.0, -1.0]),
            "green": np.array([0.0, 0.0]),
            "swir22": np.array([0.0, 1.0]),
        }
    )
    assert np.isnan(result["ndvi"]).all()
    assert np.isnan(result["nbr"]).all()
    assert np.isnan(result["ndwi_mcfeeters"][0])
    assert result["ndwi_mcfeeters"][1] == pytest.approx(-1.0)
    for index in result.values():
        assert not np.isinf(index).any()
    direct = normalized_difference(np.array([0.0, 1.0]), np.array([0.0, -1.0]))
    assert np.isnan(direct).all()


def test_values_outside_the_unit_interval_are_kept() -> None:
    result = compute_indices({"red": np.array([-0.25]), "nir": np.array([0.5])})
    assert result["ndvi"][0] == pytest.approx(3.0)


def test_integer_reflectance_is_promoted_before_subtraction() -> None:
    nir = np.array([200], dtype=np.uint8)
    red = np.array([100], dtype=np.uint8)
    result = compute_indices({"nir": nir, "red": red})
    assert result["ndvi"][0] == pytest.approx(1.0 / 3.0)
    assert result["ndvi"].dtype == np.float64
    assert nir.dtype == np.uint8
    assert red.dtype == np.uint8
    assert int(nir[0]) == 200


def test_inputs_are_not_mutated() -> None:
    nir = np.array([0.5, 0.2])
    red = np.array([0.1, 0.2])
    snapshot = (nir.copy(), red.copy())
    result = compute_indices({"nir": nir, "red": red})
    result["ndvi"][0] = 42.0
    np.testing.assert_array_equal(nir, snapshot[0])
    np.testing.assert_array_equal(red, snapshot[1])


def test_band_names_ignore_case_and_surrounding_space() -> None:
    result = compute_indices({" NIR ": np.array([0.8]), "Red": np.array([0.2])})
    assert tuple(result) == ("ndvi",)
    assert result["ndvi"][0] == pytest.approx(0.6)


def test_unused_bands_are_ignored() -> None:
    result = compute_indices(
        {
            "red": np.array([0.2]),
            "nir": np.array([0.8]),
            "blue": "not-numeric",  # type: ignore[dict-item]
            "   ": np.array([9.0]),
        }
    )
    assert tuple(result) == ("ndvi",)
    assert result["ndvi"][0] == pytest.approx(0.6)


def test_duplicate_band_name_raises() -> None:
    with pytest.raises(ValueError, match="duplicate band 'nir'"):
        compute_indices({"nir": np.array([0.5]), " NIR ": np.array([0.4]), "red": np.array([0.1])})


def test_non_string_band_name_raises() -> None:
    with pytest.raises(TypeError, match="string"):
        compute_indices({1: np.array([0.2])})  # type: ignore[dict-item]
    with pytest.raises(TypeError, match="not a string"):
        available_indices("red")


def test_non_numeric_band_raises_once_red_and_nir_are_present() -> None:
    with pytest.raises(TypeError, match="green must be a numeric array"):
        compute_indices(
            {
                "red": np.array([0.1]),
                "nir": np.array([0.4]),
                "green": [0.2, 0.3],  # type: ignore[dict-item]
            }
        )
    with pytest.raises(TypeError, match="numeric array"):
        compute_indices({"red": np.array([True]), "nir": np.array([False])})


def test_shape_mismatch_raises_and_returns_nothing() -> None:
    with pytest.raises(ValueError, match="share a shape"):
        compute_indices({"red": np.ones(2), "nir": np.ones((2, 1))})
    with pytest.raises(ValueError, match="share a shape"):
        compute_indices(
            {
                "red": np.ones(2),
                "nir": np.ones(2),
                "green": np.ones(3),
            }
        )


def test_scene_aliases_open_ndvi_only_once_red_and_nir_resolve() -> None:
    both = item_to_scene(
        {
            "id": "HLS.S30.T19TCG",
            "bbox": [-71.2, 42.2, -70.9, 42.5],
            "properties": {"datetime": "2024-07-01T15:00:00Z"},
            "assets": {
                "B04": {
                    "href": "https://example.com/B04.tif",
                    "eo:bands": [{"common_name": "red", "center_wavelength": 0.6645}],
                },
                "B08": {
                    "href": "https://example.com/B08.tif",
                    "eo:bands": [{"common_name": "nir", "center_wavelength": 0.8351}],
                },
                "B8A": {
                    "href": "https://example.com/B8A.tif",
                    "eo:bands": [{"center_wavelength": 0.8648}],
                },
            },
        }
    )
    assert both is not None
    assert available_indices(both.band_aliases) == ("ndvi",)

    nir_only = item_to_scene(
        {
            "id": "HLS.S30.T19TCG",
            "bbox": [-71.2, 42.2, -70.9, 42.5],
            "properties": {"datetime": "2024-07-01T15:00:00Z"},
            "assets": {
                "B8A": {
                    "href": "https://example.com/B8A.tif",
                    "eo:bands": [{"center_wavelength": 0.8648}],
                },
            },
        }
    )
    assert nir_only is not None
    assert available_indices(nir_only.band_aliases) == ()


def test_nodata_nan_is_rejected() -> None:
    bands = {"red": np.array([1.0, np.nan]), "nir": np.array([2.0, 3.0])}
    with pytest.raises(ValueError, match="nodata cannot be NaN"):
        compute_indices(bands, nodata=float("nan"))
    with pytest.raises(ValueError, match="nodata cannot be NaN"):
        compute_indices(bands, nodata={"nir": np.nan})


@pytest.mark.parametrize("fill", [0, 65535, -9999])
def test_nodata_fill_on_an_input_band_is_nan(fill: int) -> None:
    red = np.array([fill, 0.2], dtype=np.float64)
    nir = np.array([0.6, 0.8], dtype=np.float64)
    result = compute_indices({"red": red, "nir": nir}, nodata=fill)
    assert np.isnan(result["ndvi"][0])
    assert result["ndvi"][1] == pytest.approx((0.8 - 0.2) / (0.8 + 0.2))


def test_uint16_fill_65535_and_int_fill_minus_9999() -> None:
    red = np.array([65535, 1000], dtype=np.uint16)
    nir = np.array([4000, 3000], dtype=np.uint16)
    filled = compute_indices({"red": red, "nir": nir}, nodata=65535)
    assert np.isnan(filled["ndvi"][0])
    assert filled["ndvi"][1] == pytest.approx((3000 - 1000) / (3000 + 1000))

    signed = compute_indices(
        {"red": np.array([-9999, 0.1]), "nir": np.array([0.4, 0.7])},
        nodata=-9999,
    )
    assert np.isnan(signed["ndvi"][0])
    assert signed["ndvi"][1] == pytest.approx((0.7 - 0.1) / (0.7 + 0.1))


def test_nodata_on_one_band_blanks_only_indices_that_use_it() -> None:
    result = compute_indices(
        {
            "red": np.array([0.2, 0.2]),
            "nir": np.array([0.8, 0.8]),
            "green": np.array([-9999.0, 0.3]),
        },
        nodata={"green": -9999},
    )
    assert result["ndvi"][0] == pytest.approx((0.8 - 0.2) / (0.8 + 0.2))
    assert np.isnan(result["ndwi_mcfeeters"][0])
    assert result["ndwi_mcfeeters"][1] == pytest.approx((0.3 - 0.8) / (0.3 + 0.8))


def test_boolean_mask_marks_nodata() -> None:
    result = compute_indices(
        {"red": np.array([0.1, 0.2]), "nir": np.array([0.5, 0.6])},
        mask=np.array([True, False]),
    )
    assert np.isnan(result["ndvi"][0])
    assert result["ndvi"][1] == pytest.approx((0.6 - 0.2) / (0.6 + 0.2))


def test_per_band_mask_does_not_blank_unrelated_indices() -> None:
    result = compute_indices(
        {
            "red": np.array([0.1, 0.2]),
            "nir": np.array([0.5, 0.6]),
            "green": np.array([0.4, 0.4]),
        },
        mask={"green": np.array([True, False])},
    )
    assert np.isfinite(result["ndvi"]).all()
    assert np.isnan(result["ndwi_mcfeeters"][0])
    assert np.isfinite(result["ndwi_mcfeeters"][1])


def test_sentinel2_l2a_offset_of_minus_1000_with_scale() -> None:
    # Published BOA offset -1000 and scale 1e-4: (DN - 1000) * 1e-4.
    scale = 1e-4
    published_offset = -1000
    red = np.array([2000.0, 0.0])
    nir = np.array([6000.0, 4000.0])
    result = compute_indices(
        {"red": red, "nir": nir},
        scale=scale,
        offset=published_offset * scale,
        nodata=0,
    )
    assert result["ndvi"][0] == pytest.approx((0.5 - 0.1) / (0.5 + 0.1))
    # DN 0 is nodata, not the reflectance (0 - 1000) * 1e-4.
    assert np.isnan(result["ndvi"][1])
    red_reflectance = red * scale + published_offset * scale
    nir_reflectance = nir * scale + published_offset * scale
    assert red_reflectance[0] == pytest.approx((2000 + published_offset) * scale)
    assert nir_reflectance[0] == pytest.approx((6000 + published_offset) * scale)


def test_landsat_c2_scale_and_offset() -> None:
    scale = 2.75e-5
    offset = -0.2
    red = np.array([10000.0, 0.0])
    nir = np.array([20000.0, 8000.0])
    result = compute_indices(
        {"red": red, "nir": nir},
        scale=scale,
        offset=offset,
        nodata=0,
    )
    red_sr = 10000 * scale + offset
    nir_sr = 20000 * scale + offset
    assert red_sr == pytest.approx(0.075)
    assert nir_sr == pytest.approx(0.35)
    assert result["ndvi"][0] == pytest.approx((nir_sr - red_sr) / (nir_sr + red_sr))
    assert np.isnan(result["ndvi"][1])


def test_s2_l2a_and_landsat_c2_presets() -> None:
    assert CALIBRATION_PRESETS["s2-l2a"].scale == pytest.approx(1e-4)
    assert CALIBRATION_PRESETS["s2-l2a"].offset == pytest.approx(-1000 * 1e-4)
    assert CALIBRATION_PRESETS["landsat-c2"].scale == pytest.approx(2.75e-5)
    assert CALIBRATION_PRESETS["landsat-c2"].offset == pytest.approx(-0.2)

    red = np.array([2000.0, 0.0])
    nir = np.array([6000.0, 4000.0])
    preset = compute_indices({"red": red, "nir": nir}, preset="s2-l2a", nodata=0)
    manual = compute_indices(
        {"red": red, "nir": nir},
        scale=1e-4,
        offset=-1000 * 1e-4,
        nodata=0,
    )
    np.testing.assert_allclose(preset["ndvi"], manual["ndvi"], equal_nan=True)

    landsat_dn = {"red": np.array([10000.0]), "nir": np.array([20000.0])}
    landsat = compute_indices(landsat_dn, preset="landsat-c2")
    landsat_manual = compute_indices(landsat_dn, scale=2.75e-5, offset=-0.2)
    np.testing.assert_allclose(landsat["ndvi"], landsat_manual["ndvi"])

    with pytest.raises(ValueError, match="not both"):
        compute_indices(landsat_dn, preset="s2-l2a", scale=1.0)
    with pytest.raises(ValueError, match="unknown calibration preset"):
        compute_indices(landsat_dn, preset="modis")


def test_scale_and_offset_can_differ_per_band() -> None:
    result = compute_indices(
        {"red": np.array([0.1]), "nir": np.array([6000.0])},
        scale={"nir": 1e-4},
        offset={"nir": -1000 * 1e-4},
    )
    assert result["ndvi"][0] == pytest.approx((0.5 - 0.1) / (0.5 + 0.1))


def test_input_arrays_above_the_element_cap_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert MAX_BAND_ELEMENTS == 16_777_216
    assert MAX_BAND_BYTES == 64 * 1024 * 1024
    monkeypatch.setattr("atlas.data.indices.MAX_BAND_ELEMENTS", 2)
    bands = {"red": np.ones(3), "nir": np.ones(3)}
    with pytest.raises(ValueError, match="above the limit of 2"):
        compute_indices(bands)
    monkeypatch.setattr("atlas.data.indices.MAX_BAND_ELEMENTS", 3)
    assert compute_indices(bands)["ndvi"].shape == (3,)


def test_unknown_scale_band_and_non_boolean_mask_raise() -> None:
    bands = {"red": np.array([0.1]), "nir": np.array([0.2])}
    with pytest.raises(ValueError, match="unknown band 'blue'"):
        compute_indices(bands, scale={"blue": 1.0})
    with pytest.raises(TypeError, match="boolean"):
        compute_indices(bands, mask=np.array([1, 0]))


def test_calibration_does_not_mutate_stored_values() -> None:
    red = np.array([2000.0])
    nir = np.array([6000.0])
    snapshot = (red.copy(), nir.copy())
    compute_indices({"red": red, "nir": nir}, scale=1e-4, offset=-0.1, nodata=0)
    np.testing.assert_array_equal(red, snapshot[0])
    np.testing.assert_array_equal(nir, snapshot[1])
