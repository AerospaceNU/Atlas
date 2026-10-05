from __future__ import annotations

from typing import Any

from atlas.data.bands import BANDS, resolve_bands


def _band(wavelength: float | None = None, common_name: str | None = None) -> dict[str, Any]:
    entry: dict[str, Any] = {}
    if wavelength is not None:
        entry["center_wavelength"] = wavelength
    if common_name is not None:
        entry["common_name"] = common_name
    return {"href": "https://example.com/a.tif", "roles": ["data"], "eo:bands": [entry]}


def test_s30_nir_is_the_narrow_band_not_the_tagged_one() -> None:
    # The regression that matters. Live HLS S30 tags B08 (0.8351, broad)
    # common_name "nir" and leaves B8A (0.8648, narrow) untagged, so any
    # name-first resolver returns B08 and silently stops matching L30.
    assets = {
        "B04": _band(0.6645, "red"),
        "B05": _band(0.7039),
        "B08": _band(0.8351, "nir"),
        "B8A": _band(0.8648),
    }
    assert resolve_bands(assets) == {"red": "B04", "nir": "B8A"}


def test_l30_nir_is_b05() -> None:
    assets = {"B04": _band(0.65, "red"), "B05": _band(0.86, "nir")}
    assert resolve_bands(assets) == {"red": "B04", "nir": "B05"}


def test_s30_and_l30_nir_are_the_same_measurement() -> None:
    s30 = resolve_bands({"B08": _band(0.8351, "nir"), "B8A": _band(0.8648)})
    l30 = resolve_bands({"B05": _band(0.86, "nir")})
    assert (s30["nir"], l30["nir"]) == ("B8A", "B05")


def test_sentinel2_nir_prefers_wavelength_over_the_rededge_label() -> None:
    # PC labels S2 B8A "rededge" though it sits at 0.865. Wavelength wins,
    # which trades B08's 10 m for consistency with HLS. Stated policy.
    assets = {
        "B04": _band(0.665, "red"),
        "B08": _band(0.842, "nir"),
        "B8A": _band(0.865, "rededge"),
    }
    assert resolve_bands(assets)["nir"] == "B8A"


def test_multi_band_visual_never_matches_red() -> None:
    visual = {
        "href": "https://example.com/tci.tif",
        "eo:bands": [
            {"common_name": "red", "center_wavelength": 0.665},
            {"common_name": "green", "center_wavelength": 0.56},
            {"common_name": "blue", "center_wavelength": 0.49},
        ],
    }
    assert resolve_bands({"visual": visual}) == {}
    assert resolve_bands({"visual": visual, "B04": _band(0.6645, "red")})["red"] == "B04"


def test_names_resolve_when_no_wavelength_is_published() -> None:
    assets = {"red": _band(common_name="red"), "nir08": _band(common_name="nir08")}
    assert resolve_bands(assets) == {"red": "red", "nir": "nir08"}


def test_names_resolve_even_when_other_assets_carry_wavelengths() -> None:
    # A mixed catalog: green and coastal publish wavelengths while red and
    # nir08 do not. Deciding the method per scene would commit to wavelength
    # and resolve nothing.
    assets = {
        "coastal": _band(0.44, "coastal"),
        "green": _band(0.56, "green"),
        "red": _band(common_name="red"),
        "nir08": _band(common_name="nir08"),
    }
    assert resolve_bands(assets) == {"red": "red", "nir": "nir08"}


def test_wavelength_wins_before_names_are_consulted() -> None:
    assets = {"near": _band(0.8648), "labelled": _band(common_name="nir")}
    assert resolve_bands(assets)["nir"] == "near"


def test_assets_without_band_metadata_are_ignored() -> None:
    assets = {
        "tilejson": {"href": "https://example.com/t.json", "type": "application/json"},
        "rendered_preview": {"href": "https://example.com/p.png", "roles": ["overview"]},
        "Fmask": {"href": "https://example.com/f.tif", "eo:bands": [{"name": "Fmask"}]},
        "SZA": {"href": "https://example.com/s.tif", "eo:bands": [{"name": "SZA"}]},
        "B04": _band(0.6645, "red"),
    }
    assert resolve_bands(assets) == {"red": "B04"}


def test_rededge_is_outside_the_red_tolerance() -> None:
    assert abs(0.7039 - BANDS["red"].target) > BANDS["red"].tolerance
    assert resolve_bands({"B05": _band(0.7039)}) == {}


def test_unmatched_names_are_absent_rather_than_none() -> None:
    resolved = resolve_bands({"B11": _band(1.6137, "swir16")})
    assert resolved == {}
    assert "nir" not in resolved


def test_exact_distance_tie_breaks_on_key() -> None:
    assets = {"zzz": _band(0.86), "aaa": _band(0.87)}
    assert resolve_bands(assets)["nir"] == "aaa"


def test_malformed_assets_do_not_raise() -> None:
    assets: dict[str, Any] = {
        "none": None,
        "listy": [],
        "no_bands": {"href": "https://example.com/a.tif"},
        "bands_not_list": {"eo:bands": "B04"},
        "bands_empty": {"eo:bands": []},
        "entry_not_dict": {"eo:bands": ["B04"]},
        "bool_wavelength": {"eo:bands": [{"center_wavelength": True}]},
        "string_wavelength": {"eo:bands": [{"center_wavelength": "0.8648"}]},
        "negative": {"eo:bands": [{"center_wavelength": -0.8648}]},
        "nan": {"eo:bands": [{"center_wavelength": float("nan")}]},
        "name_not_string": {"eo:bands": [{"common_name": 4}]},
    }
    assert resolve_bands(assets) == {}


def test_common_names_are_matched_case_insensitively() -> None:
    assert resolve_bands({"x": _band(common_name="  NIR08 ")})["nir"] == "x"


def test_out_of_tolerance_band_does_not_resolve_by_name() -> None:
    assert resolve_bands({"B07": _band(0.79, "nir")}) == {}


def test_empty_assets() -> None:
    assert resolve_bands({}) == {}
