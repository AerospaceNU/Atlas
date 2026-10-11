"""The indices CLI calls compute_indices and writes one array per index."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from atlas.cli import main
from atlas.data.indices import compute_indices


def _save(path: Path, values: list[float]) -> None:
    np.save(path, np.array(values, dtype=np.float64))


def test_cli_calls_compute_indices(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    red = tmp_path / "red.npy"
    nir = tmp_path / "nir.npy"
    green = tmp_path / "green.npy"
    _save(red, [0.1, 0.0])
    _save(nir, [0.5, 0.4])
    _save(green, [0.2, 0.3])
    out = tmp_path / "out"
    calls: list[bool] = []
    real = compute_indices

    def spy(bands: object, **kwargs: object) -> dict[str, np.ndarray]:
        calls.append(True)
        return real(bands, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("atlas.cli.indices.compute_indices", spy)
    code = main(
        [
            "indices",
            "--red",
            str(red),
            "--nir",
            str(nir),
            "--green",
            str(green),
            "--out",
            str(out),
            "--nodata",
            "0",
        ]
    )
    assert code == 0
    assert calls
    expected = real(
        {
            "red": np.load(red),
            "nir": np.load(nir),
            "green": np.load(green),
        },
        nodata=0,
    )
    assert tuple(expected) == ("ndvi", "ndwi_mcfeeters")
    for name, array in expected.items():
        np.testing.assert_allclose(np.load(out / f"{name}.npy"), array, equal_nan=True)
    assert np.isnan(np.load(out / "ndvi.npy")[1])


def test_cli_applies_landsat_c2_and_sentinel2_l2a_calibration(tmp_path: Path) -> None:
    red = tmp_path / "red.npy"
    nir = tmp_path / "nir.npy"
    _save(red, [10000.0, 0.0])
    _save(nir, [20000.0, 8000.0])
    landsat = tmp_path / "landsat"
    assert (
        main(
            [
                "indices",
                "--red",
                str(red),
                "--nir",
                str(nir),
                "--out",
                str(landsat),
                "--scale",
                "2.75e-5",
                "--offset",
                "-0.2",
                "--nodata",
                "0",
            ]
        )
        == 0
    )
    expected = compute_indices(
        {"red": np.load(red), "nir": np.load(nir)},
        scale=2.75e-5,
        offset=-0.2,
        nodata=0,
    )
    np.testing.assert_allclose(np.load(landsat / "ndvi.npy"), expected["ndvi"], equal_nan=True)

    s2_red = tmp_path / "s2_red.npy"
    s2_nir = tmp_path / "s2_nir.npy"
    _save(s2_red, [2000.0, 0.0])
    _save(s2_nir, [6000.0, 4000.0])
    sentinel = tmp_path / "sentinel"
    # Published offset -1000 with scale 1e-4 is --offset -0.1.
    assert (
        main(
            [
                "indices",
                "--red",
                str(s2_red),
                "--nir",
                str(s2_nir),
                "--out",
                str(sentinel),
                "--scale",
                "1e-4",
                "--offset=-0.1",
                "--nodata",
                "-9999",
            ]
        )
        == 0
    )
    # The stored 0 is not the -9999 fill, so both pixels stay finite.
    written = np.load(sentinel / "ndvi.npy")
    assert written[0] == pytest.approx((0.5 - 0.1) / (0.5 + 0.1))
    assert np.isfinite(written[1])


def test_cli_nodata_fills(tmp_path: Path) -> None:
    red = tmp_path / "red.npy"
    nir = tmp_path / "nir.npy"
    np.save(red, np.array([65535, -9999, 0.2]))
    np.save(nir, np.array([0.6, 0.8, 0.8]))
    for fill in (0, 65535, -9999):
        out = tmp_path / f"fill-{fill}"
        assert (
            main(
                [
                    "indices",
                    "--red",
                    str(red),
                    "--nir",
                    str(nir),
                    "--out",
                    str(out),
                    "--nodata",
                    str(fill),
                ]
            )
            == 0
        )
        written = np.load(out / "ndvi.npy")
        expected = compute_indices(
            {"red": np.load(red), "nir": np.load(nir)},
            nodata=fill,
        )
        np.testing.assert_allclose(written, expected["ndvi"], equal_nan=True)


def test_cli_rejects_nan_nodata(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    red = tmp_path / "red.npy"
    nir = tmp_path / "nir.npy"
    _save(red, [0.1])
    _save(nir, [0.4])
    code = main(
        [
            "indices",
            "--red",
            str(red),
            "--nir",
            str(nir),
            "--out",
            str(tmp_path / "out"),
            "--nodata",
            "nan",
        ]
    )
    assert code == 2
    assert "NaN" in capsys.readouterr().err


def test_cli_missing_band_is_usage(tmp_path: Path) -> None:
    assert (
        main(
            [
                "indices",
                "--red",
                str(tmp_path / "missing.npy"),
                "--nir",
                str(tmp_path / "also.npy"),
                "--out",
                str(tmp_path / "out"),
            ]
        )
        == 2
    )
