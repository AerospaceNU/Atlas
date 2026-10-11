"""The spectral-indices agent tool calls the same compute_indices function."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from atlas.agent.artifacts import LocalArtifactStore, SandboxDenied
from atlas.agent.contracts import default_registry
from atlas.agent.tools.spectral_indices import SpectralIndicesTool
from atlas.data.indices import compute_indices


def _save(root: Path, name: str, values: list[float]) -> None:
    np.save(root / name, np.array(values, dtype=np.float64))


def test_tool_is_registered_and_calls_compute_indices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert "spectral_indices" in {tool.name for tool in default_registry().definitions}
    _save(tmp_path, "red.npy", [0.1, 65535.0])
    _save(tmp_path, "nir.npy", [0.5, 0.4])
    _save(tmp_path, "green.npy", [0.4, 0.2])
    calls: list[bool] = []
    real = compute_indices

    def spy(bands: object, **kwargs: object) -> dict[str, np.ndarray]:
        calls.append(True)
        return real(bands, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("atlas.agent.tools.spectral_indices.compute_indices", spy)
    store = LocalArtifactStore(tmp_path)
    result = SpectralIndicesTool().run(
        {
            "red": "red.npy",
            "nir": "nir.npy",
            "green": "green.npy",
            "nodata": 65535,
            "out": "indices",
        },
        store,
    )
    assert calls
    assert "NDWI (McFeeters)" in result.text
    assert "indices/ndwi_mcfeeters.npy" in result.artifacts
    expected = real(
        {
            "red": np.load(tmp_path / "red.npy"),
            "nir": np.load(tmp_path / "nir.npy"),
            "green": np.load(tmp_path / "green.npy"),
        },
        nodata=65535,
    )
    for name, array in expected.items():
        written = np.load(tmp_path / "indices" / f"{name}.npy")
        np.testing.assert_allclose(written, array, equal_nan=True)
    assert np.isnan(np.load(tmp_path / "indices" / "ndvi.npy")[1])


def test_tool_sentinel2_and_landsat_calibration_match_compute_indices(tmp_path: Path) -> None:
    _save(tmp_path, "red.npy", [2000.0, 0.0])
    _save(tmp_path, "nir.npy", [6000.0, 4000.0])
    store = LocalArtifactStore(tmp_path)
    tool = SpectralIndicesTool()
    sentinel = tool.run(
        {
            "red": "red.npy",
            "nir": "nir.npy",
            "scale": 1e-4,
            "offset": -1000 * 1e-4,
            "nodata": 0,
            "out": "sentinel",
        },
        store,
    )
    assert "NDVI" in sentinel.text
    expected = compute_indices(
        {"red": np.load(tmp_path / "red.npy"), "nir": np.load(tmp_path / "nir.npy")},
        scale=1e-4,
        offset=-1000 * 1e-4,
        nodata=0,
    )
    np.testing.assert_allclose(
        np.load(tmp_path / "sentinel" / "ndvi.npy"),
        expected["ndvi"],
        equal_nan=True,
    )

    _save(tmp_path, "l8_red.npy", [10000.0])
    _save(tmp_path, "l8_nir.npy", [20000.0])
    landsat = tool.run(
        {
            "red": "l8_red.npy",
            "nir": "l8_nir.npy",
            "scale": 2.75e-5,
            "offset": -0.2,
            "out": "landsat",
        },
        store,
    )
    assert landsat.artifacts == ["landsat/ndvi.npy"]
    landsat_expected = compute_indices(
        {
            "red": np.load(tmp_path / "l8_red.npy"),
            "nir": np.load(tmp_path / "l8_nir.npy"),
        },
        scale=2.75e-5,
        offset=-0.2,
    )
    np.testing.assert_allclose(
        np.load(tmp_path / "landsat" / "ndvi.npy"),
        landsat_expected["ndvi"],
    )


def test_tool_rejects_a_path_outside_the_workspace(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    with pytest.raises(SandboxDenied):
        SpectralIndicesTool().run(
            {"red": "../red.npy", "nir": "nir.npy"},
            store,
        )
