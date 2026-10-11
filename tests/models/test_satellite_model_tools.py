from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import default_registry
from atlas.models import base as model_base
from atlas.models.base import load_rgb_array, parse_plugin_toml
from atlas.models.binary_mask import burn_mask
from atlas.models.registry import iter_plugin_specs, tool_for_spec
from atlas.models.rgb_change.infer import (
    CHANGE_THRESHOLD,
    MIN_BLOB_PIXELS,
    _convex_hull,
    rgb_heatmap,
)

_ROOT = Path(__file__).resolve().parents[2]
_PACKAGED = {
    "segment_landcover",
    "rgb_change",
    "mask_clouds",
    "burn_scar",
    "flood_mask",
    "unet_water",
    "lgbm_clouds",
}
_LEARNED = {"unet_water", "lgbm_clouds"}
_HEURISTICS = ("mask_clouds", "burn_scar", "flood_mask")


def _plugin(name: str) -> Path:
    return _ROOT / "src" / "atlas" / "models" / name / "plugin.toml"


def _save(store: LocalArtifactStore, name: str, array: NDArray[np.uint8]) -> None:
    Image.fromarray(array, mode="RGB").save(store.root / name)


def _solid(size: int, color: tuple[int, int, int]) -> NDArray[np.uint8]:
    array = np.empty((size, size, 3), dtype=np.uint8)
    array[:, :] = color
    return array


def _split(
    size: int,
    left: tuple[int, int, int],
    right: tuple[int, int, int],
) -> NDArray[np.uint8]:
    array = np.empty((size, size, 3), dtype=np.uint8)
    array[:, : size // 2] = left
    array[:, size // 2 :] = right
    return array


def test_plugins_register_as_agent_tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ATLAS_WEIGHTS_DIR", str(tmp_path / "empty"))
    specs = {spec.name for spec in iter_plugin_specs()}
    definitions = {item.name: item for item in default_registry().definitions}

    assert specs == _PACKAGED
    for name in _PACKAGED - _LEARNED:
        assert definitions[name].description
    assert _LEARNED.isdisjoint(definitions)
    for name in _HEURISTICS:
        assert "heuristic" in definitions[name].description.lower()
    assert "path" in definitions["mask_clouds"].parameters["properties"]
    assert "before" in definitions["rgb_change"].parameters["properties"]
    assert "after" in definitions["rgb_change"].parameters["properties"]


def test_unsupported_runtime_raises() -> None:
    spec = parse_plugin_toml(_plugin("rgb_change").read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="Unsupported model runtime"):
        tool_for_spec(spec.model_copy(update={"runtime": "gpu"}))


def test_unknown_mask_rule_raises() -> None:
    spec = parse_plugin_toml(_plugin("mask_clouds").read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="Unknown binary mask rule"):
        tool_for_spec(spec.model_copy(update={"rule": "sar"}))


@pytest.mark.parametrize(
    ("name", "color", "positive"),
    [
        ("mask_clouds", (245, 245, 245), "cloud"),
        ("burn_scar", (45, 30, 20), "burn"),
        ("flood_mask", (30, 90, 180), "water"),
    ],
)
def test_positive_tile_is_all_ones(
    tmp_path: Path,
    name: str,
    color: tuple[int, int, int],
    positive: str,
) -> None:
    store = LocalArtifactStore(tmp_path / "workspace")
    _save(store, "tile.png", _solid(256, color))

    result = default_registry().execute(name, {"path": "tile.png"}, store)

    assert not result.text.startswith("/")
    assert all(not artifact.startswith("/") for artifact in result.artifacts)
    mask = np.asarray(Image.open(store.root / f"artifacts/{name}_mask.png"))
    assert mask.shape == (256, 256)
    assert set(np.unique(mask).tolist()) == {1}
    payload = json.loads((store.root / f"artifacts/{name}.json").read_text(encoding="utf-8"))
    assert payload["width"] == 256
    assert payload["height"] == 256
    assert payload["path"] == "tile.png"
    assert payload["positive"] == positive
    assert payload["positive_fraction"] == pytest.approx(1.0)
    assert payload["mask"] == f"artifacts/{name}_mask.png"
    assert f"{positive}=1.00" in result.text


@pytest.mark.parametrize(
    ("name", "positive_color", "negative_color"),
    [
        ("mask_clouds", (245, 245, 245), (40, 140, 50)),
        ("burn_scar", (45, 30, 20), (40, 140, 50)),
        ("flood_mask", (30, 90, 180), (140, 120, 80)),
    ],
)
def test_split_tile_keeps_native_size_and_sides(
    tmp_path: Path,
    name: str,
    positive_color: tuple[int, int, int],
    negative_color: tuple[int, int, int],
) -> None:
    store = LocalArtifactStore(tmp_path / "workspace")
    _save(store, "tile.png", _split(256, positive_color, negative_color))

    default_registry().execute(name, {"path": "tile.png"}, store)

    mask = np.asarray(Image.open(store.root / f"artifacts/{name}_mask.png"))
    assert mask.shape == (256, 256)
    assert mask[10, 10] == 1
    assert mask[10, 200] == 0
    assert set(np.unique(mask).tolist()) <= {0, 1}
    payload = json.loads((store.root / f"artifacts/{name}.json").read_text(encoding="utf-8"))
    assert payload["positive_fraction"] == pytest.approx(0.5)


def test_mask_clouds_accepts_512_and_smaller_tiles(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "workspace")
    _save(store, "big.png", _solid(512, (40, 140, 50)))

    result = default_registry().execute("mask_clouds", {"path": "big.png"}, store)

    payload = json.loads((store.root / "artifacts/mask_clouds.json").read_text(encoding="utf-8"))
    assert payload["width"] == 512
    assert payload["height"] == 512
    assert payload["positive_fraction"] == pytest.approx(0.0)
    assert "512x512" in result.text

    _save(store, "chip.png", _solid(224, (245, 245, 245)))
    default_registry().execute("mask_clouds", {"path": "chip.png"}, store)
    chip = json.loads((store.root / "artifacts/mask_clouds.json").read_text(encoding="utf-8"))
    assert chip["width"] == 224
    assert chip["height"] == 224
    assert chip["positive_fraction"] == pytest.approx(1.0)


def test_mosaic_is_tiled_and_stitched(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "workspace")
    mosaic = np.empty((40, 600, 3), dtype=np.uint8)
    mosaic[:, :300] = (245, 245, 245)
    mosaic[:, 300:] = (40, 140, 50)
    _save(store, "mosaic.png", mosaic)

    default_registry().execute("mask_clouds", {"path": "mosaic.png"}, store)

    mask = np.asarray(Image.open(store.root / "artifacts/mask_clouds_mask.png"))
    assert mask.shape == (40, 600)
    assert mask[5, 10] == 1
    assert mask[5, 520] == 0
    payload = json.loads((store.root / "artifacts/mask_clouds.json").read_text(encoding="utf-8"))
    assert payload["width"] == 600
    assert payload["height"] == 40
    assert payload["positive_fraction"] == pytest.approx(0.5)


@pytest.mark.parametrize("name", ["burn_scar", "flood_mask"])
def test_non_square_mosaic_keeps_native_size(tmp_path: Path, name: str) -> None:
    store = LocalArtifactStore(tmp_path / "workspace")
    _save(store, "mosaic.png", np.zeros((300, 400, 3), dtype=np.uint8))

    default_registry().execute(name, {"path": "mosaic.png"}, store)

    mask = np.asarray(Image.open(store.root / f"artifacts/{name}_mask.png"))
    assert mask.shape == (300, 400)
    assert int(mask.sum()) == 0


def test_burn_mask_requires_red_dominance_and_minimum_luma() -> None:
    black = np.zeros((8, 8, 3), dtype=np.uint8)
    grey = np.full((8, 8, 3), 30, dtype=np.uint8)
    dark_red = np.zeros((8, 8, 3), dtype=np.uint8)
    dark_red[..., 0] = 8
    brown = np.empty((8, 8, 3), dtype=np.uint8)
    brown[:, :] = (45, 30, 20)

    assert int(burn_mask(black).sum()) == 0
    assert int(burn_mask(grey).sum()) == 0
    assert int(burn_mask(dark_red).sum()) == 0
    assert int(burn_mask(brown).min()) == 1


def test_mask_output_must_be_png_and_cannot_escape(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "workspace")
    _save(store, "tile.png", _solid(256, (245, 245, 245)))

    with pytest.raises(ValueError, match="png"):
        default_registry().execute(
            "mask_clouds",
            {"path": "tile.png", "mask_path": "artifacts/mask.txt"},
            store,
        )
    with pytest.raises(ValueError, match="escapes"):
        default_registry().execute(
            "burn_scar",
            {"path": "../outside.png"},
            store,
        )
    with pytest.raises(ValueError, match="escapes"):
        default_registry().execute(
            "flood_mask",
            {"path": "tile.png", "json_path": "../out.json"},
            store,
        )


def test_rgb_change_half_image_heatmap_and_polygon(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "workspace")
    before = np.zeros((32, 32, 3), dtype=np.uint8)
    after = before.copy()
    after[:, 16:] = 255
    _save(store, "before.png", before)
    _save(store, "after.png", after)

    result = default_registry().execute(
        "rgb_change",
        {"before": "before.png", "after": "after.png"},
        store,
    )

    assert all(not artifact.startswith("/") for artifact in result.artifacts)
    heatmap = np.load(store.root / "artifacts/rgb_change_heatmap.npy")
    assert heatmap.dtype == np.float32
    assert heatmap.shape == (32, 32)
    assert float(heatmap.min()) >= 0.0
    assert float(heatmap.max()) <= 1.0
    assert heatmap[:, :16].max() == 0
    assert heatmap[:, 16:].min() == pytest.approx(1.0)
    payload = json.loads((store.root / "artifacts/rgb_change.json").read_text(encoding="utf-8"))
    assert payload["before"] == "before.png"
    assert payload["after"] == "after.png"
    assert payload["width"] == 32
    assert payload["height"] == 32
    assert payload["threshold"] == CHANGE_THRESHOLD
    assert payload["min_pixels"] == MIN_BLOB_PIXELS
    assert payload["changed_fraction"] == pytest.approx(0.5)
    assert payload["heatmap"] == "artifacts/rgb_change_heatmap.npy"
    assert not payload["heatmap"].startswith("/")
    assert len(payload["polygons"]) == 1
    assert payload["polygons"][0]["area"] == 16 * 32
    points = {tuple(point) for point in payload["polygons"][0]["points"]}
    assert points == {(16, 0), (31, 0), (16, 31), (31, 31)}
    assert "polygons=1" in result.text


def test_rgb_change_ignores_blobs_under_min_pixels(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "workspace")
    before = np.zeros((32, 32, 3), dtype=np.uint8)
    after = before.copy()
    after[:3, :3] = 255
    _save(store, "before.png", before)
    _save(store, "after.png", after)

    default_registry().execute(
        "rgb_change",
        {"before": "before.png", "after": "after.png"},
        store,
    )

    payload = json.loads((store.root / "artifacts/rgb_change.json").read_text(encoding="utf-8"))
    assert payload["polygons"] == []
    assert payload["changed_fraction"] == round(9 / 1024, 6)


def test_rgb_change_uses_mean_channel_delta(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "workspace")
    before = np.zeros((32, 32, 3), dtype=np.uint8)
    after = before.copy()
    after[:16, :16] = (255, 0, 0)
    _save(store, "before.png", before)
    _save(store, "after.png", after)

    default_registry().execute(
        "rgb_change",
        {"before": "before.png", "after": "after.png"},
        store,
    )

    heatmap = np.load(store.root / "artifacts/rgb_change_heatmap.npy")
    assert heatmap[0, 0] == pytest.approx(1 / 3, abs=1e-6)
    payload = json.loads((store.root / "artifacts/rgb_change.json").read_text(encoding="utf-8"))
    assert payload["changed_fraction"] == pytest.approx(0.25)
    assert payload["polygons"][0]["area"] == 16 * 16


def test_identical_pair_has_empty_polygons(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "workspace")
    image = _solid(16, (30, 90, 180))
    _save(store, "before.png", image)
    _save(store, "after.png", image)

    result = default_registry().execute(
        "rgb_change",
        {"before": "before.png", "after": "after.png"},
        store,
    )

    heatmap = np.load(store.root / "artifacts/rgb_change_heatmap.npy")
    assert heatmap.max() == 0
    payload = json.loads((store.root / "artifacts/rgb_change.json").read_text(encoding="utf-8"))
    assert payload["polygons"] == []
    assert payload["changed_fraction"] == pytest.approx(0.0)
    assert "polygons=0" in result.text


def test_convex_hull_drops_lines() -> None:
    assert _convex_hull([(0, 0), (5, 0)]) == []
    assert _convex_hull([(index, 0) for index in range(16)]) == []
    assert len(_convex_hull([(0, 0), (4, 0), (0, 4)])) == 3


def test_rgb_change_drops_two_point_polygons(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "workspace")
    before = np.zeros((32, 32, 3), dtype=np.uint8)
    after = before.copy()
    after[0, :16] = 255
    _save(store, "before.png", before)
    _save(store, "after.png", after)

    default_registry().execute(
        "rgb_change",
        {"before": "before.png", "after": "after.png"},
        store,
    )

    payload = json.loads((store.root / "artifacts/rgb_change.json").read_text(encoding="utf-8"))
    assert payload["polygons"] == []
    assert all(len(polygon["points"]) >= 3 for polygon in payload["polygons"])


def test_pixel_cap_rejects_oversized_rasters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(model_base, "_MAX_PIXELS", 10)
    path = tmp_path / "big.png"
    Image.new("RGB", (20, 20), "red").save(path)

    with pytest.raises(ValueError, match="exceeds 10"):
        load_rgb_array(path)

    before = np.zeros((4, 4, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="exceeds 10"):
        rgb_heatmap(before, before.copy())


def test_rgb_change_rejects_mismatched_shapes_and_escapes(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "workspace")
    _save(store, "before.png", _solid(32, (0, 0, 0)))
    _save(store, "after.png", _solid(16, (255, 255, 255)))

    with pytest.raises(ValueError, match="32x32 and 16x16"):
        default_registry().execute(
            "rgb_change",
            {"before": "before.png", "after": "after.png"},
            store,
        )
    with pytest.raises(ValueError, match="escapes"):
        default_registry().execute(
            "rgb_change",
            {"before": "../outside.png", "after": "after.png"},
            store,
        )
    with pytest.raises(ValueError, match="npy"):
        default_registry().execute(
            "rgb_change",
            {"before": "before.png", "after": "after.png", "heatmap_path": "heat.png"},
            store,
        )
