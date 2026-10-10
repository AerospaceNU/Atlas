from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import default_registry
from atlas.models.base import parse_plugin_toml
from atlas.models.registry import iter_plugin_specs, tool_for_spec
from atlas.models.rgb_change.infer import CHANGE_THRESHOLD, MIN_BLOB_PIXELS

_MODEL_NAMES = {"segment_landcover", "rgb_change", "mask_clouds", "burn_scar", "flood_mask"}


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


def test_plugins_register_as_agent_tools() -> None:
    specs = {spec.name for spec in iter_plugin_specs()}
    definitions = {item.name: item for item in default_registry().definitions}

    assert specs == _MODEL_NAMES
    for name in _MODEL_NAMES:
        assert definitions[name].description
    assert "path" in definitions["mask_clouds"].parameters["properties"]
    assert "before" in definitions["rgb_change"].parameters["properties"]
    assert "after" in definitions["rgb_change"].parameters["properties"]


def test_unsupported_runtime_raises() -> None:
    spec = parse_plugin_toml(
        Path("src/atlas/models/rgb_change/plugin.toml").read_text(encoding="utf-8")
    )
    with pytest.raises(ValueError, match="Unsupported model runtime"):
        tool_for_spec(spec.model_copy(update={"runtime": "gpu"}))


def test_unknown_mask_rule_raises() -> None:
    spec = parse_plugin_toml(
        Path("src/atlas/models/mask_clouds/plugin.toml").read_text(encoding="utf-8")
    )
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


def test_mask_clouds_accepts_512_and_rejects_other_sizes(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "workspace")
    _save(store, "big.png", _solid(512, (40, 140, 50)))

    result = default_registry().execute("mask_clouds", {"path": "big.png"}, store)

    payload = json.loads((store.root / "artifacts/mask_clouds.json").read_text(encoding="utf-8"))
    assert payload["width"] == 512
    assert payload["height"] == 512
    assert payload["positive_fraction"] == pytest.approx(0.0)
    assert "512x512" in result.text

    for width, height in ((224, 224), (256, 512), (128, 128)):
        array = np.zeros((height, width, 3), dtype=np.uint8)
        _save(store, "bad.png", array)
        with pytest.raises(ValueError, match=f"{width}x{height}"):
            default_registry().execute("mask_clouds", {"path": "bad.png"}, store)


@pytest.mark.parametrize("name", ["burn_scar", "flood_mask"])
def test_other_masks_reject_a_full_mosaic(tmp_path: Path, name: str) -> None:
    store = LocalArtifactStore(tmp_path / "workspace")
    _save(store, "mosaic.png", np.zeros((300, 400, 3), dtype=np.uint8))

    with pytest.raises(ValueError, match="400x300"):
        default_registry().execute(name, {"path": "mosaic.png"}, store)


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
