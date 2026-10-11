from __future__ import annotations

import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pytest
from PIL import Image

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import default_registry
from atlas.models.base import (
    PluginInput,
    PluginOutput,
    PluginSpec,
    load_weight_payload,
    parse_plugin_toml,
)
from atlas.models.lgbm_clouds.export import export_weight as export_lgbm
from atlas.models.lgbm_clouds.train import main as train_lgbm
from atlas.models.lgbm_runtime import LightGBMWeights, load_lightgbm_weights
from atlas.models.registry import tool_for_spec
from atlas.models.unet_cpu import UnetWeights, load_unet_weights, predict_unet, unet_forward
from atlas.models.unet_water.export import export_weight as export_unet
from atlas.models.unet_water.train import main as train_unet

_ROOT = Path(__file__).resolve().parents[2]


def _plugin(name: str) -> Path:
    return _ROOT / "src" / "atlas" / "models" / name / "plugin.toml"


def _center(cout: int, cin: int, src: int) -> list[list[list[list[float]]]]:
    weight = np.zeros((cout, cin, 3, 3), dtype=np.float32)
    for channel in range(cout):
        weight[channel, src + channel, 1, 1] = 1
    return weight.tolist()


def _unet_payload() -> dict[str, object]:
    hidden = 3
    head = np.zeros((2, hidden, 3, 3), dtype=np.float32)
    head[1, 2, 1, 1] = 1
    return {
        "runtime": "unet_cpu",
        "classes": ["land", "water"],
        "tile": 256,
        "scale": 1.0,
        "mean": [0.0, 0.0, 0.0],
        "std": [1.0, 1.0, 1.0],
        "enc_w": _center(hidden, 3, 0),
        "enc_b": [0, 0, 0],
        "bn_w": _center(hidden, hidden, 0),
        "bn_b": [0, 0, 0],
        "dec_w": _center(hidden, hidden * 2, hidden),
        "dec_b": [0, 0, 0],
        "head_w": head.tolist(),
        "head_b": [0, -100],
    }


def _cloud_model() -> str:
    white = np.tile(np.array([[245.0, 245.0, 245.0]]), (32, 1))
    green = np.tile(np.array([[40.0, 140.0, 50.0]]), (32, 1))
    blue = np.tile(np.array([[30.0, 90.0, 180.0]]), (32, 1))
    features = np.vstack([white, green, blue])
    labels = np.array([1] * 32 + [0] * 32 + [0] * 32)
    booster = lgb.train(
        {
            "objective": "binary",
            "verbosity": -1,
            "num_leaves": 4,
            "min_data_in_leaf": 1,
            "num_threads": 1,
        },
        lgb.Dataset(features, label=labels),
        num_boost_round=6,
    )
    return booster.model_to_string()


def _lgbm_payload() -> dict[str, object]:
    return {
        "runtime": "lightgbm",
        "classes": ["clear", "cloud"],
        "features": ["r", "g", "b"],
        "tile": 256,
        "model": _cloud_model(),
    }


def _write_weight(root: Path, relative: str, payload: dict[str, object]) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def learned_weights(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "weights"
    _write_weight(root, "unet_water/model.json", _unet_payload())
    _write_weight(root, "lgbm_clouds/model.json", _lgbm_payload())
    monkeypatch.setenv("ATLAS_WEIGHTS_DIR", str(root))
    return root


def _save(store: LocalArtifactStore, name: str, array: np.ndarray) -> None:
    Image.fromarray(array, mode="RGB").save(store.root / name)


def test_unet_tiles_odd_and_split_mosaics(tmp_path: Path, learned_weights: Path) -> None:
    store = LocalArtifactStore(tmp_path / "workspace")
    odd = np.empty((41, 33, 3), dtype=np.uint8)
    odd[:, :] = (30, 90, 180)
    _save(store, "odd.png", odd)

    default_registry().execute("unet_water", {"path": "odd.png"}, store)

    odd_mask = np.asarray(Image.open(store.root / "artifacts/unet_water_mask.png"))
    assert odd_mask.shape == (41, 33)
    assert int(odd_mask.min()) == 1

    split = np.empty((40, 64, 3), dtype=np.uint8)
    split[:, :32] = (30, 90, 180)
    split[:, 32:] = (40, 140, 50)
    _save(store, "split.png", split)
    default_registry().execute(
        "unet_water",
        {
            "path": "split.png",
            "mask_path": "artifacts/split.png",
            "json_path": "artifacts/split.json",
        },
        store,
    )
    split_mask = np.asarray(Image.open(store.root / "artifacts/split.png"))
    assert split_mask.shape == (40, 64)
    assert split_mask[0, 4] == 1
    assert split_mask[39, 60] == 0
    payload = json.loads((store.root / "artifacts/split.json").read_text(encoding="utf-8"))
    assert payload["width"] == 64
    assert payload["height"] == 40
    assert payload["positive"] == "water"
    assert payload["fractions"]["water"] == payload["positive_fraction"]
    assert payload["fractions"]["land"] == pytest.approx(1.0 - payload["positive_fraction"])
    assert str(learned_weights) not in payload["mask"]


def test_lightgbm_tiles_a_wide_mosaic(tmp_path: Path, learned_weights: Path) -> None:
    store = LocalArtifactStore(tmp_path / "workspace")
    mosaic = np.empty((20, 80, 3), dtype=np.uint8)
    mosaic[:, :40] = (245, 245, 245)
    mosaic[:, 40:] = (40, 140, 50)
    _save(store, "mosaic.png", mosaic)

    result = default_registry().execute("lgbm_clouds", {"path": "mosaic.png"}, store)

    mask = np.asarray(Image.open(store.root / "artifacts/lgbm_clouds_mask.png"))
    assert mask.shape == (20, 80)
    assert mask[3, 4] == 1
    assert mask[3, 70] == 0
    assert "80x20" in result.text
    assert learned_weights.name not in result.text


def test_weight_contract_rejects_bad_files(
    tmp_path: Path, learned_weights: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = parse_plugin_toml(_plugin("unet_water").read_text(encoding="utf-8"))
    pinned = spec.model_copy(update={"sha256": "0" * 64})
    with pytest.raises(ValueError, match="sha256 mismatch"):
        load_weight_payload(pinned)

    bad_classes = spec.model_copy(
        update={"output": PluginOutput(kind="labels", classes=["nope", "water"])}
    )
    with pytest.raises(ValueError, match="classes"):
        load_weight_payload(bad_classes)

    monkeypatch.setenv("ATLAS_WEIGHTS_DIR", str(tmp_path / "empty"))
    names = {item.name for item in default_registry().definitions}
    assert "unet_water" not in names
    assert "lgbm_clouds" not in names
    store = LocalArtifactStore(tmp_path / "workspace")
    tool = tool_for_spec(spec)
    with pytest.raises(FileNotFoundError, match=r"unet_water/model\.json") as exc:
        tool.run({"path": "missing.png"}, store)
    assert str(tmp_path) not in str(exc.value)
    assert str(learned_weights) not in str(exc.value)


def test_production_tiles_are_256_or_512() -> None:
    for name in ("unet_water", "lgbm_clouds"):
        spec = parse_plugin_toml(_plugin(name).read_text(encoding="utf-8"))
        assert spec.input.sizes == [256, 512]


def test_learned_tools_register_when_weights_exist(learned_weights: Path) -> None:
    names = {item.name for item in default_registry().definitions}
    assert {"unet_water", "lgbm_clouds"} <= names
    assert learned_weights.is_dir()


def test_unet_weight_cache_reuses_the_same_object(learned_weights: Path) -> None:
    spec = parse_plugin_toml(_plugin("unet_water").read_text(encoding="utf-8"))
    assert load_unet_weights(spec) is load_unet_weights(spec)
    assert learned_weights.is_dir()


def test_lightgbm_weight_cache_reuses_the_same_object(learned_weights: Path) -> None:
    spec = parse_plugin_toml(_plugin("lgbm_clouds").read_text(encoding="utf-8"))
    assert load_lightgbm_weights(spec) is load_lightgbm_weights(spec)
    assert learned_weights.is_dir()


def test_unet_applies_scale_mean_and_std() -> None:
    spec = PluginSpec(
        name="scaled",
        description="Normalization check.",
        runtime="unet_cpu",
        weight="scaled/model.json",
        input=PluginInput(kind="rgb_png", shape=[0, 0, 3], dtype="uint8", sizes=[32]),
        output=PluginOutput(kind="labels", classes=["land", "water"]),
    )
    raw = _unet_payload()
    raw["tile"] = 32
    shifted = dict(raw)
    shifted["mean"] = [0.0, 0.0, 180.0]
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[:, :] = (30, 90, 180)
    assert int(unet_forward(image, UnetWeights(spec, raw)).min()) == 1
    assert int(unet_forward(image, UnetWeights(spec, shifted)).max()) == 0


def test_unet_rejects_an_oversized_hidden_width() -> None:
    spec = PluginSpec(
        name="wide",
        description="Hidden-size check.",
        runtime="unet_cpu",
        weight="wide/model.json",
        input=PluginInput(kind="rgb_png", shape=[0, 0, 3], dtype="uint8", sizes=[32]),
        output=PluginOutput(kind="labels", classes=["land", "water"]),
    )
    payload = _unet_payload()
    payload["tile"] = 32
    payload["enc_w"] = np.zeros((33, 3, 3, 3), dtype=np.float32).tolist()
    with pytest.raises(ValueError, match="hidden size 33"):
        UnetWeights(spec, payload)


def test_overlapped_tiles_match_a_single_pass() -> None:
    spec = PluginSpec(
        name="seam",
        description="Overlap check.",
        runtime="unet_cpu",
        weight="seam/model.json",
        input=PluginInput(kind="rgb_png", shape=[0, 0, 3], dtype="uint8", sizes=[32]),
        output=PluginOutput(kind="labels", classes=["land", "water"]),
    )
    payload = _unet_payload()
    payload["tile"] = 32
    encoder = np.zeros((3, 3, 3, 3), dtype=np.float32)
    for channel in range(3):
        encoder[channel, channel, :, :] = np.float32(1.0 / 9.0)
    payload["enc_w"] = encoder.tolist()
    # Threshold sits between a tile-edge pad (pure color) and a 3x3 mix
    # across the seam, so a non-overlapping stitch changes the class.
    payload["head_b"] = [0.0, -150.0]
    weights = UnetWeights(spec, payload)
    image = np.empty((48, 64, 3), dtype=np.uint8)
    image[:, :32] = (30, 90, 180)
    image[:, 32:] = (40, 140, 50)
    full = unet_forward(image, weights)
    assert np.array_equal(predict_unet(image, weights), full)
    assert not np.array_equal(predict_unet(image, weights, margin=0), full)


def test_lightgbm_rejects_a_multiclass_booster() -> None:
    features = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] * 8)
    labels = np.array([0, 1, 2] * 8)
    booster = lgb.train(
        {
            "objective": "multiclass",
            "num_class": 3,
            "verbosity": -1,
            "min_data_in_leaf": 1,
            "num_threads": 1,
        },
        lgb.Dataset(features, label=labels),
        num_boost_round=2,
    )
    spec = PluginSpec(
        name="multi",
        description="Multiclass rejection.",
        runtime="lightgbm",
        weight="multi/model.json",
        input=PluginInput(kind="rgb_png", shape=[0, 0, 3], dtype="uint8", sizes=[256]),
        output=PluginOutput(kind="mask", classes=["clear", "cloud"]),
    )
    payload = {
        "runtime": "lightgbm",
        "classes": ["clear", "cloud"],
        "features": ["r", "g", "b"],
        "tile": 256,
        "model": booster.model_to_string(),
    }
    with pytest.raises(ValueError, match="num_model_per_iteration == 1"):
        LightGBMWeights(spec, payload)


def test_unet_train_and_export_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = tmp_path / "chips"
    colors = {"land": (40, 140, 50), "water": (30, 90, 180)}
    for name, color in colors.items():
        folder = data / name
        folder.mkdir(parents=True)
        Image.new("RGB", (8, 8), color).save(folder / "chip.png")
    trained = tmp_path / "trained.json"
    train_unet(["--data-dir", str(data), "--out", str(trained)])
    destination = tmp_path / "weights" / "unet_water" / "model.json"
    digest = export_unet(trained, destination)
    assert len(digest) == 64
    assert digest == hashlib.sha256(destination.read_bytes()).hexdigest()
    monkeypatch.setenv("ATLAS_WEIGHTS_DIR", str(tmp_path / "weights"))
    store = LocalArtifactStore(tmp_path / "workspace")
    for name, color in colors.items():
        _save(store, f"{name}.png", np.full((16, 16, 3), color, dtype=np.uint8))
        default_registry().execute(
            "unet_water",
            {"path": f"{name}.png", "json_path": f"artifacts/{name}.json"},
            store,
        )
        payload = json.loads((store.root / f"artifacts/{name}.json").read_text(encoding="utf-8"))
        assert payload["fractions"][name] == pytest.approx(1.0)


def test_lightgbm_train_and_export_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "chips"
    colors = {"clear": (40, 140, 50), "cloud": (245, 245, 245)}
    for name, color in colors.items():
        folder = data / name
        folder.mkdir(parents=True)
        Image.new("RGB", (8, 8), color).save(folder / "chip.png")
    trained = tmp_path / "trained.json"
    train_lgbm(["--data-dir", str(data), "--out", str(trained)])
    destination = tmp_path / "weights" / "lgbm_clouds" / "model.json"
    digest = export_lgbm(trained, destination)
    assert len(digest) == 64
    monkeypatch.setenv("ATLAS_WEIGHTS_DIR", str(tmp_path / "weights"))
    store = LocalArtifactStore(tmp_path / "workspace")
    _save(store, "cloud.png", np.full((12, 12, 3), colors["cloud"], dtype=np.uint8))
    default_registry().execute("lgbm_clouds", {"path": "cloud.png"}, store)
    mask = np.asarray(Image.open(store.root / "artifacts/lgbm_clouds_mask.png"))
    assert int(mask.min()) == 1
