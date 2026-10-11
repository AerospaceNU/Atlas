from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pytest
from PIL import Image

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import default_registry
from atlas.models.base import PluginOutput, load_weight_payload, parse_plugin_toml

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
        "tile": 32,
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
        "tile": 32,
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
    store = LocalArtifactStore(tmp_path / "workspace")
    with pytest.raises(FileNotFoundError, match=r"unet_water/model\.json") as exc:
        default_registry().execute("unet_water", {"path": "missing.png"}, store)
    assert str(tmp_path) not in str(exc.value)
    assert str(learned_weights) not in str(exc.value)
