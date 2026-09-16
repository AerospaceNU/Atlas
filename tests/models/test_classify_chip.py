from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.tools import default_registry
from atlas.models.base import parse_plugin_toml, read_weight_bytes

_CLASSES = ["bare", "built-up", "water", "vegetation", "cloud"]
_CENTROIDS = {
    "bare": [180, 140, 90],
    "built-up": [128, 128, 128],
    "water": [30, 90, 180],
    "vegetation": [40, 160, 60],
    "cloud": [245, 245, 245],
}
_PLUGIN = Path("src/atlas/models/classify_chip/plugin.toml")


def _write_weight(root: Path) -> Path:
    path = root / "classify_chip" / "model.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"classes": _CLASSES, "centroids": _CENTROIDS}, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def weights_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "weights"
    _write_weight(root)
    monkeypatch.setenv("ATLAS_WEIGHTS_DIR", str(root))
    return root


def test_default_registry_includes_classify_chip() -> None:
    names = [tool.name for tool in default_registry().definitions]
    assert "classify_chip" in names


def test_green_chip_is_vegetation(tmp_path: Path, weights_dir: Path) -> None:
    store = LocalArtifactStore(tmp_path / "workspace")
    Image.new("RGB", (224, 224), (40, 160, 60)).save(store.root / "chip.png")

    result = default_registry().execute("classify_chip", {"path": "chip.png"}, store)

    assert result.artifacts == ["artifacts/classify_chip.json"]
    assert not result.text.startswith("/")
    assert all(not artifact.startswith("/") for artifact in result.artifacts)
    payload = json.loads((store.root / "artifacts/classify_chip.json").read_text())
    assert payload["path"] == "chip.png"
    assert payload["label"] == "vegetation"
    assert payload["path"].startswith("/") is False


def test_white_chip_is_cloud(tmp_path: Path, weights_dir: Path) -> None:
    store = LocalArtifactStore(tmp_path / "workspace")
    Image.new("RGB", (224, 224), (245, 245, 245)).save(store.root / "chip.png")

    result = default_registry().execute("classify_chip", {"path": "chip.png"}, store)

    payload = json.loads((store.root / "artifacts/classify_chip.json").read_text())
    assert payload["label"] == "cloud"
    assert "classify_chip: cloud" in result.text


def test_missing_weight_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ATLAS_WEIGHTS_DIR", str(tmp_path / "empty-weights"))
    store = LocalArtifactStore(tmp_path / "workspace")
    Image.new("RGB", (32, 32), "green").save(store.root / "chip.png")

    with pytest.raises(FileNotFoundError, match=r"classify_chip/model\.json"):
        default_registry().execute("classify_chip", {"path": "chip.png"}, store)


def test_sha256_mismatch_raises(tmp_path: Path, weights_dir: Path) -> None:
    spec = parse_plugin_toml(_PLUGIN.read_text(encoding="utf-8"))
    pinned = spec.model_copy(update={"sha256": "0" * 64})
    with pytest.raises(ValueError, match="sha256 mismatch"):
        read_weight_bytes(pinned)


def test_workspace_path_cannot_escape(tmp_path: Path, weights_dir: Path) -> None:
    store = LocalArtifactStore(tmp_path / "workspace")
    with pytest.raises(ValueError, match="escapes"):
        default_registry().execute("classify_chip", {"path": "../outside.png"}, store)
