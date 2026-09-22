from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import default_registry
from atlas.models.base import parse_plugin_toml, read_weight_bytes

_CLASSES = ["bare", "built-up", "water", "vegetation", "cloud"]
_CENTROIDS = {
    "bare": [180, 140, 90],
    "built-up": [128, 128, 128],
    "water": [30, 90, 180],
    "vegetation": [40, 160, 60],
    "cloud": [245, 245, 245],
}
_PLUGIN = Path("src/atlas/models/segment_landcover/plugin.toml")


def _write_weight(root: Path) -> Path:
    path = root / "segment_landcover" / "model.json"
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


def test_default_registry_includes_segment_landcover() -> None:
    names = [tool.name for tool in default_registry().definitions]
    assert "segment_landcover" in names
    assert "classify_chip" not in names


def test_split_image_keeps_native_size_and_segments(tmp_path: Path, weights_dir: Path) -> None:
    store = LocalArtifactStore(tmp_path / "workspace")
    image = Image.new("RGB", (80, 40), (40, 160, 60))
    for x in range(40, 80):
        for y in range(40):
            image.putpixel((x, y), (30, 90, 180))
    image.save(store.root / "scene.png")

    result = default_registry().execute("segment_landcover", {"path": "scene.png"}, store)

    assert not result.text.startswith("/")
    assert all(not artifact.startswith("/") for artifact in result.artifacts)
    labels = Image.open(store.root / "artifacts/segment_landcover_labels.png")
    assert labels.size == (80, 40)
    overlay = Image.open(store.root / "artifacts/segment_landcover_overlay.png")
    assert overlay.size == (80, 40)
    payload = json.loads((store.root / "artifacts/segment_landcover.json").read_text())
    assert payload["width"] == 80
    assert payload["height"] == 40
    assert payload["path"] == "scene.png"
    left = labels.getpixel((10, 20))
    right = labels.getpixel((70, 20))
    assert left == _CLASSES.index("vegetation")
    assert right == _CLASSES.index("water")
    assert payload["fractions"]["vegetation"] == pytest.approx(0.5, abs=0.02)
    assert payload["fractions"]["water"] == pytest.approx(0.5, abs=0.02)


def test_white_scene_is_cloud(tmp_path: Path, weights_dir: Path) -> None:
    store = LocalArtifactStore(tmp_path / "workspace")
    Image.new("RGB", (32, 24), (245, 245, 245)).save(store.root / "scene.png")

    result = default_registry().execute("segment_landcover", {"path": "scene.png"}, store)

    payload = json.loads((store.root / "artifacts/segment_landcover.json").read_text())
    assert payload["height"] == 24
    assert payload["width"] == 32
    assert payload["fractions"]["cloud"] == pytest.approx(1.0)
    assert "32x24" in result.text


def test_missing_weight_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ATLAS_WEIGHTS_DIR", str(tmp_path / "empty-weights"))
    store = LocalArtifactStore(tmp_path / "workspace")
    Image.new("RGB", (32, 32), "green").save(store.root / "scene.png")

    with pytest.raises(FileNotFoundError, match=r"segment_landcover/model\.json"):
        default_registry().execute("segment_landcover", {"path": "scene.png"}, store)


def test_sha256_mismatch_raises(tmp_path: Path, weights_dir: Path) -> None:
    spec = parse_plugin_toml(_PLUGIN.read_text(encoding="utf-8"))
    pinned = spec.model_copy(update={"sha256": "0" * 64})
    with pytest.raises(ValueError, match="sha256 mismatch"):
        read_weight_bytes(pinned)


def test_workspace_path_cannot_escape(tmp_path: Path, weights_dir: Path) -> None:
    store = LocalArtifactStore(tmp_path / "workspace")
    with pytest.raises(ValueError, match="escapes"):
        default_registry().execute("segment_landcover", {"path": "../outside.png"}, store)
