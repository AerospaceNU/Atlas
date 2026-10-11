from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pytest

from atlas.agent.contracts import ToolRegistry
from atlas.models.base import PluginInput, PluginOutput, PluginSpec
from atlas.models.registry import register_model_tools, tool_for_spec


def _input(sizes: list[int]) -> PluginInput:
    return PluginInput(kind="rgb_png", shape=[0, 0, 3], dtype="uint8", sizes=sizes)


def test_fake_plugin_registers_without_packaged_manifests() -> None:
    spec = PluginSpec(
        name="fake_cloud",
        description="Heuristic cloud mask used only by this registration test.",
        runtime="binary_mask",
        rule="cloud",
        input=_input([256, 512]),
        output=PluginOutput(kind="mask", classes=["clear", "cloud"]),
    )
    registry = ToolRegistry()
    register_model_tools(registry, [spec])
    assert [item.name for item in registry.definitions] == ["fake_cloud"]


def test_fake_unet_is_hidden_until_weights_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("ATLAS_WEIGHTS_DIR", str(tmp_path))
    spec = PluginSpec(
        name="fake_unet",
        description="Synthetic U-Net that has no weight file.",
        runtime="unet_cpu",
        weight="fake_unet/model.json",
        input=_input([256, 512]),
        output=PluginOutput(kind="labels", classes=["land", "water"]),
    )
    registry = ToolRegistry()
    register_model_tools(registry, [spec])
    assert registry.definitions == []

    weight = tmp_path / "fake_unet" / "model.json"
    weight.parent.mkdir(parents=True)
    body = b"{}\n"
    weight.write_bytes(body)
    monkeypatch.delenv("ATLAS_ALLOW_UNPINNED", raising=False)
    register_model_tools(registry, [spec])
    assert registry.definitions == []

    monkeypatch.setenv("ATLAS_ALLOW_UNPINNED", "1")
    allowed = ToolRegistry()
    with caplog.at_level(logging.WARNING, logger="atlas.models.registry"):
        register_model_tools(allowed, [spec])
    assert [item.name for item in allowed.definitions] == ["fake_unet"]
    assert "unpinned" in allowed.definitions[0].description
    assert any("unpinned" in record.message for record in caplog.records)

    pinned = spec.model_copy(update={"sha256": hashlib.sha256(body).hexdigest()})
    monkeypatch.delenv("ATLAS_ALLOW_UNPINNED", raising=False)
    pinned_registry = ToolRegistry()
    register_model_tools(pinned_registry, [pinned])
    assert [item.name for item in pinned_registry.definitions] == ["fake_unet"]
    assert "unpinned" not in pinned_registry.definitions[0].description


def test_fake_lightgbm_is_skipped_when_the_extra_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ATLAS_WEIGHTS_DIR", str(tmp_path))
    weight = tmp_path / "fake_lgbm" / "model.json"
    weight.parent.mkdir(parents=True)
    weight.write_text("{}", encoding="utf-8")
    monkeypatch.setattr("atlas.models.registry.lightgbm_available", lambda: False)
    spec = PluginSpec(
        name="fake_lgbm",
        description="Synthetic LightGBM plugin.",
        runtime="lightgbm",
        weight="fake_lgbm/model.json",
        input=_input([256, 512]),
        output=PluginOutput(kind="mask", classes=["clear", "cloud"]),
    )
    registry = ToolRegistry()
    register_model_tools(registry, [spec])
    assert registry.definitions == []
    with pytest.raises(ValueError, match="lightgbm extra"):
        tool_for_spec(spec)
