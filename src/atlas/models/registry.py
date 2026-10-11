from __future__ import annotations

from importlib.resources import files

from atlas.agent.contracts import Tool, ToolRegistry
from atlas.models.base import PluginSpec, parse_plugin_toml
from atlas.models.binary_mask import BinaryMaskTool
from atlas.models.lgbm_runtime import LightGBMTool
from atlas.models.rgb_change.infer import RgbChangeTool
from atlas.models.segment_landcover.infer import SegmentLandcoverTool
from atlas.models.unet_cpu import UnetCpuTool

_MODELS_PACKAGE = "atlas.models"


def iter_plugin_specs() -> list[PluginSpec]:
    """Load every ``plugin.toml`` under ``atlas.models`` subpackages."""
    root = files(_MODELS_PACKAGE)
    specs: list[PluginSpec] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        plugin = child.joinpath("plugin.toml")
        if not plugin.is_file():
            continue
        specs.append(parse_plugin_toml(plugin.read_text(encoding="utf-8")))
    return specs


def tool_for_spec(spec: PluginSpec) -> Tool:
    """Build the agent tool for one plugin manifest.

    Args:
        spec: Parsed ``plugin.toml``.

    Returns:
        A tool whose name is ``spec.name``.

    Raises:
        ValueError: If ``spec.runtime`` has no local implementation.
    """
    if spec.runtime == "centroid_pixels":
        return SegmentLandcoverTool(spec)
    if spec.runtime == "binary_mask":
        return BinaryMaskTool(spec)
    if spec.runtime == "rgb_delta":
        return RgbChangeTool(spec)
    if spec.runtime == "unet_cpu":
        return UnetCpuTool(spec)
    if spec.runtime == "lightgbm":
        return LightGBMTool(spec)
    raise ValueError(f"Unsupported model runtime {spec.runtime!r} for {spec.name}")


def register_model_tools(registry: ToolRegistry) -> None:
    """Register each discovered model plugin on ``registry``."""
    for spec in iter_plugin_specs():
        registry.register(tool_for_spec(spec))
