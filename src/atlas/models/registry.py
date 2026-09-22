from __future__ import annotations

from importlib.resources import files

from atlas.agent.contracts import ToolRegistry
from atlas.models.base import PluginSpec, parse_plugin_toml
from atlas.models.segment_landcover.infer import SegmentLandcoverTool

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


def register_model_tools(registry: ToolRegistry) -> None:
    """Register each discovered model plugin on ``registry``."""
    for spec in iter_plugin_specs():
        if spec.runtime == "centroid_pixels":
            registry.register(SegmentLandcoverTool(spec))
            continue
        raise ValueError(f"Unsupported model runtime {spec.runtime!r} for {spec.name}")
