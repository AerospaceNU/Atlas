from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from importlib.resources import files

from atlas.agent.contracts import Tool, ToolRegistry
from atlas.models.base import PluginSpec, parse_plugin_toml, weight_path
from atlas.models.binary_mask import BinaryMaskTool
from atlas.models.rgb_change.infer import RgbChangeTool
from atlas.models.segment_landcover.infer import SegmentLandcoverTool
from atlas.models.unet_cpu import UnetCpuTool

_MODELS_PACKAGE = "atlas.models"
_WEIGHTED_RUNTIMES = frozenset({"unet_cpu", "lightgbm"})
_ALLOW_UNPINNED = "ATLAS_ALLOW_UNPINNED"
logger = logging.getLogger(__name__)


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


def lightgbm_available() -> bool:
    """Return whether the optional LightGBM extra can be imported.

    A missing package or a missing ``libgomp`` both report false. Callers skip
    the LightGBM tool instead of failing every other tool at import time.
    """
    try:
        import lightgbm  # noqa: F401
    except (ImportError, OSError):
        return False
    return True


def tool_for_spec(spec: PluginSpec) -> Tool:
    """Build the agent tool for one plugin manifest.

    Args:
        spec: Parsed ``plugin.toml``.

    Returns:
        A tool whose name is ``spec.name``.

    Raises:
        ValueError: If ``spec.runtime`` has no local implementation, or the
            LightGBM extra is not installed.
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
        if not lightgbm_available():
            raise ValueError(
                f"LightGBM runtime is unavailable for {spec.name}. Install the lightgbm extra."
            )
        from atlas.models.lgbm_runtime import LightGBMTool

        return LightGBMTool(spec)
    raise ValueError(f"Unsupported model runtime {spec.runtime!r} for {spec.name}")


def allow_unpinned() -> bool:
    """Return whether an empty ``sha256`` may still register a weighted tool."""
    return os.environ.get(_ALLOW_UNPINNED) == "1"


def should_register(spec: PluginSpec) -> bool:
    """Return whether ``spec`` should be advertised on a session registry.

    ``unet_cpu`` and ``lightgbm`` stay hidden until their weight file exists.
    An empty ``sha256`` also hides them, unless ``ATLAS_ALLOW_UNPINNED=1``.
    LightGBM stays hidden when the extra failed to import.
    """
    if spec.runtime == "lightgbm" and not lightgbm_available():
        return False
    if spec.runtime not in _WEIGHTED_RUNTIMES:
        return True
    if not weight_path(spec).is_file():
        return False
    return bool(spec.sha256.strip()) or allow_unpinned()


def _label_unpinned(spec: PluginSpec, tool: Tool) -> None:
    if spec.runtime not in _WEIGHTED_RUNTIMES or spec.sha256.strip():
        return
    logger.warning("%s weight is unpinned; set sha256 in plugin.toml", spec.name)
    if "unpinned" not in tool.description:
        tool.description = f"{tool.description} (unpinned)"


def register_model_tools(
    registry: ToolRegistry,
    specs: Sequence[PluginSpec] | None = None,
) -> None:
    """Register model plugins on ``registry``.

    Args:
        registry: Session allow-list that receives the tools.
        specs: Manifests to consider. The discovered ``plugin.toml`` files are
            used when this is omitted. Tests pass synthetic specs so they do
            not depend on the packaged plugins.
    """
    chosen = iter_plugin_specs() if specs is None else specs
    for spec in chosen:
        if not should_register(spec):
            continue
        tool = tool_for_spec(spec)
        _label_unpinned(spec, tool)
        registry.register(tool)
