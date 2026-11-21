from __future__ import annotations

import importlib.util
import logging
import os
from collections.abc import Sequence
from importlib.resources import files

from atlas.agent.contracts import Tool, ToolRegistry
from atlas.models.base import PluginSpec, parse_plugin_toml, weight_path
from atlas.models.binary_mask import BinaryMaskTool
from atlas.models.manifest import SUPPORTED_RUNTIMES, ModelManifest, iter_model_manifests
from atlas.models.rgb_change.infer import RgbChangeTool
from atlas.models.segment_landcover.infer import SegmentLandcoverTool
from atlas.models.tool import ModelTool, weights_state
from atlas.models.unet_cpu import UnetCpuTool

_MODELS_PACKAGE = "atlas.models"
_WEIGHTED_RUNTIMES = frozenset({"unet_cpu", "lightgbm"})
_ALLOW_UNPINNED = "ATLAS_ALLOW_UNPINNED"
_HEURISTIC_MODELS = "ATLAS_HEURISTIC_MODELS"
_EXTRA_MODULES = {"onnx": "onnxruntime"}
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


def heuristics_enabled() -> bool:
    """Return whether the heuristic RGB tools join the default registry."""
    return os.environ.get(_HEURISTIC_MODELS) == "1"


def _extra_available(extra: str | None) -> bool:
    if extra is None:
        return True
    module = _EXTRA_MODULES.get(extra, extra)
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def manifest_hidden_reason(manifest: ModelManifest) -> str | None:
    """Return why a v2 manifest is not registered, or None when it is.

    Baseline rules models always register. A manifest is refused when its runtime or output
    kind has no implementation, its rules do not parse, or it names an unknown QA semantic. A
    learned model also needs its extra to import and its weight file to verify; the reason
    names the pull command.
    """
    reasons = manifest.unsupported_reasons()
    if manifest.runtime.kind not in SUPPORTED_RUNTIMES:
        return "; ".join(reasons)
    if reasons:
        return "; ".join(reasons)
    if manifest.runtime.kind == "rules":
        from atlas.models.runtime_rules import rule_problems

        problems = rule_problems(manifest)
        if problems:
            return "invalid rules: " + "; ".join(problems)
    from atlas.geo.stack import SEMANTICS

    unknown = sorted(set(manifest.output.qa_planes) - set(SEMANTICS))
    if unknown:
        return f"unknown qa semantics {unknown}"
    if manifest.tier != "learned":
        return None
    if not _extra_available(manifest.extra):
        return f"needs the {manifest.extra} extra: uv sync --extra {manifest.extra}"
    state = weights_state(manifest)
    if state == "local":
        return f"local weight not installed: atlas models pull {manifest.name} --from PATH"
    if state != "ready":
        return f"weights {state}: atlas models pull {manifest.name}"
    return None


def _model_tools(registry: ToolRegistry) -> list[Tool]:
    tools: list[Tool] = []
    for manifest in iter_model_manifests(strict=False):
        if manifest.tier == "heuristic" and not heuristics_enabled():
            continue
        reason = manifest_hidden_reason(manifest)
        if reason is not None:
            logger.info("model %s not registered: %s", manifest.name, reason)
            continue
        if registry.get(manifest.name) is not None:
            logger.warning("model %s not registered: name already taken", manifest.name)
            continue
        tools.append(ModelTool(manifest))
    return tools


def register_model_tools(
    registry: ToolRegistry,
    specs: Sequence[PluginSpec] | None = None,
) -> None:
    """Register model plugins on ``registry``.

    With ``specs`` (tests pass synthetic ones) only those v1 manifests are considered, with
    no gate. By default the packaged v1 ``plugin.toml`` packs and the v2 ``model.toml``
    manifests are used; heuristic RGB tools join only when ``ATLAS_HEURISTIC_MODELS=1``.

    Args:
        registry: Session allow-list that receives the tools.
        specs: v1 manifests to consider instead of the packaged ones.
    """
    chosen = specs
    if chosen is None:
        chosen = [s for s in iter_plugin_specs() if s.tier != "heuristic" or heuristics_enabled()]
    for spec in chosen:
        if not should_register(spec):
            continue
        tool = tool_for_spec(spec)
        _label_unpinned(spec, tool)
        registry.register(tool)
    if specs is None:
        for model_tool in _model_tools(registry):
            registry.register(model_tool)
