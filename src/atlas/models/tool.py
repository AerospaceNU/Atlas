"""`ModelTool`: one agent tool per packaged v2 model manifest.

Invariants: tools are built only from packaged manifests, never from names the model
supplies; a run reads one input stack, never resamples it, and writes a NEW derived stack. No
weights are downloaded here. `atlas.geo` and `atlas.agent.runtime` load inside `run` so the
optional extras and import cycles stay out of discovery.
"""

from __future__ import annotations

import importlib
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, create_model

from atlas.agent.contracts import Tool, ToolResult

if TYPE_CHECKING:
    from atlas.agent.artifacts import LocalArtifactStore
    from atlas.geo.results import GeoResult
    from atlas.geo.stack import Stack
    from atlas.models.enforce import Roles
    from atlas.models.manifest import ModelManifest, ParamSpec
    from atlas.models.runtime_api import ParamValue, RuntimeOutput

__all__ = [
    "HEURISTIC_TIER_PREFIX",
    "MAX_SCHEMA_CHARS",
    "ModelTool",
    "weights_file",
    "weights_state",
]

HEURISTIC_TIER_PREFIX = "HEURISTIC rule, not a trained model."
MAX_SCHEMA_CHARS = 1600
_DEFAULT_SECONDS = 600.0
_RGB = ("red", "green", "blue")
_PATH_FUNCS = ("weights_path", "weight_path", "path_for")


def _weights_module() -> Any | None:
    try:
        return importlib.import_module("atlas.models.weights")
    except ImportError:
        return None


def weights_state(manifest: ModelManifest) -> str:
    """Return `none` (no weights), or the weight store status ready|missing|corrupt|unpublished.

    The weight store (`atlas.models.weights`) decides; when it is unavailable the weights
    count as missing, so a learned model stays hidden rather than run unverified.
    """
    if manifest.weights is None:
        return "none"
    module = _weights_module()
    status = getattr(module, "status", None)
    if not callable(status):
        return "missing"
    try:
        value = status(manifest)
    except Exception:
        return "corrupt"
    return str(getattr(value, "status", value))


def weights_file(manifest: ModelManifest) -> Path | None:
    """Return the verified weight file of a learned model, or None."""
    if manifest.weights is None:
        return None
    module = _weights_module()
    for name in _PATH_FUNCS:
        finder = getattr(module, name, None)
        if callable(finder):
            found = finder(manifest)
            return Path(found) if found else None
    return None


def _param_field(key: str, spec: ParamSpec) -> tuple[Any, Any]:
    description = f"{key}, default {spec.default}"
    if spec.choices:
        return Literal[tuple(spec.choices)], Field(spec.default, description=description)
    kinds: dict[str, type] = {"int": int, "float": float, "str": str, "bool": bool}
    bounds: dict[str, Any] = {}
    if spec.min is not None:
        bounds["ge"] = spec.min
    if spec.max is not None:
        bounds["le"] = spec.max
    return kinds[spec.type], Field(spec.default, description=description, **bounds)


def build_input_model(manifest: ModelManifest) -> type[BaseModel]:
    """Build the tool's input model: stack, mask_with, name, keep_layers, then parameters.

    Parameters are top-level properties (their keys are reserved-name checked by the manifest).
    """
    fields: dict[str, Any] = {
        "stack": (str, Field("latest", description="Input stack id, or latest.")),
        "mask_with": (
            list[str],
            Field(default_factory=list, description="Stack ids whose QA planes also mask pixels."),
        ),
        "name": (str | None, Field(None, description="Id for the output stack.")),
        "keep_layers": (
            list[str],
            Field(default_factory=list, description="Input layers to copy into the output."),
        ),
    }
    for key, spec in manifest.parameters.items():
        fields[key] = _param_field(key, spec)
    config = ConfigDict(extra="forbid")
    return cast(
        "type[BaseModel]",
        create_model(f"{manifest.name}_input", __config__=config, **fields),
    )


def _kind_of(manifest: ModelManifest) -> str:
    return "probability" if manifest.output.kind == "heatmap" else "categorical"


class ModelTool(Tool):
    """Run one packaged model on a prepared stack and write a derived stack."""

    domain = "satellite"

    def __init__(self, manifest: ModelManifest) -> None:
        self.manifest = manifest
        self.name = manifest.name
        prefix = f"{HEURISTIC_TIER_PREFIX} " if manifest.tier == "heuristic" else ""
        self.description = f"{prefix}{manifest.description}"
        self._input = build_input_model(manifest)

    @property
    def input_model(self) -> type[BaseModel]:
        return self._input

    # -- run ---------------------------------------------------------------------------

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        from atlas.agent.tools._geo import error_result
        from atlas.geo.errors import GeoError

        request = self._input.model_validate(arguments)
        try:
            return self._run(request, arguments, store)
        except GeoError as exc:
            return error_result(exc)

    def _run(
        self, request: BaseModel, arguments: dict[str, Any], store: LocalArtifactStore
    ) -> ToolResult:
        from atlas.agent.tools._geo import result_to_tool_result, stack_from_args
        from atlas.geo._lazy import check_cancel
        from atlas.models.enforce import check_input
        from atlas.models.runtime_api import RuntimeContext, load_runtime

        manifest = self.manifest
        args = request.model_dump()
        stack = stack_from_args(store, args["stack"])
        extra = [stack_from_args(store, ref) for ref in args["mask_with"]]
        roles = check_input(manifest, stack)
        weights = self._require_weights()
        params = {key: args[key] for key in manifest.parameters}
        valid = stack.valid("clear", layers=roles.used, extra=extra)
        deadline = time.monotonic() + self._seconds()
        check_cancel(deadline)
        ctx = RuntimeContext(
            manifest=manifest,
            stack=stack,
            layers_by_role=roles.layers_by_role,
            params=params,
            valid=valid,
            deadline=deadline,
            weights_path=weights,
        )
        output = load_runtime(manifest.runtime.kind).run(ctx)
        product = self._derive(stack, roles, output, args, params)
        result = self._summarise(stack, product, roles, output, valid, params)
        return result_to_tool_result(
            result,
            store,
            tool=self.name,
            arguments=arguments,
            stack_ids=[stack.id, product.id],
        )

    def _seconds(self) -> float:
        onnx = self.manifest.runtime.onnx
        return onnx.max_seconds if onnx is not None else _DEFAULT_SECONDS

    def _require_weights(self) -> Path | None:
        from atlas.geo.errors import GeoError, fix

        if self.manifest.weights is None:
            return None
        state = weights_state(self.manifest)
        path = weights_file(self.manifest)
        if state != "ready" or path is None:
            raise GeoError(
                fix(
                    f"{self.name} weights are {state}.",
                    f"ask the user to run: atlas models pull {self.name}",
                )
            )
        return path

    # -- product -----------------------------------------------------------------------

    def _provenance(
        self, stack: Stack, roles: Roles, params: dict[str, ParamValue]
    ) -> dict[str, Any]:
        manifest = self.manifest
        weights = manifest.weights
        return {
            "model": manifest.name,
            "version": manifest.version,
            "tier": manifest.tier,
            "runtime": manifest.runtime.kind,
            "weights_sha256": weights.sha256 if weights else None,
            "license": manifest.license.spdx,
            "params": dict(params),
            "input_stack": stack.id,
            "source_bands": dict(roles.source_bands),
        }

    def _derive(
        self,
        stack: Stack,
        roles: Roles,
        output: RuntimeOutput,
        args: dict[str, Any],
        params: dict[str, ParamValue],
    ) -> Stack:
        import numpy as np

        from atlas.geo.stack import LayerMeta, QaMeta

        manifest = self.manifest
        spec = manifest.output
        layer = LayerMeta(
            name=spec.layer,
            kind=_kind_of(manifest),
            unit="class",
            datetime=stack.layer(roles.used[-1]).datetime,
            sensor=stack.layer(roles.used[-1]).sensor,
            classes=dict(spec.classes) or None,
            processing=[f"{manifest.name} {manifest.version} ({manifest.tier})"],
        )
        planes = list(output.qa_planes.items())
        qa = [QaMeta(name=sem, semantic=sem) for sem, _ in planes]
        qa_data = np.stack([p.astype(np.uint8) for _, p in planes]) if planes else None
        citations = [c for c in (manifest.citation.text, manifest.citation.doi) if c]
        provenance = {
            "tool": manifest.name,
            "arguments": {k: v for k, v in args.items() if v not in (None, [])},
            "inputs": [{"stack": stack.id}],
            "versions": {manifest.name: manifest.version},
            "citations": citations,
            "model": self._provenance(stack, roles, params),
        }
        return stack.derive(
            op=manifest.name,
            layers=[layer],
            data=output.layers[spec.layer][None],
            params={
                "model": manifest.name,
                "version": manifest.version,
                "params": dict(params),
                "mask_with": list(args["mask_with"]),
            },
            provenance=provenance,
            name=args["name"],
            qa=qa,
            qa_data=qa_data,
            keep_layers=list(args["keep_layers"]),
        )

    # -- result ------------------------------------------------------------------------

    def _summarise(
        self,
        stack: Stack,
        product: Stack,
        roles: Roles,
        output: RuntimeOutput,
        valid: Any,
        params: dict[str, ParamValue],
    ) -> GeoResult:
        import numpy as np

        from atlas.geo.results import GeoResult, sig4

        manifest = self.manifest
        spec = manifest.output
        layer = output.layers[spec.layer]
        aoi_px = self._aoi_px(stack)
        valid_px = int(np.count_nonzero(valid))
        area = stack.pixel_area_m2 / 1e4
        classes: list[dict[str, Any]] = []
        for code, label in sorted(spec.classes.items()):
            px = int(np.count_nonzero((layer == code) & valid))
            share = 100.0 * px / valid_px if valid_px else float("nan")
            classes.append({"class": label, "px": px, "ha": px * area, "pct_valid": share})
        valid_fraction = valid_px / aoi_px if aoi_px else 0.0
        numbers: dict[str, Any] = {
            "classes": [
                {"name": c["class"], "px": c["px"], "ha": c["ha"], "pct_valid": c["pct_valid"]}
                for c in classes
            ],
            "valid_px": valid_px,
            "aoi_px": aoi_px,
            "valid_fraction": valid_fraction,
            "nodata_fraction": roles.nodata_fraction,
            "product": product.id,
            "provenance": self._provenance(stack, roles, params),
        }
        head = f"{manifest.name} {manifest.version} [{manifest.tier}] on stack {stack.id}"
        bands = ", ".join(f"{k}={v}" for k, v in roles.source_bands.items())
        summary = [
            head,
            f"valid {sig4(valid_px)} of {sig4(aoi_px)} px ({sig4(100 * valid_fraction)}%), "
            f"no-data {sig4(100 * roles.nodata_fraction)}%; bands {bands}",
            f"product: stack {product.id}, layer {spec.layer}",
            *output.notes,
        ]
        unmatched = output.extras.get("unmatched_px")
        if unmatched:
            summary.append(f"{sig4(int(unmatched))} valid px matched no class (left no-data)")
        picture, artifacts = self._picture(product, stack, spec.layer, spec.classes)
        summary.append(picture)
        summary.append(self._citation_line())
        summary.append(f"limit: {manifest.evidence.known_limits[0]}")
        warnings = list(stack.meta.warnings)
        if manifest.tier == "heuristic":
            warnings.insert(0, HEURISTIC_TIER_PREFIX)
        return GeoResult(
            summary=summary,
            table=classes,
            warnings=warnings,
            artifacts=artifacts,
            numbers=numbers,
            next=f"render_map stack={product.id} layers=[{spec.layer}]",
            table_name="classes",
        )

    @staticmethod
    def _aoi_px(stack: Stack) -> int:
        import numpy as np

        height, width = stack.grid.shape
        try:
            outside = stack.plane("outside_aoi")
        except Exception:
            return height * width
        return int(height * width - np.count_nonzero(outside))

    def _picture(
        self, product: Stack, source: Stack, layer: str, classes: dict[int, str]
    ) -> tuple[str, list[Path]]:
        """Return the overlay legend/grid text and the overlay PNG (when rendering exists)."""
        render = _render_overlay()
        if render is not None:
            try:
                base = tuple(n for n in _RGB if n in source.names())
                shown = render(
                    product,
                    layer,
                    base=source if len(base) == len(_RGB) else None,
                    base_layers=base if len(base) == len(_RGB) else (),
                )
                return shown.text, [shown.png] if shown.png is not None else []
            except Exception as exc:
                return f"overlay not drawn: {exc}\n{_grid_text(product, layer, classes)}", []
        return _grid_text(product, layer, classes), []

    def _citation_line(self) -> str:
        manifest = self.manifest
        cite = manifest.citation.text or manifest.citation.url or manifest.name
        return f"cite: {cite} (licence {manifest.license.spdx})"


def _grid_text(product: Stack, layer: str, classes: dict[int, str]) -> str:
    from atlas.geo.results import text_grid

    return text_grid(product.band(layer), classes=dict(classes) or None)


def _render_overlay() -> Callable[..., Any] | None:
    try:
        module = importlib.import_module("atlas.geo.render")
    except ImportError:
        return None
    found = getattr(module, "render_class_overlay", None)
    return found if callable(found) else None
