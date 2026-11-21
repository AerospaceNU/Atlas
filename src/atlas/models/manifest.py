"""Model manifest v2: packaged `model.toml` schema and scanner.

Invariants: this module only parses packaged TOML. It imports no runtime, no weights
code and nothing from `atlas.geo`; expressions in `[runtime.rules]` are not parsed here.
"""

from __future__ import annotations

import logging
import re
import tomllib
from importlib.resources import files
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "KINDS",
    "KINDS_WITH_RUNTIME",
    "MANIFEST_FILE",
    "RUNTIME_KINDS",
    "SUPPORTED_RUNTIMES",
    "Citation",
    "Evidence",
    "InputSpec",
    "License",
    "MeanStd",
    "ModelManifest",
    "OnnxRuntimeSpec",
    "OnnxTile",
    "OutputSpec",
    "PairRule",
    "ParamSpec",
    "RuleClass",
    "RulesRuntimeSpec",
    "RuntimeSpec",
    "Temporal",
    "Upstream",
    "Weights",
    "iter_model_manifests",
    "parse_model_toml",
]

log = logging.getLogger(__name__)

MANIFEST_FILE = "model.toml"
KINDS = (
    "mask",
    "classes",
    "heatmap",
    "scene_label",
    "chip_labels",
    "detections",
    "embedding",
)
RUNTIME_KINDS = ("rules", "onnx", "remote", "torch")
SUPPORTED_RUNTIMES = ("rules", "onnx")
KINDS_WITH_RUNTIME = ("mask", "classes", "heatmap")
RESERVED_PARAMS = ("stack", "name", "keep_layers", "mask_with")

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PARAM_KEY = re.compile(r"^[a-z][a-z0-9_]{0,29}$")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class License(_Strict):
    spdx: str
    holder: str = ""
    url: str = ""
    notice_file: str = ""


class Citation(_Strict):
    text: str = ""
    doi: str = ""
    url: str = ""


class Upstream(_Strict):
    repo: str = ""
    revision: str = ""


class Weights(_Strict):
    file: str
    url: str = ""
    sha256: str = ""
    size_bytes: int = Field(0, ge=0)
    upstream: Upstream = Field(default_factory=Upstream)

    @field_validator("sha256")
    @classmethod
    def _check_sha(cls, value: str) -> str:
        value = value.lower()
        if value and not _SHA256.match(value):
            raise ValueError("weights.sha256 must be empty or 64 lowercase hex characters")
        return value

    @field_validator("file")
    @classmethod
    def _check_file(cls, value: str) -> str:
        parts = value.split("/")
        if not value or value.startswith("/") or "\\" in value or ".." in parts:
            raise ValueError("weights.file must be a bare relative path")
        return value

    @field_validator("url")
    @classmethod
    def _check_url(cls, value: str) -> str:
        if value and not value.startswith("https://"):
            raise ValueError("weights.url must be empty or start with https://")
        return value


class MeanStd(_Strict):
    mean: list[float]
    std: list[float]

    @model_validator(mode="after")
    def _check(self) -> MeanStd:
        if len(self.mean) != len(self.std) or not self.mean:
            raise ValueError("mean and std must be non-empty and equal length")
        if any(s <= 0 for s in self.std):
            raise ValueError("std values must be > 0")
        return self


class Temporal(_Strict):
    min: int = 1
    max: int = 1

    @model_validator(mode="after")
    def _check(self) -> Temporal:
        if not 1 <= self.min <= self.max:
            raise ValueError("temporal requires 1 <= min <= max")
        return self


class PairRule(_Strict):
    require_same_grid: bool = True
    require_same_orbit: bool = False


class InputSpec(_Strict):
    mode: Literal["single", "pair", "series"] = "single"
    sensors: list[str] = Field(min_length=1)
    bands: list[str] = Field(min_length=1)
    band_sources: dict[str, dict[str, str]] = Field(default_factory=dict)
    unit: Literal["reflectance", "dB"]
    resolution_m: tuple[float, float]
    min_size: int = Field(1, ge=1)
    temporal: Temporal = Field(default_factory=Temporal)
    max_nodata_fraction: float = Field(1.0, ge=0.0, le=1.0)
    pair: PairRule = Field(default_factory=PairRule)

    @model_validator(mode="before")
    @classmethod
    def _default_pair_temporal(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("mode") == "pair" and "temporal" not in data:
            return {**data, "temporal": {"min": 2, "max": 2}}
        return data

    @field_validator("bands")
    @classmethod
    def _unique_bands(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("input.bands must be unique")
        return value

    @field_validator("resolution_m")
    @classmethod
    def _check_res(cls, value: tuple[float, float]) -> tuple[float, float]:
        if not 0 < value[0] <= value[1]:
            raise ValueError("input.resolution_m requires 0 < min <= max")
        return value

    @model_validator(mode="after")
    def _check(self) -> InputSpec:
        for sensor, mapping in self.band_sources.items():
            if sensor not in self.sensors:
                raise ValueError(f"input.band_sources sensor {sensor!r} not in input.sensors")
            extra = set(mapping) - set(self.bands)
            if extra:
                raise ValueError(f"input.band_sources[{sensor!r}] names bands not in input.bands")
        if self.mode == "pair" and self.temporal.min < 2:
            raise ValueError("input.temporal.min must be >= 2 for mode 'pair'")
        if self.mode == "single" and self.temporal.max != 1:
            raise ValueError("input.temporal.max must be 1 for mode 'single'")
        return self


class OnnxTile(_Strict):
    size: int = Field(default=1024, ge=32)
    margin: int = Field(default=0, ge=0)
    max_single_pass: int = 1_048_576

    @model_validator(mode="after")
    def _check(self) -> OnnxTile:
        if self.size <= 2 * self.margin:
            raise ValueError("runtime.onnx.tile.size must exceed 2 * margin")
        return self


class OnnxRuntimeSpec(_Strict):
    input_name: str
    output_name: str
    normalisation: Literal["in_graph", "zscore_chip"] | MeanStd = "in_graph"
    tile: OnnxTile = Field(default_factory=OnnxTile)
    threads: int = Field(0, ge=0)
    head: Literal["argmax", "sigmoid", "identity"] = "argmax"
    ms_per_mp: float = Field(0.0, ge=0)
    max_seconds: float = Field(600.0, gt=0)


class RuleClass(_Strict):
    code: int = Field(ge=0, le=255)
    name: str
    when: str


class RulesRuntimeSpec(_Strict):
    define: dict[str, str] = Field(default_factory=dict)
    classes: list[RuleClass] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique(self) -> RulesRuntimeSpec:
        codes = [c.code for c in self.classes]
        names = [c.name for c in self.classes]
        if len(set(codes)) != len(codes):
            raise ValueError("runtime.rules.classes codes must be unique")
        if len(set(names)) != len(names):
            raise ValueError("runtime.rules.classes names must be unique")
        return self


class RuntimeSpec(_Strict):
    kind: Literal["rules", "onnx", "remote", "torch"]
    onnx: OnnxRuntimeSpec | None = None
    rules: RulesRuntimeSpec | None = None

    @model_validator(mode="after")
    def _check(self) -> RuntimeSpec:
        if self.kind == "rules" and self.rules is None:
            raise ValueError("runtime kind 'rules' needs a [runtime.rules] table")
        if self.kind == "onnx" and self.onnx is None:
            raise ValueError("runtime kind 'onnx' needs a [runtime.onnx] table")
        if self.kind != "rules" and self.rules is not None:
            raise ValueError("[runtime.rules] given but runtime.kind is not 'rules'")
        if self.kind != "onnx" and self.onnx is not None:
            raise ValueError("[runtime.onnx] given but runtime.kind is not 'onnx'")
        return self


class OutputSpec(_Strict):
    kind: Literal[
        "mask",
        "classes",
        "heatmap",
        "scene_label",
        "chip_labels",
        "detections",
        "embedding",
    ]
    layer: str = Field("classes", pattern=r"^[a-z][a-z0-9_]{0,39}$")
    classes: dict[int, str] = Field(default_factory=dict)
    qa_planes: dict[str, list[str]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> OutputSpec:
        if self.kind in ("mask", "classes") and not self.classes:
            raise ValueError(f"output.classes must be non-empty for kind {self.kind!r}")
        names = set(self.classes.values())
        for plane, values in self.qa_planes.items():
            for value in values:
                if value not in names:
                    raise ValueError(
                        f"output.qa_planes[{plane!r}] value {value!r} is not an output class name"
                    )
        return self


class ParamSpec(_Strict):
    type: Literal["int", "float", "str", "bool"]
    default: int | float | str | bool
    min: float | None = None
    max: float | None = None
    choices: list[int | float | str | bool] | None = None

    @model_validator(mode="after")
    def _check(self) -> ParamSpec:
        d = self.default
        ok = {
            "bool": isinstance(d, bool),
            "int": isinstance(d, int) and not isinstance(d, bool),
            "float": isinstance(d, (int, float)) and not isinstance(d, bool),
            "str": isinstance(d, str),
        }[self.type]
        if not ok:
            raise ValueError(f"parameter default {d!r} does not match type {self.type!r}")
        if self.type in ("str", "bool"):
            if self.min is not None or self.max is not None:
                raise ValueError("min/max apply only to numeric parameters")
            return self
        assert not isinstance(d, (str, bool))
        if self.min is not None and d < self.min:
            raise ValueError("parameter default is below min")
        if self.max is not None and d > self.max:
            raise ValueError("parameter default is above max")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("parameter min exceeds max")
        return self


class Evidence(_Strict):
    known_limits: list[str] = Field(min_length=1)
    card: str = ""

    @field_validator("known_limits")
    @classmethod
    def _non_blank(cls, value: list[str]) -> list[str]:
        if any(not item.strip() for item in value):
            raise ValueError("evidence.known_limits items must be non-blank")
        return value


class ModelManifest(_Strict):
    """Validated `model.toml` (schema 2)."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: Literal[2] = Field(alias="schema")
    name: str = Field(pattern=r"^[a-z][a-z0-9_]{2,39}$")
    description: str = Field(min_length=1, max_length=400)
    version: str = Field(min_length=1)
    tier: Literal["baseline", "learned", "heuristic"]
    status: Literal["rule_based", "trained"]
    task: str
    extra: Literal["onnx"] | None = None
    license: License
    citation: Citation = Field(default_factory=Citation)
    weights: Weights | None = None
    input: InputSpec
    runtime: RuntimeSpec
    output: OutputSpec
    parameters: dict[str, ParamSpec] = Field(default_factory=dict)
    evidence: Evidence

    @field_validator("parameters")
    @classmethod
    def _check_param_keys(cls, value: dict[str, ParamSpec]) -> dict[str, ParamSpec]:
        for key in value:
            if not _PARAM_KEY.match(key):
                raise ValueError(f"parameters key {key!r} must match {_PARAM_KEY.pattern}")
            if key in RESERVED_PARAMS:
                raise ValueError(f"parameters key {key!r} is reserved")
        return value

    @model_validator(mode="after")
    def _check_tier(self) -> ModelManifest:
        if self.tier == "learned":
            if self.weights is None or not self.weights.sha256:
                raise ValueError("learned model needs weights.sha256 (64 hex)")
            if self.extra is None:
                raise ValueError("learned model needs extra")
            if self.status != "trained":
                raise ValueError("learned model needs status 'trained'")
            if not self.license.spdx:
                raise ValueError("learned model needs license.spdx")
        else:
            if self.status != "rule_based":
                raise ValueError(f"{self.tier} model needs status 'rule_based'")
            if self.runtime.kind != "rules":
                raise ValueError(f"{self.tier} model needs runtime.kind 'rules'")
        if self.runtime.kind == "onnx":
            if self.weights is None:
                raise ValueError("runtime onnx needs a [weights] table")
            if self.extra != "onnx":
                raise ValueError("runtime onnx needs extra = 'onnx'")
        elif self.runtime.kind == "rules":
            if self.weights is not None:
                raise ValueError("runtime rules must not have [weights]")
            if self.extra is not None:
                raise ValueError("runtime rules must not set extra")
        return self

    @property
    def runtime_kind(self) -> str:
        return self.runtime.kind

    def unsupported_reasons(self) -> list[str]:
        """Return why registration must refuse this manifest (empty if supported)."""
        reasons: list[str] = []
        if self.runtime.kind not in SUPPORTED_RUNTIMES:
            reasons.append(f"runtime {self.runtime.kind!r} has no implementation (SAT-16 seam)")
        if self.output.kind not in KINDS_WITH_RUNTIME:
            reasons.append(f"output kind {self.output.kind!r} has no runtime")
        return reasons

    def dump(self) -> dict[str, Any]:
        """Return a JSON-mode dict that round-trips through `model_validate`."""
        return self.model_dump(mode="json", by_alias=True, exclude_none=True)


def parse_model_toml(text: str) -> ModelManifest:
    """Parse and validate a `model.toml` document."""
    return ModelManifest.model_validate(tomllib.loads(text))


def iter_model_manifests(*, strict: bool = True) -> list[ModelManifest]:
    """Scan `atlas.models` subpackages for `model.toml`; `plugin.toml` packs are ignored."""
    found: list[ModelManifest] = []
    seen: set[str] = set()
    for child in sorted(files("atlas.models").iterdir(), key=lambda c: c.name):
        if not child.is_dir():
            continue
        manifest_file = child / MANIFEST_FILE
        if not manifest_file.is_file():
            continue
        try:
            manifest = parse_model_toml(manifest_file.read_text(encoding="utf-8"))
        except Exception as exc:
            if strict:
                raise ValueError(f"{child.name}/{MANIFEST_FILE}: {exc}") from exc
            log.warning("skipping %s/%s: %s", child.name, MANIFEST_FILE, exc)
            continue
        if manifest.name in seen:
            raise ValueError(f"duplicate model name {manifest.name!r} in {child.name}")
        seen.add(manifest.name)
        found.append(manifest)
    return found
