"""CPU forward for classify_chip: mean RGB to class scores."""

from __future__ import annotations

import json
from typing import Any

from PIL import Image
from pydantic import BaseModel, Field

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.tools import Tool, ToolResult
from atlas.models.base import PluginSpec, load_centroids, mean_rgb, softmax_neg_l2


class ChipModelInput(BaseModel):
    """Workspace-relative PNG in; JSON scores out."""

    path: str = Field(description="PNG path relative to the local artifact workspace.")
    output_path: str | None = Field(
        default=None,
        description="JSON path relative to the local artifact workspace.",
    )


class CentroidTool(Tool):
    """Classify one RGB PNG by nearest mean-color centroid."""

    def __init__(self, spec: PluginSpec) -> None:
        self.spec = spec
        self.name = spec.name
        self.description = spec.description

    @property
    def input_model(self) -> type[BaseModel]:
        return ChipModelInput

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        request = ChipModelInput.model_validate(arguments)
        relative_image = request.path
        image_path = store.resolve(relative_image)
        output_relative = request.output_path or f"artifacts/{self.name}.json"
        output_path = store.resolve(output_relative)
        if output_path.suffix.lower() != ".json":
            raise ValueError("output_path must end in .json")

        size = _spatial_size(self.spec)
        centroids = load_centroids(self.spec)
        with Image.open(image_path) as image:
            mean = mean_rgb(image, size=size)
        scores = softmax_neg_l2(mean, centroids, self.spec.output.classes)
        label = max(scores.items(), key=lambda item: item[1])[0]
        rounded = {name: round(score, 6) for name, score in scores.items()}
        payload = {
            "path": relative_image,
            "label": label,
            "scores": rounded,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        score_text = ", ".join(f"{name}={rounded[name]:.3f}" for name in self.spec.output.classes)
        return ToolResult(
            text=f"{self.name}: {label} ({score_text})",
            artifacts=[store.relative(output_path)],
        )


def _spatial_size(spec: PluginSpec) -> int:
    shape = spec.input.shape
    if len(shape) < 2:
        raise ValueError("plugin input.shape must be [H, W, C]")
    return int(shape[0])
