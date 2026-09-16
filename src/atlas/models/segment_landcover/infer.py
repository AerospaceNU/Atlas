"""Native-resolution per-pixel land-cover segmentation."""

from __future__ import annotations

import json
from typing import Any

from PIL import Image
from pydantic import BaseModel, Field

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.tools import Tool, ToolResult
from atlas.models.base import (
    PluginSpec,
    class_fractions,
    colorize_labels,
    label_pixels,
    labels_png,
    load_centroids,
)

# Display palette, index-aligned with plugin.toml classes.
CLASS_COLORS: list[tuple[int, int, int]] = [
    (210, 170, 70),
    (160, 70, 70),
    (30, 90, 180),
    (40, 140, 50),
    (230, 230, 230),
]


class SegmentInput(BaseModel):
    """Workspace-relative RGB PNG in; same-size label map and overlay out."""

    path: str = Field(description="PNG path relative to the local artifact workspace.")
    labels_path: str | None = Field(
        default=None,
        description="Palette PNG path relative to the workspace (same HxW as input).",
    )
    overlay_path: str | None = Field(
        default=None,
        description="RGB overlay PNG path relative to the workspace.",
    )
    json_path: str | None = Field(
        default=None,
        description="JSON summary path relative to the workspace.",
    )


class SegmentLandcoverTool(Tool):
    """Label every pixel of an RGB scene. Does not resize the input."""

    def __init__(self, spec: PluginSpec) -> None:
        self.spec = spec
        self.name = spec.name
        self.description = spec.description

    @property
    def input_model(self) -> type[BaseModel]:
        return SegmentInput

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        request = SegmentInput.model_validate(arguments)
        relative_image = request.path
        image_path = store.resolve(relative_image)
        labels_rel = request.labels_path or f"artifacts/{self.name}_labels.png"
        overlay_rel = request.overlay_path or f"artifacts/{self.name}_overlay.png"
        json_rel = request.json_path or f"artifacts/{self.name}.json"
        labels_path = store.resolve(labels_rel)
        overlay_path = store.resolve(overlay_rel)
        json_path = store.resolve(json_rel)
        for path, suffix in (
            (labels_path, ".png"),
            (overlay_path, ".png"),
            (json_path, ".json"),
        ):
            if path.suffix.lower() != suffix:
                raise ValueError(f"{path.suffix} output must end in {suffix}")

        centroids = load_centroids(self.spec)
        classes = self.spec.output.classes
        colors = CLASS_COLORS[: len(classes)]
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            labels = label_pixels(rgb, centroids, classes)
            colorized = colorize_labels(labels, colors)
            overlay = Image.blend(rgb, colorized, 0.45)

        if labels.shape != (height, width):
            raise RuntimeError("Label map spatial size must match the input")

        labels_path.parent.mkdir(parents=True, exist_ok=True)
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        labels_png(labels, colors).save(labels_path)
        overlay.save(overlay_path)
        fractions = class_fractions(labels, classes)
        payload = {
            "path": relative_image,
            "width": width,
            "height": height,
            "fractions": {name: round(value, 6) for name, value in fractions.items()},
            "labels": store.relative(labels_path),
            "overlay": store.relative(overlay_path),
        }
        json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        frac_text = ", ".join(f"{name}={fractions[name]:.2f}" for name in classes)
        return ToolResult(
            text=f"{self.name}: {width}x{height} ({frac_text})",
            artifacts=[
                store.relative(labels_path),
                store.relative(overlay_path),
                store.relative(json_path),
            ],
        )
