from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field, ValidationError

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import Tool, ToolResult

_TOPICS = (
    "workflow",
    "stack",
    "masks",
    "areas",
    "models",
    "render",
    "recipes",
    "scripts",
    "limits",
)


class SatelliteGuideInput(BaseModel):
    topic: Literal[
        "workflow", "stack", "masks", "areas", "models", "render", "recipes", "scripts", "limits"
    ] = Field(default="workflow", description="Guide topic; start with workflow.")


class SatelliteGuideTool(Tool):
    """Serve the satellite analysis guide as read-only text.

    No path argument, no network, no execution: the text is packaged data read through
    ``atlas.geo.__main__.guide_text``, identical to ``atlas geo guide TOPIC``.
    """

    name = "satellite_guide"
    description = (
        "Read the satellite analysis guide: call order, masks, areas, models, rendering, "
        "recipes, scripts, limits. Call with topic=workflow before satellite work."
    )
    domain: ClassVar[str] = "satellite"

    @property
    def input_model(self) -> type[BaseModel]:
        return SatelliteGuideInput

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        try:
            request = SatelliteGuideInput.model_validate(arguments)
        except ValidationError:
            return ToolResult(
                text=f"unknown topic. Topics: {', '.join(_TOPICS)}. Fix: satellite_guide topic=workflow"
            )
        from atlas.geo.__main__ import guide_text

        return ToolResult(text=guide_text(request.topic))
