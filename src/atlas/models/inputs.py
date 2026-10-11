from __future__ import annotations

from pydantic import BaseModel, Field


class SegmentInput(BaseModel):
    """Workspace-relative RGB PNG in; same-size class mask out."""

    path: str = Field(description="PNG path relative to the local artifact workspace.")
    mask_path: str | None = Field(
        default=None,
        description="Class-index PNG path relative to the workspace.",
    )
    json_path: str | None = Field(
        default=None,
        description="JSON summary path relative to the workspace.",
    )
