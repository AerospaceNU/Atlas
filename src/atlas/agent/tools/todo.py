from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar, Literal, Self

from pydantic import BaseModel, Field, model_validator

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import Tool, ToolResult

PLAN_RELATIVE = ".atlas/plan.json"

_MARKERS: dict[str, str] = {"pending": " ", "in_progress": "~", "completed": "x"}


class PlanItem(BaseModel):
    """One ordered task in the session plan."""

    text: str = Field(
        min_length=1,
        max_length=500,
        description="Short single-line description of the task (1-500 characters).",
    )
    status: Literal["pending", "in_progress", "completed"] = Field(
        description="Task state: pending, in_progress, or completed."
    )


class UpdatePlanInput(BaseModel):
    """Arguments for ``update_plan``.

    The list replaces the previous plan in full, has between 1 and 50 items,
    and exactly one item must be ``in_progress``.
    """

    items: list[PlanItem] = Field(
        min_length=1,
        max_length=50,
        description="The complete ordered task list (1-50 items).",
    )

    @model_validator(mode="after")
    def _exactly_one_in_progress(self) -> Self:
        _require_one_in_progress(self.items)
        return self


class ReadPlanInput(BaseModel):
    """Arguments for ``read_plan`` (none)."""


def _require_one_in_progress(items: Sequence[PlanItem]) -> None:
    """Refuse a plan with zero or several in-progress tasks."""
    in_progress = sum(item.status == "in_progress" for item in items)
    if in_progress != 1:
        raise ValueError(f"Plan must have exactly one in_progress item, got {in_progress}")


def _render(items: Sequence[PlanItem]) -> str:
    """Render the plan as a checklist with a count header."""
    done = sum(item.status == "completed" for item in items)
    lines = [f"plan: {len(items)} items ({done} done)"]
    lines.extend(f"[{_MARKERS[item.status]}] {item.text}" for item in items)
    return "\n".join(lines)


def _serialize(items: Sequence[PlanItem]) -> str:
    """Encode the plan as a stable JSON document."""
    document = {"items": [item.model_dump() for item in items]}
    return json.dumps(document, indent=2, ensure_ascii=False) + "\n"


def _write_atomic(path: Path, text: str) -> None:
    """Replace ``path`` with ``text`` through a temp file in the same directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(dir=path.parent, prefix=".plan-", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def _load(store: LocalArtifactStore) -> list[PlanItem] | None:
    """Return the stored plan, or ``None`` when no plan file exists."""
    path = store.resolve_read(PLAN_RELATIVE)
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Plan file is not readable JSON: {PLAN_RELATIVE}") from exc
    return UpdatePlanInput.model_validate(raw).items


class UpdatePlanTool(Tool):
    """Keep a short ordered task list for the current session.

    The whole list is replaced on every call and rendered back as a checklist
    that the caller shows to the user. The plan is stored as JSON under
    ``.atlas/plan.json`` inside the workspace, and every path goes through
    :class:`~atlas.agent.artifacts.LocalArtifactStore`.
    """

    name = "update_plan"
    description = (
        "Keep a short ordered task list for the current session. The list is shown to the user "
        "as a checklist. Each call replaces the whole list; use it to mark progress, with "
        "exactly one item in_progress."
    )
    trust: ClassVar[Literal["default", "opt_in"]] = "default"

    @property
    def input_model(self) -> type[BaseModel]:
        return UpdatePlanInput

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        request = UpdatePlanInput.model_validate(arguments)
        path = store.resolve(PLAN_RELATIVE)
        _write_atomic(path, _serialize(request.items))
        return ToolResult(text=_render(request.items))


class ReadPlanTool(Tool):
    """Read the current session task list without changing it.

    The result uses the same checklist rendering as ``update_plan``. A
    workspace with no plan file reports ``no plan yet``; the path is resolved
    only through :class:`~atlas.agent.artifacts.LocalArtifactStore`.
    """

    name = "read_plan"
    description = (
        "Read the current session task list as a checklist. The list is shown to the user; "
        "returns 'no plan yet' when no plan has been written."
    )
    trust: ClassVar[Literal["default", "opt_in"]] = "default"

    @property
    def input_model(self) -> type[BaseModel]:
        return ReadPlanInput

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        ReadPlanInput.model_validate(arguments)
        items = _load(store)
        if items is None:
            return ToolResult(text="no plan yet")
        return ToolResult(text=_render(items))
