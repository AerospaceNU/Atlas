from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import default_registry
from atlas.agent.tools.todo import PLAN_RELATIVE, ReadPlanTool, UpdatePlanTool

PLAN_ITEMS = [
    {"text": "collect scenes", "status": "pending"},
    {"text": "mosaic them", "status": "in_progress"},
    {"text": "write summary", "status": "completed"},
]


def test_update_plan_round_trips_through_read_plan(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)

    written = UpdatePlanTool().run({"items": PLAN_ITEMS}, store)
    read = ReadPlanTool().run({}, store)

    expected = "plan: 3 items (1 done)\n[ ] collect scenes\n[~] mosaic them\n[x] write summary"
    assert written.text == expected
    assert read.text == expected


def test_update_plan_rejects_several_in_progress_items(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    items = [
        {"text": "first", "status": "in_progress"},
        {"text": "second", "status": "in_progress"},
    ]

    with pytest.raises(ValueError, match="exactly one"):
        UpdatePlanTool().run({"items": items}, store)

    assert not (tmp_path / PLAN_RELATIVE).exists()


def test_update_plan_rejects_zero_in_progress_items(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    items = [{"text": "only task", "status": "pending"}]

    with pytest.raises(ValueError, match="exactly one"):
        UpdatePlanTool().run({"items": items}, store)

    assert not (tmp_path / PLAN_RELATIVE).exists()


def test_update_plan_rejects_an_empty_list(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)

    with pytest.raises(ValidationError):
        UpdatePlanTool().run({"items": []}, store)

    assert not (tmp_path / PLAN_RELATIVE).exists()


def test_update_plan_replaces_the_whole_list(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)

    UpdatePlanTool().run({"items": PLAN_ITEMS}, store)
    replaced = UpdatePlanTool().run(
        {"items": [{"text": "new task", "status": "in_progress"}]}, store
    )

    assert replaced.text == "plan: 1 items (0 done)\n[~] new task"
    assert ReadPlanTool().run({}, store).text == replaced.text


def test_plan_file_is_valid_json_at_the_workspace_path(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)

    UpdatePlanTool().run({"items": PLAN_ITEMS}, store)

    path = tmp_path / ".atlas" / "plan.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document == {"items": PLAN_ITEMS}


def test_read_plan_on_a_clean_workspace(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)

    assert ReadPlanTool().run({}, store).text == "no plan yet"


def test_plan_output_contains_no_host_absolute_path(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)

    written = UpdatePlanTool().run({"items": PLAN_ITEMS}, store)
    read = ReadPlanTool().run({}, store)

    assert str(tmp_path) not in written.text
    assert str(tmp_path) not in read.text
    assert str(tmp_path) not in (tmp_path / PLAN_RELATIVE).read_text(encoding="utf-8")


def test_default_registry_discovers_both_plan_tools() -> None:
    registry = default_registry()

    names = {definition.name for definition in registry.definitions}
    assert {"update_plan", "read_plan"} <= names
