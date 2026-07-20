from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.tools import default_registry


def test_store_rejects_paths_outside_root(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)

    with pytest.raises(ValueError, match="escapes"):
        store.resolve("../outside.png")


def test_contact_sheet_is_a_local_visual_artifact(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    Image.new("RGB", (40, 20), "red").save(tmp_path / "one.png")
    Image.new("RGB", (20, 40), "green").save(tmp_path / "two.png")

    result = default_registry().execute(
        "create_contact_sheet",
        {"paths": ["one.png", "two.png"], "output_path": "artifacts/sheet.png", "columns": 2},
        store,
    )

    assert result.artifacts == ["artifacts/sheet.png"]
    sheet = tmp_path / "artifacts/sheet.png"
    assert sheet.is_file()
    with Image.open(sheet) as image:
        assert image.format == "PNG"
        assert image.width > image.height


def test_script_proposal_is_staged_but_not_in_default_registry(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    registry = default_registry(allow_script_proposals=True)

    result = registry.execute(
        "stage_script_proposal",
        {"name": "histogram_helper", "source": "print('not executed')\n", "description": "draft"},
        store,
    )

    assert (tmp_path / "proposals/histogram_helper.py").read_text() == "print('not executed')\n"
    assert "not executable" in result.text
    assert "stage_script_proposal" not in [tool.name for tool in default_registry().definitions]


def test_script_review_writes_static_report_without_execution(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    registry = default_registry(allow_script_proposals=True)
    registry.execute(
        "stage_script_proposal",
        {
            "name": "network_helper",
            "source": "import socket\nprint('not executed')\n",
            "description": "draft",
        },
        store,
    )

    result = registry.execute("review_script_proposals", {}, store)

    assert result.artifacts == ["proposals/review.json"]
    assert "Imports socket" in (tmp_path / "proposals/review.json").read_text()
