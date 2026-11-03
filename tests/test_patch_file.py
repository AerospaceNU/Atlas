from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from atlas.agent.artifacts import LocalArtifactStore, SandboxDenied
from atlas.agent.tools.patch_file import PatchFileTool


def test_patch_file_applies_multiple_edits_in_order(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    target = tmp_path / "notes.txt"
    target.write_text("alpha beta gamma\n", encoding="utf-8")

    result = PatchFileTool().run(
        {
            "path": "notes.txt",
            "edits": [
                {"old_string": "alpha", "new_string": "first"},
                {"old_string": "first beta", "new_string": "second"},
            ],
        },
        store,
    )

    assert target.read_text(encoding="utf-8") == "second gamma\n"
    assert "2 edits" in result.text


def test_patch_file_failing_edit_leaves_the_file_byte_identical(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    target = tmp_path / "notes.txt"
    original = b"hello world\n"
    target.write_bytes(original)

    with pytest.raises(ValueError, match="Edit 1"):
        PatchFileTool().run(
            {
                "path": "notes.txt",
                "edits": [
                    {"old_string": "hello", "new_string": "goodbye"},
                    {"old_string": "missing", "new_string": "never"},
                ],
            },
            store,
        )

    assert target.read_bytes() == original


def test_patch_file_refuses_a_non_unique_old_string(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    target = tmp_path / "notes.txt"
    target.write_text("a a\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Edit 0"):
        PatchFileTool().run(
            {"path": "notes.txt", "edits": [{"old_string": "a", "new_string": "b"}]},
            store,
        )

    assert target.read_text(encoding="utf-8") == "a a\n"


def test_patch_file_replace_all_replaces_every_occurrence(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    target = tmp_path / "notes.txt"
    target.write_text("a a a\n", encoding="utf-8")

    PatchFileTool().run(
        {
            "path": "notes.txt",
            "edits": [{"old_string": "a", "new_string": "b", "replace_all": True}],
        },
        store,
    )

    assert target.read_text(encoding="utf-8") == "b b b\n"


def test_patch_file_refuses_an_absolute_path(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)

    with pytest.raises(SandboxDenied):
        PatchFileTool().run(
            {
                "path": str(tmp_path / "notes.txt"),
                "edits": [{"old_string": "a", "new_string": "b"}],
            },
            store,
        )


def test_patch_file_reports_a_workspace_relative_artifact(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    target = tmp_path / "sub" / "notes.txt"
    target.parent.mkdir()
    target.write_text("alpha\n", encoding="utf-8")

    result = PatchFileTool().run(
        {"path": "sub/notes.txt", "edits": [{"old_string": "alpha", "new_string": "beta"}]},
        store,
    )

    assert result.artifacts == ["sub/notes.txt"]
    assert not any(str(tmp_path) in artifact for artifact in result.artifacts)


def test_patch_file_requires_at_least_one_edit(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    (tmp_path / "notes.txt").write_text("text", encoding="utf-8")

    with pytest.raises(ValidationError):
        PatchFileTool().run({"path": "notes.txt", "edits": []}, store)
