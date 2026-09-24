from __future__ import annotations

from pathlib import Path

import pytest

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import default_registry
from atlas.agent.tools.BashTool import BashTool
from atlas.agent.tools.read_file import ReadFileTool


def test_store_rejects_paths_outside_root(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)

    with pytest.raises(ValueError, match="escapes"):
        store.resolve("../outside.png")


def test_read_file_returns_relative_text_contents(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    (tmp_path / "notes.txt").write_text("hello workspace", encoding="utf-8")

    result = default_registry().execute("read_file", {"path": "notes.txt"}, store)

    assert result.text == "hello workspace"


def test_read_file_returns_empty_text_for_an_empty_file(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    (tmp_path / "empty.txt").write_text("", encoding="utf-8")

    result = ReadFileTool().run({"path": "empty.txt"}, store)

    assert result.text == ""
    assert result.artifacts == []


def test_read_file_rejects_escaping_path(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    (tmp_path.parent / "outside.txt").write_text("outside", encoding="utf-8")

    with pytest.raises(ValueError, match="escapes"):
        ReadFileTool().run({"path": "../outside.txt"}, store)


def test_read_file_rejects_absolute_path(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    (tmp_path / "notes.txt").write_text("hello", encoding="utf-8")

    with pytest.raises(ValueError, match="relative"):
        ReadFileTool().run({"path": str(tmp_path / "notes.txt")}, store)


def test_read_file_missing_file_hides_the_host_path(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)

    with pytest.raises(FileNotFoundError) as excinfo:
        ReadFileTool().run({"path": "missing.txt"}, store)

    message = str(excinfo.value)
    assert "missing.txt" in message
    assert str(tmp_path) not in message


def test_read_file_rejects_oversized_file(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    (tmp_path / "big.txt").write_bytes(b"x" * 100_001)

    with pytest.raises(ValueError, match="100000"):
        ReadFileTool().run({"path": "big.txt"}, store)


def test_read_file_rejects_non_utf8_file(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    (tmp_path / "binary.txt").write_bytes(b"\xff")

    with pytest.raises(ValueError, match="UTF-8"):
        ReadFileTool().run({"path": "binary.txt"}, store)


def test_bash_reports_stdout_and_a_zero_exit(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)

    result = default_registry().execute("bash_tool", {"command": "echo hello"}, store)

    assert "exit_code: 0" in result.text
    assert "hello" in result.text


def test_bash_runs_in_the_requested_working_directory(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    (tmp_path / "notes.txt").write_text("hello workspace", encoding="utf-8")

    result = BashTool().run({"command": "cat notes.txt", "working_directory": str(tmp_path)}, store)

    assert "exit_code: 0" in result.text
    assert "hello workspace" in result.text


def test_bash_reports_a_failing_command_instead_of_raising(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)

    result = BashTool().run({"command": "echo boom >&2; exit 3"}, store)

    assert "exit_code: 3" in result.text
    assert "boom" in result.text


def test_bash_reports_a_timeout(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)

    result = BashTool().run({"command": "sleep 5", "timeout": 1}, store)

    assert "timed out after 1s" in result.text


def test_bash_reports_a_missing_working_directory(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    missing = tmp_path / "nope"

    result = BashTool().run({"command": "echo hi", "working_directory": str(missing)}, store)

    assert "Working directory not found" in result.text
    assert str(missing) in result.text
