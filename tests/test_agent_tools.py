from __future__ import annotations

from pathlib import Path

import pytest

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import default_registry
from atlas.agent.tools.edit_file import EditFileTool
from atlas.agent.tools.read_file import ReadFileTool

# read_file tests


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


# edit_file tests


def test_edit_file_rejects_escaping_path(tmp_path) -> None:
    store = LocalArtifactStore(tmp_path)
    (tmp_path.parent / "outside.txt").write_text("outside", encoding="utf-8")

    with pytest.raises(ValueError, match="escapes"):
        EditFileTool().run(
            {"path": "../outside.txt", "old_string": "outside", "new_string": "inside"},
            store,
        )


def test_edit_file_rejects_non_utf8_file(tmp_path) -> None:
    store = LocalArtifactStore(tmp_path)
    (tmp_path / "binary.txt").write_bytes(b"\xff")

    with pytest.raises(ValueError, match="UTF-8"):
        EditFileTool().run(
            {"path": "binary.txt", "old_string": "a", "new_string": "b"},
            store,
        )


def test_more_than_one_match(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    target = tmp_path / "notes.txt"
    target.write_text("hello\nhello\n", encoding="utf-8")

    with pytest.raises(ValueError, match="appears multiple times"):
        EditFileTool().run({"path": "notes.txt", "old_string": "hello", "new_string": "bye"}, store)

    assert target.read_text(encoding="utf-8") == "hello\nhello\n"


def test_edit_file_unique_replacement_successful(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    target = tmp_path / "notes.txt"
    target.write_text("my code will work", encoding="utf-8")

    EditFileTool().run({"path": "notes.txt", "old_string": "will", "new_string": "will not"}, store)

    assert target.read_text(encoding="utf-8") == "my code will not work"


def test_edit_file_missing_file_hides_the_host_path(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)

    with pytest.raises(FileNotFoundError) as excinfo:
        EditFileTool().run(
            {"path": "missing.txt", "old_string": "will", "new_string": "will not"}, store
        )

    message = str(excinfo.value)
    assert "missing.txt" in message
    assert str(tmp_path) not in message


def test_empty_old_string(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    target = tmp_path / "notes.txt"
    target.write_text("hello world", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        EditFileTool().run({"path": "notes.txt", "old_string": "", "new_string": "bye"}, store)

    assert target.read_text(encoding="utf-8") == "hello world"


def test_edit_file_rejects_oversized_file(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    (tmp_path / "notes.txt").write_bytes(b"x" * 100_001)

    with pytest.raises(ValueError, match="100000"):
        EditFileTool().run({"path": "notes.txt", "old_string": "", "new_string": "bye"}, store)
