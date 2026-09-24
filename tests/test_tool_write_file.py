from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import default_registry
from atlas.agent.tools.read_file import ReadFileTool
from atlas.agent.tools.write_file import WriteTool


def test_write_file_creates_a_new_file(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)

    result = default_registry().execute(
        "write_file", {"path": "notes.txt", "content": "hello workspace"}, store
    )

    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "hello workspace"
    assert result.artifacts == ["notes.txt"]


def test_write_file_truncates_an_existing_file(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    (tmp_path / "notes.txt").write_text("a much longer previous body", encoding="utf-8")

    WriteTool().run({"path": "notes.txt", "content": "hi"}, store)

    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "hi"


def test_write_file_creates_missing_parent_directories(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)

    result = WriteTool().run({"path": "sub/dir/notes.txt", "content": "nested"}, store)

    assert (tmp_path / "sub" / "dir" / "notes.txt").read_text(encoding="utf-8") == "nested"
    assert result.artifacts == ["sub/dir/notes.txt"]


def test_write_file_writes_an_empty_file(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)

    WriteTool().run({"path": "empty.txt", "content": ""}, store)

    assert (tmp_path / "empty.txt").is_file()
    assert (tmp_path / "empty.txt").read_text(encoding="utf-8") == ""


def test_write_file_round_trips_through_read_file(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    content = "café ☕ multibyte"

    WriteTool().run({"path": "unicode.txt", "content": content}, store)

    assert ReadFileTool().run({"path": "unicode.txt"}, store).text == content


def test_write_file_normalizes_the_reported_artifact_path(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)

    result = WriteTool().run({"path": "sub/../notes.txt", "content": "normalized"}, store)

    assert result.artifacts == ["notes.txt"]
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "normalized"


def test_write_file_rejects_escaping_path(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "workspace")

    with pytest.raises(ValueError, match="escapes"):
        WriteTool().run({"path": "../outside.txt", "content": "nope"}, store)

    assert not (tmp_path / "outside.txt").exists()


def test_write_file_rejects_absolute_path(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)

    with pytest.raises(ValueError, match="relative"):
        WriteTool().run({"path": str(tmp_path / "notes.txt"), "content": "nope"}, store)


def test_write_file_accepts_content_at_the_byte_limit(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    content = "x" * WriteTool.max_bytes

    WriteTool().run({"path": "limit.txt", "content": content}, store)

    assert (tmp_path / "limit.txt").stat().st_size == WriteTool.max_bytes


def test_write_file_rejects_oversized_content_without_creating_the_file(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    content = "x" * (WriteTool.max_bytes + 1)

    with pytest.raises(ValueError, match="write limit"):
        WriteTool().run({"path": "big.txt", "content": content}, store)

    assert not (tmp_path / "big.txt").exists()


def test_write_file_limit_counts_bytes_not_characters(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    # Each character encodes to two bytes, so this is under the limit by character
    # count and over it by byte count.
    content = "é" * (WriteTool.max_bytes // 2 + 1)
    assert len(content) < WriteTool.max_bytes

    with pytest.raises(ValueError, match="write limit"):
        WriteTool().run({"path": "multibyte.txt", "content": content}, store)


def test_write_file_rejects_missing_content_argument(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)

    with pytest.raises(ValidationError):
        WriteTool().run({"path": "notes.txt"}, store)
