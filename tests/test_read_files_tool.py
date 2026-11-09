from __future__ import annotations

from pathlib import Path

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import default_registry
from atlas.agent.tools.read_files import ReadFilesTool


def test_reads_several_files_with_line_numbers(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    (tmp_path / "a.txt").write_text("alpha\nbeta\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("gamma\n", encoding="utf-8")

    result = ReadFilesTool().run({"files": [{"path": "a.txt"}, {"path": "b.txt"}]}, store)

    assert "== a.txt (lines 1-2 of 2)" in result.text
    assert "     1| alpha" in result.text
    assert "     2| beta" in result.text
    assert "== b.txt (lines 1-1 of 1)" in result.text
    assert "     1| gamma" in result.text


def test_offset_and_limit_narrow_one_file(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    (tmp_path / "lines.txt").write_text(
        "\n".join(f"line {i}" for i in range(1, 11)), encoding="utf-8"
    )

    result = ReadFilesTool().run({"files": [{"path": "lines.txt", "offset": 4, "limit": 2}]}, store)

    assert "== lines.txt (lines 4-5 of 10)" in result.text
    assert "     4| line 4" in result.text
    assert "     5| line 5" in result.text
    assert "line 3" not in result.text
    assert "... [5 more lines]" in result.text


def test_missing_and_binary_files_do_not_fail_the_batch(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    (tmp_path / "blob.bin").write_bytes(b"\x00\x01")
    (tmp_path / "ok.txt").write_text("fine\n", encoding="utf-8")

    result = ReadFilesTool().run(
        {"files": [{"path": "missing.txt"}, {"path": "blob.bin"}, {"path": "ok.txt"}]},
        store,
    )

    assert "[missing: file does not exist]" in result.text
    assert "[skipped: binary file]" in result.text
    assert "== ok.txt" in result.text


def test_offset_past_end_is_reported(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    (tmp_path / "short.txt").write_text("one\n", encoding="utf-8")

    result = ReadFilesTool().run({"files": [{"path": "short.txt", "offset": 9}]}, store)

    assert "[offset 9 is past the end (1 lines)]" in result.text


def test_empty_file_is_labelled(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    (tmp_path / "empty.txt").write_text("", encoding="utf-8")

    result = ReadFilesTool().run({"files": [{"path": "empty.txt"}]}, store)

    assert "(empty file)" in result.text


def test_read_files_is_registered_by_default() -> None:
    names = {definition.name for definition in default_registry().definitions}

    assert {
        "read_files",
        "apply_patch",
        "outline",
        "find_references",
        "run_tests",
        "diagnostics",
    } <= names
