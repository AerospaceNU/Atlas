from __future__ import annotations

from pathlib import Path

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import default_registry
from atlas.agent.tools.glob import GlobTool
from atlas.agent.tools.grep import GrepTool
from atlas.agent.tools.ls import LsTool


def test_grep_finds_matches_with_relative_paths_and_line_numbers(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    (tmp_path / "notes.txt").write_text("first line\nneedle here\nlast\n", encoding="utf-8")
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "mod.py").write_text("import os\nneedle = 1\n", encoding="utf-8")

    result = GrepTool().run({"pattern": "needle"}, store)

    assert result.text.splitlines() == [
        "notes.txt:2:needle here",
        "pkg/mod.py:2:needle = 1",
    ]
    assert str(tmp_path) not in result.text


def test_grep_searches_an_absolute_path_and_a_parent_directory(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "workspace")
    (tmp_path / "notes.txt").write_text("needle\n", encoding="utf-8")

    absolute = GrepTool().run({"pattern": "needle", "path": str(tmp_path)}, store)
    parent = GrepTool().run({"pattern": "needle", "path": ".."}, store)

    assert "notes.txt:1:needle" in absolute.text
    assert "notes.txt:1:needle" in parent.text


def test_grep_truncates_at_max_matches(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    (tmp_path / "many.txt").write_text(
        "\n".join(f"hit {index}" for index in range(10)) + "\n", encoding="utf-8"
    )

    result = GrepTool().run({"pattern": "hit", "max_matches": 3}, store)

    lines = result.text.splitlines()
    assert lines[:3] == ["many.txt:1:hit 0", "many.txt:2:hit 1", "many.txt:3:hit 2"]
    assert lines[3] == "... [truncated 7 more matches]"


def test_grep_reports_no_matches(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    (tmp_path / "notes.txt").write_text("nothing here\n", encoding="utf-8")

    result = GrepTool().run({"pattern": "needle"}, store)

    assert result.text == "no matches"


def test_glob_matches_nested_files(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    (tmp_path / "a.py").write_text("a\n", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.py").write_text("b\n", encoding="utf-8")
    deep = sub / "deep"
    deep.mkdir()
    (deep / "c.py").write_text("c\n", encoding="utf-8")
    (deep / "notes.txt").write_text("notes\n", encoding="utf-8")

    result = GlobTool().run({"pattern": "**/*.py"}, store)

    assert result.text.splitlines() == ["a.py", "sub/b.py", "sub/deep/c.py"]
    assert str(tmp_path) not in result.text


def test_glob_matches_a_parent_traversal_pattern(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "workspace")
    (tmp_path / "outside.py").write_text("x\n", encoding="utf-8")

    result = GlobTool().run({"pattern": "../**/*.py"}, store)

    assert "outside.py" in result.text


def test_ls_lists_directories_with_a_trailing_slash(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    (tmp_path / "notes.txt").write_text("notes\n", encoding="utf-8")
    (tmp_path / "sub").mkdir()

    result = LsTool().run({}, store)

    assert result.text.splitlines() == ["notes.txt", "sub/"]
    assert str(tmp_path) not in result.text


def test_ls_lists_nested_entries_recursively(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "notes.txt").write_text("notes\n", encoding="utf-8")

    result = LsTool().run({"path": "sub", "recursive": True}, store)

    assert result.text.splitlines() == ["sub/notes.txt"]


def test_ls_lists_a_parent_directory(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "workspace")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "notes.txt").write_text("notes\n", encoding="utf-8")

    result = LsTool().run({"path": "../outside"}, store)

    assert "notes.txt" in result.text


def test_no_output_contains_a_host_absolute_path(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "notes.txt").write_text("needle\n", encoding="utf-8")

    texts = [
        GrepTool().run({"pattern": "needle"}, store).text,
        GlobTool().run({"pattern": "**/*.txt"}, store).text,
        LsTool().run({"recursive": True}, store).text,
    ]

    for text in texts:
        assert text
        assert str(tmp_path) not in text
        assert str(tmp_path.resolve()) not in text


def test_code_search_tools_are_registered() -> None:
    registry = default_registry()
    names = {definition.name for definition in registry.definitions}

    assert {"grep", "glob", "ls"} <= names
