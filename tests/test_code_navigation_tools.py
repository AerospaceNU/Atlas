from __future__ import annotations

from pathlib import Path

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.tools.find_references import FindReferencesTool
from atlas.agent.tools.outline import OutlineTool

_SOURCE = """\
import os

TIMEOUT = 30


class Runner:
    def run(self, count: int) -> str:
        return "x" * count

    async def stop(self) -> None:
        pass


def helper(value):
    return value


if True:
    def never_seen():
        pass
"""


def test_outline_lists_python_symbols_with_ranges(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    (tmp_path / "mod.py").write_text(_SOURCE, encoding="utf-8")

    result = OutlineTool().run({"path": "mod.py"}, store)

    assert "== mod.py" in result.text
    assert "class Runner" in result.text
    assert "    def run(self, count: int) -> str" in result.text
    assert "    async def stop(self) -> None" in result.text
    assert "def helper(value)" in result.text
    assert "value TIMEOUT = 30" in result.text
    assert "L3  value TIMEOUT = 30" in result.text


def test_outline_ignores_comments_and_strings(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    source = (
        'TEXT = "just text"\n'
        '"""def fake_function(): pass"""\n'
        "# class NotAClass:\n"
        "def real():\n"
        "    pass\n"
    )
    (tmp_path / "mod.py").write_text(source, encoding="utf-8")

    result = OutlineTool().run({"path": "mod.py"}, store)

    assert "def real()" in result.text
    assert "fake_function" not in result.text
    assert "NotAClass" not in result.text


def test_outline_falls_back_to_a_line_scan(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    (tmp_path / "lib.rs").write_text(
        "pub struct Thing;\n\nimpl Thing {\n    pub fn new() -> Self { Thing }\n}\n",
        encoding="utf-8",
    )

    result = OutlineTool().run({"path": "lib.rs"}, store)

    assert "struct Thing" in result.text
    assert "impl Thing" in result.text
    assert "fn new" in result.text


def test_outline_scans_a_directory_with_a_glob(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "one.py").write_text("def one():\n    pass\n", encoding="utf-8")
    (package / "two.txt").write_text("def two():\n", encoding="utf-8")

    result = OutlineTool().run({"path": "pkg", "glob": "*.py"}, store)

    assert "one.py" in result.text
    assert "two.txt" not in result.text


def test_outline_reports_a_syntax_error_and_still_scans(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    (tmp_path / "broken.py").write_text("def ok():\n    pass\n\ndef broken(:\n", encoding="utf-8")

    result = OutlineTool().run({"path": "broken.py"}, store)

    assert "parse error" in result.text
    assert "def broken" in result.text


def test_find_references_separates_definitions_from_uses(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    source = (
        "def target(value):\n"
        "    return value\n\n\n"
        "def caller():\n"
        '    note = "target is only a string here"\n'
        "    # target in a comment\n"
        "    return target(3)\n"
    )
    (tmp_path / "mod.py").write_text(source, encoding="utf-8")

    result = FindReferencesTool().run({"name": "target", "path": "."}, store)

    assert "mod.py:1:1: def: def target(value):" in result.text
    assert "mod.py:8:12: reference: return target(3)" in result.text
    assert "string here" not in result.text
    assert "comment" not in result.text


def test_find_references_can_drop_bindings(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    (tmp_path / "mod.py").write_text("value = 1\nprint(value)\n", encoding="utf-8")

    result = FindReferencesTool().run({"name": "value", "include_definition": False}, store)

    assert "mod.py:1:" not in result.text
    assert "mod.py:2:7: reference: print(value)" in result.text


def test_find_references_reports_attribute_access(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    (tmp_path / "mod.py").write_text("import os\nos.sep\n", encoding="utf-8")

    result = FindReferencesTool().run({"name": "sep"}, store)

    assert "mod.py:2:1: attribute: os.sep" in result.text


def test_find_references_reports_no_matches(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    (tmp_path / "mod.py").write_text("value = 1\n", encoding="utf-8")

    result = FindReferencesTool().run({"name": "absent"}, store)

    assert result.text == "no references to absent"
