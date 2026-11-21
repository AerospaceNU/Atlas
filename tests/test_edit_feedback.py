from __future__ import annotations

from pathlib import Path

import pytest

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.tools._edits import (
    changed_region,
    not_found_hint,
    occurrence_lines,
    python_syntax_warning,
    read_text_keeping_newlines,
)
from atlas.agent.tools.apply_patch import ApplyPatchTool
from atlas.agent.tools.edit_file import EditFileTool
from atlas.agent.tools.patch_file import PatchFileTool
from atlas.agent.tools.write_file import WriteTool


def _edit(store: LocalArtifactStore, path: str, old: str, new: str, **extra: object) -> str:
    args = {"path": path, "old_string": old, "new_string": new, **extra}
    return EditFileTool().run(args, store).text


def test_a_150_kb_file_can_be_edited(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    body = "".join(f"value_{i} = {i}\n" for i in range(10_000))
    assert len(body) > 150_000
    (tmp_path / "big.py").write_text(body, encoding="utf-8")

    text = _edit(store, "big.py", "value_5000 = 5000", "value_5000 = -1")

    assert text.splitlines()[0].startswith("File big.py edited successfully (bytes: ")
    assert "value_5000 = -1" in (tmp_path / "big.py").read_text(encoding="utf-8")


def test_result_shows_the_changed_region(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    (tmp_path / "n.txt").write_text("\n".join(f"l{i}" for i in range(1, 21)) + "\n")

    text = _edit(store, "n.txt", "l10", "TEN")

    assert text.splitlines()[0] == "File n.txt edited successfully (bytes: 71)."
    assert "Lines 7-13 now read:" in text
    assert "    10| TEN" in text
    assert "     7| l7" in text and "    13| l13" in text


def test_replace_all_reports_the_count(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    target = tmp_path / "n.txt"
    target.write_text("a\nb\na\na\n", encoding="utf-8")

    text = _edit(store, "n.txt", "a", "z", replace_all=True)

    assert target.read_text(encoding="utf-8") == "z\nb\nz\nz\n"
    assert "Replaced 3 occurrences" in text


def test_identical_strings_are_refused(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    (tmp_path / "n.txt").write_text("a\n", encoding="utf-8")

    with pytest.raises(ValueError, match="old_string and new_string are identical"):
        _edit(store, "n.txt", "a", "a")


def test_whitespace_only_difference_is_hinted_and_nothing_is_written(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    target = tmp_path / "n.py"
    original = "def f():\n    x = 1\n    return x\n"
    target.write_text(original, encoding="utf-8")

    with pytest.raises(ValueError, match=r"The input cannot be found\.") as info:
        _edit(store, "n.py", "x = 1\nreturn x", "x = 2")

    message = str(info.value)
    assert "differs only in whitespace is at lines 2-3:" in message
    assert "     2|     x = 1" in message
    assert target.read_text(encoding="utf-8") == original


def test_first_line_and_similar_line_hints() -> None:
    text = "alpha\nbeta = 1\ngamma\n"
    first = not_found_hint(text, "beta = 1\nmissing")
    assert first.startswith("The first line of old_string is at line(s) 2;")
    similar = not_found_hint(text, "beta = 2")
    assert (
        similar.startswith("The first line of old_string is at line(s)")
        or "Most similar" in similar
    )
    assert not_found_hint("alpha\n", "qqqqqqqqqq") == ""
    assert "Most similar line 2: gamma_ray" in not_found_hint("a\ngamma_ray\n", "gamma_rai")


def test_multiple_match_error_lists_lines(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    target = tmp_path / "n.txt"
    target.write_text("x\ny\nx\nz\nx\n", encoding="utf-8")

    with pytest.raises(ValueError, match="appears multiple times") as info:
        _edit(store, "n.txt", "x", "w")

    assert "It occurs 3 times, at lines 1, 3, 5." in str(info.value)
    assert occurrence_lines("x\ny\nx\n", "x") == [1, 3]


def test_changed_region_caps_long_regions() -> None:
    text = "\n".join(f"r{i}" for i in range(100)) + "\n"
    out = changed_region(text, 0, len(text) - 1, max_lines=10)
    assert out.splitlines()[-1].startswith("... [") and "more lines]" in out


def test_crlf_survives_all_three_editors(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    for name in ("e.txt", "p.txt", "a.txt"):
        (tmp_path / name).write_bytes(b"one\r\ntwo\r\nthree\r\n")

    _edit(store, "e.txt", "two", "TWO")
    PatchFileTool().run(
        {"path": "p.txt", "edits": [{"old_string": "two\nthree", "new_string": "2\n3"}]}, store
    )
    ApplyPatchTool().run(
        {"patch": "--- a/a.txt\n+++ b/a.txt\n@@ -1,3 +1,3 @@\n one\n-two\n+TWO\n three\n"}, store
    )

    assert (tmp_path / "e.txt").read_bytes() == b"one\r\nTWO\r\nthree\r\n"
    assert (tmp_path / "p.txt").read_bytes() == b"one\r\n2\r\n3\r\n"
    assert (tmp_path / "a.txt").read_bytes() == b"one\r\nTWO\r\nthree\r\n"


def test_newline_style_follows_the_majority(tmp_path: Path) -> None:
    (tmp_path / "m.txt").write_bytes(b"a\r\nb\nc\n")
    assert read_text_keeping_newlines(tmp_path / "m.txt", "m.txt", 100) == ("a\nb\nc\n", "\n")


def test_patch_file_reports_regions_and_hints(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    target = tmp_path / "n.txt"
    target.write_text("\n".join(f"l{i}" for i in range(1, 31)) + "\n", encoding="utf-8")

    result = PatchFileTool().run(
        {
            "path": "n.txt",
            "edits": [
                {"old_string": "l5", "new_string": "FIVE"},
                {"old_string": "l25", "new_string": "TWENTY"},
            ],
        },
        store,
    )

    lines = result.text.splitlines()
    assert lines[0].startswith("Applied 2 edits to n.txt (bytes: ")
    assert "     5| FIVE" in result.text and "    25| TWENTY" in result.text
    before = target.read_bytes()
    with pytest.raises(ValueError, match=r"Edit 0: old_string was not found in the file\n"):
        PatchFileTool().run(
            {"path": "n.txt", "edits": [{"old_string": "  FIVE", "new_string": "x"}]}, store
        )
    assert target.read_bytes() == before


def test_syntax_warning_only_when_the_edit_breaks_the_file(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    (tmp_path / "ok.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    (tmp_path / "bad.py").write_text("def f(:\n    return 1\n", encoding="utf-8")

    broken = _edit(store, "ok.py", "return 1", "return (")
    already = _edit(store, "bad.py", "return 1", "return 2")

    assert "warning: ok.py no longer parses as Python (line " in broken
    assert "warning" not in already
    assert python_syntax_warning("x.txt", "a", "def (") is None
    assert python_syntax_warning("x.py", None, "def (") is not None


def test_write_file_notes_an_overwrite(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    first = WriteTool().run({"path": "w.py", "content": "x = 1\n"}, store).text
    second = WriteTool().run({"path": "w.py", "content": "x = (\n"}, store).text

    assert first == "File w.py created and written successfully (bytes: 6)."
    assert second.splitlines()[0] == (
        "File w.py created and written successfully (bytes: 6). "
        "Replaced an existing file of 6 bytes."
    )
    assert "warning: w.py no longer parses as Python" in second
