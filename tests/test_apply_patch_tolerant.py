from __future__ import annotations

from pathlib import Path

import pytest

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.tools.apply_patch import ApplyPatchTool

_NOTE = "note: read leniently"


def _apply(tmp_path: Path, patch: str, **extra: object) -> str:
    return ApplyPatchTool().run({"patch": patch, **extra}, LocalArtifactStore(tmp_path)).text


def _file(tmp_path: Path, name: str = "a.txt", text: str = "one\ntwo\nthree\nfour\n") -> Path:
    target = tmp_path / name
    target.write_text(text, encoding="utf-8")
    return target


_BODY = " one\n-two\n+TWO\n three\n"


@pytest.mark.parametrize(
    "header",
    ["@@ -1,9 +1,9 @@", "@@ -1,1 +1,1 @@", "@@", "@@ def thing():", "@@ -2 +2 @@"],
    ids=["too-large", "too-small", "bare", "text", "no-counts-wrong"],
)
def test_wrong_or_missing_header_counts_are_recomputed(tmp_path: Path, header: str) -> None:
    target = _file(tmp_path)

    text = _apply(tmp_path, f"--- a/a.txt\n+++ b/a.txt\n{header}\n{_BODY}")

    assert target.read_text(encoding="utf-8") == "one\nTWO\nthree\nfour\n"
    assert text.splitlines()[0] == "applied 1 hunks to 1 files"
    assert _NOTE in text


def test_end_patch_trailer_and_fence_are_ignored(tmp_path: Path) -> None:
    target = _file(tmp_path)
    patch = (
        "```diff\n--- a/a.txt\n+++ b/a.txt\n@@ -1,9 +1,9 @@\n" + _BODY + "*** End Patch\n```\n\n"
    )

    _apply(tmp_path, patch)

    assert target.read_text(encoding="utf-8") == "one\nTWO\nthree\nfour\n"


def test_begin_patch_envelope_with_update_and_add(tmp_path: Path) -> None:
    target = _file(tmp_path)
    patch = (
        "*** Begin Patch\n*** Update File: a.txt\n@@\n"
        + _BODY
        + "*** Add File: sub/new.txt\n+hello\n+world\n*** End Patch\n"
    )

    text = _apply(tmp_path, patch)

    assert target.read_text(encoding="utf-8") == "one\nTWO\nthree\nfour\n"
    assert (tmp_path / "sub" / "new.txt").read_text(encoding="utf-8") == "hello\nworld\n"
    assert "(created)" in text


def test_git_header_lines_are_ignored(tmp_path: Path) -> None:
    target = _file(tmp_path)
    patch = "diff --git a/a.txt b/a.txt\nindex 1..2 100644\n--- a/a.txt\n+++ b/a.txt\n@@\n" + _BODY

    _apply(tmp_path, patch)

    assert target.read_text(encoding="utf-8") == "one\nTWO\nthree\nfour\n"


def test_blank_line_is_an_empty_context_line(tmp_path: Path) -> None:
    target = _file(tmp_path, text="a\n\nb\nc\n")

    _apply(tmp_path, "--- a/a.txt\n+++ b/a.txt\n@@ -1,4 +1,4 @@\n a\n\n-b\n+B\n c\n")

    assert target.read_text(encoding="utf-8") == "a\n\nB\nc\n"


def test_delete_file_is_refused(tmp_path: Path) -> None:
    target = _file(tmp_path)

    with pytest.raises(ValueError, match="deleting a file is not supported"):
        _apply(tmp_path, "*** Delete File: a.txt\n")

    assert target.exists()


def test_trailing_whitespace_context_keeps_the_file_text(tmp_path: Path) -> None:
    target = _file(tmp_path, text="one  \ntwo\nthree\t\n")

    text = _apply(tmp_path, "--- a/a.txt\n+++ b/a.txt\n@@ -1,3 +1,3 @@\n one\n-two\n+TWO\n three\n")

    assert target.read_text(encoding="utf-8") == "one  \nTWO\nthree\t\n"
    assert "note: matched ignoring trailing whitespace" in text


def test_mismatch_names_expected_and_found(tmp_path: Path) -> None:
    target = _file(tmp_path)
    before = target.read_bytes()

    with pytest.raises(ValueError, match=r"context does not match a\.txt") as info:
        _apply(tmp_path, "--- a/a.txt\n+++ b/a.txt\n@@ -1,3 +1,3 @@\n one\n-twelve\n+TWO\n three\n")

    assert "near line 2: expected 'twelve', found 'two'" in str(info.value)
    assert target.read_bytes() == before


def test_a_strictly_valid_patch_has_no_lenient_note(tmp_path: Path) -> None:
    _file(tmp_path)

    text = _apply(tmp_path, "--- a/a.txt\n+++ b/a.txt\n@@ -1,3 +1,3 @@\n" + _BODY)

    assert _NOTE not in text
    assert text.splitlines()[0] == "applied 1 hunks to 1 files"
    assert text.splitlines()[1].startswith("  a.txt: 1 hunks")


def test_a_failed_lenient_patch_leaves_every_file_byte_identical(tmp_path: Path) -> None:
    a = _file(tmp_path, "a.txt")
    b = _file(tmp_path, "b.txt", "alpha\nbeta\n")
    before = (a.read_bytes(), b.read_bytes())
    patch = (
        "--- a/a.txt\n+++ b/a.txt\n@@\n" + _BODY + "--- a/b.txt\n+++ b/b.txt\n@@\n-nothing\n+x\n"
    )

    with pytest.raises(ValueError, match=r"does not match b\.txt"):
        _apply(tmp_path, patch)

    assert (a.read_bytes(), b.read_bytes()) == before


def test_bad_body_line_names_the_line(tmp_path: Path) -> None:
    _file(tmp_path)

    with pytest.raises(ValueError, match=r"line 6: unexpected hunk line prefix 'x'"):
        _apply(tmp_path, "--- a/a.txt\n+++ b/a.txt\n@@\n one\n-two\nxoops\n")


def test_dry_run_through_the_lenient_path_writes_nothing(tmp_path: Path) -> None:
    target = _file(tmp_path)
    before = target.read_bytes()

    text = _apply(tmp_path, "--- a/a.txt\n+++ b/a.txt\n@@\n" + _BODY, dry_run=True)

    assert text.startswith("dry run: would apply 1 hunks to 1 files")
    assert _NOTE in text
    assert target.read_bytes() == before


def test_git_signature_trailer_still_applies(tmp_path: Path) -> None:
    target = _file(tmp_path, "g.txt", "a\nb\n")
    patch = "--- a/g.txt\n+++ b/g.txt\n@@ -1,2 +1,2 @@\n a\n-b\n+c\n-- \n2.39.0\n"
    text = _apply(tmp_path, patch)
    assert "applied 1 hunks" in text
    assert _NOTE not in text
    assert target.read_text(encoding="utf-8") == "a\nc\n"


def test_bare_hunk_matching_two_places_is_refused(tmp_path: Path) -> None:
    target = _file(tmp_path, text="a\nx\nb\nx\n")

    with pytest.raises(
        ValueError, match=r"hunk 1: its lines match 2 places in a.txt \(lines 2, 4\)"
    ):
        _apply(tmp_path, "--- a/a.txt\n+++ b/a.txt\n@@\n-x\n+y\n")

    assert target.read_text(encoding="utf-8") == "a\nx\nb\nx\n"


def test_bare_hunk_with_context_is_unique(tmp_path: Path) -> None:
    target = _file(tmp_path, text="a\nx\nb\nx\n")

    _apply(tmp_path, "--- a/a.txt\n+++ b/a.txt\n@@\n b\n-x\n+y\n")

    assert target.read_text(encoding="utf-8") == "a\nx\nb\ny\n"


def test_bare_pure_insertion_is_refused(tmp_path: Path) -> None:
    _file(tmp_path)

    with pytest.raises(ValueError, match="hunk 1: an insertion needs context lines or a line"):
        _apply(tmp_path, "--- a/a.txt\n+++ b/a.txt\n@@\n+new\n")


def test_two_sections_for_one_file_compose(tmp_path: Path) -> None:
    target = _file(tmp_path)
    patch = (
        "--- a/a.txt\n+++ b/a.txt\n@@ -1,1 +1,1 @@\n-one\n+ONE\n"
        "--- a/a.txt\n+++ b/a.txt\n@@ -4,1 +4,1 @@\n-four\n+FOUR\n"
    )

    text = _apply(tmp_path, patch)

    assert target.read_text(encoding="utf-8") == "ONE\ntwo\nthree\nFOUR\n"
    assert text.splitlines()[0] == "applied 2 hunks to 1 files"
    assert text.count("a.txt:") == 1
    assert "a.txt: 2 hunks" in text


def test_separator_characters_survive_apply_patch(tmp_path: Path) -> None:
    body = f"one\n{chr(12)}\nsplit{chr(0x2028)}here\ntail{chr(0x85)}\nend\n"
    target = _file(tmp_path, text=body)

    _apply(tmp_path, "--- a/a.txt\n+++ b/a.txt\n@@ -5,1 +5,1 @@\n-end\n+END\n")

    assert target.read_text(encoding="utf-8") == body.replace("end", "END")
