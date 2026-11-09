from __future__ import annotations

from pathlib import Path

import pytest

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.tools.apply_patch import ApplyPatchTool

_MULTI = """\
--- a/a.txt
+++ b/a.txt
@@ -1,3 +1,3 @@
 one
-two
+TWO
 three
--- a/b.txt
+++ b/b.txt
@@ -1,2 +1,3 @@
 alpha
+inserted
 beta
"""


def test_applies_a_patch_across_several_files(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    (tmp_path / "a.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("alpha\nbeta\n", encoding="utf-8")

    result = ApplyPatchTool().run({"patch": _MULTI}, store)

    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "one\nTWO\nthree\n"
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "alpha\ninserted\nbeta\n"
    assert result.text.splitlines()[0] == "applied 2 hunks to 2 files"
    assert set(result.artifacts) == {"a.txt", "b.txt"}


def test_a_mismatched_hunk_leaves_every_file_untouched(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    (tmp_path / "a.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("alpha\nbeta\n", encoding="utf-8")
    bad = _MULTI.replace(" alpha\n+inserted", " wrong\n+inserted")

    with pytest.raises(ValueError, match=r"hunk .* does not match b\.txt"):
        ApplyPatchTool().run({"patch": bad}, store)

    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "one\ntwo\nthree\n"
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "alpha\nbeta\n"


def test_dry_run_validates_without_writing(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    (tmp_path / "a.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("alpha\nbeta\n", encoding="utf-8")

    result = ApplyPatchTool().run({"patch": _MULTI, "dry_run": True}, store)

    assert result.text.startswith("dry run: would apply 2 hunks to 2 files")
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "one\ntwo\nthree\n"


def test_creates_a_new_file_from_dev_null(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    patch = "--- /dev/null\n+++ b/new.txt\n@@ -0,0 +1,2 @@\n+hello\n+world\n"

    result = ApplyPatchTool().run({"patch": patch}, store)

    assert (tmp_path / "new.txt").read_text(encoding="utf-8") == "hello\nworld\n"
    assert "(created)" in result.text


def test_refuses_to_overwrite_an_existing_file_as_new(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    (tmp_path / "new.txt").write_text("here\n", encoding="utf-8")
    patch = "--- /dev/null\n+++ b/new.txt\n@@ -0,0 +1 @@\n+hello\n"

    with pytest.raises(ValueError, match="already exists"):
        ApplyPatchTool().run({"patch": patch}, store)


def test_preserves_a_missing_final_newline(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    (tmp_path / "nonl.txt").write_text("old", encoding="utf-8")
    patch = (
        "--- a/nonl.txt\n+++ b/nonl.txt\n@@ -1 +1 @@\n"
        "-old\n\\ No newline at end of file\n"
        "+new\n\\ No newline at end of file\n"
    )

    ApplyPatchTool().run({"patch": patch}, store)

    assert (tmp_path / "nonl.txt").read_text(encoding="utf-8") == "new"


def test_finds_a_hunk_that_drifted_a_few_lines(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    (tmp_path / "drift.txt").write_text("new\nfiller\none\ntwo\nthree\n", encoding="utf-8")
    patch = "--- a/drift.txt\n+++ b/drift.txt\n@@ -1,3 +1,3 @@\n one\n-two\n+TWO\n three\n"

    ApplyPatchTool().run({"patch": patch}, store)

    assert (tmp_path / "drift.txt").read_text(encoding="utf-8") == "new\nfiller\none\nTWO\nthree\n"


def test_rejects_deletion_and_nul_bytes(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    (tmp_path / "gone.txt").write_text("bye\n", encoding="utf-8")
    deletion = "--- a/gone.txt\n+++ /dev/null\n@@ -1 +0,0 @@\n-bye\n"

    with pytest.raises(ValueError, match="deleting a file is not supported"):
        ApplyPatchTool().run({"patch": deletion}, store)
    with pytest.raises(ValueError, match="NUL"):
        ApplyPatchTool().run({"patch": "--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\x00b\n+c\n"}, store)


def test_rejects_an_input_without_file_headers(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)

    with pytest.raises(ValueError, match="no file sections"):
        ApplyPatchTool().run({"patch": "just some text\n"}, store)
