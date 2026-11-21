from __future__ import annotations

from atlas.harness_eval.swe_score import strip_test_created_files


def _new(path: str) -> str:
    return (
        f"diff --git a/{path} b/{path}\nnew file mode 100644\nindex 0000000..1111111\n"
        f"--- /dev/null\n+++ b/{path}\n@@ -0,0 +1 @@\n+x\n"
    )


def _mod(path: str) -> str:
    return (
        f"diff --git a/{path} b/{path}\nindex 1111111..2222222 100644\n"
        f"--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n-a\n+b\n"
    )


def test_creation_collision_stripped() -> None:
    patch = _mod("src/a.py") + _new("tests/t.py")
    out, stripped = strip_test_created_files(patch, _new("tests/t.py"))
    assert out == _mod("src/a.py")
    assert stripped == ["tests/t.py"]


def test_modification_of_existing_test_not_stripped() -> None:
    patch = _mod("tests/t.py")
    out, stripped = strip_test_created_files(patch, _mod("tests/t.py") + _new("tests/n.py"))
    assert out == patch
    assert stripped == []


def test_multiple_sections_order_preserved() -> None:
    patch = _mod("a.py") + _new("t1.py") + _mod("b.py") + _new("t2.py") + _mod("c.py")
    out, stripped = strip_test_created_files(patch, _new("t1.py") + _new("t2.py"))
    assert out == _mod("a.py") + _mod("b.py") + _mod("c.py")
    assert stripped == ["t1.py", "t2.py"]


def test_no_headers_unchanged() -> None:
    patch = "--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n"
    assert strip_test_created_files(patch, _new("x")) == (patch, [])


def test_all_stripped_is_empty() -> None:
    out, stripped = strip_test_created_files(_new("t.py"), _new("t.py"))
    assert out == ""
    assert stripped == ["t.py"]
