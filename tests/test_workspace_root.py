"""The workspace root anchors relative paths, but it does not bound access."""

from __future__ import annotations

from pathlib import Path

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import default_registry
from atlas.agent.tools.BashTool import BashTool


def _project(tmp_path: Path) -> Path:
    project = tmp_path / "myproject"
    (project / "pkg").mkdir(parents=True)
    (project / "pkg" / "module.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "parent-secret.txt").write_text("outside\n", encoding="utf-8")
    return project


def test_store_root_is_the_launch_directory(tmp_path: Path) -> None:
    project = _project(tmp_path)
    store = LocalArtifactStore(project)

    assert store.root == project.resolve()
    assert store.resolve("pkg/new.py") == (project / "pkg" / "new.py").resolve()
    assert store.resolve_read("pkg/module.py").read_text(encoding="utf-8") == "x = 1\n"


def test_a_parent_or_absolute_path_is_allowed(tmp_path: Path) -> None:
    store = LocalArtifactStore(_project(tmp_path))

    assert store.resolve("../parent-secret.txt") == (tmp_path / "parent-secret.txt").resolve()
    assert store.resolve("/etc/passwd") == Path("/etc/passwd").resolve()
    assert store.resolve_read("../parent-secret.txt").read_text(encoding="utf-8") == "outside\n"


def test_the_shell_can_read_a_parent_and_an_absolute_path(tmp_path: Path) -> None:
    project = _project(tmp_path)
    store = LocalArtifactStore(project)

    parent = BashTool().run({"command": "cat ../parent-secret.txt"}, store)
    absolute = BashTool().run({"command": f"cat {project / 'pkg' / 'module.py'}"}, store)

    assert "outside" in parent.text
    assert "x = 1" in absolute.text


def test_the_shell_inherits_the_environment(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ATLAS_SANDBOX_TEST_VALUE", "visible")
    store = LocalArtifactStore(_project(tmp_path))

    result = BashTool().run({"command": 'printf %s "$ATLAS_SANDBOX_TEST_VALUE"'}, store)

    assert "visible" in result.text


def test_file_tools_accept_absolute_paths(tmp_path: Path) -> None:
    store = LocalArtifactStore(_project(tmp_path))
    outside = tmp_path / "written.txt"

    default_registry().execute("write_file", {"path": str(outside), "content": "hello"}, store)

    assert outside.read_text(encoding="utf-8") == "hello"
