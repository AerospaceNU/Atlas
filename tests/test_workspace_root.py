"""The workspace root is the directory Atlas was launched in.

Access goes down (subdirectories) and never up: a parent path, or an absolute
path outside the root, is refused by both the file tools and the shell guard.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from atlas.agent.artifacts import LocalArtifactStore, SandboxDenied
from atlas.agent.shell_guard import guard_shell_command


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


@pytest.mark.parametrize("path", ["../parent-secret.txt", "/etc/passwd"])
def test_a_parent_or_absolute_path_is_refused(tmp_path: Path, path: str) -> None:
    store = LocalArtifactStore(_project(tmp_path))

    with pytest.raises(SandboxDenied):
        store.resolve(path)
    with pytest.raises(SandboxDenied):
        store.resolve_read(path)


def test_the_shell_cannot_read_a_parent(tmp_path: Path) -> None:
    project = _project(tmp_path)

    with pytest.raises(SandboxDenied):
        guard_shell_command(
            "cat ../parent-secret.txt",
            cwd=project,
            workspace=project,
            proposals=project / ".atlas" / "proposals",
            home=None,
        )


def test_the_shell_can_read_a_child(tmp_path: Path) -> None:
    project = _project(tmp_path)

    guard_shell_command(
        "cat pkg/module.py",
        cwd=project,
        workspace=project,
        proposals=project / ".atlas" / "proposals",
        home=None,
    )
