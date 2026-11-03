from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from atlas.agent.artifacts import LocalArtifactStore, SandboxDenied
from atlas.agent.tools.git import GitCommitTool, GitDiffTool, GitStatusTool

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


def _git(repo: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        capture_output=True,
        text=True,
        check=True,
    )


def _init_repo(tmp_path: Path) -> Path:
    """Create a repository with one committed file and a local identity."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "atlas@example.com")
    _git(repo, "config", "user.name", "Atlas Test")
    _git(repo, "config", "commit.gpgsign", "false")
    (repo / "notes.txt").write_text("original\n", encoding="utf-8")
    _git(repo, "add", "notes.txt")
    _git(repo, "commit", "-m", "initial commit")
    return repo


def test_status_reports_the_branch_and_a_modified_file(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    (repo / "notes.txt").write_text("changed\n", encoding="utf-8")
    store = LocalArtifactStore(repo)
    branch = _git(repo, "branch", "--show-current").stdout.strip()

    result = GitStatusTool().run({}, store)

    lines = result.text.splitlines()
    assert lines[0] == f"## {branch}"
    assert " M notes.txt" in lines
    assert str(repo) not in result.text


def test_diff_shows_an_added_line(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    (repo / "notes.txt").write_text("original\nadded\n", encoding="utf-8")
    store = LocalArtifactStore(repo)

    result = GitDiffTool().run({"path": "notes.txt"}, store)

    assert "+added" in result.text
    assert "notes.txt" in result.text
    assert str(repo) not in result.text


def test_diff_reads_the_index_when_staged(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    (repo / "notes.txt").write_text("original\nstaged line\n", encoding="utf-8")
    _git(repo, "add", "notes.txt")
    store = LocalArtifactStore(repo)

    assert "+staged line" in GitDiffTool().run({"staged": True}, store).text
    assert GitDiffTool().run({}, store).text == "no changes"


def test_diff_truncates_with_a_visible_marker(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    (repo / "notes.txt").write_text("original\n" + "x" * 500 + "\n", encoding="utf-8")
    store = LocalArtifactStore(repo)

    result = GitDiffTool().run({"path": "notes.txt", "max_bytes": 40}, store)

    assert "... [truncated " in result.text
    assert "bytes]" in result.text
    assert len(result.text) < 200


def test_commit_writes_a_commit_and_the_log_shows_the_message(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    (repo / "notes.txt").write_text("committed change\n", encoding="utf-8")
    store = LocalArtifactStore(repo)

    result = GitCommitTool().run({"message": "update notes", "paths": ["notes.txt"]}, store)

    log = _git(repo, "log", "--oneline").stdout
    assert "update notes" in log
    assert result.text.startswith("[")
    assert "update notes" in result.text
    assert str(repo) not in result.text


def test_commit_with_nothing_staged_does_not_create_a_commit(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    store = LocalArtifactStore(repo)
    before = _git(repo, "rev-list", "--count", "HEAD").stdout.strip()

    result = GitCommitTool().run({"message": "empty commit"}, store)

    after = _git(repo, "rev-list", "--count", "HEAD").stdout.strip()
    assert result.text == "nothing to commit"
    assert after == before


def test_absolute_paths_are_refused_with_sandbox_denied(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    (repo / "notes.txt").write_text("changed\n", encoding="utf-8")
    store = LocalArtifactStore(repo)
    absolute = str(repo / "notes.txt")

    with pytest.raises(SandboxDenied):
        GitDiffTool().run({"path": absolute}, store)

    with pytest.raises(SandboxDenied):
        GitCommitTool().run({"message": "escape", "paths": [absolute]}, store)

    assert " M notes.txt" in GitStatusTool().run({}, store).text


def test_status_and_diff_from_a_subdirectory_are_workspace_relative(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    package = repo / "pkg"
    package.mkdir()
    (package / "inside.txt").write_text("original\n", encoding="utf-8")
    _git(repo, "add", "pkg/inside.txt")
    _git(repo, "commit", "-m", "add package file")
    (package / "inside.txt").write_text("changed\n", encoding="utf-8")
    (repo / "notes.txt").write_text("outside change\n", encoding="utf-8")
    store = LocalArtifactStore(package)

    status = GitStatusTool().run({}, store).text
    diff = GitDiffTool().run({}, store).text

    assert " M inside.txt" in status.splitlines()
    assert "notes.txt" not in status
    assert "a/inside.txt" in diff
    assert "notes.txt" not in diff
    assert str(repo) not in status + diff


def test_status_outside_a_repository_reports_not_a_git_work_tree(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "plain")

    result = GitStatusTool().run({}, store)

    assert result.text == "not a git work tree"
