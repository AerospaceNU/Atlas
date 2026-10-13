"""The shell stays in the workspace and cannot read user keys or run drafts."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from atlas.agent.artifacts import LocalArtifactStore, SandboxDenied
from atlas.agent.layout import write_user_key
from atlas.agent.tools.BashTool import BashTool

_SECRET = "sk-or-shell-guard-secret-9f3c2a"


def _deny(command: str, store: LocalArtifactStore, **extra: object) -> str:
    arguments: dict[str, object] = {"command": command, **extra}
    with pytest.raises(SandboxDenied) as excinfo:
        BashTool().run(arguments, store)
    return str(excinfo.value)


def test_shell_cwd_is_the_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "notes.txt").write_text("at workspace", encoding="utf-8")
    store = LocalArtifactStore(tmp_path / "artifacts", read_root=workspace)

    result = BashTool().run({"command": "pwd"}, store)

    assert "exit_code: 0" in result.text
    assert str(workspace.resolve()) in result.text
    assert "at workspace" in BashTool().run({"command": "cat notes.txt"}, store).text


def test_shell_rejects_parent_and_absolute_escapes(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("outside-secret", encoding="utf-8")
    store = LocalArtifactStore(workspace)

    parent = _deny("cat ../secret.txt", store)
    absolute = _deny(f"cat {outside}", store)
    climbed = _deny("cd .. && cat secret.txt", store)

    assert "escapes" in parent
    assert "escapes" in absolute
    assert "escapes" in climbed
    assert "outside-secret" not in parent
    assert "outside-secret" not in absolute
    assert "outside-secret" not in climbed
    assert outside.read_text(encoding="utf-8") == "outside-secret"
    missing = tmp_path / "not-created"
    with pytest.raises(SandboxDenied):
        BashTool().run({"command": f"touch {missing}"}, store)
    assert not missing.exists()


def test_shell_allows_a_parent_segment_that_stays_inside_the_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "sub").mkdir(parents=True)
    (workspace / "notes.txt").write_text("inside", encoding="utf-8")
    store = LocalArtifactStore(workspace)

    result = BashTool().run(
        {"command": "cat ../notes.txt", "working_directory": "sub"},
        store,
    )

    assert "exit_code: 0" in result.text
    assert "inside" in result.text


def test_shell_rejects_a_symlink_that_leaves_the_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("symlink-secret", encoding="utf-8")
    (workspace / "link").symlink_to(outside)
    nested = tmp_path / "nested.txt"
    nested.write_text("nested-secret", encoding="utf-8")
    (workspace / "dirlink").symlink_to(tmp_path, target_is_directory=True)
    store = LocalArtifactStore(workspace)

    message = _deny("cat link", store)
    nested_message = _deny("cat dirlink/nested.txt", store)

    assert "escapes" in message
    assert "escapes" in nested_message
    assert "symlink-secret" not in message
    assert "nested-secret" not in nested_message


def test_shell_denies_the_user_key_file_without_returning_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("ATLAS_HOME", raising=False)
    key_path = write_user_key(_SECRET, home)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "keylink").symlink_to(key_path)
    store = LocalArtifactStore(workspace)

    commands = (
        f"cat {key_path}",
        "cat ~/.atlas/keys/openrouter_api_key",
        "cat keylink",
        "cat $HOME/.atlas/keys/openrouter_api_key",
    )
    for command in commands:
        message = _deny(command, store)
        assert _SECRET not in message
        assert key_path.read_text(encoding="utf-8").strip() == _SECRET

    # The key directory stays denied when the workspace is the home directory,
    # so a relative path that resolves there is not treated as an ordinary file.
    home_store = LocalArtifactStore(home)
    message = _deny("cat .atlas/keys/openrouter_api_key", home_store)
    assert _SECRET not in message
    assert key_path.read_text(encoding="utf-8").strip() == _SECRET


def test_shell_rejects_a_hardlink_that_may_leave_the_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("hardlink-secret", encoding="utf-8")
    os.link(outside, workspace / "notes.txt")
    store = LocalArtifactStore(workspace)

    message = _deny("cat notes.txt", store)

    assert "hardlinked" in message
    assert "hardlink-secret" not in message
    assert outside.read_text(encoding="utf-8") == "hardlink-secret"


def test_shell_does_not_execute_a_drafted_tool(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    artifact = workspace / ".atlas" / "artifacts" / "session"
    proposals = artifact / "proposals"
    proposals.mkdir(parents=True)
    marker = workspace / "draft-ran.txt"
    script = proposals / "helper.py"
    script.write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('RAN_DRAFT_TOOL\\n', encoding='utf-8')\n",
        encoding="utf-8",
    )
    shell_script = proposals / "helper.sh"
    shell_script.write_text(
        f"#!/bin/sh\necho RAN_DRAFT_TOOL > {marker}\n",
        encoding="utf-8",
    )
    shell_script.chmod(0o755)
    store = LocalArtifactStore(artifact, read_root=workspace)
    relative = script.relative_to(workspace)
    shell_relative = shell_script.relative_to(workspace)

    for command in (
        f"python3 {relative}",
        f"python3 {script}",
        f"bash {shell_relative}",
        f"{shell_script}",
    ):
        message = _deny(command, store)
        assert "executable" in message
        assert "RAN_DRAFT_TOOL" not in message
        assert not marker.exists()

    artifact_store = LocalArtifactStore(artifact)
    for command in (
        "python3 proposals/helper.py",
        "./proposals/helper.sh",
        "sh proposals/helper.sh",
        "python3 -c \"open('proposals/helper.py').read()\"",
    ):
        message = _deny(command, artifact_store)
        assert "executable" in message
        assert not marker.exists()

    message = _deny(
        "echo hi",
        store,
        working_directory=os.fspath(proposals.relative_to(workspace)),
    )
    assert "executable" in message
    assert not marker.exists()
