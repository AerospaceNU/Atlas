from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError

from atlas.agent.artifacts import LocalArtifactStore, SandboxDenied
from atlas.agent.contracts import default_registry
from atlas.agent.layout import (
    AtlasPathError,
    ensure_project_layout,
    ensure_user_layout,
    list_entries,
    remove_entry,
    write_user_key,
)
from atlas.agent.tools.BashTool import BashTool
from atlas.agent.tools.read_file import ReadFileTool


def test_store_rejects_paths_outside_root(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)

    with pytest.raises(ValueError, match="escapes"):
        store.resolve("../outside.png")

    # A symlink that sits lexically under root but resolves outside it must
    # be rejected the same way as a lexical ".." escape.
    outside = tmp_path.parent / f"{tmp_path.name}-symlink-target.txt"
    outside.write_text("secret", encoding="utf-8")
    (tmp_path / "link.txt").symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        store.resolve("link.txt")

    # A hardlink resolves to itself, so it is refused for having another name.
    os.link(outside, tmp_path / "hardlink.txt")
    with pytest.raises(ValueError, match="hardlinked"):
        store.resolve("hardlink.txt")


def test_read_file_rejects_a_hardlink_to_an_outside_file(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    store = LocalArtifactStore(workspace)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    os.link(outside, workspace / "notes.txt")

    with pytest.raises(ValueError, match="hardlinked"):
        ReadFileTool().run({"path": "notes.txt"}, store)


def test_list_images_skips_a_hardlinked_image(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    store = LocalArtifactStore(workspace)
    Image.new("RGB", (4, 4), "red").save(tmp_path / "secret.png")
    os.link(tmp_path / "secret.png", workspace / "linked.png")
    Image.new("RGB", (4, 4), "blue").save(workspace / "visible.png")

    images = store.list_images()

    assert [image.path for image in images] == ["visible.png"]


def test_list_images_skips_a_symlinked_escape_instead_of_raising(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    Image.new("RGB", (4, 4), "red").save(outside_dir / "secret.png")
    (workspace / "escape_link").symlink_to(outside_dir)
    Image.new("RGB", (4, 4), "blue").save(workspace / "visible.png")
    store = LocalArtifactStore(workspace)

    images = store.list_images()

    assert [image.path for image in images] == ["visible.png"]


def test_remove_rejects_escape_and_keeps_the_user_store(tmp_path: Path) -> None:
    project = tmp_path / "project"
    home = tmp_path / "home"
    data = project / ".atlas" / "data" / "keep.txt"
    data.parent.mkdir(parents=True)
    data.write_text("table", encoding="utf-8")
    weight = home / ".atlas" / "weights" / "model.json"
    weight.parent.mkdir(parents=True)
    weight.write_text("{}", encoding="utf-8")
    ensure_project_layout(project)
    ensure_user_layout(home)
    assert data.read_text(encoding="utf-8") == "table"
    assert weight.read_text(encoding="utf-8") == "{}"
    assert (project / ".atlas" / "sessions").is_dir()
    assert (project / ".atlas" / "artifacts").is_dir()
    assert (project / ".atlas" / "data").is_dir()
    assert (home / ".atlas" / "keys").is_dir()
    assert (home / ".atlas" / "weights").is_dir()
    assert (home / ".atlas" / "defaults").is_dir()

    session = project / ".atlas" / "sessions" / "s1"
    artifact = project / ".atlas" / "artifacts" / "s1"
    session.mkdir()
    artifact.mkdir()
    (artifact / "chip.txt").write_text("chip", encoding="utf-8")
    assert list_entries(project / ".atlas", "sessions") == ["s1"]
    assert list_entries(project / ".atlas", "artifacts") == ["s1"]
    outside = tmp_path / "secret.txt"
    outside.write_text("nope", encoding="utf-8")
    key = write_user_key("user-key", home)
    assert key.stat().st_mode & 0o777 == 0o600

    with pytest.raises(AtlasPathError, match="confirmation required"):
        remove_entry(project / ".atlas", "sessions", "s1", confirm=False)
    assert session.is_dir()

    with pytest.raises(AtlasPathError, match="escapes"):
        remove_entry(project / ".atlas", "sessions", "../secret.txt", confirm=True)
    with pytest.raises(AtlasPathError, match="escapes"):
        remove_entry(project / ".atlas", "artifacts", "/etc/passwd", confirm=True)
    assert outside.is_file()
    assert key.is_file()
    assert weight.is_file()

    remove_entry(project / ".atlas", "sessions", "s1", confirm=True)
    assert not session.exists()
    assert not artifact.exists()
    assert data.is_file()
    assert key.read_text(encoding="utf-8").strip() == "user-key"
    assert weight.is_file()
    assert list_entries(project / ".atlas", "sessions") == []
    assert list_entries(project / ".atlas", "artifacts") == []
    remove_entry(home / ".atlas", "weights", "model.json", confirm=True)
    assert not weight.exists()
    assert key.is_file()
    assert data.is_file()
    assert "user-key" not in "\n".join(
        path.read_text(encoding="utf-8")
        for path in (project / ".atlas").rglob("*")
        if path.is_file()
    )


def test_read_file_returns_relative_text_contents(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    (tmp_path / "notes.txt").write_text("hello workspace", encoding="utf-8")

    result = default_registry().execute("read_file", {"path": "notes.txt"}, store)

    assert result.text == "hello workspace"


def test_read_file_returns_empty_text_for_an_empty_file(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    (tmp_path / "empty.txt").write_text("", encoding="utf-8")

    result = ReadFileTool().run({"path": "empty.txt"}, store)

    assert result.text == ""
    assert result.artifacts == []


def test_read_file_rejects_escaping_path(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    (tmp_path.parent / "outside.txt").write_text("outside", encoding="utf-8")

    with pytest.raises(ValueError, match="escapes"):
        ReadFileTool().run({"path": "../outside.txt"}, store)


def test_read_file_rejects_absolute_path(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    (tmp_path / "notes.txt").write_text("hello", encoding="utf-8")

    with pytest.raises(ValueError, match="relative"):
        ReadFileTool().run({"path": str(tmp_path / "notes.txt")}, store)


def test_read_file_missing_file_hides_the_host_path(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)

    with pytest.raises(FileNotFoundError) as excinfo:
        ReadFileTool().run({"path": "missing.txt"}, store)

    message = str(excinfo.value)
    assert "missing.txt" in message
    assert str(tmp_path) not in message


def test_read_file_rejects_oversized_file(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    (tmp_path / "big.txt").write_bytes(b"x" * 100_001)

    with pytest.raises(ValueError, match="100000"):
        ReadFileTool().run({"path": "big.txt"}, store)


def test_read_file_rejects_non_utf8_file(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    (tmp_path / "binary.txt").write_bytes(b"\xff")

    with pytest.raises(ValueError, match="UTF-8"):
        ReadFileTool().run({"path": "binary.txt"}, store)


def test_bash_reports_stdout_and_a_zero_exit(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)

    result = default_registry().execute("bash_tool", {"command": "echo hello"}, store)

    assert "exit_code: 0" in result.text
    assert "hello" in result.text


def test_bash_runs_in_the_requested_working_directory(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "notes.txt").write_text("hello workspace", encoding="utf-8")

    result = BashTool().run({"command": "cat notes.txt", "working_directory": "sub"}, store)

    assert "exit_code: 0" in result.text
    assert "hello workspace" in result.text


def test_bash_defaults_to_the_artifact_root(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    (tmp_path / "notes.txt").write_text("hello root", encoding="utf-8")

    result = BashTool().run({"command": "cat notes.txt"}, store)

    assert "hello root" in result.text


def test_bash_reports_a_failing_command_instead_of_raising(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)

    result = BashTool().run({"command": "echo boom >&2; exit 3"}, store)

    assert "exit_code: 3" in result.text
    assert "boom" in result.text


def test_bash_reports_a_timeout(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)

    result = BashTool().run({"command": "sleep 5", "timeout": 1}, store)

    assert "timed out after 1s" in result.text


def test_bash_timeout_kills_a_backgrounded_child_too(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    marker = tmp_path / "still_running.marker"

    result = BashTool().run(
        {"command": f"(sleep 3 && touch {marker}) & disown; echo launched", "timeout": 1},
        store,
    )

    assert "timed out after 1s" in result.text
    time.sleep(4)
    assert not marker.exists()


def test_bash_kills_a_detached_background_child_when_the_call_returns(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    marker = tmp_path / "still_running.marker"

    result = BashTool().run(
        {
            "command": f"(sleep 1 && touch {marker}) >/dev/null 2>&1 </dev/null & echo launched",
            "timeout": 10,
        },
        store,
    )

    assert "exit_code: 0" in result.text
    time.sleep(2)
    assert not marker.exists()


def test_bash_scrubs_ambient_environment_variables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = LocalArtifactStore(tmp_path)
    monkeypatch.setenv("SOME_SECRET_TOKEN", "not-actually-a-secret")

    result = BashTool().run({"command": "env"}, store)

    assert "SOME_SECRET_TOKEN" not in result.text
    assert "PATH=" in result.text


def test_bash_truncates_oversized_output(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)

    result = BashTool().run({"command": "yes X | head -c 300000"}, store)

    assert "truncated" in result.text
    assert len(result.text) < 300_000


def test_bash_strips_ansi_escape_sequences(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)

    result = BashTool().run({"command": "printf '\\033[31mred\\033[0m'"}, store)

    assert "\x1b[" not in result.text
    assert "red" in result.text


def test_bash_reports_a_missing_working_directory(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)

    result = BashTool().run({"command": "echo hi", "working_directory": "nope"}, store)

    assert "Working directory not found" in result.text
    assert "nope" in result.text
    assert str(tmp_path) not in result.text


def test_bash_rejects_absolute_or_escaping_working_directory(tmp_path: Path) -> None:
    # Raised rather than returned, so the agent loop records it in the audit log.
    store = LocalArtifactStore(tmp_path)

    with pytest.raises(SandboxDenied, match="relative"):
        BashTool().run({"command": "echo hi", "working_directory": "/etc"}, store)
    with pytest.raises(SandboxDenied, match="escapes"):
        BashTool().run({"command": "echo hi", "working_directory": "../outside"}, store)


def test_bash_defaults_to_the_workspace_root(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "notes.txt").write_text("at workspace", encoding="utf-8")
    store = LocalArtifactStore(tmp_path / "artifact", read_root=workspace)

    result = BashTool().run({"command": "cat notes.txt"}, store)

    assert "at workspace" in result.text


def test_read_file_falls_back_to_the_read_root(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "README.md").write_text("project readme", encoding="utf-8")
    store = LocalArtifactStore(tmp_path / "artifact", read_root=workspace)

    result = ReadFileTool().run({"path": "README.md"}, store)

    assert result.text == "project readme"


def test_read_file_prefers_the_write_root_over_the_read_root(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    workspace = tmp_path / "workspace"
    artifact.mkdir()
    workspace.mkdir()
    (artifact / "notes.txt").write_text("artifact", encoding="utf-8")
    (workspace / "notes.txt").write_text("workspace", encoding="utf-8")
    store = LocalArtifactStore(artifact, read_root=workspace)

    assert ReadFileTool().run({"path": "notes.txt"}, store).text == "artifact"


def test_read_file_still_rejects_escapes_and_absolute_paths_with_a_read_root(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = LocalArtifactStore(tmp_path / "artifact", read_root=workspace)

    with pytest.raises(ValueError, match="escapes"):
        ReadFileTool().run({"path": "../outside.txt"}, store)
    with pytest.raises(ValueError, match="relative"):
        ReadFileTool().run({"path": str(workspace / "notes.txt")}, store)


def test_read_file_rejects_a_hardlink_in_the_read_root(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    os.link(outside, workspace / "notes.txt")
    store = LocalArtifactStore(tmp_path / "artifact", read_root=workspace)

    with pytest.raises(SandboxDenied, match="hardlinked"):
        ReadFileTool().run({"path": "notes.txt"}, store)


def test_read_file_missing_file_with_read_root_hides_host_paths(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = LocalArtifactStore(tmp_path / "artifact", read_root=workspace)

    with pytest.raises(FileNotFoundError) as excinfo:
        ReadFileTool().run({"path": "missing.txt"}, store)

    message = str(excinfo.value)
    assert "missing.txt" in message
    assert str(tmp_path) not in message


def test_write_file_never_writes_to_the_read_root(tmp_path: Path) -> None:
    from atlas.agent.tools.write_file import WriteTool

    artifact = tmp_path / "artifact"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = LocalArtifactStore(artifact, read_root=workspace)

    WriteTool().run({"path": "out.txt", "content": "data"}, store)

    assert (artifact / "out.txt").is_file()
    assert not (workspace / "out.txt").exists()


def test_bash_does_not_read_the_sessions_stdin(tmp_path: Path) -> None:
    # The session reads the user's protocol messages from stdin, so run the
    # tool in a child process whose stdin holds one such message.
    script = (
        "import sys; from pathlib import Path;"
        "from atlas.agent.artifacts import LocalArtifactStore;"
        "from atlas.agent.tools.BashTool import BashTool;"
        "print(BashTool().run({'command': 'head -1'}, LocalArtifactStore(Path(sys.argv[1]))).text)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        input='{"type": "user", "text": "next message"}\n',
        capture_output=True,
        text=True,
        check=True,
    )

    assert "exit_code: 0" in completed.stdout
    assert "next message" not in completed.stdout


@pytest.mark.parametrize("timeout", [0, -5, 301, 10**9])
def test_bash_rejects_a_timeout_outside_the_allowed_range(tmp_path: Path, timeout: int) -> None:
    store = LocalArtifactStore(tmp_path)

    with pytest.raises(ValidationError):
        BashTool().run({"command": "echo hi", "timeout": timeout}, store)


def test_bash_reports_non_utf8_output_instead_of_raising(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)

    result = BashTool().run({"command": "printf 'ok\\377'"}, store)

    assert "exit_code: 0" in result.text
    assert "ok�" in result.text
