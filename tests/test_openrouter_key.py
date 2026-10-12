"""ATLAS-50: set an OpenRouter key for this session or for the user."""

from __future__ import annotations

import io
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any

import pytest

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import ModelResponse, ToolRegistry
from atlas.agent.layout import (
    AtlasPathError,
    OpenRouterKeyWarning,
    ensure_user_layout,
    read_user_key,
    resolve_openrouter_key,
    write_user_key,
)
from atlas.agent.model import MISSING_KEY_MESSAGE, ModelConfig, OpenRouterModel
from atlas.agent.runtime import Agent
from atlas.agent.session import AgentSession, main, serve

_USER_FILE = Path(".atlas") / "keys" / "openrouter_api_key"


class _Response:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.status_code = 200
        self.elapsed_seconds = 0.5
        self.time_to_first_token_seconds = None

    def json(self) -> dict[str, Any]:
        return self.payload


class _Client:
    """Records OpenRouter calls and returns one canned completion."""

    def __init__(self) -> None:
        self.posts: list[dict[str, Any]] = []

    def post(self, url: str, **kwargs: Any) -> _Response:
        self.posts.append({"url": url, **kwargs})
        return _Response(
            {
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                "choices": [{"message": {"content": "ok", "tool_calls": []}}],
            }
        )

    def close(self) -> None:
        return None


def _session(tmp_path: Path) -> tuple[AgentSession, _Client, Path, Path]:
    home = tmp_path / "home"
    project = tmp_path / "project"
    project.mkdir()
    client = _Client()
    model = OpenRouterModel(ModelConfig(model="example/model", api_key=""), client)
    session = AgentSession(Agent(model, ToolRegistry(), LocalArtifactStore(project)))
    session.home = home
    session.workspace = project
    session.requires_api_key = True
    return session, client, home, project


def _run(session: AgentSession, lines: list[str]) -> list[dict[str, Any]]:
    stdout = io.StringIO()
    code = serve(session, io.StringIO("".join(f"{line}\n" for line in lines)), stdout)
    assert code == 0
    return [json.loads(line) for line in stdout.getvalue().splitlines()]


def _set_key(scope: str, key: str) -> str:
    return json.dumps({"type": "set_key", "scope": scope, "key": key})


def _bearer(client: _Client) -> list[str]:
    return [post["headers"]["Authorization"] for post in client.posts]


def _project_text(project: Path) -> str:
    root = project / ".atlas"
    if not root.exists():
        return ""
    return "\n".join(path.read_text(encoding="utf-8") for path in root.rglob("*") if path.is_file())


def test_resolution_prefers_session_then_environment_then_the_user_file(tmp_path: Path) -> None:
    home = tmp_path / "home"
    write_user_key("  sk-or-user-key  ", home)

    assert read_user_key(home) == "sk-or-user-key"
    assert (
        resolve_openrouter_key(session_key=" sk-or-session-key ", env_key="env-key", home=home)
        == "sk-or-session-key"
    )
    assert resolve_openrouter_key(session_key="  ", env_key=" env-key ", home=home) == "env-key"
    assert resolve_openrouter_key(session_key=None, env_key=None, home=home) == "sk-or-user-key"
    assert resolve_openrouter_key(session_key=None, env_key=" ", home=home) == "sk-or-user-key"
    path = home / _USER_FILE
    assert path.is_file()
    assert path.stat().st_mode & 0o777 == 0o600
    assert (home / ".atlas" / "keys").stat().st_mode & 0o777 == 0o700
    assert path.read_text(encoding="utf-8") == "sk-or-user-key\n"


def test_session_key_is_sent_on_the_next_completion_and_is_not_stored(tmp_path: Path) -> None:
    session, client, home, project = _session(tmp_path)

    events = _run(
        session,
        [
            _set_key("session", "  sk-or-session-key  "),
            '{"type":"user","text":"hi"}',
            '{"type":"new"}',
            '{"type":"user","text":"again"}',
        ],
    )

    assert _bearer(client) == ["Bearer sk-or-session-key", "Bearer sk-or-session-key"]
    assert session.session_key == "sk-or-session-key"
    assert "sk-or-session-key" in session.secrets
    assert events[1] == {"type": "key", "scope": "session", "saved": True}
    ready = next(event for event in events if event["type"] == "ready" and event is not events[0])
    assert ready["key_set"] is True
    assert "sk-or-session-key" not in json.dumps(events)
    assert not (home / _USER_FILE).exists()
    assert "sk-or-session-key" not in _project_text(project)


def test_user_key_is_saved_for_later_and_used_when_no_session_key_is_set(tmp_path: Path) -> None:
    session, client, home, project = _session(tmp_path)

    events = _run(
        session,
        [
            _set_key("user", "  sk-or-user-key  "),
            '{"type":"user","text":"hi"}',
        ],
    )

    assert _bearer(client) == ["Bearer sk-or-user-key"]
    assert events[1] == {"type": "key", "scope": "user", "saved": True}
    path = home / _USER_FILE
    assert path.read_text(encoding="utf-8") == "sk-or-user-key\n"
    assert path.stat().st_mode & 0o777 == 0o600
    assert (home / ".atlas" / "keys").stat().st_mode & 0o777 == 0o700
    assert "sk-or-user-key" not in json.dumps(events)
    assert "sk-or-user-key" not in _project_text(project)


def test_a_later_user_key_does_not_replace_the_session_key(tmp_path: Path) -> None:
    session, client, home, project = _session(tmp_path)

    events = _run(
        session,
        [
            _set_key("user", "sk-or-user-key"),
            _set_key("session", "sk-or-session-key"),
            _set_key("user", "sk-or-later-user-key"),
            '{"type":"user","text":"hi"}',
        ],
    )

    assert _bearer(client) == ["Bearer sk-or-session-key"]
    assert session.session_key == "sk-or-session-key"
    assert (home / _USER_FILE).read_text(encoding="utf-8").strip() == "sk-or-later-user-key"
    dumped = json.dumps(events)
    assert "sk-or-user-key" not in dumped
    assert "sk-or-session-key" not in dumped
    assert "sk-or-later-user-key" not in dumped
    assert "sk-or-session-key" not in _project_text(project)
    assert "sk-or-later-user-key" not in _project_text(project)


def test_a_blank_or_unknown_scope_is_rejected_without_echoing_the_key(tmp_path: Path) -> None:
    session, client, home, project = _session(tmp_path)
    secret = "rejected-key"

    events = _run(
        session,
        [
            _set_key("session", "   "),
            _set_key("elsewhere", secret),
            json.dumps({"type": "set_key", "scope": "user"}),
            json.dumps({"type": "set_key", "scope": "user", "key": 1}),
            '{"type":"user","text":"hi"}',
        ],
    )

    assert client.posts == []
    assert not (home / _USER_FILE).exists()
    assert secret not in json.dumps(events)
    errors = [event for event in events if event["type"] == "error"]
    assert len(errors) == 5
    assert errors[0]["message"] == "set_key needs scope session or user and a key"
    assert errors[-1]["message"] == MISSING_KEY_MESSAGE
    assert "rejected-key" not in _project_text(project)


def test_user_key_write_refuses_a_symlink(tmp_path: Path) -> None:
    home = tmp_path / "home"
    outside = tmp_path / "outside"
    outside.write_text("original", encoding="utf-8")
    keys = home / ".atlas" / "keys"
    keys.mkdir(parents=True)
    link = keys / "openrouter_api_key"
    link.symlink_to(outside)

    with pytest.raises(AtlasPathError, match="symlink"):
        write_user_key("planted-key", home)

    assert outside.read_text(encoding="utf-8") == "original"
    assert link.is_symlink()
    assert read_user_key(home) is None

    stolen = tmp_path / "stolen"
    stolen.mkdir()
    home2 = tmp_path / "home2"
    atlas = home2 / ".atlas"
    atlas.mkdir(parents=True)
    (atlas / "keys").symlink_to(stolen, target_is_directory=True)
    with pytest.raises(AtlasPathError, match="symlink"):
        write_user_key("planted-key", home2)
    assert list(stolen.iterdir()) == []
    assert read_user_key(home2) is None


def test_set_key_refuses_a_symlinked_user_file_without_installing_it(tmp_path: Path) -> None:
    session, client, home, _project = _session(tmp_path)
    outside = tmp_path / "outside"
    outside.write_text("original", encoding="utf-8")
    keys = home / ".atlas" / "keys"
    keys.mkdir(parents=True)
    (keys / "openrouter_api_key").symlink_to(outside)

    events = _run(session, [_set_key("user", "planted-key"), '{"type":"user","text":"hi"}'])

    assert outside.read_text(encoding="utf-8") == "original"
    assert client.posts == []
    assert events[1]["type"] == "error"
    assert events[1]["message"] == "Could not save the key: AtlasPathError"
    assert "planted-key" not in json.dumps(events)
    assert session.agent.model.config.api_key == ""


def test_main_loads_the_user_key_and_prefers_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    write_user_key("sk-or-saved-user-key", home)
    seen: list[str] = []
    real = OpenRouterModel

    def capture(config: ModelConfig, client: Any = None) -> OpenRouterModel:
        seen.append(config.api_key)
        return real(config, client)

    monkeypatch.setattr("atlas.agent.session.OpenRouterModel", capture)
    monkeypatch.setattr("atlas.agent.session.Path.home", lambda: home)
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"type":"quit"}\n'))

    assert main(["--workspace", str(workspace)]) == 0
    captured = capsys.readouterr()
    ready = json.loads(captured.out.splitlines()[0])
    assert ready["type"] == "ready"
    assert ready["key_set"] is True
    assert seen == ["sk-or-saved-user-key"]
    assert "sk-or-saved-user-key" not in captured.out
    assert "sk-or-saved-user-key" not in captured.err

    seen.clear()
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-wins-key")
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"type":"quit"}\n'))
    assert main(["--workspace", str(workspace)]) == 0
    captured = capsys.readouterr()
    assert seen == ["env-wins-key"]
    assert "env-wins-key" not in captured.out
    assert "sk-or-saved-user-key" not in captured.out
    assert (home / _USER_FILE).read_text(encoding="utf-8").strip() == "sk-or-saved-user-key"


def test_a_session_key_does_not_survive_into_the_next_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    session, _client, home, _project = _session(tmp_path)
    _run(session, [_set_key("session", "sk-or-session-only")])
    assert not (home / _USER_FILE).exists()

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr("atlas.agent.session.Path.home", lambda: home)
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"type":"quit"}\n'))

    assert main(["--workspace", str(workspace)]) == 0
    captured = capsys.readouterr()
    ready = json.loads(captured.out.splitlines()[0])
    assert ready["key_set"] is False
    assert "sk-or-session-only" not in captured.out
    assert "sk-or-session-only" not in captured.err


def test_private_key_file_stays_mode_600_when_the_umask_is_permissive(tmp_path: Path) -> None:
    home = tmp_path / "home"
    previous = os.umask(0)
    try:
        path = write_user_key("sk-or-user-key", home)
    finally:
        os.umask(previous)
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700


def test_keys_directory_is_created_mode_700_and_an_existing_one_is_tightened(
    tmp_path: Path,
) -> None:
    if os.name != "posix":
        pytest.skip("Windows does not enforce Unix directory modes")
    home = tmp_path / "home"
    keys = home / ".atlas" / "keys"
    keys.mkdir(parents=True)
    os.chmod(keys, 0o755)

    ensure_user_layout(home)

    assert keys.stat().st_mode & 0o777 == 0o700
    previous = os.umask(0)
    try:
        fresh = tmp_path / "fresh"
        write_user_key("sk-or-fresh-key", fresh)
    finally:
        os.umask(previous)
    assert (fresh / ".atlas" / "keys").stat().st_mode & 0o777 == 0o700


def test_replacing_a_loose_key_file_tightens_it_before_the_atomic_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if os.name != "posix":
        pytest.skip("Windows does not enforce Unix file modes")
    home = tmp_path / "home"
    keys = home / ".atlas" / "keys"
    keys.mkdir(parents=True)
    path = keys / "openrouter_api_key"
    path.write_text("old-loose-key\n", encoding="utf-8")
    os.chmod(path, 0o644)
    real_chmod = os.chmod
    real_open = os.open
    real_replace = os.replace
    real_fsync = os.fsync
    seen: list[tuple[str, str, int]] = []
    opened: list[tuple[str, int, int]] = []
    replaced: list[tuple[str, str]] = []
    synced: list[int] = []

    def spy_chmod(target: os.PathLike[str] | str, mode: int) -> None:
        candidate = Path(target)
        if candidate.name == "openrouter_api_key" and candidate.exists():
            seen.append(
                (
                    candidate.read_text(encoding="utf-8"),
                    candidate.stat().st_mode & 0o777,
                    mode,
                )
            )
        real_chmod(target, mode)

    def spy_open(target: os.PathLike[str] | str, flags: int, mode: int = 0o777) -> int:
        name = Path(target).name
        opened.append((name, flags, mode))
        if name == "openrouter_api_key":
            raise AssertionError("the existing key file was opened in place")
        return real_open(target, flags, mode)

    def spy_replace(src: os.PathLike[str] | str, dst: os.PathLike[str] | str) -> None:
        replaced.append((Path(src).name, Path(dst).name))
        real_replace(src, dst)

    def spy_fsync(fd: int) -> None:
        synced.append(fd)
        real_fsync(fd)

    monkeypatch.setattr(os, "chmod", spy_chmod)
    monkeypatch.setattr(os, "open", spy_open)
    monkeypatch.setattr(os, "replace", spy_replace)
    monkeypatch.setattr(os, "fsync", spy_fsync)

    write_user_key("sk-or-new-key", home)

    assert seen[0] == ("old-loose-key\n", 0o644, 0o600)
    created = [item for item in opened if item[0] != "keys"]
    assert created
    assert all(flags & os.O_EXCL and mode == 0o600 for _name, flags, mode in created)
    assert synced
    assert replaced[0][1] == "openrouter_api_key"
    assert replaced[0][0] != "openrouter_api_key"
    assert path.read_text(encoding="utf-8") == "sk-or-new-key\n"
    assert path.stat().st_mode & 0o777 == 0o600
    assert [item.name for item in keys.iterdir()] == ["openrouter_api_key"]


def test_the_keys_directory_is_fsynced_after_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if os.name != "posix":
        pytest.skip("Windows has no directory fsync")
    home = tmp_path / "home"
    events: list[str] = []
    real_replace = os.replace
    real_fsync = os.fsync

    def spy_replace(src: os.PathLike[str] | str, dst: os.PathLike[str] | str) -> None:
        events.append("replace")
        real_replace(src, dst)

    def spy_fsync(fd: int) -> None:
        kind = "dir" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file"
        events.append(f"fsync-{kind}")
        real_fsync(fd)

    monkeypatch.setattr(os, "replace", spy_replace)
    monkeypatch.setattr(os, "fsync", spy_fsync)

    path = write_user_key("sk-or-dir-sync", home)

    assert events == ["fsync-file", "replace", "fsync-dir"]
    assert path.read_text(encoding="utf-8") == "sk-or-dir-sync\n"


def test_a_failed_directory_fsync_raises_after_the_new_key_is_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if os.name != "posix":
        pytest.skip("Windows has no directory fsync")
    home = tmp_path / "home"
    real_fsync = os.fsync

    def spy_fsync(fd: int) -> None:
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError("directory fsync failed")
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", spy_fsync)
    with pytest.raises(OSError, match="directory fsync failed"):
        write_user_key("sk-or-dir-sync", home)
    path = home / _USER_FILE
    assert path.read_text(encoding="utf-8") == "sk-or-dir-sync\n"
    assert [item.name for item in path.parent.iterdir()] == ["openrouter_api_key"]


def test_a_failed_atomic_write_leaves_the_old_key_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    keys = home / ".atlas" / "keys"
    keys.mkdir(parents=True)
    path = keys / "openrouter_api_key"
    path.write_text("old-loose-key\n", encoding="utf-8")
    os.chmod(path, 0o644)

    def boom(_fd: int) -> None:
        raise OSError("fsync failed")

    monkeypatch.setattr(os, "fsync", boom)
    with pytest.raises(OSError, match="fsync failed"):
        write_user_key("sk-or-new-key", home)
    assert path.read_text(encoding="utf-8") == "old-loose-key\n"
    if os.name == "posix":
        assert path.stat().st_mode & 0o777 == 0o600
    assert [item.name for item in keys.iterdir()] == ["openrouter_api_key"]


def test_read_warns_when_the_key_file_is_too_loose(tmp_path: Path) -> None:
    if os.name != "posix":
        pytest.skip("Windows does not report Unix permission bits")
    home = tmp_path / "home"
    path = write_user_key("sk-or-user-key", home)
    os.chmod(path, 0o644)

    with pytest.warns(OpenRouterKeyWarning, match="too loose"):
        assert read_user_key(home) == "sk-or-user-key"


def test_a_newline_is_rejected_and_a_missing_prefix_is_warned(tmp_path: Path) -> None:
    home = tmp_path / "home"
    with pytest.raises(ValueError, match="newline"):
        write_user_key("sk-or-abc\nmore", home)
    assert not (home / _USER_FILE).exists()

    with pytest.warns(OpenRouterKeyWarning, match="sk-or-"):
        path = write_user_key("plain-user-key", home)
    assert path.read_text(encoding="utf-8") == "plain-user-key\n"

    session, client, session_home, _project = _session(tmp_path)
    secret = "sk-or-line\nsecret"
    events = _run(session, [_set_key("user", secret), _set_key("session", "plain-session-key")])
    assert secret not in json.dumps(events)
    assert events[1]["type"] == "error"
    assert events[1]["message"] == "set_key rejected a key that contains a newline"
    assert events[2]["warning"] == "OpenRouter key does not start with sk-or-"
    assert "plain-session-key" in session.secrets
    assert "plain-session-key" not in json.dumps(events)
    assert (session_home / _USER_FILE).read_text(encoding="utf-8") == "plain-user-key\n"
    assert client.posts == []


def test_delete_key_removes_the_file_and_keeps_redacting_it(tmp_path: Path) -> None:
    session, client, home, project = _session(tmp_path)

    class Echo:
        def __init__(self) -> None:
            self.config = ModelConfig(model="example/model", api_key="")
            self.reply = "leak sk-or-delete-me"

        def complete(self, messages: list[Any], tools: list[Any]) -> ModelResponse:
            del messages, tools
            return ModelResponse(content=self.reply)

    echo = Echo()
    session.agent.model = echo
    events = _run(
        session,
        [
            _set_key("user", "sk-or-delete-me"),
            '{"type":"user","text":"hi"}',
            '{"type":"delete_key"}',
            '{"type":"user","text":"again"}',
        ],
    )
    assert not (home / _USER_FILE).exists()
    assert session.session_key is None
    assert session.active_key is None
    assert echo.config.api_key == ""
    deleted = next(event for event in events if event.get("scope") == "deleted")
    assert deleted == {
        "type": "key",
        "scope": "deleted",
        "saved": False,
        "warning": ("OPENROUTER_API_KEY in the environment will still be used on the next launch"),
    }
    assert "sk-or-delete-me" not in json.dumps(events)
    assert "***" in json.dumps(events)
    assert client.posts == []
    missing = [event for event in events if event["type"] == "error"]
    assert missing[0]["message"] == MISSING_KEY_MESSAGE
    assert "sk-or-delete-me" not in _project_text(project)


def test_a_key_present_at_startup_is_redacted_without_a_separate_redact_argument(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()

    class Echo:
        def __init__(self) -> None:
            self.config = ModelConfig(model="example/model", api_key="sk-or-startup-key")

        def complete(self, messages: list[Any], tools: list[Any]) -> ModelResponse:
            del messages, tools
            return ModelResponse(content="see sk-or-startup-key")

    session = AgentSession(Agent(Echo(), ToolRegistry(), LocalArtifactStore(project)))
    assert "sk-or-startup-key" in session.secrets
    events = _run(session, ['{"type":"user","text":"hi"}'])
    assert "sk-or-startup-key" not in json.dumps(events)
    assert "***" in json.dumps(events)
