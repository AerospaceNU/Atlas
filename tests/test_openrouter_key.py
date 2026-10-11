"""ATLAS-50: set an OpenRouter key for this session or for the user."""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import ToolRegistry
from atlas.agent.layout import (
    AtlasPathError,
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
    write_user_key("  user-key  ", home)

    assert read_user_key(home) == "user-key"
    assert resolve_openrouter_key(session_key=" session-key ", env_key="env-key", home=home) == (
        "session-key"
    )
    assert resolve_openrouter_key(session_key="  ", env_key=" env-key ", home=home) == "env-key"
    assert resolve_openrouter_key(session_key=None, env_key=None, home=home) == "user-key"
    assert resolve_openrouter_key(session_key=None, env_key=" ", home=home) == "user-key"
    path = home / _USER_FILE
    assert path.is_file()
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.read_text(encoding="utf-8") == "user-key\n"


def test_session_key_is_sent_on_the_next_completion_and_is_not_stored(tmp_path: Path) -> None:
    session, client, home, project = _session(tmp_path)

    events = _run(
        session,
        [
            _set_key("session", "  session-key  "),
            '{"type":"user","text":"hi"}',
            '{"type":"new"}',
            '{"type":"user","text":"again"}',
        ],
    )

    assert _bearer(client) == ["Bearer session-key", "Bearer session-key"]
    assert session.session_key == "session-key"
    assert events[1] == {"type": "key", "scope": "session", "saved": True}
    ready = next(event for event in events if event["type"] == "ready" and event is not events[0])
    assert ready["key_set"] is True
    assert "session-key" not in json.dumps(events)
    assert not (home / _USER_FILE).exists()
    assert "session-key" not in _project_text(project)


def test_user_key_is_saved_for_later_and_used_when_no_session_key_is_set(tmp_path: Path) -> None:
    session, client, home, project = _session(tmp_path)

    events = _run(
        session,
        [
            _set_key("user", "  user-key  "),
            '{"type":"user","text":"hi"}',
        ],
    )

    assert _bearer(client) == ["Bearer user-key"]
    assert events[1] == {"type": "key", "scope": "user", "saved": True}
    path = home / _USER_FILE
    assert path.read_text(encoding="utf-8") == "user-key\n"
    assert path.stat().st_mode & 0o777 == 0o600
    assert "user-key" not in json.dumps(events)
    assert "user-key" not in _project_text(project)


def test_a_later_user_key_does_not_replace_the_session_key(tmp_path: Path) -> None:
    session, client, home, project = _session(tmp_path)

    events = _run(
        session,
        [
            _set_key("user", "user-key"),
            _set_key("session", "session-key"),
            _set_key("user", "later-user-key"),
            '{"type":"user","text":"hi"}',
        ],
    )

    assert _bearer(client) == ["Bearer session-key"]
    assert session.session_key == "session-key"
    assert (home / _USER_FILE).read_text(encoding="utf-8").strip() == "later-user-key"
    dumped = json.dumps(events)
    assert "user-key" not in dumped
    assert "session-key" not in dumped
    assert "later-user-key" not in dumped
    assert "session-key" not in _project_text(project)
    assert "later-user-key" not in _project_text(project)


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
    write_user_key("saved-user-key", home)
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
    assert seen == ["saved-user-key"]
    assert "saved-user-key" not in captured.out
    assert "saved-user-key" not in captured.err

    seen.clear()
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-wins-key")
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"type":"quit"}\n'))
    assert main(["--workspace", str(workspace)]) == 0
    captured = capsys.readouterr()
    assert seen == ["env-wins-key"]
    assert "env-wins-key" not in captured.out
    assert "saved-user-key" not in captured.out
    assert (home / _USER_FILE).read_text(encoding="utf-8").strip() == "saved-user-key"


def test_a_session_key_does_not_survive_into_the_next_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    session, _client, home, _project = _session(tmp_path)
    _run(session, [_set_key("session", "session-only")])
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
    assert "session-only" not in captured.out
    assert "session-only" not in captured.err


def test_private_key_file_stays_mode_600_when_the_umask_is_permissive(tmp_path: Path) -> None:
    home = tmp_path / "home"
    previous = os.umask(0)
    try:
        path = write_user_key("user-key", home)
    finally:
        os.umask(previous)
    assert path.stat().st_mode & 0o777 == 0o600
