"""ATLAS-104: store EARTHDATA_TOKEN and FIRMS_MAP_KEY like the OpenRouter key."""

from __future__ import annotations

import io
import json
import os
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import ToolRegistry
from atlas.agent.layout import (
    EARTHDATA_TOKEN,
    FIRMS_MAP_KEY,
    CredentialKeyWarning,
    bind_session_credentials,
    resolve_credential,
    write_user_key,
)
from atlas.agent.model import ModelConfig, OpenRouterModel
from atlas.agent.runtime import Agent
from atlas.agent.session import AgentSession, serve
from atlas.cli.data import run_data
from atlas.data.base import Asset, BBox, PullRequest, Scene, SceneKind
from atlas.data.firms import FirmsClient
from atlas.data.sources import fldas as fldas_module
from atlas.data.sources.fldas import MISSING_TOKEN_MESSAGE, FldasAuthError, FldasClient

_FILES = {
    EARTHDATA_TOKEN: Path(".atlas") / "keys" / "earthdata_token",
    FIRMS_MAP_KEY: Path(".atlas") / "keys" / "firms_map_key",
}
_TOKENS = {
    EARTHDATA_TOKEN: "ed-session-token",
    FIRMS_MAP_KEY: "firms-session-key",
}


class _Client:
    def post(self, url: str, **kwargs: Any) -> object:
        del url, kwargs
        raise AssertionError("these tests do not call the model")

    def close(self) -> None:
        return None


class _Gets:
    def __init__(self) -> None:
        self.urls: list[str] = []

    async def get(self, url: str) -> _Body:
        self.urls.append(url)
        return _Body()


class _Body:
    status_code = 200

    def raise_for_status(self) -> None:
        return None

    text = (
        "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,"
        "satellite,instrument,confidence,version,bright_ti5,frp,daynight\n"
        "42.35,-71.08,330.1,0.4,0.4,2024-07-01,1342,N,VIIRS,n,2.0NRT,290.1,4.2,D\n"
    )


def _session(tmp_path: Path) -> tuple[AgentSession, Path, Path]:
    home = tmp_path / "home"
    project = tmp_path / "project"
    project.mkdir()
    model = OpenRouterModel(ModelConfig(model="example/model", api_key=""), _Client())
    session = AgentSession(Agent(model, ToolRegistry(), LocalArtifactStore(project)))
    session.home = home
    session.workspace = project
    return session, home, project


def _run(session: AgentSession, lines: list[str]) -> list[dict[str, Any]]:
    stdout = io.StringIO()
    code = serve(session, io.StringIO("".join(f"{line}\n" for line in lines)), stdout)
    assert code == 0
    return [json.loads(line) for line in stdout.getvalue().splitlines()]


def _set_key(name: str, scope: str, key: str) -> str:
    return json.dumps({"type": "set_key", "scope": scope, "name": name, "key": key})


def _project_text(project: Path) -> str:
    root = project / ".atlas"
    if not root.exists():
        return ""
    return "\n".join(path.read_text(encoding="utf-8") for path in root.rglob("*") if path.is_file())


def _pull() -> PullRequest:
    return PullRequest(
        start_date=date(2024, 7, 1),
        end_date=date(2024, 7, 2),
        bbox=BBox(west=-71.2, south=42.2, east=-71.0, north=42.4),
        limit=1,
    )


def _scene() -> Scene:
    return Scene(
        id="g1",
        source="fldas",
        collection="FLDAS_NOAH01_C_GL_M",
        kind=SceneKind.landsurface,
        datetime=datetime(2020, 1, 2, tzinfo=UTC),
        bbox=BBox(west=-180, south=-60, east=180, north=90),
        platform="FLDAS",
        assets={"opendap": Asset(href="https://example.test/opendap")},
    )


@pytest.fixture(autouse=True)
def _hide_saved_keys(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv(EARTHDATA_TOKEN, raising=False)
    monkeypatch.delenv(FIRMS_MAP_KEY, raising=False)
    home = tmp_path / "empty-home"
    home.mkdir()
    monkeypatch.setattr("atlas.agent.layout.Path.home", lambda: home)


@pytest.mark.parametrize("name", [EARTHDATA_TOKEN, FIRMS_MAP_KEY])
def test_resolution_prefers_session_then_environment_then_the_user_file(
    tmp_path: Path, name: str
) -> None:
    home = tmp_path / "home"
    write_user_key(f"  {_TOKENS[name]}-user  ", home, name=name)

    assert (
        resolve_credential(
            name,
            session_key=f" {_TOKENS[name]} ",
            env_key="env-token",
            home=home,
        )
        == _TOKENS[name]
    )
    assert (
        resolve_credential(name, session_key="  ", env_key=" env-token ", home=home) == "env-token"
    )
    assert resolve_credential(name, session_key=None, env_key=" ", home=home) == (
        f"{_TOKENS[name]}-user"
    )
    path = home / _FILES[name]
    assert path.is_file()
    assert path.read_text(encoding="utf-8") == f"{_TOKENS[name]}-user\n"
    if os.name == "posix":
        assert path.stat().st_mode & 0o777 == 0o600
        assert path.parent.stat().st_mode & 0o777 == 0o700


@pytest.mark.parametrize("name", [EARTHDATA_TOKEN, FIRMS_MAP_KEY])
def test_session_key_stays_in_memory_and_is_not_printed(tmp_path: Path, name: str) -> None:
    session, home, project = _session(tmp_path)
    secret = _TOKENS[name]

    events = _run(session, [_set_key(name, "session", f"  {secret}  ")])

    assert session.credential_keys[name] == secret
    assert secret in session.secrets
    assert not (home / _FILES[name]).exists()
    assert secret not in json.dumps(events)
    assert secret not in _project_text(project)
    assert events[1] == {"type": "key", "scope": "session", "saved": True, "name": name}
    client = FirmsClient() if name == FIRMS_MAP_KEY else None
    if client is not None:
        assert client._map_key == ""
    bind_session_credentials(session.credential_keys)
    if name == FIRMS_MAP_KEY:
        assert FirmsClient()._map_key == secret
    else:
        assert FldasClient()._token == secret


@pytest.mark.parametrize("name", [EARTHDATA_TOKEN, FIRMS_MAP_KEY])
def test_user_key_is_private_and_a_later_session_value_wins(tmp_path: Path, name: str) -> None:
    session, home, project = _session(tmp_path)
    user_secret = f"{_TOKENS[name]}-user"
    session_secret = _TOKENS[name]

    events = _run(
        session,
        [
            _set_key(name, "user", f"  {user_secret}  "),
            _set_key(name, "session", session_secret),
        ],
    )

    path = home / _FILES[name]
    assert path.read_text(encoding="utf-8") == f"{user_secret}\n"
    if os.name == "posix":
        assert path.stat().st_mode & 0o777 == 0o600
        assert (home / ".atlas" / "keys").stat().st_mode & 0o777 == 0o700
    assert session.credential_keys[name] == session_secret
    dumped = json.dumps(events)
    assert user_secret not in dumped
    assert session_secret not in dumped
    assert user_secret not in _project_text(project)
    assert session_secret not in _project_text(project)
    bind_session_credentials(session.credential_keys)
    monkeypatch_home = home
    assert resolve_credential(name, home=monkeypatch_home) == session_secret


def test_a_user_key_does_not_replace_the_session_value(tmp_path: Path) -> None:
    session, home, project = _session(tmp_path)
    events = _run(
        session,
        [
            _set_key(EARTHDATA_TOKEN, "session", "ed-session-token"),
            _set_key(EARTHDATA_TOKEN, "user", "ed-later-user-token"),
        ],
    )
    assert session.credential_keys[EARTHDATA_TOKEN] == "ed-session-token"
    assert (home / _FILES[EARTHDATA_TOKEN]).read_text(encoding="utf-8").strip() == (
        "ed-later-user-token"
    )
    dumped = json.dumps(events)
    assert "ed-session-token" not in dumped
    assert "ed-later-user-token" not in dumped
    assert "ed-later-user-token" not in _project_text(project)
    bind_session_credentials(session.credential_keys)
    assert resolve_credential(EARTHDATA_TOKEN, home=home) == "ed-session-token"


def test_errors_name_the_variable_and_redact_the_value(tmp_path: Path) -> None:
    session, home, project = _session(tmp_path)
    secret = "rejected-earthdata-token"
    events = _run(
        session,
        [
            _set_key(EARTHDATA_TOKEN, "session", "   "),
            _set_key(EARTHDATA_TOKEN, "elsewhere", secret),
            json.dumps({"type": "set_key", "name": secret, "scope": "user", "key": secret}),
            _set_key(FIRMS_MAP_KEY, "user", "firms-line\nsecret"),
        ],
    )
    dumped = json.dumps(events)
    assert secret not in dumped
    assert "firms-line" not in dumped
    assert "rejected-earthdata-token" not in _project_text(project)
    assert not (home / _FILES[EARTHDATA_TOKEN]).exists()
    assert not (home / _FILES[FIRMS_MAP_KEY]).exists()
    messages = [event["message"] for event in events if event["type"] == "error"]
    assert messages == [
        "EARTHDATA_TOKEN needs scope session or user and a key",
        "EARTHDATA_TOKEN needs scope session or user and a key",
        "set_key name must be OPENROUTER_API_KEY, EARTHDATA_TOKEN, or FIRMS_MAP_KEY",
        "FIRMS_MAP_KEY is empty or contains a newline",
    ]


def test_a_symlinked_user_file_is_refused_without_echoing_the_value(tmp_path: Path) -> None:
    session, home, _project = _session(tmp_path)
    outside = tmp_path / "outside"
    outside.write_text("original", encoding="utf-8")
    keys = home / ".atlas" / "keys"
    keys.mkdir(parents=True)
    (keys / "earthdata_token").symlink_to(outside)

    events = _run(session, [_set_key(EARTHDATA_TOKEN, "user", "planted-earthdata-token")])

    assert outside.read_text(encoding="utf-8") == "original"
    assert events[1]["message"] == "Could not save EARTHDATA_TOKEN: AtlasPathError"
    assert "planted-earthdata-token" not in json.dumps(events)
    assert EARTHDATA_TOKEN not in session.credential_keys


def test_private_key_file_stays_mode_600_when_the_umask_is_permissive(tmp_path: Path) -> None:
    if os.name != "posix":
        pytest.skip("Windows does not enforce Unix file modes")
    home = tmp_path / "home"
    previous = os.umask(0)
    try:
        path = write_user_key("firms-user-key", home, name=FIRMS_MAP_KEY)
    finally:
        os.umask(previous)
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700


def test_read_warns_when_the_key_file_is_too_loose(tmp_path: Path) -> None:
    if os.name != "posix":
        pytest.skip("Windows does not report Unix permission bits")
    home = tmp_path / "home"
    path = write_user_key("ed-user-token", home, name=EARTHDATA_TOKEN)
    os.chmod(path, 0o644)
    with pytest.warns(CredentialKeyWarning, match="EARTHDATA_TOKEN file permissions are too loose"):
        assert resolve_credential(EARTHDATA_TOKEN, env_key=" ", home=home) == "ed-user-token"


def test_delete_removes_the_file_and_keeps_redacting_the_value(tmp_path: Path) -> None:
    session, home, project = _session(tmp_path)
    secret = "firms-delete-me"
    events = _run(
        session,
        [
            _set_key(FIRMS_MAP_KEY, "user", secret),
            json.dumps({"type": "delete_key", "name": FIRMS_MAP_KEY}),
        ],
    )
    assert not (home / _FILES[FIRMS_MAP_KEY]).exists()
    assert FIRMS_MAP_KEY not in session.credential_keys
    deleted = next(event for event in events if event.get("scope") == "deleted")
    assert deleted == {
        "type": "key",
        "scope": "deleted",
        "saved": False,
        "name": FIRMS_MAP_KEY,
        "warning": ("FIRMS_MAP_KEY in the environment will still be used on the next launch"),
    }
    assert secret not in json.dumps(events)
    assert secret not in _project_text(project)


@pytest.mark.asyncio
async def test_firms_uses_the_session_key_and_keeps_the_missing_key_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    home = tmp_path / "home"
    write_user_key("firms-user-key", home, name=FIRMS_MAP_KEY)
    monkeypatch.setattr("atlas.agent.layout.Path.home", lambda: home)
    monkeypatch.setenv(FIRMS_MAP_KEY, "firms-env-key")
    bind_session_credentials({FIRMS_MAP_KEY: "firms-session-key"})
    transport = _Gets()
    client = FirmsClient(client=transport)  # type: ignore[arg-type]
    result = await client.search(_pull())
    assert len(result.scenes) == 1
    assert transport.urls
    assert "firms-session-key" in transport.urls[0]
    assert "firms-env-key" not in transport.urls[0]
    assert "firms-user-key" not in transport.urls[0]

    bind_session_credentials({})
    monkeypatch.delenv(FIRMS_MAP_KEY, raising=False)
    explicit = FirmsClient(map_key="", client=transport)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="FIRMS_MAP_KEY is not set") as excinfo:
        await explicit.search(_pull())
    assert str(excinfo.value) == "FIRMS_MAP_KEY is not set"
    assert "firms-session-key" not in str(excinfo.value)

    from_file = FirmsClient()
    assert from_file._map_key == "firms-user-key"


@pytest.mark.asyncio
async def test_missing_earthdata_token_error_is_unchanged_and_a_user_token_is_used(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    home = tmp_path / "home"
    monkeypatch.setattr("atlas.agent.layout.Path.home", lambda: home)
    monkeypatch.setattr(fldas_module, "load_repo_env", lambda: None)
    client = FldasClient()
    with pytest.raises(FldasAuthError) as excinfo:
        await client.extract_training_table(_pull(), [_scene()], variables=["Evap_tavg"])
    assert str(excinfo.value) == MISSING_TOKEN_MESSAGE
    assert "ed-user-token" not in str(excinfo.value)

    write_user_key("ed-user-token", home, name=EARTHDATA_TOKEN)
    monkeypatch.setenv(EARTHDATA_TOKEN, "ed-env-token")
    bind_session_credentials({EARTHDATA_TOKEN: "ed-session-token"})
    assert FldasClient()._token == "ed-session-token"
    bind_session_credentials({})
    assert FldasClient()._token == "ed-env-token"
    monkeypatch.delenv(EARTHDATA_TOKEN, raising=False)
    assert FldasClient()._token == "ed-user-token"
    assert FldasClient(token="explicit-token")._token == "explicit-token"


def test_a_saved_user_token_satisfies_the_fldas_precheck_without_printing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    home = tmp_path / "home"
    secret = "ed-user-token"
    write_user_key(secret, home, name=EARTHDATA_TOKEN)
    monkeypatch.setattr("atlas.agent.layout.Path.home", lambda: home)

    async def _pull_ok(**_kwargs: object) -> int:
        return 0

    monkeypatch.setattr("atlas.cli.data.execute_pull", _pull_ok)
    code = run_data(
        _namespace(
            tmp_path,
            satellite="fldas",
            kind="landsurface",
            date="2020-01-01:2020-01-31",
            coords="-71.2,42.2,-70.9,42.5",
        )
    )
    assert code == 0
    captured = capsys.readouterr()
    assert MISSING_TOKEN_MESSAGE not in captured.err
    assert secret not in captured.out
    assert secret not in captured.err


def _namespace(tmp_path: Path, **overrides: object) -> Any:
    import argparse

    values: dict[str, Any] = {
        "list_catalog": False,
        "satellite": "fldas",
        "kind": "landsurface",
        "date": "2020-01-01:2020-01-31",
        "coords": "-71.2,42.2,-70.9,42.5",
        "polygon": None,
        "out": tmp_path,
        "limit": 1,
        "min_cloud_cover": None,
        "max_cloud_cover": None,
        "concurrency": 2,
        "search_only": False,
        "asset": None,
        "table": None,
        "variables": None,
        "points": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)
