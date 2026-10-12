"""ATLAS-104: store EARTHDATA_TOKEN and FIRMS_MAP_KEY like the OpenRouter key."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import traceback
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import (
    ChatMessage,
    ModelResponse,
    Tool,
    ToolCall,
    ToolRegistry,
    ToolResult,
)
from atlas.agent.layout import (
    EARTHDATA_TOKEN,
    FIRMS_MAP_KEY,
    CredentialKeyWarning,
    bind_session_credentials,
    bind_session_secrets,
    resolve_credential,
    unbind_session_secrets,
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
# Shapes checked against the FIRMS area tutorial (32 hex characters) and the
# Earthdata Login user-token docs (a JWT of three base64url segments).
_TOKENS = {
    EARTHDATA_TOKEN: "eyJhbGciOiJSUzI1NiJ9.eyJzZXNzaW9uIjoidG9rZW4ifQ.c2lnbmF0dXJl",
    FIRMS_MAP_KEY: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
}
_USER_TOKENS = {
    EARTHDATA_TOKEN: "eyJhbGciOiJSUzI1NiJ9.eyJ1c2VyIjoidG9rZW4ifQ.c2lnbmF0dXJl",
    FIRMS_MAP_KEY: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
}
_LATER_EARTHDATA = "eyJhbGciOiJSUzI1NiJ9.eyJsYXRlciI6InVzZXIifQ.c2lnbmF0dXJl"
_PLANTED_EARTHDATA = "eyJhbGciOiJSUzI1NiJ9.eyJwbGFudGVkIjoidG9rZW4ifQ.c2lnbmF0dXJl"
_FIRMS_DELETE = "dddddddddddddddddddddddddddddddd"
_FIRMS_ENV = "cccccccccccccccccccccccccccccccc"


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
    write_user_key(f"  {_USER_TOKENS[name]}  ", home, name=name)

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
    assert resolve_credential(name, session_key=None, env_key=" ", home=home) == _USER_TOKENS[name]
    path = home / _FILES[name]
    assert path.is_file()
    assert path.read_text(encoding="utf-8") == f"{_USER_TOKENS[name]}\n"
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
    user_secret = _USER_TOKENS[name]
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
            _set_key(EARTHDATA_TOKEN, "session", _TOKENS[EARTHDATA_TOKEN]),
            _set_key(EARTHDATA_TOKEN, "user", _LATER_EARTHDATA),
        ],
    )
    assert session.credential_keys[EARTHDATA_TOKEN] == _TOKENS[EARTHDATA_TOKEN]
    assert (home / _FILES[EARTHDATA_TOKEN]).read_text(encoding="utf-8").strip() == _LATER_EARTHDATA
    dumped = json.dumps(events)
    assert _TOKENS[EARTHDATA_TOKEN] not in dumped
    assert _LATER_EARTHDATA not in dumped
    assert _LATER_EARTHDATA not in _project_text(project)
    bind_session_credentials(session.credential_keys)
    assert resolve_credential(EARTHDATA_TOKEN, home=home) == _TOKENS[EARTHDATA_TOKEN]


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

    events = _run(session, [_set_key(EARTHDATA_TOKEN, "user", _PLANTED_EARTHDATA)])

    assert outside.read_text(encoding="utf-8") == "original"
    assert events[1]["message"] == "Could not save EARTHDATA_TOKEN: AtlasPathError"
    assert _PLANTED_EARTHDATA not in json.dumps(events)
    assert EARTHDATA_TOKEN not in session.credential_keys


def test_private_key_file_stays_mode_600_when_the_umask_is_permissive(tmp_path: Path) -> None:
    if os.name != "posix":
        pytest.skip("Windows does not enforce Unix file modes")
    home = tmp_path / "home"
    previous = os.umask(0)
    try:
        path = write_user_key(_USER_TOKENS[FIRMS_MAP_KEY], home, name=FIRMS_MAP_KEY)
    finally:
        os.umask(previous)
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700


def test_read_warns_when_the_key_file_is_too_loose(tmp_path: Path) -> None:
    if os.name != "posix":
        pytest.skip("Windows does not report Unix permission bits")
    home = tmp_path / "home"
    path = write_user_key(_USER_TOKENS[EARTHDATA_TOKEN], home, name=EARTHDATA_TOKEN)
    os.chmod(path, 0o644)
    with pytest.warns(CredentialKeyWarning, match="EARTHDATA_TOKEN file permissions are too loose"):
        assert (
            resolve_credential(EARTHDATA_TOKEN, env_key=" ", home=home)
            == (_USER_TOKENS[EARTHDATA_TOKEN])
        )


def test_delete_removes_the_file_and_keeps_redacting_the_value(tmp_path: Path) -> None:
    session, home, project = _session(tmp_path)
    secret = _FIRMS_DELETE
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
    write_user_key(_USER_TOKENS[FIRMS_MAP_KEY], home, name=FIRMS_MAP_KEY)
    monkeypatch.setattr("atlas.agent.layout.Path.home", lambda: home)
    monkeypatch.setenv(FIRMS_MAP_KEY, _FIRMS_ENV)
    bind_session_credentials({FIRMS_MAP_KEY: _TOKENS[FIRMS_MAP_KEY]})
    transport = _Gets()
    client = FirmsClient(client=transport)  # type: ignore[arg-type]
    result = await client.search(_pull())
    assert len(result.scenes) == 1
    assert transport.urls
    assert _TOKENS[FIRMS_MAP_KEY] in transport.urls[0]
    assert _FIRMS_ENV not in transport.urls[0]
    assert _USER_TOKENS[FIRMS_MAP_KEY] not in transport.urls[0]

    bind_session_credentials({})
    monkeypatch.delenv(FIRMS_MAP_KEY, raising=False)
    explicit = FirmsClient(map_key="", client=transport)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="FIRMS_MAP_KEY is not set") as excinfo:
        await explicit.search(_pull())
    assert str(excinfo.value) == "FIRMS_MAP_KEY is not set"
    assert _TOKENS[FIRMS_MAP_KEY] not in str(excinfo.value)

    from_file = FirmsClient()
    assert from_file._map_key == _USER_TOKENS[FIRMS_MAP_KEY]


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
    assert _USER_TOKENS[EARTHDATA_TOKEN] not in str(excinfo.value)

    write_user_key(_USER_TOKENS[EARTHDATA_TOKEN], home, name=EARTHDATA_TOKEN)
    monkeypatch.setenv(EARTHDATA_TOKEN, "ed-env-token")
    bind_session_credentials({EARTHDATA_TOKEN: _TOKENS[EARTHDATA_TOKEN]})
    assert FldasClient()._token == _TOKENS[EARTHDATA_TOKEN]
    bind_session_credentials({})
    assert FldasClient()._token == "ed-env-token"
    monkeypatch.delenv(EARTHDATA_TOKEN, raising=False)
    assert FldasClient()._token == _USER_TOKENS[EARTHDATA_TOKEN]
    assert FldasClient(token="explicit-token")._token == "explicit-token"


def test_a_saved_user_token_satisfies_the_fldas_precheck_without_printing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    home = tmp_path / "home"
    secret = _USER_TOKENS[EARTHDATA_TOKEN]
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


class _Empty(BaseModel):
    pass


class _ScriptedModel:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = responses

    def complete(self, messages: list[Any], tools: list[Any]) -> ModelResponse:
        del messages, tools
        return self.responses.pop(0)


class _EchoKeys(Tool):
    name = "echo_keys"
    description = "Return the supplied credential text."

    def __init__(self, text: str) -> None:
        self._text = text

    @property
    def input_model(self) -> type[BaseModel]:
        return _Empty

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        del arguments, store
        return ToolResult(text=self._text)


class _FirmsSearch(Tool):
    name = "firms_search"
    description = "Search FIRMS and surface an HTTP failure."

    def __init__(self, map_key: str) -> None:
        self._map_key = map_key

    @property
    def input_model(self) -> type[BaseModel]:
        return _Empty

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        del arguments, store
        asyncio.run(_failing_firms_search(self._map_key))
        return ToolResult(text="ok")


class _Forbidden:
    async def get(self, url: str) -> httpx.Response:
        return httpx.Response(403, request=httpx.Request("GET", url))


async def _failing_firms_search(map_key: str) -> None:
    client = FirmsClient(map_key=map_key, client=_Forbidden())  # type: ignore[arg-type]
    await client.search(_pull())


def test_a_firms_http_error_in_a_tool_result_does_not_expose_the_key(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    key = _TOKENS[FIRMS_MAP_KEY]
    project = tmp_path / "project"
    project.mkdir()
    registry = ToolRegistry()
    registry.register(_FirmsSearch(key))
    model = _ScriptedModel(
        [
            ModelResponse(tool_calls=[ToolCall(id="call-1", name="firms_search", arguments={})]),
            ModelResponse(content="done"),
        ]
    )
    agent = Agent(model, registry, LocalArtifactStore(project))
    messages = [ChatMessage(role="user", content="search fires")]

    with caplog.at_level(logging.DEBUG):
        result = agent.advance(messages)

    assert result.steps[0].error is not None
    assert key not in result.steps[0].error
    assert "403" in result.steps[0].error
    tool_text = next(message.content or "" for message in messages if message.role == "tool")
    assert key not in tool_text

    captured = ""
    error: httpx.HTTPStatusError | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(403)

    async def logged() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http:
            client = FirmsClient(map_key=key, client=http)
            await client.search(_pull())

    with caplog.at_level(logging.DEBUG):
        try:
            asyncio.run(logged())
        except httpx.HTTPStatusError as exc:
            error = exc
            captured = traceback.format_exc()
    assert error is not None
    assert key not in str(error)
    assert key not in str(error.request.url)
    assert key not in captured
    assert "HTTPStatusError" in captured
    assert key not in caplog.text
    assert logging.getLogger("httpx").level >= logging.WARNING


def test_env_and_file_keys_are_redacted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    firms = _TOKENS[FIRMS_MAP_KEY]
    earth = _USER_TOKENS[EARTHDATA_TOKEN]
    home = tmp_path / "home"
    write_user_key(earth, home, name=EARTHDATA_TOKEN)
    monkeypatch.setenv(FIRMS_MAP_KEY, firms)
    monkeypatch.setattr("atlas.agent.layout.Path.home", lambda: home)

    remembered: list[str] = []
    bind_session_secrets(remembered)
    assert resolve_credential(FIRMS_MAP_KEY, home=home) == firms
    assert resolve_credential(EARTHDATA_TOKEN, env_key=" ", home=home) == earth
    assert remembered == [firms, earth]
    unbind_session_secrets()

    project = tmp_path / "project"
    project.mkdir()
    registry = ToolRegistry()
    registry.register(_EchoKeys(f"{firms}\n{earth}"))
    model = _ScriptedModel(
        [
            ModelResponse(tool_calls=[ToolCall(id="call-1", name="echo_keys", arguments={})]),
            ModelResponse(content="done"),
        ]
    )
    session = AgentSession(Agent(model, registry, LocalArtifactStore(project)))
    session.home = home
    session.workspace = project
    events = _run(
        session,
        [
            json.dumps({"type": "user", "text": "show the keys"}),
            json.dumps({"type": "quit"}),
        ],
    )
    dumped = json.dumps(events)
    assert firms not in dumped
    assert earth not in dumped
    assert firms in session.secrets
    assert earth in session.secrets
    step = next(event for event in events if event["type"] == "step")
    assert step["text"] == "***\n***"


def test_a_malformed_firms_key_is_rejected(tmp_path: Path) -> None:
    session, home, project = _session(tmp_path)
    malformed = "not-a-firms-key"
    events = _run(session, [_set_key(FIRMS_MAP_KEY, "session", malformed)])
    dumped = json.dumps(events)
    assert malformed not in dumped
    assert malformed not in _project_text(project)
    assert FIRMS_MAP_KEY not in session.credential_keys
    assert not (home / _FILES[FIRMS_MAP_KEY]).exists()
    assert events[1] == {
        "type": "error",
        "message": "FIRMS_MAP_KEY must be 32 hexadecimal characters",
    }
    with pytest.raises(ValueError, match="32 hexadecimal") as excinfo:
        write_user_key("gggggggggggggggggggggggggggggggg", home, name=FIRMS_MAP_KEY)
    assert "gggggggggggggggggggggggggggggggg" not in str(excinfo.value)
    assert not (home / _FILES[FIRMS_MAP_KEY]).exists()


@pytest.mark.asyncio
async def test_firms_percent_encodes_the_map_key() -> None:
    transport = _Gets()
    client = FirmsClient(map_key="abc/def", client=transport)  # type: ignore[arg-type]
    await client.search(_pull())
    assert transport.urls
    assert "/abc%2Fdef/" in transport.urls[0]
    assert "/abc/def/" not in transport.urls[0]


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
