"""A missing search credential is a source error that names the variable."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

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
    bind_session_credentials,
    unbind_session_credentials,
    write_user_key,
)
from atlas.agent.runtime import Agent
from atlas.cli.data import add_data_parser, run_data
from atlas.compile.aggregate import aggregate
from atlas.compile.product import SceneCatalog, SourceScenes
from atlas.data.base import BBox, DataPullClient, PullRequest, PullResult, Scene
from atlas.data.cdse import CdseSentinel3OlciClient
from atlas.data.cmr_stac import GediL2AClient, IceSat2Atl03Client
from atlas.data.credentials import (
    CDSE_PASSWORD,
    CDSE_USERNAME,
    CredentialRejectedError,
    MissingCredentialError,
)
from atlas.data.firms import FirmsClient
from atlas.data.modis_vegetation import MODISVegetationClient
from atlas.data.sources import fldas as fldas_module
from atlas.data.sources.fldas import FldasClient

_SECRET = "planted-credential-value"
_EARTH_MISSING = "EARTHDATA_TOKEN is not set; set it with /key earthdata or the env var"
_FIRMS_MISSING = "FIRMS_MAP_KEY is not set; set it with /key firms or the env var"
_CDSE_BOTH_MISSING = "CDSE_USERNAME and CDSE_PASSWORD are not set; set the env vars"
_CDSE_USER_MISSING = "CDSE_USERNAME is not set; set the env var"
_SESSION_EARTH = "eyJhbGciOiJSUzI1NiJ9.eyJzZXNzaW9uIjoidG9rZW4ifQ.c2lnbmF0dXJl"
_FILE_EARTH = "eyJhbGciOiJSUzI1NiJ9.eyJ1c2VyIjoidG9rZW4ifQ.c2lnbmF0dXJl"
_SESSION_FIRMS = "0123456789abcdef0123456789abcdef"
_FILE_FIRMS = "fedcba9876543210fedcba9876543210"
_AOI = BBox(west=-71.12, south=42.32, east=-71.02, north=42.40)
_CDSE_CURSOR = json.dumps(
    {
        "body": {"token": "cdse-page"},
        "merge": True,
        "method": "POST",
    },
    sort_keys=True,
    separators=(",", ":"),
)


def _request() -> PullRequest:
    return PullRequest(
        start_date=date(2024, 7, 1),
        end_date=date(2024, 7, 3),
        bbox=_AOI,
        limit=2,
    )


def _auth_transport(status: int) -> httpx.MockTransport:
    """Return ``status`` and plant the secret in the URL and body."""

    def handler(request: httpx.Request) -> httpx.Response:
        leaked = httpx.Request(request.method, f"{request.url}?leak={quote(_SECRET)}")
        return httpx.Response(status, request=leaked, text=_SECRET)

    return httpx.MockTransport(handler)


@pytest.fixture
def isolated_credentials(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Hide a developer key file and the credential variables under test."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr("atlas.agent.layout.Path.home", lambda: home)
    monkeypatch.setattr(fldas_module, "load_repo_env", lambda: None)
    unbind_session_credentials()
    for name in (
        EARTHDATA_TOKEN,
        FIRMS_MAP_KEY,
        CDSE_USERNAME,
        CDSE_PASSWORD,
        "EARTHDATA_USERNAME",
        "EARTHDATA_PASSWORD",
    ):
        monkeypatch.delenv(name, raising=False)
    return home


def _assert_names_secret(error: str | None, *names: str) -> None:
    assert error is not None
    for name in names:
        assert name in error
    assert "is not set" in error or "are not set" in error
    assert _SECRET not in error
    assert "://" not in error


@pytest.mark.asyncio
@pytest.mark.usefixtures("isolated_credentials")
async def test_missing_firms_key_is_a_source_error_without_the_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(FIRMS_MAP_KEY, _SECRET)
    client = FirmsClient(map_key="   ", client=httpx.AsyncClient(transport=_auth_transport(401)))
    catalog = await aggregate(_request(), {"firms_viirs": client})
    _assert_names_secret(catalog.sources[0].error, FIRMS_MAP_KEY)
    assert catalog.sources[0].error == _FIRMS_MISSING
    assert catalog.sources[0].scenes == []


@pytest.mark.asyncio
@pytest.mark.usefixtures("isolated_credentials")
async def test_firms_provider_error_keeps_the_status_and_hides_the_key() -> None:
    client = FirmsClient(
        map_key=_SECRET,
        client=httpx.AsyncClient(transport=_auth_transport(403)),
    )
    catalog = await aggregate(_request(), {"firms_viirs": client})
    error = catalog.sources[0].error
    assert error == "FIRMS_MAP_KEY was rejected (HTTP 403)"
    assert _SECRET not in error
    assert "is not set" not in error


@pytest.mark.asyncio
@pytest.mark.usefixtures("isolated_credentials")
@pytest.mark.parametrize(
    "factory",
    [GediL2AClient, IceSat2Atl03Client, MODISVegetationClient, FldasClient],
    ids=["cmr-lpcloud", "cmr-nsidc", "modis", "fldas"],
)
async def test_missing_earthdata_token_replaces_the_provider_error(factory: type) -> None:
    client = factory(client=httpx.AsyncClient(transport=_auth_transport(401)))
    catalog = await aggregate(_request(), {"earthdata": client})
    _assert_names_secret(catalog.sources[0].error, EARTHDATA_TOKEN)
    assert catalog.sources[0].error == _EARTH_MISSING
    assert "401" not in (catalog.sources[0].error or "")


@pytest.mark.asyncio
@pytest.mark.usefixtures("isolated_credentials")
async def test_earthdata_token_that_is_set_stays_a_provider_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(EARTHDATA_TOKEN, _SECRET)
    client = GediL2AClient(client=httpx.AsyncClient(transport=_auth_transport(403)))
    catalog = await aggregate(_request(), {"gedi": client})
    error = catalog.sources[0].error
    assert error == "EARTHDATA_TOKEN was rejected (HTTP 403)"
    assert _SECRET not in error
    assert "is not set" not in error


@pytest.mark.asyncio
@pytest.mark.usefixtures("isolated_credentials")
async def test_fldas_explicit_token_is_not_named_missing() -> None:
    client = FldasClient(
        token=_SECRET,
        client=httpx.AsyncClient(transport=_auth_transport(401)),
    )
    catalog = await aggregate(_request(), {"fldas": client})
    error = catalog.sources[0].error
    assert error == "EARTHDATA_TOKEN was rejected (HTTP 401)"
    assert _SECRET not in error


@pytest.mark.asyncio
@pytest.mark.usefixtures("isolated_credentials")
async def test_missing_cdse_pair_is_a_source_error_without_the_value() -> None:
    client = CdseSentinel3OlciClient(client=httpx.AsyncClient(transport=_auth_transport(403)))
    catalog = await aggregate(_request(), {"cdse_sentinel3_olci": client})
    _assert_names_secret(catalog.sources[0].error, CDSE_USERNAME, CDSE_PASSWORD)
    assert catalog.sources[0].error == _CDSE_BOTH_MISSING
    assert "403" not in (catalog.sources[0].error or "")
    assert "/key" not in (catalog.sources[0].error or "")


@pytest.mark.asyncio
@pytest.mark.usefixtures("isolated_credentials")
async def test_one_missing_cdse_variable_names_only_that_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(CDSE_PASSWORD, _SECRET)
    client = CdseSentinel3OlciClient(client=httpx.AsyncClient(transport=_auth_transport(401)))
    catalog = await aggregate(_request(), {"cdse_sentinel3_olci": client})
    error = catalog.sources[0].error
    _assert_names_secret(error, CDSE_USERNAME)
    assert error == _CDSE_USER_MISSING
    assert "CDSE_PASSWORD" not in error
    assert "/key" not in error


@pytest.mark.asyncio
@pytest.mark.usefixtures("isolated_credentials")
async def test_cdse_pair_that_is_set_stays_a_provider_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(CDSE_USERNAME, "cdse-user-secret")
    monkeypatch.setenv(CDSE_PASSWORD, _SECRET)
    client = CdseSentinel3OlciClient(client=httpx.AsyncClient(transport=_auth_transport(401)))
    catalog = await aggregate(_request(), {"cdse_sentinel3_olci": client})
    error = catalog.sources[0].error
    assert error == "CDSE_USERNAME and CDSE_PASSWORD were rejected (HTTP 401)"
    assert _SECRET not in error
    assert "cdse-user-secret" not in error


@pytest.mark.asyncio
@pytest.mark.usefixtures("isolated_credentials")
async def test_cdse_resume_names_the_pair_and_keeps_the_cursor() -> None:
    client = CdseSentinel3OlciClient(client=httpx.AsyncClient(transport=_auth_transport(401)))
    catalog = await aggregate(
        _request(),
        {"cdse_sentinel3_olci": client},
        page_cursors={"cdse_sentinel3_olci": _CDSE_CURSOR},
    )
    source = catalog.sources[0]
    _assert_names_secret(source.error, CDSE_USERNAME, CDSE_PASSWORD)
    assert source.cursor_rejected is False
    assert source.next_cursor == _CDSE_CURSOR


@pytest.mark.asyncio
@pytest.mark.usefixtures("isolated_credentials")
async def test_server_error_is_not_relabeled_as_a_missing_credential() -> None:
    client = GediL2AClient(client=httpx.AsyncClient(transport=_auth_transport(500)))
    catalog = await aggregate(_request(), {"gedi": client})
    error = catalog.sources[0].error
    assert error == "500 Internal Server Error"
    assert error is not None
    assert _SECRET not in error
    assert EARTHDATA_TOKEN not in error


def test_credential_error_drops_a_value_passed_as_a_name() -> None:
    error = MissingCredentialError(_SECRET, EARTHDATA_TOKEN)
    assert str(error) == _EARTH_MISSING
    assert error.names == (EARTHDATA_TOKEN,)
    assert _SECRET not in str(error)
    blank = MissingCredentialError(_SECRET)
    assert str(blank) == "credential is not set"
    assert _SECRET not in str(blank)
    rejected = CredentialRejectedError(401, _SECRET, EARTHDATA_TOKEN)
    assert str(rejected) == "EARTHDATA_TOKEN was rejected (HTTP 401)"
    assert rejected.names == (EARTHDATA_TOKEN,)
    assert _SECRET not in str(rejected)
    rejected_blank = CredentialRejectedError(403, _SECRET)
    assert str(rejected_blank) == "credential was rejected (HTTP 403)"
    assert _SECRET not in str(rejected_blank)


def _by_source(catalog: SceneCatalog) -> dict[str, SourceScenes]:
    return {src.source: src for src in catalog.sources}


class _WorkingClient(DataPullClient):
    """One scene, with no credential and no network."""

    async def search(self, request: PullRequest) -> PullResult:
        scene = Scene(
            id="kept-scene",
            datetime=datetime(2024, 7, 1, tzinfo=UTC),
            bbox=_AOI,
            platform="sentinel-2",
        )
        return PullResult(request=request, scenes=[scene])


class _EmptyInput(BaseModel):
    pass


class _ScriptedModel:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = responses

    def complete(self, messages: list[Any], tools: list[Any]) -> ModelResponse:
        del messages, tools
        return self.responses.pop(0)


class _SecretSearch(Tool):
    """Run one search whose HTTP body and URL contain the planted secret."""

    name = "search_with_secret"
    description = "Search a source that may echo a credential."

    def __init__(self, search: Any) -> None:
        self._search = search

    @property
    def input_model(self) -> type[BaseModel]:
        return _EmptyInput

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        del arguments, store
        asyncio.run(self._search())
        return ToolResult(text="ok")


def _cli_args(tmp_path: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="atlas")
    sub = parser.add_subparsers(dest="group")
    add_data_parser(sub)
    return parser.parse_args(
        [
            "data",
            "firms_viirs",
            "detection",
            "--date",
            "2024-07-01:2024-07-03",
            "--coords",
            "0,0,1,1",
            "--out",
            str(tmp_path),
            "--search-only",
        ]
    )


@pytest.mark.asyncio
@pytest.mark.usefixtures("isolated_credentials")
async def test_session_bound_key_is_not_reported_missing() -> None:
    bind_session_credentials({EARTHDATA_TOKEN: _SESSION_EARTH, FIRMS_MAP_KEY: _SESSION_FIRMS})
    try:
        catalog = await aggregate(
            _request(),
            {
                "gedi": GediL2AClient(client=httpx.AsyncClient(transport=_auth_transport(401))),
                "modis_vegetation": MODISVegetationClient(
                    client=httpx.AsyncClient(transport=_auth_transport(401))
                ),
                "firms_viirs": FirmsClient(
                    client=httpx.AsyncClient(transport=_auth_transport(403))
                ),
            },
        )
    finally:
        unbind_session_credentials()
    found = _by_source(catalog)
    assert found["gedi"].error == "EARTHDATA_TOKEN was rejected (HTTP 401)"
    assert found["modis_vegetation"].error == "EARTHDATA_TOKEN was rejected (HTTP 401)"
    assert found["firms_viirs"].error == "FIRMS_MAP_KEY was rejected (HTTP 403)"
    rendered = " ".join(src.error or "" for src in catalog.sources)
    assert _SESSION_EARTH not in rendered
    assert _SESSION_FIRMS not in rendered
    assert "is not set" not in rendered


@pytest.mark.asyncio
@pytest.mark.usefixtures("isolated_credentials")
async def test_modis_token_resolved_at_init_is_not_reported_missing() -> None:
    bind_session_credentials({EARTHDATA_TOKEN: _SESSION_EARTH})
    try:
        client = MODISVegetationClient(client=httpx.AsyncClient(transport=_auth_transport(401)))
    finally:
        unbind_session_credentials()
    catalog = await aggregate(_request(), {"modis_vegetation": client})
    error = catalog.sources[0].error
    assert error == "EARTHDATA_TOKEN was rejected (HTTP 401)"
    assert _SESSION_EARTH not in (error or "")


@pytest.mark.asyncio
async def test_user_key_file_is_not_reported_missing(isolated_credentials: Path) -> None:
    write_user_key(_FILE_EARTH, isolated_credentials, name=EARTHDATA_TOKEN)
    write_user_key(_FILE_FIRMS, isolated_credentials, name=FIRMS_MAP_KEY)
    catalog = await aggregate(
        _request(),
        {
            "gedi": GediL2AClient(client=httpx.AsyncClient(transport=_auth_transport(401))),
            "modis_vegetation": MODISVegetationClient(
                client=httpx.AsyncClient(transport=_auth_transport(403))
            ),
            "firms_viirs": FirmsClient(client=httpx.AsyncClient(transport=_auth_transport(401))),
        },
    )
    found = _by_source(catalog)
    assert found["gedi"].error == "EARTHDATA_TOKEN was rejected (HTTP 401)"
    assert found["modis_vegetation"].error == "EARTHDATA_TOKEN was rejected (HTTP 403)"
    assert found["firms_viirs"].error == "FIRMS_MAP_KEY was rejected (HTTP 401)"
    rendered = " ".join(src.error or "" for src in catalog.sources)
    assert _FILE_EARTH not in rendered
    assert _FILE_FIRMS not in rendered
    assert "is not set" not in rendered


@pytest.mark.asyncio
@pytest.mark.usefixtures("isolated_credentials")
async def test_whitespace_only_env_var_counts_as_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(EARTHDATA_TOKEN, " \t")
    monkeypatch.setenv(FIRMS_MAP_KEY, "   ")
    monkeypatch.setenv(CDSE_USERNAME, " ")
    monkeypatch.setenv(CDSE_PASSWORD, "\t")
    firms_called = False

    def firms_handler(request: httpx.Request) -> httpx.Response:
        nonlocal firms_called
        firms_called = True
        return httpx.Response(401, request=request, text=_SECRET)

    catalog = await aggregate(
        _request(),
        {
            "gedi": GediL2AClient(client=httpx.AsyncClient(transport=_auth_transport(401))),
            "cdse_sentinel3_olci": CdseSentinel3OlciClient(
                client=httpx.AsyncClient(transport=_auth_transport(403))
            ),
            "firms_viirs": FirmsClient(
                client=httpx.AsyncClient(transport=httpx.MockTransport(firms_handler))
            ),
        },
    )
    found = _by_source(catalog)
    assert found["gedi"].error == _EARTH_MISSING
    assert found["cdse_sentinel3_olci"].error == _CDSE_BOTH_MISSING
    assert found["firms_viirs"].error == _FIRMS_MISSING
    assert firms_called is False
    rendered = " ".join(src.error or "" for src in catalog.sources)
    assert _SECRET not in rendered


@pytest.mark.asyncio
@pytest.mark.usefixtures("isolated_credentials")
async def test_failing_credential_source_keeps_the_working_source_scenes() -> None:
    catalog = await aggregate(
        _request(),
        {
            "gedi": GediL2AClient(client=httpx.AsyncClient(transport=_auth_transport(401))),
            "sentinel2": _WorkingClient(),
        },
    )
    found = _by_source(catalog)
    assert found["gedi"].error == _EARTH_MISSING
    assert found["gedi"].scenes == []
    assert found["sentinel2"].error is None
    assert [scene.id for scene in found["sentinel2"].scenes] == ["kept-scene"]


@pytest.mark.usefixtures("isolated_credentials")
def test_secret_is_absent_from_the_manifest_and_cli_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("atlas.cli.data.load_repo_env", lambda: None)
    clients = {
        "firms_viirs": FirmsClient(
            map_key=_SECRET,
            client=httpx.AsyncClient(transport=_auth_transport(401)),
        ),
        "gedi": GediL2AClient(client=httpx.AsyncClient(transport=_auth_transport(401))),
    }
    code = run_data(_cli_args(tmp_path), clients=clients)
    assert code == 1
    manifest = (tmp_path / "manifest.json").read_text(encoding="utf-8")
    captured = capsys.readouterr()
    visible = manifest + captured.out + captured.err
    assert _SECRET not in visible
    assert "FIRMS_MAP_KEY was rejected (HTTP 401)" in manifest
    assert _EARTH_MISSING in manifest
    assert "FIRMS_MAP_KEY was rejected (HTTP 401)" in captured.err
    assert _EARTH_MISSING in captured.err


@pytest.mark.usefixtures("isolated_credentials")
def test_secret_is_absent_from_tool_output(tmp_path: Path) -> None:
    async def firms_search() -> None:
        client = FirmsClient(
            map_key=_SECRET,
            client=httpx.AsyncClient(transport=_auth_transport(403)),
        )
        await client.search(_request())

    async def earthdata_search() -> None:
        client = GediL2AClient(client=httpx.AsyncClient(transport=_auth_transport(401)))
        await client.search(_request())

    project = tmp_path / "project"
    project.mkdir()
    firms_tool = _SecretSearch(firms_search)
    firms_tool.name = "search_firms"
    earth_tool = _SecretSearch(earthdata_search)
    earth_tool.name = "search_earthdata"
    registry = ToolRegistry()
    registry.register(firms_tool)
    registry.register(earth_tool)
    model = _ScriptedModel(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(id="call-1", name="search_firms", arguments={}),
                    ToolCall(id="call-2", name="search_earthdata", arguments={}),
                ]
            ),
            ModelResponse(content="done"),
        ]
    )
    agent = Agent(model, registry, LocalArtifactStore(project))
    messages = [ChatMessage(role="user", content="search")]
    result = agent.advance(messages)

    rendered = "\n".join(step.error or "" for step in result.steps)
    rendered += "\n".join(message.content or "" for message in messages if message.role == "tool")
    assert _SECRET not in rendered
    assert "FIRMS_MAP_KEY was rejected (HTTP 403)" in rendered
    assert _EARTH_MISSING in rendered
