"""A missing search credential is a source error that names the variable."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from urllib.parse import quote

import httpx
import pytest

from atlas.agent.layout import EARTHDATA_TOKEN, FIRMS_MAP_KEY
from atlas.compile.aggregate import aggregate
from atlas.data.base import BBox, PullRequest
from atlas.data.cdse import CdseSentinel3OlciClient
from atlas.data.cmr_stac import GediL2AClient, IceSat2Atl03Client
from atlas.data.credentials import (
    CDSE_PASSWORD,
    CDSE_USERNAME,
    MissingCredentialError,
)
from atlas.data.firms import FirmsClient
from atlas.data.modis_vegetation import MODISVegetationClient
from atlas.data.sources import fldas as fldas_module
from atlas.data.sources.fldas import FldasClient

_SECRET = "planted-credential-value"
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
def isolated_credentials(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Hide a developer key file and the credential variables under test."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr("atlas.agent.layout.Path.home", lambda: home)
    monkeypatch.setattr(fldas_module, "load_repo_env", lambda: None)
    for name in (
        EARTHDATA_TOKEN,
        FIRMS_MAP_KEY,
        CDSE_USERNAME,
        CDSE_PASSWORD,
        "EARTHDATA_USERNAME",
        "EARTHDATA_PASSWORD",
    ):
        monkeypatch.delenv(name, raising=False)


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
    assert error == "403 Forbidden"
    assert error is not None
    assert _SECRET not in error
    assert "FIRMS_MAP_KEY is not set" not in error


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
    assert error == "403 Forbidden"
    assert error is not None
    assert _SECRET not in error
    assert "EARTHDATA_TOKEN is not set" not in error


@pytest.mark.asyncio
@pytest.mark.usefixtures("isolated_credentials")
async def test_fldas_explicit_token_is_not_named_missing() -> None:
    client = FldasClient(
        token=_SECRET,
        client=httpx.AsyncClient(transport=_auth_transport(401)),
    )
    catalog = await aggregate(_request(), {"fldas": client})
    error = catalog.sources[0].error
    assert error == "401 Unauthorized"
    assert error is not None
    assert _SECRET not in error


@pytest.mark.asyncio
@pytest.mark.usefixtures("isolated_credentials")
async def test_missing_cdse_pair_is_a_source_error_without_the_value() -> None:
    client = CdseSentinel3OlciClient(client=httpx.AsyncClient(transport=_auth_transport(403)))
    catalog = await aggregate(_request(), {"cdse_sentinel3_olci": client})
    _assert_names_secret(catalog.sources[0].error, CDSE_USERNAME, CDSE_PASSWORD)
    assert "403" not in (catalog.sources[0].error or "")


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
    assert error is not None
    assert "CDSE_PASSWORD" not in error


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
    assert error == "401 Unauthorized"
    assert error is not None
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
    assert str(error) == "EARTHDATA_TOKEN is not set"
    assert error.names == (EARTHDATA_TOKEN,)
    assert _SECRET not in str(error)
    blank = MissingCredentialError(_SECRET)
    assert str(blank) == "credential is not set"
    assert _SECRET not in str(blank)
