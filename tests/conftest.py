from __future__ import annotations

import pytest

from atlas.agent.model import clear_model_catalog_cache


@pytest.fixture(autouse=True)
def _clear_openrouter_catalog_cache() -> None:
    """Keep one test's catalog cache from answering the next test."""
    clear_model_catalog_cache()


@pytest.fixture(autouse=True)
def _clear_atlas_home(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep a developer ``ATLAS_HOME`` from redirecting tests that pass a home."""
    monkeypatch.delenv("ATLAS_HOME", raising=False)


@pytest.fixture(autouse=True)
def _clear_session_credentials():
    """Drop a previous test's in-memory Earthdata or FIRMS session key."""
    from atlas.agent.layout import unbind_session_credentials, unbind_session_secrets

    unbind_session_credentials()
    unbind_session_secrets()
    yield
    unbind_session_credentials()
    unbind_session_secrets()
