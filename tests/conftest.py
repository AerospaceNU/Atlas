from __future__ import annotations

import pytest

from atlas.agent.model import clear_model_catalog_cache


@pytest.fixture(autouse=True)
def _clear_openrouter_catalog_cache() -> None:
    """Keep one test's catalog cache from answering the next test."""
    clear_model_catalog_cache()


@pytest.fixture(autouse=True)
def _clear_session_credentials():
    """Drop a previous test's in-memory Earthdata or FIRMS session key."""
    from atlas.agent.layout import unbind_session_credentials

    unbind_session_credentials()
    yield
    unbind_session_credentials()
