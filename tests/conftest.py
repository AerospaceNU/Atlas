from __future__ import annotations

import pytest

from atlas.agent.model import clear_model_catalog_cache


@pytest.fixture(autouse=True)
def _clear_openrouter_catalog_cache() -> None:
    """Keep one test's catalog cache from answering the next test."""
    clear_model_catalog_cache()
