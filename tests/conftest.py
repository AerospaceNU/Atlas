from __future__ import annotations

from typing import NoReturn

import pytest

from atlas.agent.model import clear_model_catalog_cache

_NETWORK_REFUSED = "default tests must not use the network"


@pytest.fixture(autouse=True)
def _clear_openrouter_catalog_cache() -> None:
    """Keep one test's catalog cache from answering the next test."""
    clear_model_catalog_cache()


@pytest.fixture(autouse=True)
def _clear_atlas_home(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep a developer ``ATLAS_HOME`` from redirecting tests that pass a home."""
    monkeypatch.delenv("ATLAS_HOME", raising=False)


@pytest.fixture(autouse=True)
def _forbid_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuse an INET connect from any test that is not marked ``evals``.

    AGENTS.md states the default suite never calls OpenRouter, Planetary
    Computer, or any other network, but nothing enforced it: a test that dialled
    out paid real latency and could pass or fail with the weather. Tests marked
    ``evals`` opt in to live APIs and keep their sockets.
    """
    if request.node.get_closest_marker("evals") is not None:
        return

    import socket

    inet = (socket.AF_INET, socket.AF_INET6)
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def refuse(address: object) -> NoReturn:
        raise AssertionError(f"{_NETWORK_REFUSED}; a socket dialled {address!r}")

    def connect(self: socket.socket, address: object) -> None:
        if self.family in inet:
            refuse(address)
        real_connect(self, address)

    def connect_ex(self: socket.socket, address: object) -> int:
        if self.family in inet:
            refuse(address)
        return real_connect_ex(self, address)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)


@pytest.fixture(autouse=True)
def _clear_session_credentials():
    """Drop a previous test's in-memory Earthdata or FIRMS session key."""
    from atlas.agent.layout import unbind_session_credentials, unbind_session_secrets

    unbind_session_credentials()
    unbind_session_secrets()
    yield
    unbind_session_credentials()
    unbind_session_secrets()
