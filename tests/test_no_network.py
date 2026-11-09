"""Proof that the default suite cannot reach the network.

AGENTS.md states that default tests never call OpenRouter, Planetary Computer,
or any other network. Nothing enforced that before ``tests/conftest.py`` grew
``_forbid_network``: a test that dialled out paid real latency and could pass
or fail with the weather. These tests fail if that guard is removed.

Every address here is RFC 5737 TEST-NET-1, which is reserved and never routed,
so a regression reaches no real host.
"""

from __future__ import annotations

import shutil
import socket
import tempfile
from pathlib import Path

import httpx
import pytest

_REFUSED = "must not use the network"
_UNROUTABLE = "192.0.2.1"


def test_a_raw_socket_cannot_reach_the_network() -> None:
    with pytest.raises(AssertionError, match=_REFUSED):
        socket.create_connection((_UNROUTABLE, 80), timeout=1)


def test_an_http_client_cannot_reach_the_network() -> None:
    """The guard sits under httpx too, so a client call cannot slip past it."""
    with pytest.raises(AssertionError, match=_REFUSED):
        httpx.get(f"http://{_UNROUTABLE}:80/", timeout=1)


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="AF_UNIX is unavailable")
def test_local_unix_sockets_still_connect() -> None:
    """Local IPC is not a network call, so the guard leaves it alone.

    The socket lives in a short ``/tmp`` directory: an AF_UNIX path is capped
    near 104 bytes, and pytest's ``tmp_path`` is already longer than that on
    macOS.
    """
    short = Path(tempfile.mkdtemp(prefix="atlas", dir=tempfile.gettempdir()))
    path = short / "listener.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        listener.bind(str(path))
        listener.listen(1)
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            client.connect(str(path))
        finally:
            client.close()
    finally:
        listener.close()
        shutil.rmtree(short, ignore_errors=True)
