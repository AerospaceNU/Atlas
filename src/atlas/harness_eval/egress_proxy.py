"""Allowlisting HTTPS proxy for benchmark rollout containers.

Rollout containers sit on a Docker network with no route out. This proxy is
their only way to the internet, and it tunnels ``CONNECT`` requests to the
allowed hosts on port 443 and refuses everything else. That keeps a rollout
able to call its model while a package index, a code host or a search engine
stays out of reach: any of them can hand the agent the upstream fix.

Standard library only, and it imports nothing from Atlas, so it runs as a
plain script in any Python image::

    python egress_proxy.py --allow openrouter.ai --port 8080
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import socket

_MAX_HEADER_BYTES = 16_384
_CONNECT_ATTEMPTS = 4
_CONNECT_TIMEOUT_SECONDS = 10


def allowed(host: str, allow: tuple[str, ...]) -> bool:
    """Return whether ``host`` is an allowed name or a subdomain of one."""
    name = host.lower().rstrip(".")
    return any(name == entry or name.endswith("." + entry) for entry in allow)


async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while data := await reader.read(65_536):
            writer.write(data)
            await writer.drain()
    except (ConnectionError, asyncio.CancelledError):
        pass
    finally:
        with contextlib.suppress(Exception):
            writer.close()


async def _refuse(writer: asyncio.StreamWriter, status: str) -> None:
    writer.write(f"HTTP/1.1 {status}\r\nConnection: close\r\n\r\n".encode())
    with contextlib.suppress(ConnectionError):
        await writer.drain()
    writer.close()


async def _connect(host: str) -> tuple[asyncio.StreamReader, asyncio.StreamWriter] | None:
    """Open the upstream connection over IPv4, retrying a transient failure.

    IPv4 only: a container network usually has no IPv6 route, and an address
    that cannot be reached there hangs instead of failing.
    """
    for attempt in range(1, _CONNECT_ATTEMPTS + 1):
        try:
            return await asyncio.wait_for(
                asyncio.open_connection(host, 443, family=socket.AF_INET),
                timeout=_CONNECT_TIMEOUT_SECONDS,
            )
        except (OSError, TimeoutError) as exc:
            print(f"upstream {host} attempt {attempt} failed: {exc!r}", flush=True)
            await asyncio.sleep(attempt)
    return None


async def handle(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter, allow: tuple[str, ...]
) -> None:
    """Serve one client connection: tunnel an allowed ``CONNECT``, refuse the rest."""
    try:
        head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=30)
    except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, TimeoutError):
        writer.close()
        return
    parts = head.split(b"\r\n", 1)[0].decode("latin-1").split()
    if len(parts) < 2 or parts[0].upper() != "CONNECT":
        await _refuse(writer, "403 Forbidden")
        return
    host, _, port = parts[1].rpartition(":")
    if port != "443" or not allowed(host, allow):
        print(f"refused {parts[1]}", flush=True)
        await _refuse(writer, "403 Forbidden")
        return
    upstream = await _connect(host)
    if upstream is None:
        await _refuse(writer, "502 Bad Gateway")
        return
    upstream_reader, upstream_writer = upstream
    writer.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
    await writer.drain()
    await asyncio.gather(_pipe(reader, upstream_writer), _pipe(upstream_reader, writer))


async def serve(port: int, allow: tuple[str, ...]) -> None:
    """Listen on every interface and serve until cancelled."""
    server = await asyncio.start_server(
        lambda reader, writer: handle(reader, writer, allow),
        "0.0.0.0",
        port,
        limit=_MAX_HEADER_BYTES,
    )
    print(f"egress proxy on :{port}, allowing {', '.join(allow)}", flush=True)
    async with server:
        await server.serve_forever()


def main(argv: list[str] | None = None) -> int:
    """Run the proxy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow", action="append", required=True, help="allowed host")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args(argv)
    asyncio.run(serve(args.port, tuple(entry.lower() for entry in args.allow)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
