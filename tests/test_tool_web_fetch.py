from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from atlas.agent.artifacts import LocalArtifactStore, SandboxDenied
from atlas.agent.contracts import build_registry, default_registry
from atlas.agent.tools.web_fetch import WebFetchTool


def _handler(response: httpx.Response, calls: list[httpx.Request] | None = None):
    def handle(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(request)
        return response

    return handle


def _tool(handler: Any) -> WebFetchTool:
    return WebFetchTool(
        httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)
    )


def _html(body: str, status: int = 200) -> httpx.Response:
    return httpx.Response(status, headers={"content-type": "text/html; charset=utf-8"}, text=body)


def test_web_fetch_tool_is_opt_in() -> None:
    names = [definition.name for definition in default_registry().definitions]
    opted_in = [
        definition.name
        for definition in build_registry(("atlas.agent.tools",), include_opt_in=True).definitions
    ]

    assert WebFetchTool.trust == "opt_in"
    assert "web_fetch" not in names
    assert "web_fetch" in opted_in


def test_returns_the_body_as_text(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    result = _tool(_handler(_html("<p>hello</p>"))).run({"url": "https://example.com/page"}, store)

    assert "HTTP 200" in result.text
    assert "text/html" in result.text
    assert result.text.endswith("<p>hello</p>")
    assert result.artifacts == []


def test_does_not_follow_a_redirect(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    calls: list[httpx.Request] = []
    redirect = httpx.Response(302, headers={"location": "https://elsewhere.example/other"})
    result = _tool(_handler(redirect, calls)).run({"url": "https://example.com/old"}, store)

    assert "Redirect not followed" in result.text
    assert "https://elsewhere.example/other" in result.text
    assert [str(call.url) for call in calls] == ["https://example.com/old"]


def test_reports_an_http_error(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    tool = _tool(_handler(httpx.Response(500, text="boom")))

    with pytest.raises(RuntimeError, match="HTTP 500"):
        tool.run({"url": "https://example.com/broken"}, store)


def test_refuses_a_non_text_response(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    response = httpx.Response(
        200, headers={"content-type": "application/octet-stream"}, content=b"\x00\x01"
    )

    with pytest.raises(ValueError, match="non-text"):
        _tool(_handler(response)).run({"url": "https://example.com/blob"}, store)


def test_truncates_a_large_body(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    oversized = "x" * (WebFetchTool.max_bytes + 5_000)
    result = _tool(_handler(_html(oversized))).run({"url": "https://example.com/big"}, store)

    assert f"truncated at {WebFetchTool.max_bytes} bytes" in result.text
    assert len(result.text) < WebFetchTool.max_bytes + 200


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.com/x",
        "https://",
        "https://user:secret@example.com/",
    ],
)
def test_rejects_unfetchable_urls(tmp_path: Path, url: str) -> None:
    store = LocalArtifactStore(tmp_path)
    tool = _tool(_handler(_html("never returned")))

    with pytest.raises(ValueError):
        tool.run({"url": url}, store)


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "169.254.169.254"])
def test_rejects_local_hosts(tmp_path: Path, host: str) -> None:
    store = LocalArtifactStore(tmp_path)
    tool = _tool(_handler(_html("never returned")))

    with pytest.raises(SandboxDenied):
        tool.run({"url": f"http://{host}/admin"}, store)


def test_strips_terminal_escape_sequences(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    result = _tool(_handler(_html("\x1b[31mred\x1b[0m"))).run(
        {"url": "https://example.com/page"}, store
    )

    assert "red" in result.text
    assert "\x1b" not in result.text


@pytest.mark.parametrize("timeout", [0, 300])
def test_bounds_the_timeout(tmp_path: Path, timeout: int) -> None:
    store = LocalArtifactStore(tmp_path)
    tool = _tool(_handler(_html("never returned")))

    with pytest.raises(ValidationError):
        tool.run({"url": "https://example.com/page", "timeout": timeout}, store)


def test_reads_a_body_without_a_content_type(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    response = httpx.Response(200, content=b"plain bytes, no declared type")
    result = _tool(_handler(response)).run({"url": "https://example.com/opaque"}, store)

    assert "unspecified content type" in result.text
    assert result.text.endswith("plain bytes, no declared type")
