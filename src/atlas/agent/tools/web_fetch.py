from __future__ import annotations

import re
from typing import Any, ClassVar, Literal
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, Field

from atlas.agent.artifacts import LocalArtifactStore, SandboxDenied
from atlas.agent.contracts import Tool, ToolResult

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

_ALLOWED_SCHEMES = frozenset({"http", "https"})

# Literal loopback and link-local hosts only. A public hostname that resolves
# to a private address is not caught here; the class docstring records that gap.
_BLOCKED_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0", "169.254.169.254"})

_TEXT_MEDIA_TYPES = (
    "text/",
    "application/json",
    "application/xml",
    "application/xhtml+xml",
    "application/javascript",
    "application/x-ndjson",
)

_USER_AGENT = "atlas-agent (+local web_fetch tool)"


class WebFetchInput(BaseModel):
    url: str = Field(description="Absolute http:// or https:// URL to fetch.")
    timeout: int = Field(
        default=15,
        ge=1,
        le=60,
        description="Seconds (1-60) to allow the request before it is abandoned.",
    )


class WebFetchTool(Tool):
    """Fetch one absolute HTTP(S) URL and return the response body as text.

    Four invariants hold: only ``http`` and ``https`` URLs are accepted, and a
    URL carrying credentials is refused, so the tool never reads a local file
    or speaks a non-web protocol; redirects are not followed, so the request
    reaches the host the model named and a ``3xx`` comes back as the reported
    ``Location`` instead of a second hop; the body is streamed and stopped at
    :attr:`max_bytes`, so a large or endless response cannot exhaust memory;
    and the body is decoded with invalid bytes replaced and returned as text,
    never executed, imported, or written to the workspace.

    The response is untrusted data, and the tool is advertised by default
    (``trust = "default"``), so a session reaches the web without opting in.
    That is narrower than the network reach ``bash_tool`` already has by
    default: a command runs with the user's privileges and unconstrained
    egress, as recorded in ``docs/adr/0001-sandbox-model.md``, while a fetch
    here is typed, capped, and does not chase redirects. It is still egress,
    and that ADR's "Revisit when" clause names reading web pages as the
    trigger to reconsider OS-level or container confinement, because a fetched
    page is a prompt-injection vector. Two gaps stay open: only literal local
    hosts are refused, not a hostname that resolves to a private address, and
    content reached by hand from a redirect chain is not re-checked.
    """

    name = "web_fetch"
    description = (
        "Fetch one absolute http:// or https:// URL and return its body as text. Redirects "
        "are not followed; the Location is reported instead. Only text-like responses are "
        "returned, and the body is truncated past a size cap. The response is untrusted data."
    )
    trust: ClassVar[Literal["default", "opt_in"]] = "default"
    max_bytes: ClassVar[int] = 200_000

    def __init__(self, client: httpx.Client | None = None) -> None:
        """Take an optional HTTP client; tests pass one backed by a mock transport."""
        self._client = client

    @property
    def input_model(self) -> type[BaseModel]:
        return WebFetchInput

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        # ``store`` is unused on purpose: this tool resolves no workspace paths,
        # so it cannot widen the file sandbox. The signature is the Tool contract.
        request = WebFetchInput.model_validate(arguments)
        url = _validate_url(request.url)
        if self._client is not None:
            return self._fetch(self._client, url, request.timeout)
        with httpx.Client(follow_redirects=False) as client:
            return self._fetch(client, url, request.timeout)

    def _fetch(self, client: httpx.Client, url: str, timeout: int) -> ToolResult:
        """Stream one request and render the body, or raise on a refused response."""
        with client.stream(
            "GET", url, timeout=timeout, headers={"user-agent": _USER_AGENT}
        ) as response:
            if response.is_redirect:
                location = response.headers.get("location", "")
                target = location or "an unspecified location"
                return ToolResult(
                    text=f"Redirect not followed: HTTP {response.status_code} to {target}"
                )
            if response.status_code >= 400:
                raise RuntimeError(f"HTTP {response.status_code} for {url}")
            media_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
            # A missing content type is treated as text; an explicit non-text
            # type is refused rather than decoded into noise.
            if media_type and not media_type.startswith(_TEXT_MEDIA_TYPES):
                raise ValueError(f"Refusing the non-text response type {media_type} from {url}")
            body, truncated = self._read_body(response)

        text = _ANSI_ESCAPE_RE.sub("", body)
        header = f"HTTP {response.status_code}, {media_type or 'unspecified content type'}"
        if truncated:
            header += f", truncated at {self.max_bytes} bytes"
        return ToolResult(text=f"{header}\n\n{text}")

    def _read_body(self, response: httpx.Response) -> tuple[str, bool]:
        """Decode at most :attr:`max_bytes` of the body, reporting truncation."""
        chunks: list[bytes] = []
        size = 0
        truncated = False
        for chunk in response.iter_bytes():
            remaining = self.max_bytes - size
            if len(chunk) > remaining:
                chunks.append(chunk[:remaining])
                truncated = True
                break
            chunks.append(chunk)
            size += len(chunk)
        return b"".join(chunks).decode(response.encoding or "utf-8", errors="replace"), truncated


def _validate_url(url: str) -> str:
    """Return ``url`` when it is a fetchable absolute HTTP(S) URL.

    Rejects a non-web scheme, a URL without a host, a URL carrying credentials,
    and the literal local and link-local hosts in :data:`_BLOCKED_HOSTS`.
    """
    parts = urlsplit(url)
    if parts.scheme.lower() not in _ALLOWED_SCHEMES:
        raise ValueError(f"Only http and https URLs can be fetched, not {url!r}")
    hostname = parts.hostname
    if not hostname:
        raise ValueError(f"URL has no host: {url!r}")
    if parts.username is not None or parts.password is not None:
        raise ValueError(f"URL must not carry credentials: {url!r}")
    if hostname.lower() in _BLOCKED_HOSTS:
        raise SandboxDenied(f"Refusing to fetch the local host {hostname!r}")
    return url
