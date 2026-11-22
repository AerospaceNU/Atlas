"""Unified secret, credential, and URL redaction for logs, RPC, state, and skills."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

_HTTP_URL = re.compile(r"https?://[^\s\"'\\<>]+", re.IGNORECASE)
_FILE_URL = re.compile(r"file://[^\s\"'\\<>]+", re.IGNORECASE)
_ABS_UNIX = re.compile(r"(?<!:/)(?<!://)(?<![A-Za-z0-9_])/(?:[A-Za-z0-9._+-]+/)*[A-Za-z0-9._+-]+")
_ABS_WINDOWS = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/](?:[^\\/\s\"']+[\\/])*(?:[^\\/\s\"']+)")
_FIRMS_PATH_KEY = re.compile(
    r"(https?://firms\.modaps\.eosdis\.nasa\.gov/api/area/csv/)[^/\s\"'?#]+",
    re.IGNORECASE,
)
_USERINFO_FALLBACK = re.compile(r"^(https?://)[^/?#\s@]+@", re.IGNORECASE)

# Prefixed credentials and JWTs. Bare hex is left alone so sha256 hashes and scene ids survive.
_KEY_SHAPED = re.compile(
    r"(?i)(?:"
    r"sk-(?:or-v1-|ant-(?:api|admin)\d{2}-|proj-)?[A-Za-z0-9_-]{16,}"
    r"|ghp_[A-Za-z0-9]{20,}"
    r"|github_pat_[A-Za-z0-9_]{20,}"
    r"|AKIA[0-9A-Z]{16}"
    r"|AIza[0-9A-Za-z_-]{20,}"
    r"|xox[baprs]-[A-Za-z0-9-]{10,}"
    r"|Bearer\s+[A-Za-z0-9._-]{8,}"
    r"|eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"
    r")"
)

_SECRET_QUERY_NAMES = frozenset(
    {
        "api_key",
        "apikey",
        "api-key",
        "token",
        "access_token",
        "refresh_token",
        "id_token",
        "password",
        "passwd",
        "secret",
        "client_secret",
        "sig",
        "signature",
        "x-amz-signature",
        "x-amz-credential",
        "x-amz-security-token",
        "x-goog-signature",
        "key",
        "auth",
        "code",
    }
)

_FRAGMENT_SECRET_RE = re.compile(
    r"(?i)#("
    + "|".join(re.escape(name) for name in sorted(_SECRET_QUERY_NAMES, key=len, reverse=True))
    + r")=([^\s\"'\\&#]*)"
)


def _scrub_fragment(text: str, mask: str) -> str:
    return _FRAGMENT_SECRET_RE.sub(lambda match: f"#{match.group(1)}={mask}", text)


def redact_url(url: str, *, mask: str = "[redacted]") -> str:
    """Strip userinfo, secret query/fragment parameters, and embedded keys from ``url``."""
    try:
        parts = urlsplit(url)
    except ValueError:
        cleaned = _USERINFO_FALLBACK.sub(r"\1", url)
        cleaned = _FIRMS_PATH_KEY.sub(lambda m: f"{m.group(1)}{mask}", cleaned)
        return _KEY_SHAPED.sub(mask, _scrub_fragment(cleaned, mask))

    netloc = parts.netloc
    if "@" in netloc:
        netloc = netloc.rsplit("@", 1)[1]

    query = ""
    if parts.query:
        kept = [
            (k, v)
            for k, v in parse_qsl(parts.query, keep_blank_values=True)
            if k.lower() not in _SECRET_QUERY_NAMES
        ]
        query = urlencode(kept, doseq=True)

    fragment = parts.fragment
    if fragment:
        fragment = _scrub_fragment(f"#{fragment}", mask)[1:]

    rebuilt = urlunsplit((parts.scheme, netloc, parts.path, query, fragment))
    rebuilt = _FIRMS_PATH_KEY.sub(lambda m: f"{m.group(1)}{mask}", rebuilt)
    return _KEY_SHAPED.sub(mask, rebuilt)


def redact_text(
    text: str,
    secrets: Iterable[str] = (),
    *,
    mask: str = "[redacted]",
    scrub_paths: bool = False,
) -> str:
    """Redact explicit secrets, credential-shaped tokens, and secret URL components."""
    for secret in sorted({item for item in secrets if item}, key=len, reverse=True):
        text = text.replace(secret, mask)

    held: list[str] = []

    def _hold(match: re.Match[str]) -> str:
        held.append(match.group(0))
        return f"\ue000url{len(held) - 1}\ue001"

    text = _scrub_fragment(text, mask)
    text = _HTTP_URL.sub(_hold, text)
    text = _KEY_SHAPED.sub(mask, text)
    if scrub_paths:
        text = _FILE_URL.sub(mask, text)
        text = _ABS_WINDOWS.sub(mask, text)
        text = _ABS_UNIX.sub(mask, text)
    for index, url in enumerate(held):
        text = text.replace(f"\ue000url{index}\ue001", redact_url(url, mask=mask))
    return text


def redact_value(
    value: Any,
    secrets: Iterable[str] = (),
    *,
    mask: str = "[redacted]",
    scrub_paths: bool = False,
) -> Any:
    """Recursively redact strings inside dicts, lists, and tuples."""
    if isinstance(value, str):
        return redact_text(value, secrets, mask=mask, scrub_paths=scrub_paths)
    if isinstance(value, dict):
        return {
            (
                redact_text(k, secrets, mask=mask, scrub_paths=scrub_paths)
                if isinstance(k, str)
                else k
            ): redact_value(v, secrets, mask=mask, scrub_paths=scrub_paths)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [redact_value(item, secrets, mask=mask, scrub_paths=scrub_paths) for item in value]
    if isinstance(value, tuple):
        return tuple(
            redact_value(item, secrets, mask=mask, scrub_paths=scrub_paths) for item in value
        )
    return value
