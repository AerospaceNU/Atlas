"""Model weight store: layout, status, verification and human-run pulls.

Invariants: downloads happen only through `pull`, which no agent tool calls. A sha256 pin
is mandatory; a file is installed only after its hash matches the pin. Remote sources are
https only (loopback http is allowed for local smoke tests) and redirects stay on https.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from urllib.parse import urljoin, urlparse

if TYPE_CHECKING:
    from atlas.models.manifest import ModelManifest

__all__ = [
    "VERIFIED_FILE",
    "WeightError",
    "WeightMismatch",
    "WeightNetworkError",
    "WeightState",
    "WeightUnpublished",
    "WeightUsageError",
    "load_manifests",
    "pull",
    "status",
    "verify",
    "weight_path",
    "weights_dir",
]

VERIFIED_FILE = ".verified.json"
_CHUNK = 1 << 20
_MAX_REDIRECTS = 5
_LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})
_FALLBACK_CAP = 8 << 30

Progress = Callable[[int, int | None], None]


class WeightError(Exception):
    """Base class; `exit_code` is the `atlas models` process exit code."""

    exit_code = 1

    def __init__(self, message: str) -> None:
        super().__init__(message)


class WeightUsageError(WeightError):
    """Bad usage or a licence gate that was not accepted (exit 2)."""

    exit_code = 2


class WeightMismatch(WeightError):
    """Hash or size does not match the pin (exit 3)."""

    exit_code = 3


class WeightNetworkError(WeightError):
    """Transfer failed or a URL was refused (exit 4)."""

    exit_code = 4


class WeightUnpublished(WeightError):
    """No pin or no source is published for these weights (exit 5)."""

    exit_code = 5


@dataclass(frozen=True)
class WeightState:
    """Install state of one model's weights.

    Attributes:
        status: ready, missing, corrupt or unpublished.
        path: Expected file location (None when the model has no weights).
        detail: Human-readable reason.
    """

    status: Literal["ready", "missing", "corrupt", "unpublished"]
    path: Path | None
    detail: str = ""


def load_manifests() -> list[ModelManifest]:
    """Return every packaged v2 manifest (the single seam tests patch)."""
    from atlas.models.manifest import iter_model_manifests

    return list(iter_model_manifests())


def weights_dir() -> Path:
    """Return the weight root: `ATLAS_WEIGHTS_DIR` or the user weights directory.

    The working directory is never probed.
    """
    env = os.environ.get("ATLAS_WEIGHTS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    from atlas.agent.layout import user_weights_dir

    return user_weights_dir().resolve()


def weight_path(manifest: ModelManifest) -> Path:
    """Return `<dir>/<name>/<version>/<file>` for a manifest with weights.

    Raises:
        WeightUsageError: If the model has no weights or the file name escapes the directory.
    """
    spec = manifest.weights
    if spec is None:
        raise WeightUsageError(f"{manifest.name} has no weights")
    if "/" in spec.file or "\\" in spec.file or ".." in spec.file:
        raise WeightUsageError(f"{manifest.name}: weights.file {spec.file!r} must be a bare name")
    root = weights_dir()
    path = (root / manifest.name / manifest.version / spec.file).resolve()
    if not path.is_relative_to(root):
        raise WeightUsageError(f"{manifest.name}: weights.file escapes the weights directory")
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _marker(path: Path) -> Path:
    return path.parent / VERIFIED_FILE


def _marker_matches(path: Path, pin: str) -> bool:
    try:
        data = json.loads(_marker(path).read_text(encoding="utf-8"))
        stat = path.stat()
        return bool(
            data.get("sha256") == pin
            and data.get("size") == stat.st_size
            and data.get("mtime_ns") == stat.st_mtime_ns
        )
    except (OSError, ValueError, AttributeError):
        return False


def _write_marker(path: Path, sha: str, url: str) -> None:
    try:
        stat = path.stat()
        record = {"sha256": sha, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "url": url}
        tmp = path.parent / f"{VERIFIED_FILE}.tmp-{os.getpid()}"
        tmp.write_text(json.dumps(record), encoding="utf-8")
        os.replace(tmp, _marker(path))
    except OSError:
        return


def _inspect(manifest: ModelManifest, *, rehash: bool) -> WeightState:
    spec = manifest.weights
    if spec is None:
        return WeightState("ready", None, "no weights")
    path = weight_path(manifest)
    if not spec.sha256:
        return WeightState("unpublished", path, "no sha256 pin")
    if not path.is_file():
        if not spec.url:
            return WeightState("unpublished", path, "no url; use --from PATH")
        return WeightState("missing", path, "not installed")
    size = path.stat().st_size
    if spec.size_bytes and size != spec.size_bytes:
        return WeightState("corrupt", path, f"size {size} != {spec.size_bytes}")
    if not rehash and _marker_matches(path, spec.sha256):
        return WeightState("ready", path, "verified")
    got = _sha256_file(path)
    if got != spec.sha256:
        return WeightState("corrupt", path, f"sha256 {got[:12]} != pin {spec.sha256[:12]}")
    _write_marker(path, got, spec.url)
    return WeightState("ready", path, "verified")


def status(manifest: ModelManifest) -> WeightState:
    """Return the weight state, trusting a matching `.verified.json` marker."""
    return _inspect(manifest, rehash=False)


def verify(manifest: ModelManifest) -> WeightState:
    """Return the weight state after always rehashing the file."""
    return _inspect(manifest, rehash=True)


def _check_url(url: str, *, first: bool) -> None:
    parsed = urlparse(url)
    if parsed.scheme == "https" and parsed.hostname:
        return
    if first and parsed.scheme == "http" and parsed.hostname in _LOOPBACK:
        return
    raise WeightNetworkError(f"refusing non-https URL: {url}")


def _size_cap(size_bytes: int) -> int:
    return math.ceil(1.01 * size_bytes) if size_bytes > 0 else _FALLBACK_CAP


def _write_stream(
    chunks: Iterable[bytes], part: Path, cap: int, total: int | None, progress: Progress | None
) -> str:
    """Write chunks to `part` with an incremental sha256 and a size cap; return the hash."""
    digest = hashlib.sha256()
    count = 0
    part.parent.mkdir(parents=True, exist_ok=True)
    with part.open("wb") as handle:
        for chunk in chunks:
            count += len(chunk)
            if count > cap:
                raise WeightMismatch(
                    f"download exceeds the size cap ({cap} bytes); nothing installed"
                )
            digest.update(chunk)
            handle.write(chunk)
            if progress is not None:
                progress(count, total)
    return digest.hexdigest()


def _http_chunks(url: str, cap: int) -> Iterator[bytes]:
    import httpx

    timeout = httpx.Timeout(30.0, read=60.0)
    try:
        with httpx.Client(timeout=timeout, follow_redirects=False) as client:
            current = url
            for hop in range(_MAX_REDIRECTS + 1):
                _check_url(current, first=hop == 0)
                with client.stream("GET", current) as response:
                    if response.is_redirect:
                        location = response.headers.get("location")
                        if not location or hop == _MAX_REDIRECTS:
                            raise WeightNetworkError(f"bad or too many redirects from {url}")
                        current = urljoin(current, location)
                        continue
                    if response.status_code != 200:
                        raise WeightNetworkError(f"HTTP {response.status_code} from {current}")
                    declared = response.headers.get("content-length")
                    if declared and declared.isdigit() and int(declared) > cap:
                        raise WeightMismatch(
                            f"server reports {declared} bytes, over the size cap {cap}; "
                            "nothing installed"
                        )
                    yield from response.iter_bytes(_CHUNK)
                    return
    except httpx.HTTPError as exc:
        raise WeightNetworkError(f"download failed: {exc}") from exc


def _file_chunks(src: Path) -> Iterator[bytes]:
    try:
        with src.open("rb") as handle:
            while chunk := handle.read(_CHUNK):
                yield chunk
    except OSError as exc:
        raise WeightUsageError(f"cannot read {src}: {exc}") from exc


def _licence_gate(manifest: ModelManifest, accept: bool) -> None:
    lic = manifest.license
    spec = manifest.weights
    size = f"{spec.size_bytes / 1e6:.1f} MB" if spec and spec.size_bytes else "unknown size"
    print(f"{manifest.name} weights: licence {lic.spdx}  holder {lic.holder or '-'}")
    print(f"licence url: {lic.url or '-'}  size: {size}")
    if not accept:
        raise WeightUsageError(
            f"licence not accepted for {manifest.name}; pass --accept-license to continue"
        )


def pull(
    manifest: ModelManifest,
    *,
    url: str | None = None,
    src: Path | None = None,
    accept_license: bool = False,
    progress: Progress | None = None,
) -> Path:
    """Install weights from a URL or a local file, verifying the pinned sha256.

    Args:
        manifest: A manifest with a `[weights]` table.
        url: Override source URL (defaults to `weights.url`).
        src: Local file to install instead of downloading.
        accept_license: Whether the user accepted the licence terms.
        progress: Optional callback (bytes so far, total or None).

    Returns:
        The installed weights path.

    Raises:
        WeightUnpublished: No pin, or no source.
        WeightUsageError: Licence not accepted.
        WeightMismatch: Hash or size does not match the pin; nothing is installed.
        WeightNetworkError: Transfer failed or the URL was refused.
    """
    spec = manifest.weights
    if spec is None:
        raise WeightUsageError(f"{manifest.name} has no weights to pull")
    if not spec.sha256:
        raise WeightUnpublished(f"{manifest.name} is not published: no sha256 pin")
    target = weight_path(manifest)
    if src is None and url is None and status(manifest).status == "ready":
        print(f"{manifest.name} already installed at {target}")
        return target
    source = url or spec.url
    if src is None and not source:
        raise WeightUnpublished(f"{manifest.name} is not published; use --from PATH")
    _licence_gate(manifest, accept_license)
    cap = _size_cap(spec.size_bytes)
    part = target.parent / f"{target.name}.part-{os.getpid()}"
    try:
        if src is not None:
            source = str(src)
            got = _write_stream(_file_chunks(src), part, cap, None, progress)
        else:
            _check_url(source, first=True)
            got = _write_stream(_http_chunks(source, cap), part, cap, None, progress)
        if got != spec.sha256:
            raise WeightMismatch(f"sha256 {got[:12]} != pin {spec.sha256[:12]}; nothing installed")
        os.replace(part, target)
        _write_marker(target, got, source)
    finally:
        with contextlib.suppress(OSError):
            part.unlink()
    return target
