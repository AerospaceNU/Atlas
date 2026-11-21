"""Subprocess helper shared by the execution tools.

Invariants: stdin is /dev/null, output goes to temporary files rather than
pipes, and the child's whole process group is killed and reaped on every
exit path, so a backgrounded descendant can neither hold the call open nor
outlive it.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from os import PathLike
from typing import IO

from atlas.agent.scope import cancel_requested

DEFAULT_MAX_CHARS = 200_000
MAX_CAPTURE_BYTES = 64 * 1024 * 1024
OUTPUT_LIMIT_NOTE = "note: output exceeded 64 MB and the command was stopped"
_POLL_SECONDS = 0.1
_HEAD_FRACTION = 0.4


@dataclass(frozen=True)
class Captured:
    """Result of one captured process run."""

    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool
    output_limited: bool = False
    cancelled: bool = False


def run_captured(
    argv: Sequence[str],
    *,
    cwd: str | PathLike[str],
    timeout: float,
    env: Mapping[str, str] | None = None,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> Captured:
    """Run ``argv`` without a shell and return its bounded output.

    The call returns once the direct child exits, even if a background
    descendant still holds the output files. On timeout the output written so
    far is returned with ``timed_out=True``. When either output file grows past
    :data:`MAX_CAPTURE_BYTES` the process group is killed and ``output_limited``
    is set; the child itself is never given a resource limit. When the current
    :class:`~atlas.agent.scope.RunScope` is cancelled the group is killed, the
    output so far is returned with ``cancelled=True`` and ``exit_code=None``; an
    already-cancelled scope returns without spawning.
    """
    if cancel_requested():
        return Captured(None, "", "", False, cancelled=True)
    with tempfile.TemporaryFile() as out_file, tempfile.TemporaryFile() as err_file:
        process = subprocess.Popen(
            list(argv),
            cwd=cwd,
            env=None if env is None else dict(env),
            stdin=subprocess.DEVNULL,
            stdout=out_file,
            stderr=err_file,
            start_new_session=True,
        )
        # The session leader's pid is the group id; capture it before reaping.
        pgid = process.pid
        timed_out = False
        output_limited = False
        cancelled = False
        try:
            timed_out, output_limited, cancelled = _wait(process, out_file, err_file, timeout)
        finally:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pgid, signal.SIGKILL)
            process.wait()
        return Captured(
            exit_code=None if (timed_out or cancelled) else process.returncode,
            stdout=_read_bounded(out_file, max_chars),
            stderr=_read_bounded(err_file, max_chars),
            timed_out=timed_out,
            output_limited=output_limited,
            cancelled=cancelled,
        )


def _wait(
    process: subprocess.Popen[bytes], out_file: IO[bytes], err_file: IO[bytes], timeout: float
) -> tuple[bool, bool, bool]:
    """Wait for the direct child; return ``(timed_out, output_limited, cancelled)``.

    Waits in short slices so the size of both capture files can be checked
    between them. Reaching the byte cap stops the wait; the caller kills the group. A cancelled scope is polled each slice and also
    stops the wait; a child that already exited is never reported as cancelled.
    """
    deadline = time.monotonic() + timeout
    while True:
        if cancel_requested():
            return False, False, True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return True, False, False
        try:
            process.wait(timeout=min(_POLL_SECONDS, remaining))
        except subprocess.TimeoutExpired:
            if cancel_requested():
                return False, False, True
            if _over_cap(out_file) or _over_cap(err_file):
                return False, True, False
            continue
        return False, False, False


def _over_cap(handle: IO[bytes]) -> bool:
    return os.fstat(handle.fileno()).st_size > MAX_CAPTURE_BYTES


def _read_bounded(handle: IO[bytes], max_chars: int) -> str:
    """Read at most ``max_chars`` characters, keeping the first 40% and last 60%.

    The file is sized and sought, never loaded whole when it is larger than the
    cap. The omitted count is measured in bytes once the file exceeds the cap.
    """
    size = handle.seek(0, os.SEEK_END)
    if size <= max_chars:
        handle.seek(0)
        text = handle.read().decode("utf-8", errors="replace")
        if len(text) <= max_chars:
            return text
        return _join(text[: _head(max_chars)], text[len(text) - _tail(max_chars) :], len(text))
    head_bytes = _head(max_chars)
    tail_bytes = _tail(max_chars)
    handle.seek(0)
    head = handle.read(head_bytes).decode("utf-8", errors="replace")
    handle.seek(size - tail_bytes)
    tail = handle.read(tail_bytes).decode("utf-8", errors="replace")
    return _join(head, tail, size, kept=head_bytes + tail_bytes)


def _head(max_chars: int) -> int:
    return int(max_chars * _HEAD_FRACTION)


def _tail(max_chars: int) -> int:
    return max_chars - _head(max_chars)


def _join(head: str, tail: str, total: int, kept: int | None = None) -> str:
    omitted = total - (len(head) + len(tail) if kept is None else kept)
    return f"{head}\n... [truncated {omitted} characters] ...\n{tail}"
