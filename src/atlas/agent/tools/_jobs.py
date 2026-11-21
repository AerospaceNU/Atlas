"""Managed background jobs for the bash tool.

Invariants: jobs are started only by an approved bash call and only signalled or
read afterwards; nothing here executes caller code on its own. State changes happen
under one lock and only from ``running`` (first writer wins), so the janitor,
``stop_job`` and ``stop_all`` cannot double-kill or overwrite a state. POSIX only.
"""

from __future__ import annotations

import atexit
import contextlib
import itertools
import os
import re
import signal
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

MAX_RUNNING_JOBS = 8
DEFAULT_LIFETIME = 3600
MIN_LIFETIME = 60
MAX_LIFETIME = 86400
MAX_LOG_BYTES = 64 * 1024 * 1024
GRACE_SECONDS = 3.0
MAX_FINISHED_KEPT = 32
MAX_WAIT = 60

_TICK = 0.5
_POLL = 0.1
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


class JobError(ValueError):
    """A user-facing refusal; the message is returned to the model as text."""


@dataclass
class Job:
    id: str
    owner: str
    command: str
    cwd: str
    pid: int
    log_path: Path
    started: float
    lifetime: int
    proc: subprocess.Popen[bytes]
    state: str = "running"
    exit_code: int | None = None


@dataclass(frozen=True)
class OutputSlice:
    text: str
    start: int
    next_offset: int
    running: bool
    exit_code: int | None
    state: str


_LOCK = threading.Lock()
_JOBS: dict[str, Job] = {}
_COUNTER = itertools.count(1)
_state: dict[str, object] = {"janitor": None, "atexit": False}


def _running_ids(owner: str | None = None) -> list[str]:
    return [j.id for j in _JOBS.values() if j.state == "running" and owner in (None, j.owner)]


def start_job(
    argv: Sequence[str],
    *,
    command: str,
    cwd: str | os.PathLike[str],
    env: Mapping[str, str],
    artifact_root: str | os.PathLike[str],
    lifetime: int = DEFAULT_LIFETIME,
) -> Job:
    """Start ``argv`` in its own session with stdout and stderr in a log file."""
    if os.name == "nt":
        raise JobError("background jobs need a POSIX system")
    owner = str(artifact_root)
    with _LOCK:
        if len(_running_ids()) >= MAX_RUNNING_JOBS:
            mine = ",".join(_running_ids(owner))
            raise JobError(
                f"background job limit reached ({MAX_RUNNING_JOBS} running; yours: {mine}). "
                "Fix: jobs action=stop job_id=<id>"
            )
        job_id = f"j{next(_COUNTER)}"
        log_path = Path(artifact_root) / "jobs" / f"{os.getpid()}-{job_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_APPEND, 0o644)
        try:
            proc = subprocess.Popen(
                list(argv),
                cwd=cwd,
                env=dict(env),
                stdin=subprocess.DEVNULL,
                stdout=fd,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            os.close(fd)
        job = Job(
            id=job_id,
            owner=owner,
            command=command,
            cwd=str(cwd),
            pid=proc.pid,
            log_path=log_path,
            started=time.monotonic(),
            lifetime=max(MIN_LIFETIME, min(MAX_LIFETIME, int(lifetime))),
            proc=proc,
        )
        _JOBS[job_id] = job
        _trim_finished()
        _ensure_janitor()
    return job


def _trim_finished() -> None:
    finished = [j for j in _JOBS.values() if j.state != "running"]
    for old in finished[: max(0, len(finished) - MAX_FINISHED_KEPT)]:
        del _JOBS[old.id]


def _ensure_janitor() -> None:
    if not _state["atexit"]:
        atexit.register(stop_all, "interpreter exit")
        _state["atexit"] = True
    thread = _state["janitor"]
    if isinstance(thread, threading.Thread) and thread.is_alive():
        return
    thread = threading.Thread(target=_janitor, name="atlas-jobs-janitor", daemon=True)
    _state["janitor"] = thread
    thread.start()


def _janitor() -> None:
    while True:
        time.sleep(_TICK)
        with _LOCK:
            running = [j for j in _JOBS.values() if j.state == "running"]
            for job in running:
                _check(job)
            if not any(j.state == "running" for j in running):
                _state["janitor"] = None
                return


def _check(job: Job) -> None:
    """Advance one running job; the caller holds the lock."""
    code = job.proc.poll()
    if code is not None:
        job.state, job.exit_code = "exited", code
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(job.pid, signal.SIGKILL)
    elif time.monotonic() - job.started > job.lifetime:
        _terminate(job, "expired")
    elif _log_size(job) > MAX_LOG_BYTES:
        _terminate(job, "log_cap")


def _log_size(job: Job) -> int:
    try:
        return job.log_path.stat().st_size
    except OSError:
        return 0


def _terminate(job: Job, state: str) -> None:
    """SIGTERM the group, then SIGKILL after the grace period; the caller holds the lock."""
    if job.state != "running":
        return
    job.state = state
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(job.pid, signal.SIGTERM)
    deadline = time.monotonic() + GRACE_SECONDS
    while time.monotonic() < deadline and job.proc.poll() is None:
        time.sleep(_POLL)
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(job.pid, signal.SIGKILL)
    job.exit_code = job.proc.wait()


def get_job(job_id: str, owner: str) -> Job:
    with _LOCK:
        job = _JOBS.get(job_id.strip())
        if job is None or job.owner != owner:
            mine = ",".join(j.id for j in _JOBS.values() if j.owner == owner) or "none"
            raise JobError(f"no job {job_id}. Yours: {mine}")
        return job


def list_jobs(owner: str) -> list[Job]:
    with _LOCK:
        for job in _JOBS.values():
            if job.state == "running":
                _check(job)
        return [j for j in _JOBS.values() if j.owner == owner]


def _cancelled() -> bool:
    from atlas.agent.scope import cancel_requested

    return cancel_requested()


def _refresh(job: Job) -> None:
    with _LOCK:
        if job.state == "running":
            _check(job)


def read_output(
    job: Job, *, from_byte: int | None, max_chars: int, wait_seconds: float = 0
) -> OutputSlice:
    """Return a byte slice of the job log.

    ``from_byte=None`` returns the last ``max_chars`` bytes. While the job runs or more
    bytes remain, the slice end is backed off to a UTF-8 boundary so ``next_offset`` never
    splits a character. ``wait_seconds`` (clamped to 0..60) waits for new bytes.
    """
    deadline = time.monotonic() + max(0.0, min(float(wait_seconds), float(MAX_WAIT)))
    while True:
        _refresh(job)
        size = _log_size(job)
        start = max(0, size - max_chars) if from_byte is None else from_byte
        if start < size or job.state != "running" or time.monotonic() >= deadline:
            break
        if _cancelled():
            break
        time.sleep(_POLL)
    start = min(start, size)
    data = _read_range(job.log_path, start, min(size, start + max_chars))
    end = start + len(data)
    if end < size or job.state == "running":
        end = start + _utf8_safe_length(data)
        data = data[: end - start]
    text = _ANSI_ESCAPE_RE.sub("", data.decode("utf-8", errors="replace"))
    return OutputSlice(
        text=text,
        start=start,
        next_offset=end,
        running=job.state == "running",
        exit_code=job.exit_code,
        state=job.state,
    )


def _read_range(path: Path, start: int, end: int) -> bytes:
    try:
        with path.open("rb") as handle:
            handle.seek(start)
            return handle.read(max(0, end - start))
    except OSError:
        return b""


def _utf8_safe_length(data: bytes) -> int:
    """Length of ``data`` minus an incomplete trailing UTF-8 sequence (at most 3 bytes)."""
    n = len(data)
    for back in range(1, min(4, n) + 1):
        byte = data[n - back]
        if byte & 0xC0 == 0x80:
            continue
        need = 1 if byte < 0x80 else 2 if byte >> 5 == 0b110 else 3 if byte >> 4 == 0b1110 else 4
        return n if back >= need else n - back
    return n


def stop_job(job: Job, reason: str = "stopped") -> Job:
    with _LOCK:
        if job.state == "running":
            code = job.proc.poll()
            if code is not None:
                job.state, job.exit_code = "exited", code
            else:
                _terminate(job, "stopped")
    return job


def stop_all(reason: str) -> int:
    """Stop every running job of every owner and return how many were running."""
    count = 0
    with _LOCK:
        for job in list(_JOBS.values()):
            if job.state == "running" and job.proc.poll() is None:
                _terminate(job, "stopped")
                count += 1
    return count
