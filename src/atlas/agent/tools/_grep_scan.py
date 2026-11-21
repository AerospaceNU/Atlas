"""Parent side of the grep worker.

Invariants: the child is a one-shot stdlib process in its own session; it is always killed
(process group) on cancel, deadline and exit from `run_worker`. No caller code is executed:
the child runs a fixed packaged script.
"""

from __future__ import annotations

import json
import os
import selectors
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path

from atlas.agent.scope import cancel_requested

_WORKER = Path(__file__).with_name("_grep_worker.py")


@dataclass(frozen=True)
class WorkerConfig:
    pattern: str
    flags: int
    max_scan_bytes: int
    scan_seconds: float
    max_file_bytes: int
    max_hits: int
    per_file_seconds: float = 2.0
    slow_abort: int = 3


@dataclass
class WorkerResult:
    """Outcome of one worker run; `hits` are (path index, first lines, total matches)."""

    hits: list[tuple[int, list[int], int]] = field(default_factory=list)
    files: int = 0
    stopped: bool = False
    binary: int = 0
    slow: list[int] = field(default_factory=list)
    aborted: bool = False
    cancelled: bool = False
    error: str | None = None


def worker_available() -> bool:
    """Return True when a child interpreter and the worker script exist."""
    return bool(sys.executable) and _WORKER.is_file()


def _parse(line: bytes, result: WorkerResult, state: dict[str, int]) -> bool:
    """Fold one worker record into `result`; return True for the final record."""
    try:
        record = json.loads(line)
    except ValueError:
        return False
    if not isinstance(record, dict):
        return False
    if "d" in record:
        if "err" in record:
            result.error = f"invalid pattern: {record['err']}"
        result.files = int(record.get("files", 0))
        result.stopped = bool(record.get("stopped"))
        result.binary = int(record.get("binary", 0))
        result.aborted = bool(record.get("aborted"))
        return True
    if "slow" in record:
        result.slow.append(int(record["p"]))
    elif "h" in record:
        result.hits.append((int(record["p"]), [int(n) for n in record["h"]], int(record["n"])))
    elif "f" in record:
        state["files"] = int(record["f"])
    return False


def _pump(
    proc: subprocess.Popen[bytes], result: WorkerResult, state: dict[str, int], deadline: float
) -> bool:
    """Read the child's records until EOF, cancel or deadline; return True on a final record."""
    assert proc.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(proc.stdout, selectors.EVENT_READ)
    buffer = b""
    final = False
    try:
        while not final:
            if cancel_requested():
                result.cancelled = True
                return False
            if time.monotonic() > deadline:
                result.stopped = True
                return False
            if not selector.select(timeout=0.1):
                continue
            chunk = os.read(proc.stdout.fileno(), 65536)
            if not chunk:
                return False
            buffer += chunk
            *complete, buffer = buffer.split(b"\n")
            for line in complete:
                final = _parse(line, result, state) or final
    finally:
        selector.close()
    return final


def run_worker(config: WorkerConfig, paths: Sequence[str], *, deadline: float) -> WorkerResult:
    """Run the worker over `paths`; always reaps and kills the child's process group."""
    result = WorkerResult()
    state = {"files": 0}
    with tempfile.TemporaryDirectory(prefix="atlas-grep-") as workdir:
        (Path(workdir) / "config.json").write_text(json.dumps(config.__dict__), encoding="utf-8")
        (Path(workdir) / "paths.bin").write_bytes(b"\0".join(os.fsencode(p) for p in paths))
        argv = [sys.executable, "-I", "-S", str(_WORKER), workdir]
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            env={"PYTHONIOENCODING": "utf-8"},
        )
        try:
            final = _pump(proc, result, state, time.monotonic() + deadline)
            killed = result.cancelled or result.stopped
            if not final and not killed:
                with suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=1)
                result.error = f"grep worker failed (exit {proc.poll()})"
            if not final:
                result.files = state["files"]
        finally:
            with suppress(ProcessLookupError, PermissionError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            if proc.stdout is not None:
                proc.stdout.close()
    return result
