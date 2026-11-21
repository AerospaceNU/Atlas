"""One-shot grep child: scans a list of files for a regex, killable by its parent.

Invariants: stdlib only, no atlas imports (it runs as `python -I -S _grep_worker.py WORKDIR`),
one JSON record per stdout line. A per-file SIGALRM interrupts a catastrophic match
(SRE checks signals regularly); the parent's SIGKILL is the backstop.
"""

from __future__ import annotations

import json
import os
import re
import signal
import sys
import time
from types import FrameType
from typing import Any


class _Slow(Exception):
    """Raised by the alarm handler when one file takes too long."""


def _on_alarm(signum: int, frame: FrameType | None) -> None:
    raise _Slow


def _split_lines(text: str) -> list[str]:
    """Split on newline only; keep in sync with `atlas.agent.tools.grep._split_lines`."""
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return [line[:-1] if line.endswith("\r") else line for line in lines]


def _emit(record: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(record, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _read_paths(workdir: str) -> list[str]:
    with open(os.path.join(workdir, "paths.bin"), "rb") as handle:
        raw = handle.read()
    return [os.fsdecode(item) for item in raw.split(b"\0") if item]


def _scan_file(
    pattern: re.Pattern[str], text: str, max_hits: int, per_file_seconds: float
) -> tuple[list[int], int]:
    """Return the first `max_hits` matching line numbers and the total; raise _Slow on timeout."""
    hits: list[int] = []
    total = 0
    signal.setitimer(signal.ITIMER_REAL, per_file_seconds)
    try:
        for number, line in enumerate(_split_lines(text), 1):
            if pattern.search(line):
                total += 1
                if len(hits) < max_hits:
                    hits.append(number)
        signal.setitimer(signal.ITIMER_REAL, 0)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    return hits, total


def main() -> int:
    workdir = sys.argv[1]
    with open(os.path.join(workdir, "config.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    try:
        pattern = re.compile(config["pattern"], config["flags"])
    except re.error as exc:
        _emit({"d": 1, "err": str(exc)})
        return 0
    paths = _read_paths(workdir)
    signal.signal(signal.SIGALRM, _on_alarm)
    started = time.monotonic()
    scanned = files = binary = slow_count = 0
    stopped = aborted = False
    for index, path in enumerate(paths):
        if (
            scanned >= config["max_scan_bytes"]
            or time.monotonic() - started > config["scan_seconds"]
        ):
            stopped = True
            break
        try:
            if os.stat(path).st_size > config["max_file_bytes"]:
                continue
            with open(path, "rb") as handle:
                data = handle.read()
        except OSError:
            continue
        if b"\x00" in data:
            binary += 1
            continue
        scanned += len(data)
        files += 1
        try:
            try:
                hits, total = _scan_file(
                    pattern,
                    data.decode("utf-8", errors="replace"),
                    config["max_hits"],
                    config["per_file_seconds"],
                )
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
        except _Slow:
            _emit({"p": index, "slow": 1})
            slow_count += 1
            if slow_count >= config["slow_abort"]:
                aborted = True
                break
            continue
        if total:
            _emit({"p": index, "h": hits, "n": total})
        if files % 50 == 0:
            _emit({"f": files})
    _emit({"d": 1, "files": files, "stopped": stopped, "binary": binary, "aborted": aborted})
    return 0


if __name__ == "__main__":
    sys.exit(main())
