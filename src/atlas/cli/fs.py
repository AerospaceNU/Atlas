"""Filesystem helpers shared by the data CLI writers."""

from __future__ import annotations

import os
import time

# A Windows antivirus or indexer can hold the destination open for a moment,
# so an immediate os.replace raises PermissionError. A few short retries cover
# that without hiding a real permission failure.
REPLACE_ATTEMPTS = 5
REPLACE_DELAY_SECONDS = 0.05


def replace_with_retry(
    source: str | os.PathLike[str],
    target: str | os.PathLike[str],
    *,
    attempts: int = REPLACE_ATTEMPTS,
    delay: float = REPLACE_DELAY_SECONDS,
) -> None:
    """Replace ``target`` with ``source``, retrying a transient sharing error.

    Args:
        source: Finished file to move.
        target: Final name.
        attempts: Total tries, at least one.
        delay: Seconds to wait after a failed try.

    Raises:
        OSError: The replace still failed on the last attempt.
    """
    total = max(1, attempts)
    for attempt in range(1, total + 1):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt >= total:
                raise
            time.sleep(delay)
