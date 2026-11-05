"""Append-only audit log for tool and model activity.

Every tool execution and every model call appends one JSON object to
``<workspace>/.atlas/audit.log``. A record holds the event, the tool or model
name, the outcome, and the durations and status available at that point. Paths
are workspace-relative and no tool arguments are stored, so a key that appears
in a command is not copied into the log.

The log is intentionally plain JSONL: it is easy to tail, grep, or hand to
another tool, and a line that fails to parse does not stop the session.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
from pathlib import Path
from typing import Any

from atlas.agent.artifacts import LocalArtifactStore

_LOGGER = logging.getLogger("atlas.audit")
_AUDIT_RELATIVE = Path(".atlas") / "audit.log"


def audit_path(store: LocalArtifactStore) -> Path:
    """Return the audit log path for this workspace."""
    return store.resolve(_AUDIT_RELATIVE.as_posix())


def record(store: LocalArtifactStore, event: str, **fields: Any) -> None:
    """Append one event to the workspace audit log.

    A failure to write is logged and swallowed: auditing must never change the
    outcome of the turn it is describing.

    Args:
        store: Session store whose workspace holds the log.
        event: Event name, such as ``tool`` or ``model``.
        **fields: Values placed in the record. Non-JSON values are stringified.
    """
    payload = {
        "time": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "event": event,
        **fields,
    }
    try:
        path = audit_path(store)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(payload, default=str)
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except Exception:
        _LOGGER.exception("Could not write the audit record for %s", event)
