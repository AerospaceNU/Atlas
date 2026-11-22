"""Append-only JSONL audit log for tool and model activity."""

from __future__ import annotations

import datetime
import json
import logging
import os
from pathlib import Path
from typing import Any

from atlas.agent.artifacts import LocalArtifactStore

_LOGGER = logging.getLogger("atlas.audit")


def audit_path(store: LocalArtifactStore) -> Path:
    """Return the audit log path for this workspace."""
    return store.resolve(".atlas/audit.log")


def record(store: LocalArtifactStore, event: str, **fields: Any) -> None:
    """Append one event to ``<workspace>/.atlas/audit.log``, swallowing write errors."""
    payload = {
        "time": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "event": event,
        **fields,
    }
    try:
        path = audit_path(store)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str) + "\n")
    except Exception:
        _LOGGER.exception("Could not write the audit record for %s", event)
