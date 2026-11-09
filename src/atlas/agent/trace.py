"""A versioned on-disk record of one agent run.

A benchmark replays and compares runs long after the harness that produced them
changed. ``AgentRun`` round-trips through JSON on its own, but it carries no
version, so a reader cannot tell which build wrote a record or whether the shape
still matches. ``TraceRecord`` adds that version, and ``load_trace`` refuses a
record it cannot interpret instead of silently dropping fields it does not know.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from atlas.agent.runtime import AgentRun

# Bump when a change makes an older record unreadable. Adding a field with a
# default is compatible and keeps the number.
TRACE_SCHEMA = 1

__all__ = [
    "TRACE_SCHEMA",
    "TraceRecord",
    "TraceSchemaError",
    "dump_trace",
    "load_trace",
    "read_trace",
    "write_trace",
]


class TraceSchemaError(ValueError):
    """A record is malformed or was written by an unreadable schema."""


class TraceRecord(BaseModel):
    """One agent run plus the schema version that shapes it."""

    schema_version: int = Field(default=TRACE_SCHEMA)
    run: AgentRun


def dump_trace(run: AgentRun) -> str:
    """Serialize a run to a versioned JSON record."""
    return TraceRecord(run=run).model_dump_json()


def load_trace(text: str) -> TraceRecord:
    """Parse a versioned record, refusing a schema this build cannot read."""
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise TraceSchemaError(f"trace is not JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise TraceSchemaError("trace must be a JSON object")
    version = raw.get("schema_version")
    if version != TRACE_SCHEMA:
        raise TraceSchemaError(
            f"trace schema {version!r} is not readable by this build (expected {TRACE_SCHEMA})"
        )
    return TraceRecord.model_validate(raw)


def write_trace(path: Path, run: AgentRun) -> Path:
    """Write a versioned record to ``path``, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_trace(run) + "\n", encoding="utf-8")
    return path


def read_trace(path: Path) -> TraceRecord:
    """Read a versioned record from ``path``."""
    return load_trace(path.read_text(encoding="utf-8"))
