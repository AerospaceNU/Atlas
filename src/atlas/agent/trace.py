"""Versioned on-disk JSON record for replaying and comparing agent runs."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from atlas.agent.runtime import AgentRun

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
    """A trace record is malformed or uses an unsupported schema version."""


class TraceRecord(BaseModel):
    """One agent run plus its schema version."""

    schema_version: int = Field(default=TRACE_SCHEMA)
    run: AgentRun


def dump_trace(run: AgentRun) -> str:
    return TraceRecord(run=run).model_dump_json()


def load_trace(text: str) -> TraceRecord:
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_trace(run) + "\n", encoding="utf-8")
    return path


def read_trace(path: Path) -> TraceRecord:
    return load_trace(path.read_text(encoding="utf-8"))
