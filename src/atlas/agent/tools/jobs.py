from __future__ import annotations

import time
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field, ValidationError

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import Tool, ToolResult
from atlas.agent.tools import _jobs

MAX_OUTPUT_CHARS = 20_000


class JobsInput(BaseModel):
    action: Literal["list", "output", "stop"] = Field(description="list, output or stop")
    job_id: str = Field(default="", description="Job id such as j1; required for output and stop.")
    wait_seconds: int = Field(
        default=0,
        ge=0,
        description="output only: wait up to this many seconds (max 60) for new text.",
    )
    from_byte: int | None = Field(
        default=None, ge=0, description="output only: byte offset; default is the log tail."
    )
    max_chars: int = Field(
        default=4000, ge=1, description="output only: characters to return (max 20000)."
    )


class JobsTool(Tool):
    """List, read, or stop background jobs a bash call started; it never starts one."""

    name = "jobs"
    description = (
        "List, read the output of, or stop background jobs started by bash with "
        "run_in_background. It cannot start anything; commands must run in the foreground."
    )
    trust: ClassVar[Literal["default", "opt_in"]] = "default"
    input_model = JobsInput

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        try:
            request = JobsInput.model_validate(arguments)
        except ValidationError as exc:
            first = exc.errors()[0]
            where = ".".join(str(part) for part in first.get("loc", ()))
            return ToolResult(
                text=f"invalid arguments: {where}: {first.get('msg', exc)}. Fix: jobs action=list"
            )
        owner = str(store.artifact_root)
        try:
            if request.action == "list":
                return ToolResult(text=_list(owner))
            if not request.job_id:
                return ToolResult(
                    text=f"job_id is required. Fix: jobs action={request.action} job_id=j1"
                )
            job = _jobs.get_job(request.job_id, owner)
            if request.action == "stop":
                return ToolResult(text=_stop(job))
            return ToolResult(text=_output(job, request))
        except _jobs.JobError as exc:
            return ToolResult(text=str(exc))


def _status(job: _jobs.Job) -> str:
    if job.state == "exited":
        return f"exited (code {job.exit_code})"
    return job.state


def _list(owner: str) -> str:
    jobs = _jobs.list_jobs(owner)
    if not jobs:
        return "no background jobs"
    now = time.monotonic()
    return "\n".join(
        f"{j.id} {_status(j)} pid {j.pid} {int(now - j.started)}s/{j.lifetime}s: {j.command[:80]}"
        for j in jobs
    )


def _stop(job: _jobs.Job) -> str:
    was_running = job.state == "running"
    _jobs.stop_job(job)
    if was_running and job.state == "stopped":
        return f"job {job.id} stopped (was running)"
    return f"job {job.id} already finished: {_status(job)}"


def _output(job: _jobs.Job, request: JobsInput) -> str:
    max_chars = min(request.max_chars, MAX_OUTPUT_CHARS)
    clamp = (
        f"note: max_chars clamped to {MAX_OUTPUT_CHARS}\n" if request.max_chars > max_chars else ""
    )
    piece = _jobs.read_output(
        job, from_byte=request.from_byte, max_chars=max_chars, wait_seconds=request.wait_seconds
    )
    status = _status(job)
    return (
        f"{clamp}job {job.id} {status}, bytes {piece.start}-{piece.next_offset}\n"
        f"{piece.text}\nnext_offset: {piece.next_offset}"
    )
