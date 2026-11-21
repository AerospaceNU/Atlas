"""One-shot ``atlas run``: a single agent turn with exit codes and a JSON result.

Built on :func:`atlas.agent.session.build_session`. Invariants: approval is explicit
(default deny) and the workspace ``agent.toml`` is never written or able to loosen it;
nothing touches disk before the workspace, prompt and key checks pass; output is redacted
with the session secrets.
"""

from __future__ import annotations

import contextlib
import json
import os
import shlex
import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Literal

from atlas.agent.contracts import ToolCapableModel
from atlas.agent.layout import resolve_openrouter_key
from atlas.agent.model import MISSING_KEY_MESSAGE, MissingOpenRouterKey
from atlas.agent.runtime import AgentRun, AgentStep
from atlas.agent.session import (
    AgentSession,
    SessionNotFound,
    _find_repo_root,
    build_session,
    done_event,
    redact_value,
    resolve_model,
    session_bound,
    status_fields,
    step_event,
)
from atlas.agent.sessions import call_summary, newest_session

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_TOOL_LIMIT = 3
EXIT_BUDGET = 4
EXIT_CONTEXT = 5
EXIT_CANCELLED = 6
EXIT_NOT_CONFIGURED = 7

SCHEMA_VERSION = 1
_STATUS_NAMES = {
    EXIT_OK: "ok",
    EXIT_ERROR: "error",
    EXIT_USAGE: "usage",
    EXIT_TOOL_LIMIT: "tool_call_limit",
    EXIT_BUDGET: "budget",
    EXIT_CONTEXT: "context",
    EXIT_CANCELLED: "cancelled",
    EXIT_NOT_CONFIGURED: "not_configured",
}


@dataclass(frozen=True)
class RunOptions:
    """Parsed ``atlas run`` arguments."""

    prompt: str | None
    prompt_file: Path | None
    workspace: Path
    model_id: str | None = None
    approval: Literal["allow", "deny"] = "deny"
    max_tool_calls: int | None = None
    max_spend_usd: float | None = None
    timeout: float | None = None
    resume: str | None = None
    output: Literal["text", "json"] = "text"
    quiet: bool = False
    trace: Path | None = None


class _Usage(Exception):
    """A usage error that maps to exit 2."""


def _emit_early(options: RunOptions, stdout: IO[str], stderr: IO[str], code: int, msg: str) -> int:
    """Report a failure that happened before any session existed."""
    if options.output == "json":
        payload = _payload(
            options,
            code=code,
            response="",
            error=msg,
            session_id=None,
            workspace=options.workspace,
            model=options.model_id,
            elapsed=0.0,
            steps=[],
            run=None,
            usage={},
            resume_cmd=None,
        )
        stdout.write(json.dumps(payload) + "\n")
        stdout.flush()
    if not options.quiet or options.output == "text":
        stderr.write(f"atlas: {msg}\n")
        stderr.flush()
    return code


def _read_prompt(options: RunOptions) -> str | None:
    """Return the prompt text, or None for a prompt-less resume; raise _Usage otherwise."""
    if options.prompt_file is not None:
        if options.prompt not in (None, ""):
            raise _Usage("pass either PROMPT or --prompt-file, not both")
        try:
            text = options.prompt_file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise _Usage(f"cannot read --prompt-file {options.prompt_file}: {exc}") from exc
    elif options.prompt == "-":
        text = sys.stdin.read()
    elif options.prompt is not None:
        text = options.prompt
    elif options.resume is not None:
        return None
    elif sys.stdin.isatty():
        raise _Usage("no prompt: pass PROMPT, - or --prompt-file")
    else:
        text = sys.stdin.read()
    text = text.strip()
    if text:
        return text
    if options.resume is not None and options.prompt is None and options.prompt_file is None:
        return None
    raise _Usage("no prompt: pass PROMPT, - or --prompt-file")


def _validate(options: RunOptions) -> str | None:
    """Check flags and the prompt; return the prompt (None for resume) or raise _Usage."""
    workspace = options.workspace
    if not workspace.exists() or not workspace.is_dir():
        raise _Usage(f"workspace {workspace} is not an existing directory")
    if options.max_tool_calls is not None and options.max_tool_calls < 1:
        raise _Usage("--max-tool-calls must be at least 1")
    if options.timeout is not None and options.timeout <= 0:
        raise _Usage("--timeout must be greater than 0")
    if options.max_spend_usd is not None and options.max_spend_usd < 0:
        raise _Usage("--max-spend-usd must not be negative")
    return _read_prompt(options)


def _load_checkout_env() -> None:
    """Load the checkout ``.env`` (never the workspace one), without overriding."""
    root = _find_repo_root(Path(__file__).resolve().parent) or _find_repo_root(Path.cwd())
    if root is None:
        return
    from dotenv import load_dotenv

    load_dotenv(root / ".env", override=False)


def _classify(run: AgentRun | None, timed_out: bool, error: str | None) -> int:
    """Map the outcome to an exit code; the first match wins."""
    if timed_out or (run is not None and run.cancelled):
        return EXIT_CANCELLED
    if run is not None:
        if run.stopped_for_limit:
            return EXIT_TOOL_LIMIT
        if run.stopped_for_budget:
            return EXIT_BUDGET
        if run.stopped_for_context:
            return EXIT_CONTEXT
        if run.error:
            return EXIT_ERROR
    if error is not None:
        return EXIT_ERROR
    return EXIT_OK


def _is_refused(step: AgentStep) -> bool:
    return step.error_kind == "denied" or "is refused:" in (step.error or "")


def _step_line(index: int, step: AgentStep) -> str:
    summary = call_summary(step.call.name, step.call.arguments)
    refused = _is_refused(step)
    state = "ok" if step.result is not None else ("refused" if refused else "error")
    label = summary if summary.startswith(step.call.name) else f"{step.call.name} {summary}"
    return f"[{index}] {label} {state}"


def _resume_command(code: int, session_id: str, workspace: Path) -> str:
    base = f"atlas run --resume {session_id} --workspace {shlex.quote(str(workspace))}"
    return base if code == EXIT_TOOL_LIMIT else f'{base} "continue"'


def _payload(
    options: RunOptions,
    *,
    code: int,
    response: str,
    error: str | None,
    session_id: str | None,
    workspace: Path,
    model: str | None,
    elapsed: float,
    steps: list[AgentStep],
    run: AgentRun | None,
    usage: dict[str, Any],
    resume_cmd: str | None,
) -> dict[str, Any]:
    """Build the schema-1 JSON result (unredacted)."""
    return {
        "schema": SCHEMA_VERSION,
        "ok": code == EXIT_OK,
        "exit_code": code,
        "status": _STATUS_NAMES[code],
        "response": response,
        "error": error,
        "session_id": session_id,
        "workspace": str(workspace),
        "model": model,
        "elapsed_seconds": round(elapsed, 3),
        "steps": [
            {
                "name": step.call.name,
                "call_id": step.call.id,
                "summary": call_summary(step.call.name, step.call.arguments),
                "ok": step.result is not None,
                "error": step.error,
                "error_kind": step.error_kind,
            }
            for step in steps
        ],
        "refused": sum(1 for step in steps if _is_refused(step)),
        "stopped_for_limit": bool(run and run.stopped_for_limit),
        "stopped_for_budget": bool(run and run.stopped_for_budget),
        "stopped_for_context": bool(run and run.stopped_for_context),
        "cancelled": bool(run and run.cancelled) or code == EXIT_CANCELLED,
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "spend": usage.get("spend"),
        },
        "resume": resume_cmd,
    }


def _footer(
    code: int,
    elapsed: float,
    steps: list[AgentStep],
    usage: dict[str, Any],
    session_id: str,
    resume_cmd: str | None,
    approval: str,
) -> list[str]:
    spend = usage.get("spend")
    cost = f", ${spend:.4f}" if isinstance(spend, int | float) else ""
    lines = [
        f"atlas: {_STATUS_NAMES[code]} in {elapsed:.0f}s, {len(steps)} steps{cost}, "
        f"session {session_id}"
    ]
    refused = sum(1 for step in steps if _is_refused(step))
    if refused and approval == "deny":
        lines.append(
            f"atlas: {refused} edit(s) refused: approval is deny. Re-run with --approval allow"
        )
    if resume_cmd:
        lines.append(f"atlas: resume with: {resume_cmd}")
    return lines


class _Cancel:
    """SIGINT and timeout wiring that sets the session cancel event."""

    def __init__(self, session: AgentSession, timeout: float | None) -> None:
        self.session = session
        self.timeout = timeout
        self.timed_out = False
        self._timer: threading.Timer | None = None
        self._previous: Any = None
        self._installed = False

    def _on_sigint(self, signum: int, frame: Any) -> None:
        self.session.cancel_event.set()
        signal.signal(signal.SIGINT, signal.default_int_handler)

    def _on_timeout(self) -> None:
        self.timed_out = True
        self.session.cancel_event.set()

    def start(self) -> None:
        if threading.current_thread() is threading.main_thread():
            self._previous = signal.signal(signal.SIGINT, self._on_sigint)
            self._installed = True
        if self.timeout is not None:
            self._timer = threading.Timer(self.timeout, self._on_timeout)
            self._timer.daemon = True
            self._timer.start()

    def stop(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
        if self._installed:
            signal.signal(signal.SIGINT, self._previous or signal.SIG_DFL)


def _resolve_resume(options: RunOptions) -> str | None:
    if options.resume is None:
        return None
    if options.resume == "last":
        found = newest_session(options.workspace)
        if found is None:
            raise SessionNotFound("no saved session last")
        return found
    return options.resume


def _write_trace(trace: IO[str] | None, secrets: list[str], event: dict[str, Any]) -> None:
    if trace is None:
        return
    trace.write(json.dumps(redact_value(event, secrets), default=str) + "\n")
    trace.flush()


def run_oneshot(
    options: RunOptions,
    *,
    stdout: IO[str],
    stderr: IO[str],
    model: ToolCapableModel | None = None,
) -> int:
    """Run one agent turn and return the process exit code (0-7).

    Args:
        options: Parsed flags.
        stdout: Receives the answer (text) or exactly one JSON line.
        stderr: Receives progress and the footer.
        model: Test seam; an injected model skips the key check and is not closed.

    Returns:
        Exit code: 0 ok, 1 error, 2 usage, 3 tool-call limit, 4 budget, 5 context stop,
        6 cancelled, 7 not configured.
    """
    try:
        prompt = _validate(options)
    except _Usage as exc:
        return _emit_early(options, stdout, stderr, EXIT_USAGE, str(exc))
    workspace = options.workspace.resolve()
    if sys.platform == "win32" and os.environ.get("ATLAS_ALLOW_NATIVE_WINDOWS") != "1":
        from atlas.cli.tui import native_windows_message

        return _emit_early(options, stdout, stderr, EXIT_NOT_CONFIGURED, native_windows_message())
    if model is None:
        _load_checkout_env()
        key = resolve_openrouter_key(
            session_key=None, env_key=os.environ.get("OPENROUTER_API_KEY"), home=Path.home()
        )
        if key is None:
            return _emit_early(options, stdout, stderr, EXIT_NOT_CONFIGURED, MISSING_KEY_MESSAGE)
    try:
        resume_id = _resolve_resume(options)
        bundle = build_session(
            workspace,
            model_id=resolve_model(workspace, options.model_id),
            approval=options.approval,
            max_tool_calls=options.max_tool_calls,
            max_spend_usd=options.max_spend_usd,
            resume=resume_id,
            write_config=False,
            home=Path.home(),
            model=model,
        )
    except SessionNotFound as exc:
        msg = f"{exc}; list: ls .atlas/sessions"
        return _emit_early(options, stdout, stderr, EXIT_USAGE, msg)
    try:
        return _run_session(options, bundle.session, workspace, prompt, stdout, stderr)
    finally:
        bundle.close()


def _run_session(
    options: RunOptions,
    session: AgentSession,
    workspace: Path,
    prompt: str | None,
    stdout: IO[str],
    stderr: IO[str],
) -> int:
    secrets = session.secrets
    steps: list[AgentStep] = []
    trace: IO[str] | None = None
    if options.trace is not None:
        try:
            trace = options.trace.open("a", encoding="utf-8")
        except OSError as exc:
            return _emit_early(options, stdout, stderr, EXIT_USAGE, f"cannot open --trace: {exc}")

    def on_step(step: AgentStep) -> None:
        steps.append(step)
        _write_trace(trace, secrets, step_event(step))
        if not options.quiet:
            line = redact_value(_step_line(len(steps), step), secrets)
            stderr.write(line + "\n")
            stderr.flush()

    cancel = _Cancel(session, options.timeout)
    run: AgentRun | None = None
    error: str | None = None
    started = time.monotonic()
    with session_bound(session):
        cancel.start()
        try:
            if prompt is not None:
                run = session.turn(prompt, on_step=on_step)
            else:
                run = session.resume(on_step=on_step)
        except KeyboardInterrupt:
            cancel.timed_out = True
        except MissingOpenRouterKey:
            _finish_trace(trace)
            return _emit_early(options, stdout, stderr, EXIT_NOT_CONFIGURED, MISSING_KEY_MESSAGE)
        except RuntimeError as exc:
            if prompt is None and "not stopped" in str(exc):
                _finish_trace(trace)
                return _emit_early(
                    options, stdout, stderr, EXIT_USAGE, "nothing to resume: give a PROMPT"
                )
            error = str(exc) or type(exc).__name__
        except Exception as exc:
            error = str(exc) or type(exc).__name__
        finally:
            cancel.stop()
        elapsed = time.monotonic() - started
        if run is None:
            partial = getattr(session.agent, "partial_run", None)
            if partial is not None:
                steps = list(getattr(partial, "steps", steps)) or steps
        code = _classify(run, cancel.timed_out, error)
        with contextlib.suppress(Exception):
            session.save_state()
        if run is not None and trace is not None:
            _write_trace(trace, secrets, done_event(session, run))
        _finish_trace(trace)
        usage = status_fields(session)
        _report(
            options, session, workspace, code, run, error, steps, usage, elapsed, stdout, stderr
        )
    return code


def _finish_trace(trace: IO[str] | None) -> None:
    if trace is not None:
        trace.close()


def _report(
    options: RunOptions,
    session: AgentSession,
    workspace: Path,
    code: int,
    run: AgentRun | None,
    error: str | None,
    steps: list[AgentStep],
    usage: dict[str, Any],
    elapsed: float,
    stdout: IO[str],
    stderr: IO[str],
) -> None:
    secrets = session.secrets
    session_id = session.session_id or ""
    saved = (
        bool(session.session_id)
        and (workspace / ".atlas" / "sessions" / session_id / "state.json").is_file()
    )
    resume_cmd = _resume_command(code, session_id, workspace) if saved and code else None
    response = run.response if run is not None else ""
    err = error or (run.error if run is not None else None)
    if options.output == "json":
        payload = _payload(
            options,
            code=code,
            response=response,
            error=err,
            session_id=session.session_id,
            workspace=workspace,
            model=session.model_id or usage.get("model"),
            elapsed=elapsed,
            steps=steps,
            run=run,
            usage=usage,
            resume_cmd=resume_cmd,
        )
        stdout.write(json.dumps(redact_value(payload, secrets)) + "\n")
        stdout.flush()
    elif code == EXIT_OK:
        stdout.write(redact_value(response, secrets).rstrip("\n") + "\n")
        stdout.flush()
    if options.quiet:
        return
    if options.output == "text" and code != EXIT_OK and response.strip():
        stderr.write(redact_value(response, secrets).rstrip("\n") + "\n")
    if err:
        stderr.write(f"atlas: error: {redact_value(err, secrets)}\n")
    for line in _footer(code, elapsed, steps, usage, session_id, resume_cmd, options.approval):
        stderr.write(redact_value(line, secrets) + "\n")
    stderr.flush()


__all__: list[str] = [
    "EXIT_BUDGET",
    "EXIT_CANCELLED",
    "EXIT_CONTEXT",
    "EXIT_ERROR",
    "EXIT_NOT_CONFIGURED",
    "EXIT_OK",
    "EXIT_TOOL_LIMIT",
    "EXIT_USAGE",
    "RunOptions",
    "run_oneshot",
]
