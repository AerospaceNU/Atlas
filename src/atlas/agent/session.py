from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import IO, Any

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import ChatMessage, Role, default_registry
from atlas.agent.model import ModelConfig, OpenRouterModel
from atlas.agent.runtime import (
    _SYSTEM_PROMPT,
    Agent,
    AgentRun,
    AgentStep,
    ensure_agent_config,
    load_model,
)

_REDACTED = "***"


class AgentSession:
    """A persistent agent conversation that survives across turns."""

    def __init__(self, agent: Agent) -> None:
        self.agent = agent
        self.messages: list[ChatMessage] = [ChatMessage(role=Role.SYSTEM, content=_SYSTEM_PROMPT)]
        self.stopped_for_limit = False

    def turn(
        self,
        request: str,
        *,
        on_step: Callable[[AgentStep], None] | None = None,
    ) -> AgentRun:
        """Append the user message and advance the shared history one turn."""
        self.messages.append(ChatMessage(role=Role.USER, content=request))
        run = self.agent.advance(self.messages, on_step=on_step)
        self.stopped_for_limit = run.stopped_for_limit
        return run

    def resume(self, *, on_step: Callable[[AgentStep], None] | None = None) -> AgentRun:
        """Continue a turn that stopped at the tool-call limit, with a fresh budget."""
        if not self.stopped_for_limit:
            raise RuntimeError("Session is not stopped for the tool-call limit")
        run = self.agent.advance(self.messages, on_step=on_step)
        self.stopped_for_limit = run.stopped_for_limit
        return run


def _write_line(stdout: IO[str], payload: dict[str, Any], redact: str | None) -> None:
    text = json.dumps(_redact_value(payload, redact), ensure_ascii=False)
    stdout.write(text + "\n")
    stdout.flush()


def _redact_value(value: Any, redact: str | None) -> Any:
    if not redact:
        return value
    if isinstance(value, str):
        return value.replace(redact, _REDACTED)
    if isinstance(value, dict):
        return {key: _redact_value(item, redact) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_value(item, redact) for item in value]
    return value


def _step_event(step: AgentStep) -> dict[str, Any]:
    result = step.result
    return {
        "type": "step",
        "name": step.call.name,
        "call_id": step.call.id,
        "arguments": step.call.arguments,
        "ok": result is not None,
        "text": result.text if result is not None else None,
        "error": step.error,
        "error_kind": step.error_kind,
        "artifacts": result.artifacts if result is not None else [],
    }


def serve(
    session: AgentSession,
    stdin: IO[str],
    stdout: IO[str],
    *,
    redact: str | None = None,
) -> int:
    """Run the line-delimited JSON protocol until EOF or ``quit``.

    Recoverable problems are reported as ``error`` events and the loop keeps
    reading; only ``quit`` (or EOF) ends the session.
    """
    _write_line(
        stdout,
        {"type": "ready", "tools": [tool.model_dump() for tool in session.agent.tools.definitions]},
        redact,
    )
    for line in stdin:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            message = json.loads(stripped)
        except json.JSONDecodeError:
            _write_line(stdout, {"type": "error", "message": "Invalid JSON input"}, redact)
            continue
        if not isinstance(message, dict):
            _write_line(stdout, {"type": "error", "message": "Expected a JSON object"}, redact)
            continue
        kind = message.get("type")
        if kind == "quit":
            return 0
        if kind == "user":
            text = message.get("text")
            if not isinstance(text, str) or not text:
                _write_line(
                    stdout,
                    {"type": "error", "message": "user text must be a non-empty string"},
                    redact,
                )
                continue
            on_step = _step_writer(stdout, redact)
            try:
                run = session.turn(text, on_step=on_step)
            except Exception as exc:
                _write_line(
                    stdout,
                    {"type": "error", "message": f"{type(exc).__name__}: {exc}"},
                    redact,
                )
                continue
            _write_line(stdout, _done_event(run), redact)
            continue
        if kind == "resume":
            on_step = _step_writer(stdout, redact)
            try:
                run = session.resume(on_step=on_step)
            except Exception as exc:
                _write_line(
                    stdout,
                    {"type": "error", "message": f"{type(exc).__name__}: {exc}"},
                    redact,
                )
                continue
            _write_line(stdout, _done_event(run), redact)
            continue
        _write_line(stdout, {"type": "error", "message": f"Unknown message type: {kind!r}"}, redact)
    return 0


def _step_writer(stdout: IO[str], redact: str | None) -> Callable[[AgentStep], None]:
    def write(step: AgentStep) -> None:
        _write_line(stdout, _step_event(step), redact)

    return write


def _done_event(run: AgentRun) -> dict[str, Any]:
    return {
        "type": "done",
        "response": run.response,
        "stopped_for_limit": run.stopped_for_limit,
    }


def _find_repo_root(start: Path) -> Path | None:
    for directory in (start, *start.parents):
        pyproject = directory / "pyproject.toml"
        if pyproject.is_file() and 'name = "atlas"' in pyproject.read_text(encoding="utf-8"):
            return directory
    return None


def resolve_model(workspace: Path, cli_model: str | None) -> str:
    """Choose the OpenRouter model id for a session.

    ``--model`` wins, then ``ATLAS_MODEL``, then ``model`` in the workspace
    ``.atlas/agent.toml``, then :data:`atlas.agent.runtime.DEFAULT_MODEL`.
    """
    if cli_model is not None and cli_model.strip():
        return cli_model.strip()
    env_model = os.environ.get("ATLAS_MODEL", "").strip()
    if env_model:
        return env_model
    return load_model(workspace)


def main(argv: Sequence[str] | None = None) -> int:
    """Console entry for ``python -m atlas.agent.session``."""
    parser = argparse.ArgumentParser(prog="atlas.agent.session")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--model", default=None)
    args = parser.parse_args(argv)

    workspace = Path(args.workspace).resolve()
    repo_root = _find_repo_root(workspace) or _find_repo_root(Path.cwd())
    if repo_root is not None:
        from dotenv import load_dotenv

        load_dotenv(repo_root / ".env", override=False)

    ensure_agent_config(workspace)
    api_key = os.environ.get("OPENROUTER_API_KEY") or ""
    if not api_key.strip():
        _write_line(sys.stdout, {"type": "error", "message": "OPENROUTER_API_KEY is not set"}, None)
        return 1

    model_id = resolve_model(workspace, args.model)
    model = OpenRouterModel(ModelConfig(model=model_id, api_key=api_key))
    try:
        agent = Agent(model, default_registry(), LocalArtifactStore(workspace))
        session = AgentSession(agent)
        return serve(session, sys.stdin, sys.stdout, redact=api_key)
    finally:
        model.close()


if __name__ == "__main__":
    raise SystemExit(main())
