from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any

from pydantic import ValidationError

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import CatalogModel, ChatMessage, Role, ToolRegistry, default_registry
from atlas.agent.layout import (
    AtlasPathError,
    ensure_project_layout,
    ensure_user_layout,
    list_entries,
    project_atlas_root,
    remove_entry,
    resolve_openrouter_key,
    user_atlas_root,
    write_user_key,
)
from atlas.agent.metrics import SessionTotals, apply_call, apply_http_error, format_tui_status
from atlas.agent.model import (
    MISSING_KEY_MESSAGE,
    MissingOpenRouterKey,
    ModelConfig,
    OpenRouterHTTPError,
    OpenRouterModel,
)
from atlas.agent.runtime import (
    Agent,
    AgentRun,
    AgentStep,
    ensure_agent_config,
    load_home_skills_enabled,
    load_max_tool_calls,
    load_model,
)
from atlas.agent.skill_author import normalize_skill_name
from atlas.agent.skills import Skill, load_skills, register_skill_tool

_REDACTED = "***"


class AgentSession:
    """A persistent agent conversation that survives across turns."""

    def __init__(self, agent: Agent) -> None:
        self.agent = agent
        self.messages: list[ChatMessage] = [
            ChatMessage(role=Role.SYSTEM, content=self.agent.system_prompt())
        ]
        self.stopped_for_limit = False
        self.session_key: str | None = None
        self.home: Path | None = None
        self.workspace: Path | None = None
        self.requires_api_key = False
        self.active_key: str | None = None
        self.secrets: list[str] = []
        self.model_id = ""
        self.catalog: list[CatalogModel] = []
        self.active_skills: list[str] = []
        self.totals = SessionTotals()
        self.fetch_models: Callable[[], list[CatalogModel]] | None = None
        config = getattr(agent.model, "config", None)
        if config is not None:
            self.model_id = str(getattr(config, "model", "") or "")
            self.totals.model = self.model_id
            key = str(getattr(config, "api_key", "") or "").strip()
            self.active_key = key or None

    def turn(
        self,
        request: str,
        *,
        on_step: Callable[[AgentStep], None] | None = None,
        on_phase: Callable[[str, str | None], None] | None = None,
        on_thought: Callable[[str, float | None], None] | None = None,
    ) -> AgentRun:
        """Append the user message and advance the shared history one turn."""
        self._require_key()
        self.messages.append(ChatMessage(role=Role.USER, content=request))
        run = self.agent.advance(
            self.messages, on_step=on_step, on_phase=on_phase, on_thought=on_thought
        )
        self.stopped_for_limit = run.stopped_for_limit
        self._note(run)
        return run

    def resume(
        self,
        *,
        on_step: Callable[[AgentStep], None] | None = None,
        on_phase: Callable[[str, str | None], None] | None = None,
        on_thought: Callable[[str, float | None], None] | None = None,
    ) -> AgentRun:
        """Continue a turn that stopped at the tool-call limit, with a fresh budget."""
        self._require_key()
        if not self.stopped_for_limit:
            raise RuntimeError("Session is not stopped for the tool-call limit")
        run = self.agent.advance(
            self.messages, on_step=on_step, on_phase=on_phase, on_thought=on_thought
        )
        self.stopped_for_limit = run.stopped_for_limit
        self._note(run)
        return run

    def start_fresh(self) -> None:
        """Start a new conversation without reopening the process.

        History and running totals return to their initial state. A skill
        applied during this conversation leaves with that history. The skill
        catalog on the system prompt stays. The current model id and context
        window are kept, and writes move to a new session artifact directory
        inside the same workspace. Reads keep the previous read root, so this
        never widens write access to the workspace.
        """
        self.messages = [ChatMessage(role=Role.SYSTEM, content=self.agent.system_prompt())]
        self.active_skills = []
        self.stopped_for_limit = False
        self.totals = SessionTotals(
            model=self.model_id or self.totals.model,
            context_limit=self.totals.context_limit,
        )
        if self.workspace is None:
            return
        session_id = _remember_session(self.workspace, self.model_id)
        artifact_root = project_atlas_root(self.workspace) / "artifacts" / session_id
        read_root = self.agent.artifacts.read_root
        self.agent.artifacts = LocalArtifactStore(artifact_root, read_root=read_root)

    def use_skill(self, name: str, argument: str | None = None) -> Skill:
        """Apply a named skill for later turns in this conversation.

        The name is looked up in the catalog loaded at session start
        (``self.agent.skills``). A skill written after that catalog was built
        is not available. The first activation re-reads the file and refuses
        it when the bytes changed, then appends the tool text as a user
        message. A later activation of the same skill appends only a new
        ``Argument:`` line. The system prompt stays the catalog.

        Args:
            name: Skill name from the catalog, matched with
                :func:`atlas.agent.skill_author.normalize_skill_name`.
            argument: Optional text from the user, kept outside the skill body.

        Returns:
            The skill that was activated.

        Raises:
            ValueError: ``name`` is empty, fails
                :func:`atlas.agent.skill_author.normalize_skill_name`, the
                file changed, or it cannot be parsed.
            FileNotFoundError: No skill with that name is in the catalog.
        """
        cleaned = name.strip()
        if not cleaned:
            raise ValueError("skill name must be a non-empty string")
        requested, errors = normalize_skill_name(cleaned)
        if requested is None or errors:
            raise ValueError("; ".join(errors))
        skills = list(self.agent.skills)
        skill = next((item for item in skills if item.name == requested), None)
        if skill is None:
            raise FileNotFoundError(f"Skill {requested} is missing")
        extra = argument.strip() if isinstance(argument, str) else ""
        if skill.name in self.active_skills:
            if extra:
                self.messages.append(ChatMessage(role=Role.USER, content=f"Argument: {extra}"))
            return skill
        registry = ToolRegistry()
        register_skill_tool(registry, skills)
        result = registry.execute("use_skill", {"name": requested}, self.agent.artifacts)
        content = result.text
        if extra:
            content = f"{content}\n\nArgument: {extra}"
        self.messages.append(ChatMessage(role=Role.USER, content=content))
        self.active_skills.append(skill.name)
        return skill

    def _require_key(self) -> None:
        if self.requires_api_key and not (self.active_key and self.active_key.strip()):
            raise MissingOpenRouterKey(MISSING_KEY_MESSAGE)

    def _note(self, run: AgentRun) -> None:
        for call in run.calls:
            apply_call(self.totals, call)


def _write_line(stdout: IO[str], payload: dict[str, Any], secrets: list[str]) -> None:
    text = json.dumps(_redact_value(payload, secrets), ensure_ascii=False)
    stdout.write(text + "\n")
    stdout.flush()


def _redact_value(value: Any, secrets: list[str]) -> Any:
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, _REDACTED)
        return value
    if isinstance(value, dict):
        return {key: _redact_value(item, secrets) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_value(item, secrets) for item in value]
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
    reading; only ``quit`` (or EOF) ends the session. Key material in
    ``redact`` and in ``session.secrets`` is replaced before a line is written.
    """
    if redact and redact not in session.secrets:
        session.secrets.append(redact)
    _write_line(stdout, _ready_event(session), session.secrets)
    for line in stdin:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            message = json.loads(stripped)
        except json.JSONDecodeError:
            _write_line(stdout, {"type": "error", "message": "Invalid JSON input"}, session.secrets)
            continue
        if not isinstance(message, dict):
            _write_line(
                stdout, {"type": "error", "message": "Expected a JSON object"}, session.secrets
            )
            continue
        kind = message.get("type")
        if kind == "quit":
            return 0
        if kind == "user":
            _handle_user(session, stdout, message)
            continue
        if kind == "resume":
            _handle_resume(session, stdout)
            continue
        if kind == "new":
            _handle_new(session, stdout)
            continue
        if kind == "skill":
            _handle_skill(session, stdout, message)
            continue
        if kind == "models":
            _handle_models(session, stdout)
            continue
        if kind == "select_model":
            _handle_select_model(session, stdout, message)
            continue
        if kind == "set_key":
            _handle_set_key(session, stdout, message)
            continue
        if kind == "list":
            _handle_list(session, stdout, message)
            continue
        if kind == "remove":
            _handle_remove(session, stdout, message)
            continue
        _write_line(
            stdout,
            {"type": "error", "message": f"Unknown message type: {kind!r}"},
            session.secrets,
        )
    return 0


def _handle_user(session: AgentSession, stdout: IO[str], message: dict[str, Any]) -> None:
    text = message.get("text")
    if not isinstance(text, str) or not text:
        _write_line(
            stdout,
            {"type": "error", "message": "user text must be a non-empty string"},
            session.secrets,
        )
        return
    on_step = _step_writer(stdout, session.secrets)
    on_phase = _phase_writer(stdout, session.secrets)
    on_thought = _thought_writer(stdout, session.secrets)
    try:
        run = session.turn(text, on_step=on_step, on_phase=on_phase, on_thought=on_thought)
    except MissingOpenRouterKey as exc:
        _write_line(stdout, {"type": "error", "message": str(exc)}, session.secrets)
        return
    except OpenRouterHTTPError as exc:
        _write_http_error(session, stdout, exc)
        return
    except Exception as exc:
        _write_line(
            stdout,
            {"type": "error", "message": f"{type(exc).__name__}: {exc}"},
            session.secrets,
        )
        return
    _write_line(stdout, _done_event(session, run), session.secrets)


def _handle_resume(session: AgentSession, stdout: IO[str]) -> None:
    on_step = _step_writer(stdout, session.secrets)
    on_phase = _phase_writer(stdout, session.secrets)
    on_thought = _thought_writer(stdout, session.secrets)
    try:
        run = session.resume(on_step=on_step, on_phase=on_phase, on_thought=on_thought)
    except MissingOpenRouterKey as exc:
        _write_line(stdout, {"type": "error", "message": str(exc)}, session.secrets)
        return
    except OpenRouterHTTPError as exc:
        _write_http_error(session, stdout, exc)
        return
    except Exception as exc:
        _write_line(
            stdout,
            {"type": "error", "message": f"{type(exc).__name__}: {exc}"},
            session.secrets,
        )
        return
    _write_line(stdout, _done_event(session, run), session.secrets)


def _handle_new(session: AgentSession, stdout: IO[str]) -> None:
    """Start a fresh conversation and report the reset status."""
    session.start_fresh()
    _write_line(stdout, _ready_event(session), session.secrets)


def _handle_skill(session: AgentSession, stdout: IO[str], message: dict[str, Any]) -> None:
    """Apply a named skill without calling the model."""
    name = message.get("name")
    argument = message.get("argument")
    if not isinstance(name, str) or not name.strip():
        _write_line(
            stdout,
            {"type": "error", "message": "skill name must be a non-empty string"},
            session.secrets,
        )
        return
    if argument is not None and not isinstance(argument, str):
        _write_line(
            stdout,
            {"type": "error", "message": "skill argument must be a string"},
            session.secrets,
        )
        return
    try:
        skill = session.use_skill(name, argument if isinstance(argument, str) else None)
    except OSError:
        _write_line(
            stdout,
            {"type": "error", "message": f"Skill {name.strip()} could not be read"},
            session.secrets,
        )
        return
    except (ValueError, ValidationError, KeyError) as exc:
        _write_line(stdout, {"type": "error", "message": str(exc)}, session.secrets)
        return
    _write_line(
        stdout,
        {
            "type": "skill",
            "name": skill.name,
            "description": skill.description,
            "location": skill.directory,
            "active": list(session.active_skills),
        },
        session.secrets,
    )


def _handle_models(session: AgentSession, stdout: IO[str]) -> None:
    """Fetch the catalog. A failure leaves the selected model id unchanged."""
    previous = session.model_id
    fetcher = session.fetch_models
    if fetcher is None:
        _write_line(
            stdout,
            {
                "type": "models",
                "ok": False,
                "message": "model catalog is unavailable",
                "models": [],
            },
            session.secrets,
        )
        return
    try:
        catalog = fetcher()
    except Exception as exc:
        session.model_id = previous
        _write_line(
            stdout,
            {"type": "models", "ok": False, "message": _public_error(exc), "models": []},
            session.secrets,
        )
        return
    session.catalog = catalog
    _apply_known_context_limit(session)
    payload: dict[str, Any] = {
        "type": "models",
        "ok": True,
        "message": "",
        "models": [model.model_dump() for model in catalog],
    }
    payload.update(_status_fields(session))
    _write_line(stdout, payload, session.secrets)


def _handle_select_model(session: AgentSession, stdout: IO[str], message: dict[str, Any]) -> None:
    model_id = message.get("id")
    previous = session.model_id
    previous_limit = session.totals.context_limit
    if not isinstance(model_id, str) or not model_id.strip():
        _write_line(
            stdout,
            {"type": "error", "message": "model id must be a non-empty string"},
            session.secrets,
        )
        return
    chosen = next((item for item in session.catalog if item.id == model_id), None)
    if chosen is None:
        session.model_id = previous
        session.totals.context_limit = previous_limit
        _write_line(
            stdout,
            {"type": "error", "message": f"Unknown model: {model_id}"},
            session.secrets,
        )
        return
    config = getattr(session.agent.model, "config", None)
    if config is not None:
        config.model = chosen.id
    session.model_id = chosen.id
    session.totals.model = chosen.id
    session.totals.context_limit = chosen.context_length
    payload = {"type": "model", "id": chosen.id, "context_length": chosen.context_length}
    payload.update(_status_fields(session))
    _write_line(stdout, payload, session.secrets)


def _handle_set_key(session: AgentSession, stdout: IO[str], message: dict[str, Any]) -> None:
    scope = message.get("scope")
    key = message.get("key")
    if scope not in {"session", "user"} or not isinstance(key, str) or not key.strip():
        _write_line(
            stdout,
            {"type": "error", "message": "set_key needs scope session or user and a key"},
            session.secrets,
        )
        return
    cleaned = key.strip()
    if cleaned not in session.secrets:
        session.secrets.append(cleaned)
    try:
        if scope == "user":
            write_user_key(cleaned, session.home)
            if not (session.session_key and session.session_key.strip()):
                _install_key(session, cleaned)
        else:
            session.session_key = cleaned
            _install_key(session, cleaned)
    except Exception as exc:
        _write_line(
            stdout,
            {"type": "error", "message": f"Could not save the key: {type(exc).__name__}"},
            session.secrets,
        )
        return
    _write_line(stdout, {"type": "key", "scope": scope, "saved": True}, session.secrets)


def _handle_list(session: AgentSession, stdout: IO[str], message: dict[str, Any]) -> None:
    try:
        root = _atlas_root(session, message.get("scope"))
        names = list_entries(root, str(message.get("kind")))
    except (AtlasPathError, TypeError) as exc:
        _write_line(stdout, {"type": "error", "message": str(exc)}, session.secrets)
        return
    _write_line(
        stdout,
        {
            "type": "listing",
            "scope": message.get("scope"),
            "kind": message.get("kind"),
            "names": names,
        },
        session.secrets,
    )


def _handle_remove(session: AgentSession, stdout: IO[str], message: dict[str, Any]) -> None:
    name = message.get("name")
    kind = message.get("kind")
    if not isinstance(name, str) or not isinstance(kind, str):
        _write_line(
            stdout,
            {"type": "error", "message": "remove needs a kind and a name"},
            session.secrets,
        )
        return
    try:
        root = _atlas_root(session, message.get("scope"))
        remove_entry(root, kind, name, confirm=message.get("confirm") is True)
    except (AtlasPathError, FileNotFoundError, TypeError) as exc:
        _write_line(stdout, {"type": "error", "message": str(exc)}, session.secrets)
        return
    _write_line(
        stdout,
        {"type": "removed", "scope": message.get("scope"), "kind": kind, "name": name},
        session.secrets,
    )


def _install_key(session: AgentSession, key: str) -> None:
    session.active_key = key
    config = getattr(session.agent.model, "config", None)
    if config is not None:
        config.api_key = key


def _atlas_root(session: AgentSession, scope: object) -> Path:
    if scope == "project":
        if session.workspace is None:
            raise AtlasPathError("project workspace is not set")
        return project_atlas_root(session.workspace)
    if scope == "user":
        return user_atlas_root(session.home)
    raise AtlasPathError("scope must be project or user")


def _write_http_error(session: AgentSession, stdout: IO[str], exc: OpenRouterHTTPError) -> None:
    apply_http_error(
        session.totals,
        http_status=exc.status_code,
        elapsed_seconds=exc.elapsed_seconds,
        time_to_first_token_seconds=exc.time_to_first_token_seconds,
    )
    payload: dict[str, Any] = {
        "type": "error",
        "message": str(exc),
        "http_status": exc.status_code,
        "elapsed_seconds": exc.elapsed_seconds,
        "time_to_first_token_seconds": exc.time_to_first_token_seconds,
        "error_body": exc.body,
    }
    payload.update(_status_fields(session))
    _write_line(stdout, payload, session.secrets)


def _public_error(exc: Exception) -> str:
    if isinstance(exc, MissingOpenRouterKey | OpenRouterHTTPError):
        return str(exc)
    text = f"{type(exc).__name__}: {exc}"
    return text[:300]


def _step_writer(stdout: IO[str], secrets: list[str]) -> Callable[[AgentStep], None]:
    def write(step: AgentStep) -> None:
        _write_line(stdout, _step_event(step), secrets)

    return write


def _thought_writer(stdout: IO[str], secrets: list[str]) -> Callable[[str, float | None], None]:
    def write(text: str, speed: float | None) -> None:
        if not text.strip():
            return
        payload: dict[str, Any] = {"type": "thought", "text": text}
        if speed is not None and speed > 0:
            payload["tokens_per_second"] = speed
        _write_line(stdout, payload, secrets)

    return write


def _phase_writer(stdout: IO[str], secrets: list[str]) -> Callable[[str, str | None], None]:
    def write(phase: str, name: str | None) -> None:
        payload: dict[str, Any] = {"type": "phase", "phase": phase}
        if name:
            payload["name"] = name
        _write_line(stdout, payload, secrets)

    return write


def _key_set(session: AgentSession) -> bool:
    if not session.requires_api_key:
        return True
    return bool(session.active_key and session.active_key.strip())


def _apply_known_context_limit(session: AgentSession) -> None:
    """Copy the active model's catalog window onto the dashboard totals.

    The limit stays unset when the catalog has not been loaded or has no row
    for the model. A known window replaces ``unknown`` on the next status line.
    """
    model_id = session.totals.model or session.model_id
    if not model_id:
        return
    chosen = next((item for item in session.catalog if item.id == model_id), None)
    if chosen is not None and chosen.context_length is not None:
        session.totals.context_limit = chosen.context_length


def _status_fields(session: AgentSession) -> dict[str, Any]:
    _apply_known_context_limit(session)
    totals = session.totals
    return {
        "status_line": format_tui_status(totals),
        "model": totals.model,
        "context_used": totals.context_used,
        "context_limit": totals.context_limit,
        "prompt_tokens": totals.prompt_tokens,
        "completion_tokens": totals.completion_tokens,
        "total_tokens": totals.total_tokens,
        "spend": totals.cost,
        "tokens_per_second": totals.tokens_per_second,
        "http_status": totals.last_http_status,
        "elapsed_seconds": totals.last_elapsed_seconds,
        "time_to_first_token_seconds": totals.last_ttft_seconds,
        "recent": list(totals.recent),
    }


def _ready_event(session: AgentSession) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": "ready",
        "tools": [tool.model_dump() for tool in session.agent.tools.definitions],
        "key_set": _key_set(session),
    }
    payload.update(_status_fields(session))
    return payload


def _done_event(session: AgentSession, run: AgentRun) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": "done",
        "response": run.response,
        "stopped_for_limit": run.stopped_for_limit,
    }
    payload.update(_status_fields(session))
    return payload


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


def _remember_session(workspace: Path, model_id: str) -> str:
    """Create a project session directory and a matching artifact directory."""
    root = ensure_project_layout(workspace)
    session_id = uuid.uuid4().hex[:12]
    session_dir = root / "sessions" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    (root / "artifacts" / session_id).mkdir(parents=True, exist_ok=True)
    (session_dir / "meta.json").write_text(
        json.dumps(
            {
                "id": session_id,
                "model": model_id,
                "created": datetime.now(UTC).isoformat(),
            }
        ),
        encoding="utf-8",
    )
    return session_id


def main(argv: Sequence[str] | None = None) -> int:
    """Console entry for ``python -m atlas.agent.session``."""
    parser = argparse.ArgumentParser(prog="atlas.agent.session")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--model", default=None)
    args = parser.parse_args(argv)

    workspace = Path(args.workspace).resolve()
    home = Path.home()
    repo_root = _find_repo_root(workspace) or _find_repo_root(Path.cwd())
    if repo_root is not None:
        from dotenv import load_dotenv

        load_dotenv(repo_root / ".env", override=False)

    ensure_agent_config(workspace)
    ensure_user_layout(home)
    api_key = (
        resolve_openrouter_key(
            session_key=None,
            env_key=os.environ.get("OPENROUTER_API_KEY"),
            home=home,
        )
        or ""
    )
    model_id = resolve_model(workspace, args.model)
    session_id = _remember_session(workspace, model_id)
    # Writes stay inside this session's artifact directory so removing the
    # session removes them and cannot touch the workspace root. Reads may
    # fall back to the project workspace via the store's read_root.
    artifact_root = project_atlas_root(workspace) / "artifacts" / session_id
    store = LocalArtifactStore(artifact_root, read_root=workspace)
    # Home skills are opt-in. The default session sees only the workspace.
    skill_home = home if load_home_skills_enabled(workspace) else None
    skills = load_skills(store, home=skill_home)
    registry = default_registry()
    register_skill_tool(registry, skills)
    model = OpenRouterModel(ModelConfig(model=model_id, api_key=api_key))
    try:
        agent = Agent(
            model,
            registry,
            store,
            # The store root is the session artifact directory, which has no
            # agent.toml. The budget lives in the workspace config.
            max_tool_calls=load_max_tool_calls(workspace),
            skills=skills,
        )
        session = AgentSession(agent)
        session.requires_api_key = True
        session.home = home
        session.workspace = workspace
        session.active_key = api_key or None
        session.fetch_models = model.list_models
        return serve(session, sys.stdin, sys.stdout, redact=api_key or None)
    finally:
        model.close()


if __name__ == "__main__":
    raise SystemExit(main())
