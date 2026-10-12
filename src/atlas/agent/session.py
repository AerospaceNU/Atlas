from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import uuid
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any

from pydantic import ValidationError

from atlas.agent.analysis_skill import (
    AnalysisStep,
    AnalysisTurn,
    FinishedAnalysis,
    SavedSkill,
    SkillPreview,
    SkillSaveError,
    enable_command,
    enable_skill,
    preview_skill,
    save_skill,
    skill_slug,
    successful_steps,
)
from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import CatalogModel, ChatMessage, Role, ToolRegistry, default_registry
from atlas.agent.images import ingest_images
from atlas.agent.layout import (
    EARTHDATA_TOKEN,
    FIRMS_MAP_KEY,
    KEY_VARIABLES,
    OPENROUTER_API_KEY,
    AtlasPathError,
    bind_session_credentials,
    bind_session_secrets,
    delete_user_key,
    ensure_project_layout,
    ensure_user_layout,
    list_entries,
    prepare_credential,
    prepare_openrouter_key,
    project_atlas_root,
    quiet_http_client_logs,
    read_user_key,
    remove_entry,
    resolve_credential,
    resolve_openrouter_key,
    unbind_session_credentials,
    unbind_session_secrets,
    user_atlas_root,
    write_user_key,
)
from atlas.agent.metrics import (
    SessionTotals,
    apply_call,
    apply_http_error,
    format_tui_status,
    reported_spend,
)
from atlas.agent.model import (
    _CATALOG_TIME_LIMIT,
    MISSING_KEY_MESSAGE,
    CatalogFetch,
    MissingOpenRouterKey,
    ModelConfig,
    OpenRouterHTTPError,
    OpenRouterModel,
    peek_model_catalog,
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
_STDOUT_LOCK = threading.Lock()


class AgentSession:
    """A persistent agent conversation that survives across turns.

    The first message is the trusted system prompt. When the project root
    contains an honored instruction file, the next message is that file as
    delimited project guidance.
    """

    def __init__(self, agent: Agent) -> None:
        self.agent = agent
        self.messages: list[ChatMessage] = self.agent.opening_messages()
        self.stopped_for_limit = False
        self.session_key: str | None = None
        self.credential_keys: dict[str, str] = {}
        self.home: Path | None = None
        self.workspace: Path | None = None
        self.requires_api_key = False
        self.active_key: str | None = None
        self.secrets: list[str] = []
        self.model_id = ""
        self.catalog: list[CatalogModel] = []
        self.active_skills: list[str] = []
        self.totals = SessionTotals()
        self.fetch_models: Callable[[], CatalogFetch | list[CatalogModel]] | None = None
        self._catalog_thread: threading.Thread | None = None
        self._catalog_inflight = False
        self._catalog_lock = threading.Lock()
        config = getattr(agent.model, "config", None)
        if config is not None:
            self.model_id = str(getattr(config, "model", "") or "")
            self.totals.model = self.model_id
            key = str(getattr(config, "api_key", "") or "").strip()
            self.active_key = key or None
            if key:
                self.secrets.append(key)
        self.skill_catalog_note = ""
        self._skill_previews: dict[tuple[str, str], str] = {}
        self._analysis_turns: list[AnalysisTurn] = []
        self._analysis_open = False
        self._analysis_request = ""
        self._analysis_steps: list[AnalysisStep] = []
        self._analysis_response = ""

    def turn(
        self,
        request: str,
        *,
        images: Sequence[str] = (),
        on_step: Callable[[AgentStep], None] | None = None,
        on_phase: Callable[[str, str | None], None] | None = None,
        on_thought: Callable[[str, float | None], None] | None = None,
    ) -> AgentRun:
        """Append the user message and advance the shared history one turn.

        ``images`` are paths in the write sandbox or the project workspace.
        A path the sandbox rejects is not appended.
        """
        self._require_key()
        message = ChatMessage(role=Role.USER, content=request)
        if images:
            message.images = ingest_images(self.agent.artifacts, list(images))
        self._begin_analysis(request)
        self.messages.append(message)
        run = self.agent.advance(
            self.messages, on_step=on_step, on_phase=on_phase, on_thought=on_thought
        )
        self.stopped_for_limit = run.stopped_for_limit
        self._note(run)
        self._capture_analysis(run)
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
        self._capture_analysis(run)
        return run

    def start_fresh(self) -> None:
        """Start a new conversation without reopening the process.

        History and running totals return to their initial state. A skill
        applied during this conversation leaves with that history. The skill
        catalog on the system prompt stays. Project instructions are loaded
        again from the project root after writes move; they are not copied
        from the previous transcript. The current model id and context
        window are kept, and writes move to a new session artifact directory
        inside the same workspace. Reads keep the previous read root, so this
        never widens write access to the workspace.
        """
        self.active_skills = []
        self.stopped_for_limit = False
        self.totals = SessionTotals(
            model=self.model_id or self.totals.model,
            context_limit=self.totals.context_limit,
        )
        self._clear_analysis()
        if self.workspace is not None:
            session_id = _remember_session(self.workspace, self.model_id)
            artifact_root = project_atlas_root(self.workspace) / "artifacts" / session_id
            read_root = self.agent.artifacts.read_root
            self.agent.artifacts = LocalArtifactStore(artifact_root, read_root=read_root)
        self.messages = self.agent.opening_messages()

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
        prompt_price, completion_price, cache_read_price = _catalog_prices(self)
        for call in run.calls:
            apply_call(
                self.totals,
                call,
                prompt_price=prompt_price,
                completion_price=completion_price,
                cache_read_price=cache_read_price,
            )

    def save_finished_skill(
        self,
        name: str,
        *,
        description: str | None = None,
        replace: bool = False,
        scope: str = "project",
    ) -> SavedSkill:
        """Save the finished analysis as a draft in the shared drafts directory."""
        if self._analysis_open:
            raise SkillSaveError("analysis is not finished")
        if not self._analysis_turns:
            raise SkillSaveError("no finished analysis to save")
        if self.workspace is None:
            raise SkillSaveError("project scope needs a workspace")
        return save_skill(
            self.workspace,
            FinishedAnalysis(turns=list(self._analysis_turns)),
            name,
            description=description,
            replace=replace,
            secrets=self.secrets,
            scope=scope,
            home=self.home,
        )

    def preview_saved_skill(self, name: str, *, scope: str = "project") -> SkillPreview:
        """Show a draft's description and body hash without promoting it."""
        self._require_user_skills(scope)
        if self.workspace is None:
            raise SkillSaveError("project scope needs a workspace")
        slug = skill_slug(name)
        preview = preview_skill(self.workspace, slug, scope=scope, home=self.home)
        self._skill_previews[(slug, scope)] = preview.content_sha256
        return preview

    def enable_saved_skill(
        self,
        name: str,
        expected_sha256: str,
        *,
        scope: str = "project",
    ) -> SavedSkill:
        """Move a previewed draft into the live skills directory.

        ``expected_sha256`` is the digest from :meth:`preview_saved_skill`, or
        its first eight hex characters. Enabling without that preview is refused.
        """
        self._require_user_skills(scope)
        if self.workspace is None:
            raise SkillSaveError("project scope needs a workspace")
        slug = skill_slug(name)
        full_hash = self._expected_preview_hash(slug, scope, expected_sha256)
        saved = enable_skill(
            self.workspace,
            slug,
            full_hash,
            scope=scope,
            home=self.home,
        )
        self._skill_previews.pop((saved.name, scope), None)
        self.skill_catalog_note = _add_enabled_skill(self, saved.name, scope)
        return saved

    def _expected_preview_hash(self, name: str, scope: str, presented: str) -> str:
        stored = self._skill_previews.get((name, scope))
        if stored is None:
            raise SkillSaveError("enable needs a preview")
        token = presented.strip().lower()
        if token != stored and not (len(token) == 8 and stored.startswith(token)):
            raise SkillSaveError("skill hash does not match the preview")
        return stored

    def _require_user_skills(self, scope: str) -> None:
        if scope != "user":
            return
        if self.workspace is None or not load_home_skills_enabled(self.workspace):
            raise SkillSaveError(
                "home skills are off; set ATLAS_HOME_SKILLS or home_skills = true "
                "in ~/.atlas/config.toml"
            )

    def _begin_analysis(self, request: str) -> None:
        # A turn that stopped for the tool-call limit is not finished.
        # Starting another turn drops it instead of recording it.
        self._analysis_open = True
        self._analysis_request = request
        self._analysis_steps = []
        self._analysis_response = ""

    def _capture_analysis(self, run: AgentRun) -> None:
        if not self._analysis_open:
            return
        self._analysis_steps.extend(successful_steps(run))
        self._analysis_response = run.response
        if run.stopped_for_limit:
            return
        self._analysis_turns.append(
            AnalysisTurn(
                request=self._analysis_request,
                response=self._analysis_response,
                steps=list(self._analysis_steps),
            )
        )
        self._analysis_open = False
        self._analysis_request = ""
        self._analysis_steps = []
        self._analysis_response = ""

    def _clear_analysis(self) -> None:
        self._analysis_turns = []
        self._analysis_open = False
        self._analysis_request = ""
        self._analysis_steps = []
        self._analysis_response = ""


def _write_line(stdout: IO[str], payload: dict[str, Any], secrets: list[str]) -> None:
    text = json.dumps(_redact_value(payload, secrets), ensure_ascii=False)
    with _STDOUT_LOCK:
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
    bind_session_credentials(session.credential_keys)
    bind_session_secrets(session.secrets)
    quiet_http_client_logs()
    try:
        _remember_resolved_credentials(session)
        return _serve_bound(session, stdin, stdout)
    finally:
        unbind_session_credentials()
        unbind_session_secrets()


def _serve_bound(session: AgentSession, stdin: IO[str], stdout: IO[str]) -> int:
    """Read the protocol after the session credential map is bound."""
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
            _join_catalog_fetch(session)
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
        if kind == "delete_key":
            _handle_delete_key(session, stdout, message)
            continue
        if kind == "list":
            _handle_list(session, stdout, message)
            continue
        if kind == "remove":
            _handle_remove(session, stdout, message)
            continue
        if kind == "save_skill":
            _handle_save_skill(session, stdout, message)
            continue
        if kind == "enable_skill":
            _handle_enable_skill(session, stdout, message)
            continue
        _write_line(
            stdout,
            {"type": "error", "message": f"Unknown message type: {kind!r}"},
            session.secrets,
        )
    _join_catalog_fetch(session)
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
    """Start a catalog fetch. A failure leaves the selected model id unchanged.

    A fresh cache hit is written immediately. Otherwise paging runs on a
    background thread so the session loop can keep reading. The thread is
    joined when the loop ends.
    """
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
    key = _catalog_api_key(session)
    cached = peek_model_catalog(key) if key else None
    if cached is not None:
        _publish_catalog(session, stdout, CatalogFetch(models=cached))
        return
    if session._catalog_inflight:
        return
    session._catalog_inflight = True
    thread = threading.Thread(
        target=_catalog_worker,
        args=(session, stdout),
        daemon=True,
        name="atlas-model-catalog",
    )
    session._catalog_thread = thread
    thread.start()


def _catalog_worker(session: AgentSession, stdout: IO[str]) -> None:
    try:
        fetcher = session.fetch_models
        if fetcher is None:
            fetched = CatalogFetch(models=[], error="model catalog is unavailable")
        else:
            fetched = _coerce_catalog(fetcher())
        _publish_catalog(session, stdout, fetched)
    except Exception as exc:
        _write_line(
            stdout,
            {"type": "models", "ok": False, "message": _public_error(exc), "models": []},
            session.secrets,
        )
    finally:
        session._catalog_inflight = False


def _join_catalog_fetch(session: AgentSession) -> None:
    thread = session._catalog_thread
    if thread is not None and thread is not threading.current_thread():
        thread.join(timeout=_CATALOG_TIME_LIMIT + 5)


def _coerce_catalog(value: CatalogFetch | list[CatalogModel]) -> CatalogFetch:
    if isinstance(value, CatalogFetch):
        return value
    return CatalogFetch(models=list(value))


def _include_selected_model(
    session: AgentSession, models: list[CatalogModel]
) -> list[CatalogModel]:
    """Keep the active model in the list when a fetch does not return it.

    A short or empty catalog would otherwise leave the picker with nothing to
    select. A row already stored for that id keeps its context and tool flag.
    """
    selected = session.model_id.strip()
    if not selected or any(model.id == selected for model in models):
        return list(models)
    previous = next((model for model in session.catalog if model.id == selected), None)
    kept = list(models)
    if previous is not None:
        kept.append(previous)
    else:
        kept.append(CatalogModel(id=selected, supports_tools=True))
    return kept


def _publish_catalog(session: AgentSession, stdout: IO[str], fetched: CatalogFetch) -> None:
    with session._catalog_lock:
        models = _include_selected_model(session, fetched.models)
        session.catalog = models
    _apply_known_context_limit(session)
    payload: dict[str, Any] = {
        "type": "models",
        "ok": fetched.error is None,
        "message": fetched.error or "",
        "models": [model.model_dump() for model in models],
    }
    payload.update(_status_fields(session))
    _write_line(stdout, payload, session.secrets)


def _catalog_api_key(session: AgentSession) -> str:
    config = getattr(session.agent.model, "config", None)
    if config is not None:
        key = str(getattr(config, "api_key", "") or "").strip()
        if key:
            return key
    return (session.active_key or "").strip()


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
    with session._catalog_lock:
        chosen = next((item for item in session.catalog if item.id == model_id), None)
        if chosen is not None:
            chosen = chosen.model_copy()
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


_KEY_NAME_ERROR = "set_key name must be OPENROUTER_API_KEY, EARTHDATA_TOKEN, or FIRMS_MAP_KEY"


def _credential_name(message: dict[str, Any]) -> str:
    """Return the key variable named by ``message``, defaulting to OpenRouter.

    Raises:
        ValueError: if ``name`` is present and not an allowed variable. The
        message does not echo the supplied name.
    """
    if "name" not in message or message.get("name") is None:
        return OPENROUTER_API_KEY
    name = message.get("name")
    if not isinstance(name, str) or name not in KEY_VARIABLES:
        raise ValueError(_KEY_NAME_ERROR)
    return name


def _handle_set_key(session: AgentSession, stdout: IO[str], message: dict[str, Any]) -> None:
    try:
        name = _credential_name(message)
    except ValueError:
        _write_line(stdout, {"type": "error", "message": _KEY_NAME_ERROR}, session.secrets)
        return
    if name != OPENROUTER_API_KEY:
        _handle_set_named_key(session, stdout, message, name)
        return
    scope = message.get("scope")
    key = message.get("key")
    if scope not in {"session", "user"} or not isinstance(key, str) or not key.strip():
        _write_line(
            stdout,
            {"type": "error", "message": "set_key needs scope session or user and a key"},
            session.secrets,
        )
        return
    try:
        # User scope warns inside write_user_key. Session scope warns here.
        cleaned, warning = prepare_openrouter_key(key, warn=scope != "user")
    except ValueError:
        _write_line(
            stdout,
            {"type": "error", "message": "set_key rejected a key that contains a newline"},
            session.secrets,
        )
        return
    _remember_secret(session, cleaned)
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
    payload: dict[str, Any] = {"type": "key", "scope": scope, "saved": True}
    if warning:
        payload["warning"] = warning
    _write_line(stdout, payload, session.secrets)


def _handle_set_named_key(
    session: AgentSession, stdout: IO[str], message: dict[str, Any], name: str
) -> None:
    """Store ``EARTHDATA_TOKEN`` or ``FIRMS_MAP_KEY`` for the session or the user.

    The value is never written into the reply. A session value stays in
    ``session.credential_keys``. A user value is the private file under
    ``~/.atlas/keys``. The session value is left in place when a user value
    is saved.
    """
    scope = message.get("scope")
    key = message.get("key")
    if scope not in {"session", "user"} or not isinstance(key, str) or not key.strip():
        _write_line(
            stdout,
            {"type": "error", "message": f"{name} needs scope session or user and a key"},
            session.secrets,
        )
        return
    try:
        cleaned = prepare_credential(key, name=name)
    except ValueError as exc:
        _write_line(
            stdout,
            {"type": "error", "message": str(exc)},
            session.secrets,
        )
        return
    _remember_secret(session, cleaned)
    try:
        if scope == "user":
            write_user_key(cleaned, session.home, name=name)
        else:
            session.credential_keys[name] = cleaned
    except Exception as exc:
        _write_line(
            stdout,
            {"type": "error", "message": f"Could not save {name}: {type(exc).__name__}"},
            session.secrets,
        )
        return
    _write_line(
        stdout,
        {"type": "key", "scope": scope, "saved": True, "name": name},
        session.secrets,
    )


def _handle_delete_key(
    session: AgentSession, stdout: IO[str], message: dict[str, Any] | None = None
) -> None:
    """Remove the saved user key and stop using any key in this process.

    The reply warns that the environment variable is still used on the next
    launch. ``EARTHDATA_TOKEN`` and ``FIRMS_MAP_KEY`` name that variable.
    OpenRouter keeps its existing reply. The value is not printed.
    """
    payload = message or {}
    try:
        name = _credential_name(payload)
    except ValueError:
        _write_line(stdout, {"type": "error", "message": _KEY_NAME_ERROR}, session.secrets)
        return
    if name != OPENROUTER_API_KEY:
        _handle_delete_named_key(session, stdout, name)
        return
    try:
        delete_user_key(session.home)
    except Exception as exc:
        _write_line(
            stdout,
            {"type": "error", "message": f"Could not delete the key: {type(exc).__name__}"},
            session.secrets,
        )
        return
    session.session_key = None
    _clear_installed_key(session)
    _write_line(
        stdout,
        {
            "type": "key",
            "scope": "deleted",
            "saved": False,
            "warning": (
                "OPENROUTER_API_KEY in the environment will still be used on the next launch"
            ),
        },
        session.secrets,
    )


def _handle_delete_named_key(session: AgentSession, stdout: IO[str], name: str) -> None:
    """Remove the saved user file and the in-memory session value for ``name``.

    The value is remembered for redaction and is not written into the reply.
    The environment variable is left set; the reply says the next launch
    still reads it.
    """
    current = session.credential_keys.get(name)
    if current:
        _remember_secret(session, current)
    try:
        stored = read_user_key(session.home, name=name)
    except ValueError:
        stored = None
    if stored:
        _remember_secret(session, stored)
    try:
        delete_user_key(session.home, name=name)
    except Exception as exc:
        _write_line(
            stdout,
            {"type": "error", "message": f"Could not delete {name}: {type(exc).__name__}"},
            session.secrets,
        )
        return
    session.credential_keys.pop(name, None)
    _write_line(
        stdout,
        {
            "type": "key",
            "scope": "deleted",
            "saved": False,
            "name": name,
            "warning": (f"{name} in the environment will still be used on the next launch"),
        },
        session.secrets,
    )


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


def _handle_save_skill(session: AgentSession, stdout: IO[str], message: dict[str, Any]) -> None:
    name = message.get("name")
    if not isinstance(name, str) or not name.strip():
        _write_line(
            stdout, {"type": "error", "message": "save_skill needs a name"}, session.secrets
        )
        return
    description = message.get("description")
    if description is not None and not isinstance(description, str):
        _write_line(
            stdout,
            {"type": "error", "message": "description must be a string"},
            session.secrets,
        )
        return
    try:
        scope = _skill_scope(message.get("scope"))
        saved = session.save_finished_skill(
            name,
            description=description,
            replace=message.get("replace") is True,
            scope=scope,
        )
    except (SkillSaveError, AtlasPathError) as exc:
        _write_line(stdout, {"type": "error", "message": str(exc)}, session.secrets)
        return
    _write_line(
        stdout,
        {
            "type": "saved_skill",
            "name": saved.name,
            "description": saved.description,
            "scope": scope,
            "path": saved.path,
            "enabled": saved.enabled,
            "enable": enable_command(saved.name),
            "note": saved.note,
        },
        session.secrets,
    )


def _handle_enable_skill(session: AgentSession, stdout: IO[str], message: dict[str, Any]) -> None:
    name = message.get("name")
    if not isinstance(name, str) or not name.strip():
        _write_line(
            stdout, {"type": "error", "message": "enable_skill needs a name"}, session.secrets
        )
        return
    try:
        scope = _skill_scope(message.get("scope"))
        if message.get("confirm") is not True:
            preview = session.preview_saved_skill(name, scope=scope)
            _write_line(
                stdout,
                {
                    "type": "enable_preview",
                    "name": preview.name,
                    "description": preview.description,
                    "scope": scope,
                    "path": preview.path,
                    "content_sha256": preview.content_sha256,
                    "body": preview.body,
                    "body_characters": preview.body_characters,
                    "body_lines": preview.body_lines,
                    "resources": [item.model_dump() for item in preview.resources],
                    "confirm": preview.confirm,
                },
                session.secrets,
            )
            return
        expected = message.get("expected_sha256")
        if not isinstance(expected, str) or not expected.strip():
            raise SkillSaveError("enable needs a preview")
        saved = session.enable_saved_skill(name, expected, scope=scope)
        catalog = session.skill_catalog_note
    except OSError:
        _write_line(stdout, {"type": "error", "message": "could not enable skill"}, session.secrets)
        return
    except (SkillSaveError, AtlasPathError) as exc:
        _write_line(stdout, {"type": "error", "message": str(exc)}, session.secrets)
        return
    _write_line(
        stdout,
        {
            "type": "enabled_skill",
            "name": saved.name,
            "description": saved.description,
            "scope": scope,
            "path": saved.path,
            "enabled": saved.enabled,
            "catalog": catalog,
        },
        session.secrets,
    )


def _skill_scope(scope: object) -> str:
    if scope is None:
        return "project"
    if scope in {"project", "user"}:
        return str(scope)
    raise SkillSaveError("scope must be project or user")


def _add_enabled_skill(session: AgentSession, name: str, scope: str) -> str:
    """Add ``name`` to the live catalog without rehashing skills already loaded."""
    from atlas.agent.skills import UseSkillTool, load_skill_directory

    restart = "Restart Atlas to load this skill."
    workspace = session.workspace
    if workspace is None:
        return restart
    if scope == "user":
        if session.home is None or not load_home_skills_enabled(workspace):
            return restart
        base = session.home
        label = "~/"
    else:
        base = workspace
        label = ""
    try:
        skill = load_skill_directory(
            base,
            f".atlas/skills/{name}",
            label_prefix=label,
            artifact_root=project_atlas_root(workspace) / "artifacts",
        )
    except OSError:
        return restart
    if skill is None:
        return restart
    skills = [item for item in session.agent.skills if item.name != skill.name]
    skills.append(skill)
    skills.sort(key=lambda item: item.name)
    session.agent.skills = skills
    session.agent.tools.replace(UseSkillTool(skills))
    if session.messages and session.messages[0].role == Role.SYSTEM:
        session.messages[0] = ChatMessage(
            role=Role.SYSTEM,
            content=session.agent.system_prompt(),
        )
    return "The skill catalog was reloaded."


def _remember_resolved_credentials(session: AgentSession) -> None:
    """Remember env and file credentials before any tool result is written.

    In-session ``set_key`` already records the value it stores. A key that
    arrived from the environment or ``~/.atlas/keys`` is resolved here, and
    ``resolve_credential`` appends it to the bound secret list.
    """
    home = session.home
    for name in (EARTHDATA_TOKEN, FIRMS_MAP_KEY):
        try:
            resolve_credential(name, home=home)
        except ValueError:
            continue
    resolve_openrouter_key(
        session_key=session.session_key,
        env_key=os.environ.get(OPENROUTER_API_KEY),
        home=home,
    )


def _remember_secret(session: AgentSession, key: str) -> None:
    if key and key not in session.secrets:
        session.secrets.append(key)


def _install_key(session: AgentSession, key: str) -> None:
    _remember_secret(session, key)
    session.active_key = key
    config = getattr(session.agent.model, "config", None)
    if config is not None:
        config.api_key = key


def _clear_installed_key(session: AgentSession) -> None:
    session.active_key = None
    config = getattr(session.agent.model, "config", None)
    if config is not None:
        config.api_key = ""


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


def _catalog_prices(session: AgentSession) -> tuple[str | None, str | None, str | None]:
    """Return prompt, completion, and cache-read prices for the active model.

    Any rate the catalog row does not carry is None. A missing cache-read
    price means cached tokens are priced at the prompt rate.
    """
    model_id = session.totals.model or session.model_id
    if not model_id:
        return None, None, None
    with session._catalog_lock:
        chosen = next((item for item in session.catalog if item.id == model_id), None)
    if chosen is None:
        return None, None, None
    return chosen.prompt_price, chosen.completion_price, chosen.cache_read_price


def _apply_known_context_limit(session: AgentSession) -> None:
    """Copy the active model's catalog window onto the dashboard totals.

    The limit stays unset when the catalog has not been loaded or has no row
    for the model. A known window replaces ``unknown`` on the next status line.
    """
    model_id = session.totals.model or session.model_id
    if not model_id:
        return
    with session._catalog_lock:
        chosen = next((item for item in session.catalog if item.id == model_id), None)
        context_length = chosen.context_length if chosen is not None else None
    if context_length is not None:
        session.totals.context_limit = context_length


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
        "cached_tokens": totals.cached_tokens,
        "reasoning_tokens": totals.reasoning_tokens,
        "spend": reported_spend(totals),
        "estimated": totals.estimated,
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
