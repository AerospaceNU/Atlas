"""Pydantic-typed, non-blocking v2 JSON-lines RPC transport for ``AgentSession``."""

from __future__ import annotations

import contextlib
import json
import os
import queue
import threading
from collections.abc import Callable, Sequence
from typing import IO, Annotated, Any, Literal

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from atlas.agent import layout
from atlas.agent.analysis_skill import SkillSaveError
from atlas.agent.checkpoints import restore_latest
from atlas.agent.contracts import CatalogModel
from atlas.agent.metrics import apply_http_error
from atlas.agent.model import (
    CatalogFetch,
    MissingOpenRouterKey,
    OpenRouterHTTPError,
    clear_model_catalog_cache,
    peek_model_catalog,
)
from atlas.agent.redaction import redact_value
from atlas.agent.runtime import AgentRun, AgentStep, PermissionDecision, tool_summary
from atlas.agent.session import AgentSession, _stop_jobs, session_bound
from atlas.agent.sessions import discard_unused_session, list_sessions, newest_session

_STDOUT_LOCK = threading.Lock()
_PROVIDERS = {
    "openrouter": ("openrouter", layout.OPENROUTER_API_KEY),
    "openrouter_api_key": ("openrouter", layout.OPENROUTER_API_KEY),
    "earthdata": ("earthdata", layout.EARTHDATA_TOKEN),
    "earthdata_token": ("earthdata", layout.EARTHDATA_TOKEN),
    "firms": ("firms", layout.FIRMS_MAP_KEY),
    "firms_map_key": ("firms", layout.FIRMS_MAP_KEY),
}


class SimpleMessage(BaseModel):
    type: Literal["resume", "cancel", "undo", "new", "compact", "quit"]


class UserMessage(BaseModel):
    type: Literal["user"]
    text: Any = None


class PermissionMessage(BaseModel):
    type: Literal["permission"]
    id: str
    decision: Literal["allow", "allow_session", "deny"] | None = None
    reason: str | None = None
    allow: bool | None = None


class YoloMessage(BaseModel):
    type: Literal["yolo"]
    enabled: bool | None = None


class LoadSessionMessage(BaseModel):
    type: Literal["load_session"]
    id: Any = None


class ModelsMessage(BaseModel):
    type: Literal["models"]
    refresh: bool = False


class SelectModelMessage(BaseModel):
    type: Literal["select_model"]
    model: Any = None
    id: Any = None


class GoalMessage(BaseModel):
    type: Literal["goal"]
    text: Any = None
    clear: Any = None


class SkillMessage(BaseModel):
    type: Literal["skill"]
    name: Any = None
    argument: Any = None


class SaveSkillMessage(BaseModel):
    type: Literal["save_skill"]
    name: Any = None
    description: Any = None
    replace: bool = False
    scope: Any = None


class EnableSkillMessage(BaseModel):
    type: Literal["enable_skill"]
    name: Any = None
    sha256: Any = None
    expected_sha256: Any = None
    confirm: Any = None
    scope: Any = None


class ListMessage(BaseModel):
    type: Literal["list"]
    kind: Any = None
    scope: Any = None


class SetKeyMessage(BaseModel):
    type: Literal["set_key"]
    key: Any = None
    provider: Any = None
    name: Any = None
    scope: Any = None


class DeleteKeyMessage(BaseModel):
    type: Literal["delete_key"]
    provider: Any = None
    name: Any = None
    scope: Any = None


ClientMessage = Annotated[
    SimpleMessage
    | UserMessage
    | PermissionMessage
    | YoloMessage
    | LoadSessionMessage
    | ModelsMessage
    | SelectModelMessage
    | GoalMessage
    | SkillMessage
    | SaveSkillMessage
    | EnableSkillMessage
    | ListMessage
    | SetKeyMessage
    | DeleteKeyMessage,
    Field(discriminator="type"),
]
_CLIENT_ADAPTER: TypeAdapter[ClientMessage] = TypeAdapter(ClientMessage)
EmitFn = Callable[[dict[str, Any]], None]


def _err(emit: EmitFn, message: str) -> None:
    emit({"type": "error", "message": message})


def _write_line(stdout: IO[str], payload: dict[str, Any], secrets: list[str]) -> None:
    text = json.dumps(redact_value(payload, secrets), ensure_ascii=False)
    with _STDOUT_LOCK:
        stdout.write(text + "\n")
        stdout.flush()


def _ready_event(session: AgentSession) -> dict[str, Any]:
    return {
        "type": "ready",
        "session_id": session.session_id or "",
        "model": session.totals.model or session.model_id,
        "workspace": str(session.workspace or session.agent.artifacts.root),
        "tools": [t.name for t in session.agent.tools.definitions],
        "key_set": session.key_set,
        "status": session.status_snapshot(),
        "goal": session.goal,
        "active_skills": list(session.active_skills),
        "yolo_enabled": session.yolo_enabled,
    }


def _session_loaded_event(session: AgentSession, sid: str = "") -> dict[str, Any]:
    messages, omitted = session.replay_items_with_omitted(limit=40, max_chars=1500)
    return {
        "type": "session_loaded",
        "session_id": session.session_id or sid,
        "model": session.model_id,
        "goal": session.goal,
        "active_skills": list(session.active_skills),
        "status": session.status_snapshot(),
        "messages": messages,
        "stopped_for_limit": session.stopped_for_limit,
        "interrupted": session.interrupted,
        "omitted": omitted,
    }


def step_event(step: AgentStep) -> dict[str, Any]:
    res = step.result
    text = res.text if res is not None else ""
    diff = res.diff if res is not None else None
    if (
        diff is None
        and res is not None
        and step.call.name in {"git_diff", "apply_patch"}
        and text.startswith(("diff --git ", "--- "))
    ):
        diff = text
    return {
        "type": "step",
        "call_id": step.call.id,
        "name": step.call.name,
        "summary": (res.summary if res and res.summary else None) or tool_summary(step.call),
        "ok": res is not None,
        "duration_ms": max(0, int(step.duration_ms)),
        "text": text,
        "error": step.error,
        "error_kind": step.error_kind,
        "artifacts": list(res.artifacts) if res is not None else [],
        "notes": list(step.notes),
        "diff": diff,
        "lines": len(text.splitlines()) if text else 0,
        "omitted": None,
        "arguments": step.call.arguments,
    }


def done_event(
    session: AgentSession,
    run: AgentRun,
    *,
    thoughts: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "type": "done",
        "response": run.response,
        "thoughts": "\n\n".join(t for t in thoughts if t.strip()) or None,
        "cancelled": run.cancelled,
        "stopped_for_limit": run.stopped_for_limit,
        "stopped_for_budget": run.stopped_for_budget,
        "stopped_for_context": run.stopped_for_context,
        "compaction": run.compaction.model_dump(),
        "empty_replies": run.empty_replies,
        "intent_nudges": run.intent_nudges,
        "status": session.status_snapshot(),
    }


def _save_after_failure(session: AgentSession) -> None:
    """Persist history appended before a failed turn; a save error never masks the cause."""
    with contextlib.suppress(Exception):
        session.save_state()


def serve(
    session: AgentSession,
    stdin: IO[str],
    stdout: IO[str],
    *,
    redact: str | None = None,
    resumed: bool = False,
) -> int:
    """Run the non-blocking v2 JSON-lines protocol until ``quit`` or EOF."""
    if redact:
        session.remember_secret(redact)
    emit: EmitFn = lambda ev: _write_line(stdout, ev, session.secrets)  # noqa: E731
    session.agent.on_permission = lambda req: session.ask_permission(req, emit)
    with session_bound(session):
        for name in (layout.EARTHDATA_TOKEN, layout.FIRMS_MAP_KEY):
            with contextlib.suppress(ValueError):
                layout.resolve_credential(name, home=session.home)
        layout.resolve_openrouter_key(
            session_key=session.session_key,
            env_key=os.environ.get(layout.OPENROUTER_API_KEY),
            home=session.home,
        )
        should_emit_loaded = resumed or session.loaded_at_launch
        session.loaded_at_launch = False
        try:
            code = _serve_bound(session, stdin, emit, resumed=should_emit_loaded)
        finally:
            _stop_jobs("session ended")
        if session.has_conversation():
            with contextlib.suppress(Exception):
                session.save_state(only_if_dir_exists=True)
        return code


def _serve_bound(
    session: AgentSession, stdin: IO[str], emit: EmitFn, *, resumed: bool = False
) -> int:
    emit(_ready_event(session))
    if resumed:
        emit(_session_loaded_event(session))
    work: queue.Queue[ClientMessage | None] = queue.Queue()

    def worker_loop() -> None:
        while (item := work.get()) is not None:
            _dispatch(session, emit, item)

    worker = threading.Thread(target=worker_loop, daemon=True, name="atlas-rpc-worker")
    worker.start()

    explicit_quit = False
    for raw_line in stdin:
        if not (stripped := raw_line.strip()):
            continue
        try:
            msg = _CLIENT_ADAPTER.validate_json(stripped)
        except ValidationError as exc:
            _emit_parse_error(emit, stripped, exc)
            continue

        if isinstance(msg, SimpleMessage) and msg.type == "cancel":
            session.cancel()
        elif isinstance(msg, SimpleMessage) and msg.type == "quit":
            explicit_quit = True
            break
        elif isinstance(msg, PermissionMessage):
            decision = (
                PermissionDecision(decision=msg.decision, reason=msg.reason)
                if msg.decision is not None
                else PermissionDecision(
                    decision="allow" if msg.allow else "deny",
                    reason=msg.reason if not msg.allow else None,
                )
            )
            session.resolve_permission(msg.id, decision)
        elif isinstance(msg, YoloMessage):
            session.set_yolo(msg.enabled, emit)
        elif isinstance(msg, ModelsMessage):
            _handle_models(session, emit, refresh=msg.refresh)
        else:
            work.put(msg)

    work.put(None)
    worker.join(timeout=2.0 if explicit_quit else None)
    if not explicit_quit:
        thread = session._catalog_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=3.0)
    return 0


def _emit_parse_error(emit: EmitFn, stripped: str, exc: ValidationError) -> None:
    if any(err.get("type") == "json_invalid" for err in exc.errors()):
        _err(emit, "Invalid JSON input")
    elif not stripped.startswith("{"):
        _err(emit, "Expected a JSON object")
    elif tag_err := next((e for e in exc.errors() if e.get("type") == "union_tag_invalid"), None):
        _err(emit, f"Unknown message type: {tag_err.get('ctx', {}).get('tag')!r}")
    else:
        _err(emit, "Invalid protocol message")


def _dispatch(session: AgentSession, emit: EmitFn, msg: ClientMessage) -> None:
    if isinstance(msg, UserMessage):
        _handle_turn(session, emit, msg.text)
    elif isinstance(msg, SimpleMessage):
        if msg.type == "resume":
            _handle_turn(session, emit, None)
        elif msg.type == "undo":
            restored = restore_latest(session.agent.artifacts)
            emit(
                {"type": "undo", "ok": True, "path": restored, "message": f"restored {restored}"}
                if restored is not None
                else {
                    "type": "undo",
                    "ok": False,
                    "path": None,
                    "message": "no checkpoint to restore",
                }
            )
        elif msg.type == "new":
            session.start_fresh()
            emit(_ready_event(session))
        elif msg.type == "compact":
            _handle_compact(session, emit)
    elif isinstance(msg, LoadSessionMessage):
        _handle_load_session(session, emit, msg)
    elif isinstance(msg, SelectModelMessage):
        _handle_select_model(session, emit, msg)
    elif isinstance(msg, GoalMessage):
        _handle_goal(session, emit, msg)
    elif isinstance(msg, SkillMessage):
        _handle_skill(session, emit, msg)
    elif isinstance(msg, SaveSkillMessage):
        _handle_save_skill(session, emit, msg)
    elif isinstance(msg, EnableSkillMessage):
        _handle_enable_skill(session, emit, msg)
    elif isinstance(msg, ListMessage):
        _handle_list(session, emit, msg)
    elif isinstance(msg, SetKeyMessage):
        _handle_set_key(session, emit, msg)
    elif isinstance(msg, DeleteKeyMessage):
        _handle_delete_key(session, emit, msg)


def _handle_compact(session: AgentSession, emit: EmitFn) -> None:
    try:
        compaction = session.compact()
        session.save_state()
    except Exception as exc:
        return _err(emit, f"{type(exc).__name__}: {exc}")
    emit(
        {
            "type": "compacted",
            "compaction": compaction.model_dump(),
            "status": session.status_snapshot(),
        }
    )


def _handle_turn(session: AgentSession, emit: EmitFn, text: Any) -> None:
    if text is not None and (not isinstance(text, str) or not text):
        return _err(emit, "user text must be a non-empty string")
    thoughts: list[str] = []
    streamed_thought = False

    def on_phase(phase: str, detail: str | None) -> None:
        nonlocal streamed_thought
        if phase in {"thinking", "working"}:
            streamed_thought = False
        emit({"type": "phase", "phase": phase, **({"detail": detail} if detail else {})})

    def on_thought_delta(chunk: str) -> None:
        nonlocal streamed_thought
        streamed_thought = True
        emit({"type": "thought_delta", "text": chunk})

    def on_thought(thought: str, _speed: float | None) -> None:
        if thought.strip():
            thoughts.append(thought)
            if not streamed_thought:
                emit({"type": "thought_delta", "text": thought})

    callbacks: dict[str, Any] = {
        "on_step": lambda step: emit(step_event(step)),
        "on_phase": on_phase,
        "on_thought": on_thought,
        "on_content_delta": lambda chunk: emit({"type": "content_delta", "text": chunk}),
        "on_thought_delta": on_thought_delta,
        "on_tool_start": lambda cid, name, summary: emit(
            {"type": "tool_start", "call_id": cid, "name": name, "summary": summary}
        ),
        "on_tool_output": lambda cid, chunk: emit(
            {"type": "tool_output_delta", "call_id": cid, "chunk": chunk}
        ),
        "on_status": lambda status: emit({"type": "status", "status": status}),
    }
    try:
        run = session.resume(**callbacks) if text is None else session.turn(text, **callbacks)
    except MissingOpenRouterKey as exc:
        return _err(emit, str(exc))
    except KeyboardInterrupt:
        _save_after_failure(session)
        raise
    except OpenRouterHTTPError as exc:
        _save_after_failure(session)
        apply_http_error(
            session.totals,
            http_status=exc.status_code,
            elapsed_seconds=exc.elapsed_seconds,
            time_to_first_token_seconds=exc.time_to_first_token_seconds,
        )
        emit(
            {
                "type": "error",
                "message": str(exc),
                "http_status": exc.status_code,
                "elapsed_seconds": exc.elapsed_seconds,
                "error_body": exc.body,
                "status": session.status_snapshot(),
            }
        )
        return
    except Exception as exc:
        _save_after_failure(session)
        return _err(emit, f"{type(exc).__name__}: {exc}")
    session.save_state()
    emit(done_event(session, run, thoughts=thoughts))


def _handle_load_session(session: AgentSession, emit: EmitFn, msg: LoadSessionMessage) -> None:
    if not isinstance(msg.id, str) or not msg.id.strip():
        return _err(emit, "session id must be a non-empty string")
    wanted = msg.id.strip()
    previous = session.session_id
    if wanted == "last" and session.workspace is not None:
        newest = newest_session(session.workspace, exclude=previous)
        if newest is None:
            return _err(emit, "No saved session")
        wanted = newest
    try:
        loaded = session.load_state(wanted)
    except (OSError, ValueError, ValidationError) as exc:
        return _err(emit, f"Could not reopen session: {exc}")
    if not loaded:
        return _err(emit, f"No saved session {wanted}")
    if (
        session.workspace is not None
        and isinstance(previous, str)
        and previous != session.session_id
    ):
        discard_unused_session(session.workspace, previous)
    emit(_session_loaded_event(session, wanted))


def _handle_goal(session: AgentSession, emit: EmitFn, msg: GoalMessage) -> None:
    usage = "usage: /goal [text] | /goal clear"
    if (msg.clear is not None and msg.clear is not True) or (
        msg.clear is True and msg.text is not None
    ):
        return _err(emit, usage)
    if msg.clear is True or msg.text == "":
        session.clear_goal()
        emit({"type": "goal", "text": "", "cleared": True, "status": session.status_snapshot()})
        return
    if msg.text is None:
        emit(
            {
                "type": "goal",
                "text": session.goal,
                "cleared": False,
                "status": session.status_snapshot(),
            }
        )
        return
    if not isinstance(msg.text, str):
        return _err(emit, usage)
    try:
        session.set_goal(msg.text)
    except ValueError as exc:
        return _err(emit, str(exc))
    emit(
        {
            "type": "goal",
            "text": session.goal,
            "cleared": False,
            "status": session.status_snapshot(),
        }
    )


def _handle_skill(session: AgentSession, emit: EmitFn, msg: SkillMessage) -> None:
    if not isinstance(msg.name, str) or not msg.name.strip():
        return _err(emit, "skill name must be a non-empty string")
    if msg.argument is not None and not isinstance(msg.argument, str):
        return _err(emit, "skill argument must be a string")
    try:
        skill = session.use_skill(msg.name, msg.argument)
    except OSError:
        return _err(emit, f"Skill {msg.name.strip()} could not be read")
    except (ValueError, ValidationError, KeyError) as exc:
        return _err(emit, str(exc))
    emit(
        {
            "type": "skill",
            "name": skill.name,
            "active": True,
            "note": skill.description or None,
            "status": session.status_snapshot(),
        }
    )


def _handle_models(session: AgentSession, emit: EmitFn, *, refresh: bool = False) -> None:
    if (fetcher := session.fetch_models) is None:
        return _err(emit, "model catalog is unavailable")
    if refresh:
        clear_model_catalog_cache()
    cfg = getattr(session.agent.model, "config", None)
    key = (
        str(getattr(cfg, "api_key", "") or "").strip()
        if cfg
        else (session.active_key or "").strip()
    )
    if not refresh and key and (cached := peek_model_catalog(key)) is not None:
        _publish_catalog(session, emit, CatalogFetch(models=cached))
        return
    if session._catalog_inflight:
        return
    session._catalog_inflight = True

    def worker() -> None:
        try:
            raw = fetcher()
            _publish_catalog(
                session,
                emit,
                raw if isinstance(raw, CatalogFetch) else CatalogFetch(models=list(raw)),
            )
        except Exception as exc:
            err = (
                str(exc)
                if isinstance(exc, MissingOpenRouterKey | OpenRouterHTTPError)
                else f"{type(exc).__name__}: {exc}"[:300]
            )
            _err(emit, err)
        finally:
            session._catalog_inflight = False

    thread = threading.Thread(target=worker, daemon=True, name="atlas-model-catalog")
    session._catalog_thread = thread
    thread.start()


def _publish_catalog(session: AgentSession, emit: EmitFn, fetched: CatalogFetch) -> None:
    with session._catalog_lock:
        selected = session.model_id.strip()
        models = list(fetched.models)
        if selected and not any(m.id == selected for m in models):
            prev = next((m for m in session.catalog if m.id == selected), None)
            models.append(prev or CatalogModel(id=selected, supports_tools=True))
        session.catalog = models
    session.apply_known_context_limit()
    emit(
        {
            "type": "models",
            "models": [
                {
                    "id": m.id,
                    "name": m.name,
                    "context_length": m.context_length,
                    "prompt_price": m.prompt_price,
                    "completion_price": m.completion_price,
                    "supports_tools": m.supports_tools,
                }
                for m in models
            ],
            "selected": session.model_id,
        }
    )
    if fetched.error:
        _err(emit, fetched.error)


def _handle_select_model(session: AgentSession, emit: EmitFn, msg: SelectModelMessage) -> None:
    raw = msg.model if msg.model is not None else msg.id
    if not isinstance(raw, str) or not raw.strip():
        return _err(emit, "model id must be a non-empty string")
    model_id = raw.strip()
    with session._catalog_lock:
        chosen = next((m.model_copy() for m in session.catalog if m.id == model_id), None)
    if chosen is None:
        return _err(emit, f"Unknown model: {model_id}")
    if (cfg := getattr(session.agent.model, "config", None)) is not None:
        cfg.model = chosen.id
    session.model_id = session.totals.model = chosen.id
    session.totals.context_limit = session.agent.context_limit = chosen.context_length
    emit(
        {
            "type": "model_selected",
            "model": chosen.id,
            "context_length": chosen.context_length,
            "status": session.status_snapshot(),
        }
    )


def _resolve_provider(provider: Any, name: Any) -> tuple[str, str]:
    raw = provider if provider is not None else name
    if raw is None:
        return ("openrouter", layout.OPENROUTER_API_KEY)
    if not isinstance(raw, str) or (found := _PROVIDERS.get(raw.strip().lower())) is None:
        raise ValueError("provider must be openrouter, earthdata, or firms")
    return found


def _handle_set_key(session: AgentSession, emit: EmitFn, msg: SetKeyMessage) -> None:
    try:
        provider, env_var = _resolve_provider(msg.provider, msg.name)
    except ValueError as exc:
        return _err(emit, str(exc))
    scope = "session" if msg.scope is None else msg.scope
    if scope not in {"session", "user"} or not isinstance(msg.key, str) or not msg.key.strip():
        return _err(emit, f"{env_var} needs scope session or user and a key")
    warning = ""
    try:
        if env_var == layout.OPENROUTER_API_KEY:
            cleaned, warning = layout.prepare_openrouter_key(msg.key, warn=scope != "user")
        else:
            cleaned = layout.prepare_credential(msg.key, name=env_var)
    except ValueError as exc:
        return _err(emit, str(exc))
    session.remember_secret(cleaned)
    try:
        if scope == "user":
            layout.write_user_key(cleaned, session.home, name=env_var)
            if env_var == layout.OPENROUTER_API_KEY and not (
                session.session_key and session.session_key.strip()
            ):
                session.install_key(cleaned)
        elif env_var == layout.OPENROUTER_API_KEY:
            session.session_key = cleaned
            session.install_key(cleaned)
        else:
            session.credential_keys[env_var] = cleaned
    except Exception as exc:
        return _err(emit, f"Could not save {env_var}: {type(exc).__name__}")
    emit({"type": "key_set", "provider": provider, "key_set": True, "note": warning or None})


def _handle_delete_key(session: AgentSession, emit: EmitFn, msg: DeleteKeyMessage) -> None:
    try:
        provider, env_var = _resolve_provider(msg.provider, msg.name)
    except ValueError as exc:
        return _err(emit, str(exc))
    if msg.scope is not None and msg.scope not in {"session", "user"}:
        return _err(emit, "scope must be session or user")
    if env_var != layout.OPENROUTER_API_KEY:
        if cur := session.credential_keys.get(env_var):
            session.remember_secret(cur)
        with contextlib.suppress(ValueError):
            if stored := layout.read_user_key(session.home, name=env_var):
                session.remember_secret(stored)
    try:
        if msg.scope != "session":
            layout.delete_user_key(session.home, name=env_var)
    except Exception as exc:
        return _err(emit, f"Could not delete {env_var}: {type(exc).__name__}")
    if env_var == layout.OPENROUTER_API_KEY:
        if msg.scope != "user":
            session.session_key = None
            session.clear_installed_key()
        elif not session.session_key:
            session.clear_installed_key()
    elif msg.scope != "user":
        session.credential_keys.pop(env_var, None)
    emit(
        {
            "type": "key_set",
            "provider": provider,
            "key_set": False,
            "note": f"{env_var} in the environment will still be used on the next launch",
        }
    )


def _session_rows(session: AgentSession) -> list[dict[str, Any]]:
    if session.workspace is None:
        return []
    try:
        found = list_sessions(session.workspace, limit=50)
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for item in found:
        row = item.model_dump()
        prompt = str(row.get("prompt", ""))
        row["prompt"] = prompt if len(prompt) <= 120 else prompt[:117] + "..."
        rows.append(row)
    return rows


def _handle_list(session: AgentSession, emit: EmitFn, msg: ListMessage) -> None:
    if not isinstance(msg.kind, str) or not msg.kind.strip():
        return _err(emit, "list needs a kind")
    kind = msg.kind.strip()
    scope = "project" if msg.scope is None else msg.scope
    sessions_rows: list[dict[str, Any]] = []
    if kind == "skills":
        items = [s.name for s in session.agent.skills]
    elif kind == "tools":
        items = [t.name for t in session.agent.tools.definitions]
    elif kind == "drafts":
        d = layout.project_skill_drafts_root(session.workspace) if session.workspace else None
        items = sorted(p.name for p in d.iterdir() if p.is_dir()) if d and d.is_dir() else []
    else:
        try:
            if scope == "project":
                if session.workspace is None:
                    raise layout.AtlasPathError("project workspace is not set")
                root = layout.project_atlas_root(session.workspace)
            elif scope == "user":
                root = layout.user_atlas_root(session.home)
            else:
                raise layout.AtlasPathError("scope must be project or user")
            items = layout.list_entries(root, kind)
        except (layout.AtlasPathError, TypeError) as exc:
            return _err(emit, str(exc))
        if kind == "sessions" and scope == "project":
            sessions_rows = _session_rows(session)
    payload: dict[str, Any] = {"type": "list", "kind": kind, "items": items, "note": None}
    if sessions_rows:
        payload["sessions"] = sessions_rows
    emit(payload)


def _handle_save_skill(session: AgentSession, emit: EmitFn, msg: SaveSkillMessage) -> None:
    name = (
        msg.name.strip()
        if isinstance(msg.name, str) and msg.name.strip()
        else session.default_skill_name()
    )
    if msg.description is not None and not isinstance(msg.description, str):
        return _err(emit, "description must be a string")
    try:
        scope = "project" if msg.scope is None else str(msg.scope)
        if scope not in {"project", "user"}:
            raise SkillSaveError("scope must be project or user")
        saved = session.save_finished_skill(
            name, description=msg.description, replace=msg.replace, scope=scope
        )
    except (SkillSaveError, layout.AtlasPathError) as exc:
        return _err(emit, str(exc))
    emit({"type": "save_skill", "name": saved.name, "path": saved.path, "note": saved.note or None})


def _handle_enable_skill(session: AgentSession, emit: EmitFn, msg: EnableSkillMessage) -> None:
    if not isinstance(msg.name, str) or not msg.name.strip():
        return _err(emit, "enable_skill needs a name")
    token = msg.sha256 if msg.sha256 is not None else msg.expected_sha256
    try:
        scope = "project" if msg.scope is None else str(msg.scope)
        if scope not in {"project", "user"}:
            raise SkillSaveError("scope must be project or user")
        if token is None and msg.confirm is not True:
            preview = session.preview_saved_skill(msg.name, scope=scope)
            emit(
                {
                    "type": "enable_preview",
                    "name": preview.name,
                    "sha256": preview.content_sha256,
                    "files": [r.path for r in preview.resources],
                    "note": preview.description or preview.confirm,
                }
            )
            return
        if not isinstance(token, str) or not token.strip():
            raise SkillSaveError("enable needs a preview")
        saved = session.enable_saved_skill(msg.name, token, scope=scope)
    except OSError:
        return _err(emit, "could not enable skill")
    except (SkillSaveError, layout.AtlasPathError) as exc:
        return _err(emit, str(exc))
    emit(
        {
            "type": "enabled_skill",
            "name": saved.name,
            "enabled": saved.enabled,
            "note": session.skill_catalog_note or None,
        }
    )
