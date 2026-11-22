from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from atlas.agent import layout
from atlas.agent.analysis_skill import (
    AnalysisStep,
    AnalysisTurn,
    FinishedAnalysis,
    SavedSkill,
    SkillPreview,
    SkillSaveError,
    enable_skill,
    preview_skill,
    save_skill,
    skill_slug,
    successful_steps,
)
from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.audit import record as audit_event
from atlas.agent.contracts import (
    CatalogModel,
    ChatMessage,
    ModelCall,
    Role,
    ToolCapableModel,
    ToolRegistry,
    ToolResult,
    default_registry,
)
from atlas.agent.images import ingest_images
from atlas.agent.metrics import SessionTotals, apply_call, reported_spend
from atlas.agent.model import (
    MISSING_KEY_MESSAGE,
    CatalogFetch,
    MissingOpenRouterKey,
    ModelConfig,
    OpenRouterModel,
)
from atlas.agent.redaction import redact_text
from atlas.agent.redaction import redact_value as redact_value
from atlas.agent.runtime import (
    Agent,
    AgentRun,
    AgentStep,
    ContextCompaction,
    PermissionDecision,
    PermissionRequest,
    _compact,
    _estimate_tokens,
    ensure_agent_config,
    load_approval,
    load_home_skills_enabled,
    load_max_tool_calls,
    load_model,
    load_optional_tokens,
    load_read_only,
    load_reasoning_effort,
    load_spend_limit,
    tool_summary,
)
from atlas.agent.sessions import (
    discard_unused_session,
    first_prompt,
    repair_history,
    write_meta,
)
from atlas.agent.skill_author import normalize_skill_name
from atlas.agent.skills import Skill, load_skills, register_skill_tool

_GOAL_MAX_CHARS = 2_000
_GOAL_OPEN = "<session_goal>"
_GOAL_CLOSE = "</session_goal>"
_PROJECT_INSTRUCTION_PREFIX = "<untrusted_project_instructions"
PERMISSION_TIMEOUT = 300.0


def session_goal_text(goal: str) -> str:
    return f"{_GOAL_OPEN}\n{goal}\n{_GOAL_CLOSE}"


def _find_repo_root(start: Path | None = None) -> Path | None:
    return layout.find_repo_root(start)


def _git_branch(workspace: Path | None) -> str | None:
    if workspace is None:
        return None
    for candidate in (workspace, *workspace.parents):
        dotgit = candidate / ".git"
        if dotgit.is_dir():
            head_file = dotgit / "HEAD"
        elif dotgit.is_file():
            try:
                text = dotgit.read_text(encoding="utf-8").strip()
            except OSError:
                return None
            if not text.startswith("gitdir:"):
                return None
            gitdir = Path(text[7:].strip())
            head_file = (
                gitdir if gitdir.is_absolute() else (candidate / gitdir).resolve()
            ) / "HEAD"
        else:
            continue
        try:
            head = head_file.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        return head[16:] if head.startswith("ref: refs/heads/") else (head[:8] or None)
    return None


def _stop_jobs(reason: str) -> None:
    """Stop background bash jobs this process started; never raises."""
    with contextlib.suppress(Exception):
        from atlas.agent.tools._jobs import stop_all

        stop_all(reason)


class AgentSession:
    """A persistent agent conversation surviving across turns and RPC commands."""

    AUTOSAVE_SECONDS: float = 5.0

    def __init__(self, agent: Agent) -> None:
        self.agent = agent
        self.messages: list[ChatMessage] = self.agent.opening_messages()
        self.stopped_for_limit = False
        self.interrupted = False
        self._last_autosave: float | None = None
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
        self.goal = ""
        self._goal_message: ChatMessage | None = None
        self.totals = SessionTotals()
        self.fetch_models: Callable[..., CatalogFetch | list[CatalogModel]] | None = None
        self._catalog_thread: threading.Thread | None = None
        self._catalog_inflight = False
        self._catalog_lock = threading.Lock()
        if (config := getattr(agent.model, "config", None)) is not None:
            self.model_id = str(getattr(config, "model", "") or "")
            self.totals.model = self.model_id
            key = str(getattr(config, "api_key", "") or "").strip()
            self.active_key = key or None
            if key:
                self.secrets.append(key)
        # Bound to the live list, so a key added later through set_key is covered too.
        self.agent.redact_output = lambda text: redact_text(text, self.secrets)
        self.skill_catalog_note = ""
        self._skill_previews: dict[tuple[str, str], str] = {}
        self._analysis_turns: list[AnalysisTurn] = []
        self._analysis_open = False
        self._analysis_request = ""
        self._analysis_steps: list[AnalysisStep] = []
        self._analysis_response = ""
        self.cancel_event = threading.Event()
        self.quit_turn: int | None = None
        self._turn_no = 0
        self.loaded_at_launch = False
        self.session_id: str | None = None
        self.permission_slots: dict[str, list[object]] = {}
        self.permission_lock = threading.RLock()

    @property
    def yolo_enabled(self) -> bool:
        return self.agent.approval == "allow"

    @property
    def key_set(self) -> bool:
        return (not self.requires_api_key) or bool(self.active_key and self.active_key.strip())

    def remember_secret(self, key: str) -> None:
        if (cleaned := key.strip()) and cleaned not in self.secrets:
            self.secrets.append(cleaned)

    def install_key(self, key: str) -> None:
        self.remember_secret(key)
        self.active_key = key
        if (config := getattr(self.agent.model, "config", None)) is not None:
            config.api_key = key

    def clear_installed_key(self) -> None:
        self.active_key = None
        if (config := getattr(self.agent.model, "config", None)) is not None:
            config.api_key = ""

    def _active_catalog_row(self) -> CatalogModel | None:
        model_id = self.totals.model or self.model_id
        if not model_id:
            return None
        with self._catalog_lock:
            return next((item for item in self.catalog if item.id == model_id), None)

    def apply_known_context_limit(self) -> None:
        chosen = self._active_catalog_row()
        if chosen is not None and chosen.context_length is not None:
            self.totals.context_limit = chosen.context_length
            self.agent.context_limit = chosen.context_length

    def status_snapshot(self) -> dict[str, Any]:
        """Build the v2 ``StatusSnapshot`` payload dict."""
        self.apply_known_context_limit()
        t, ws = self.totals, self.workspace or self.agent.artifacts.root
        return {
            "model": t.model or self.model_id,
            "context_used": t.context_used,
            "context_limit": t.context_limit,
            "prompt_tokens": t.prompt_tokens,
            "completion_tokens": t.completion_tokens,
            "total_tokens": t.total_tokens,
            "cached_tokens": t.cached_tokens,
            "reasoning_tokens": t.reasoning_tokens,
            "spend": reported_spend(t),
            "tokens_per_second": t.tokens_per_second,
            "yolo_enabled": self.yolo_enabled,
            "estimated": t.estimated,
            "branch": _git_branch(ws),
            "workspace": str(ws) if ws is not None else None,
            "goal": self.goal or None,
            "active_skills": list(self.active_skills),
        }

    def set_yolo(
        self, enabled: bool | None, emit: Callable[[dict[str, Any]], None] | None = None
    ) -> bool:
        with self.permission_lock:
            target = (not self.yolo_enabled) if enabled is None else bool(enabled)
            self.agent.approval = "allow" if target else "ask"
            if emit is not None:
                emit({"type": "yolo", "enabled": target, "status": self.status_snapshot()})
            if target:
                for slot in self.permission_slots.values():
                    slot[1] = PermissionDecision(decision="allow")
                    assert isinstance(slot[0], threading.Event)
                    slot[0].set()
            return target

    def toggle_yolo(self, emit: Callable[[dict[str, Any]], None]) -> None:
        self.set_yolo(None, emit)

    def cancel(self) -> None:
        """Cancel the active turn and immediately wake any waiting permission prompt."""
        self.cancel_event.set()
        with self.permission_lock:
            for slot in self.permission_slots.values():
                slot[1] = PermissionDecision(decision="deny", reason="cancelled by the user")
                assert isinstance(slot[0], threading.Event)
                slot[0].set()

    def ask_permission(
        self, request: PermissionRequest, emit: Callable[[dict[str, Any]], None]
    ) -> PermissionDecision:
        slot: list[object] = [threading.Event(), PermissionDecision(decision="deny")]
        with self.permission_lock:
            if self.yolo_enabled:
                return PermissionDecision(decision="allow")
            if self.cancel_event.is_set():
                return PermissionDecision(decision="deny", reason="cancelled by the user")
            self.permission_slots[request.id] = slot
            emit(
                {
                    "type": "permission_request",
                    "id": request.id,
                    "tool": request.tool,
                    "summary": request.summary,
                    "preview": request.preview,
                    "preview_kind": request.preview_kind,
                    "arguments": request.arguments,
                }
            )
        assert isinstance(slot[0], threading.Event)
        slot[0].wait(PERMISSION_TIMEOUT)
        with self.permission_lock:
            self.permission_slots.pop(request.id, None)
        assert isinstance(slot[1], PermissionDecision)
        return slot[1]

    def resolve_permission(
        self, request_id: str, allow: bool | PermissionDecision, *, reason: str | None = None
    ) -> bool:
        decision = (
            allow
            if isinstance(allow, PermissionDecision)
            else PermissionDecision(
                decision="allow" if allow else "deny", reason=reason if not allow else None
            )
        )
        with self.permission_lock:
            if (slot := self.permission_slots.get(request_id)) is None:
                return False
            slot[1] = decision
            assert isinstance(slot[0], threading.Event)
            slot[0].set()
        return True

    def _should_cancel(self) -> bool:
        quit_turn = self.quit_turn
        return self.cancel_event.is_set() or (quit_turn is not None and quit_turn >= self._turn_no)

    def has_conversation(self) -> bool:
        """Return whether any real user message has been added."""
        return any(m.role is Role.USER and not m.tool_observation for m in self.messages)

    def _autosaving_phase(
        self, on_phase: Callable[[str, str | None], None] | None
    ) -> Callable[[str, str | None], None]:
        def wrapped(phase: str, detail: str | None = None) -> None:
            if phase in ("thinking", "working"):
                self._autosave()
            if on_phase is not None:
                on_phase(phase, detail)

        return wrapped

    def _autosave(self, *, force: bool = False) -> None:
        """Save at a round boundary, at most every AUTOSAVE_SECONDS; never raises."""
        now = time.monotonic()
        last = self._last_autosave
        if not force and last is not None and now - last < self.AUTOSAVE_SECONDS:
            return
        self._last_autosave = now
        with contextlib.suppress(Exception):
            self.save_state(only_if_dir_exists=True)

    def _record_call(
        self, call: ModelCall, on_status: Callable[[dict[str, Any]], None] | None = None
    ) -> None:
        row = self._active_catalog_row()
        apply_call(
            self.totals,
            call,
            prompt_price=row.prompt_price if row else None,
            completion_price=row.completion_price if row else None,
            cache_read_price=row.cache_read_price if row else None,
        )
        timing = call.timing
        audit_event(
            self.agent.artifacts,
            "model",
            model=self.model_id or self.totals.model,
            http_status=timing.http_status if timing else None,
            duration_s=timing.elapsed_seconds if timing else None,
            prompt_tokens=call.usage.prompt_tokens,
            completion_tokens=call.usage.completion_tokens,
        )
        if on_status is not None:
            on_status(self.status_snapshot())

    def _advance(
        self,
        *,
        on_status: Callable[[dict[str, Any]], None] | None,
        callbacks: dict[str, Any],
    ) -> AgentRun:
        self._turn_no += 1
        self.cancel_event.clear()
        self.apply_known_context_limit()
        recorded_calls = 0

        def handle_model_call(call: ModelCall) -> None:
            nonlocal recorded_calls
            recorded_calls += 1
            self._record_call(call, on_status)

        on_phase = callbacks.pop("on_phase", None)
        try:
            run = self.agent.advance(
                self.messages,
                on_phase=self._autosaving_phase(on_phase),
                on_model_call=handle_model_call,
                should_cancel=self._should_cancel,
                cancel_event=self.cancel_event,
                **callbacks,
            )
        except BaseException:
            self._analysis_open = False
            self._analysis_request, self._analysis_steps, self._analysis_response = "", [], ""
            partial = self.agent.partial_run
            if partial is not None:
                with contextlib.suppress(Exception):
                    for call in partial.calls[recorded_calls:]:
                        self._record_call(call, None)
            raise
        for call in run.calls[recorded_calls:]:
            self._record_call(call, on_status)
        self.stopped_for_limit = run.stopped_for_limit
        self.interrupted = False
        if self._analysis_open:
            self._analysis_steps.extend(successful_steps(run))
            self._analysis_response = run.response
            if not run.stopped_for_limit:
                self._analysis_turns.append(
                    AnalysisTurn(
                        request=self._analysis_request,
                        response=self._analysis_response,
                        steps=list(self._analysis_steps),
                    )
                )
                self._analysis_open = False
                self._analysis_request, self._analysis_steps, self._analysis_response = "", [], ""
        return run

    def turn(
        self,
        request: str,
        *,
        images: Sequence[str] = (),
        on_status: Callable[[dict[str, Any]], None] | None = None,
        **callbacks: Any,
    ) -> AgentRun:
        """Append the user message and advance the shared history one turn."""
        if not self.key_set:
            raise MissingOpenRouterKey(MISSING_KEY_MESSAGE)
        msg = ChatMessage(role=Role.USER, content=request)
        if images:
            msg.images = ingest_images(self.agent.artifacts, list(images))
        (
            self._analysis_open,
            self._analysis_request,
            self._analysis_steps,
            self._analysis_response,
        ) = (True, request, [], "")
        self.messages.append(msg)
        self.interrupted = True
        self._autosave(force=True)
        return self._advance(on_status=on_status, callbacks=callbacks)

    def resume(
        self, *, on_status: Callable[[dict[str, Any]], None] | None = None, **callbacks: Any
    ) -> AgentRun:
        """Continue a turn that stopped at the tool-call limit or was interrupted."""
        if not self.key_set:
            raise MissingOpenRouterKey(MISSING_KEY_MESSAGE)
        if not (self.stopped_for_limit or self.interrupted):
            raise RuntimeError("Session is not stopped for the tool-call limit")
        self.interrupted = True
        self._autosave(force=True)
        return self._advance(on_status=on_status, callbacks=callbacks)

    def compact(self) -> ContextCompaction:
        """Halve the estimated context when possible, keeping protected messages."""
        limit = max(1, _estimate_tokens(self.messages) // 2)
        change = _compact(self.messages, root=self.agent.artifacts.root, limit=limit)
        if change.occurred:
            self.totals.context_used = _estimate_tokens(self.messages)
        return change

    def start_fresh(self) -> None:
        """Start a new conversation without reopening the process."""
        _stop_jobs("new conversation")
        old_id = self.session_id
        self.goal, self._goal_message, self.active_skills = "", None, []
        self.stopped_for_limit = False
        self.interrupted = False
        self._last_autosave = None
        self.agent.session_allowed_tools.clear()
        self.totals = SessionTotals(
            model=self.model_id or self.totals.model, context_limit=self.totals.context_limit
        )
        self._analysis_turns, self._analysis_open = [], False
        self._analysis_request, self._analysis_steps, self._analysis_response = "", [], ""
        if self.workspace is not None:
            self.session_id = _remember_session(self.workspace, self.model_id)
            self.agent.artifacts = LocalArtifactStore(
                self.workspace,
                artifact_root=layout.project_artifact_dir(self.workspace, self.session_id),
            )
            if isinstance(old_id, str) and old_id != self.session_id:
                discard_unused_session(self.workspace, old_id)
        self.messages = self.agent.opening_messages()

    def _goal_index(self) -> int | None:
        if self._goal_message is None:
            return None
        for idx, message in enumerate(self.messages):
            if message is self._goal_message:
                return idx
        return None

    def set_goal(self, text: str) -> None:
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("goal text must be a non-empty string")
        if len(cleaned) > _GOAL_MAX_CHARS:
            raise ValueError(f"goal text must be at most {_GOAL_MAX_CHARS} characters")
        self.goal = cleaned
        self._place_goal_message()

    def clear_goal(self) -> None:
        self.goal = ""
        if (idx := self._goal_index()) is not None:
            del self.messages[idx]
        self._goal_message = None

    def _place_goal_message(self) -> None:
        content = session_goal_text(self.goal)
        if (idx := self._goal_index()) is not None:
            self.messages[idx].content = content
            return
        self._goal_message = ChatMessage(role=Role.USER, content=content)
        end = 0
        if self.messages and self.messages[0].role == Role.SYSTEM:
            end = (
                2
                if len(self.messages) > 1
                and (self.messages[1].content or "").startswith(_PROJECT_INSTRUCTION_PREFIX)
                else 1
            )
        self.messages.insert(end, self._goal_message)

    def use_skill(self, name: str, argument: str | None = None) -> Skill:
        if not (cleaned := name.strip()):
            raise ValueError("skill name must be a non-empty string")
        requested, errors = normalize_skill_name(cleaned)
        if requested is None or errors:
            raise ValueError("; ".join(errors))
        skills = list(self.agent.skills)
        skill = next((s for s in skills if s.name == requested), None)
        if skill is None:
            raise FileNotFoundError(f"Skill {requested} is missing")
        extra = argument.strip() if isinstance(argument, str) else ""
        if skill.name in self.active_skills:
            if extra:
                self.messages.append(ChatMessage(role=Role.USER, content=f"Argument: {extra}"))
            return skill
        reg = ToolRegistry()
        register_skill_tool(reg, skills)
        result = reg.execute("use_skill", {"name": requested}, self.agent.artifacts)
        content = f"{result.text}\n\nArgument: {extra}" if extra else result.text
        self.messages.append(ChatMessage(role=Role.USER, content=content))
        self.active_skills.append(skill.name)
        return skill

    def save_state(self, *, only_if_dir_exists: bool = False) -> None:
        """Write this conversation to ``.atlas/sessions/<id>/state.json``."""
        if self.workspace is None or self.session_id is None:
            return
        directory = layout.project_session_dir(self.workspace, self.session_id)
        if only_if_dir_exists and not directory.is_dir():
            return
        directory.mkdir(parents=True, exist_ok=True)
        snapshot = list(self.messages)
        saved_at = datetime.now(UTC).isoformat()
        payload = {
            "id": self.session_id,
            "model": self.model_id,
            "goal": self.goal,
            "active_skills": list(self.active_skills),
            "messages": [{**m.model_dump(mode="json"), "images": []} for m in snapshot],
            "totals": self.totals.model_dump(mode="json"),
            "saved": saved_at,
            "interrupted": self.interrupted,
            "stopped_for_limit": self.stopped_for_limit,
        }
        text = redact_text(json.dumps(payload, indent=2), self.secrets)
        temp = directory / ".state.json.tmp"
        temp.write_text(text + "\n", encoding="utf-8")
        os.replace(temp, directory / "state.json")
        prompt = redact_text(first_prompt(snapshot), self.secrets)
        with contextlib.suppress(OSError):
            write_meta(
                directory,
                session_id=self.session_id,
                model=self.model_id,
                messages=sum(1 for item in snapshot if item.role is not Role.SYSTEM),
                saved=saved_at,
                interrupted=self.interrupted,
                stopped_for_limit=self.stopped_for_limit,
                prompt=prompt,
            )

    def load_state(self, session_id: str, *, adopt_model: bool = True) -> bool:
        """Restore a saved conversation and rebind model, goal, skills, and artifact store."""
        if self.workspace is None:
            return False
        path = layout.project_session_dir(self.workspace, session_id) / "state.json"
        if not path.is_file():
            return False
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("saved session must be a JSON object")
        messages = [ChatMessage.model_validate(item) for item in payload.get("messages", [])]
        inserted = repair_history(messages)
        self.session_id = session_id
        self.messages = messages
        self.goal = str(payload.get("goal") or "")
        raw_skills = payload.get("active_skills")
        self.active_skills = (
            [s for s in raw_skills if isinstance(s, str)] if isinstance(raw_skills, list) else []
        )
        if isinstance(totals := payload.get("totals"), dict):
            self.totals = SessionTotals.model_validate(totals)
        if adopt_model and isinstance(model := payload.get("model"), str) and model.strip():
            self.model_id = model.strip()
            self.totals.model = self.model_id
            if (cfg := getattr(self.agent.model, "config", None)) is not None:
                cfg.model = self.model_id
        self.agent.context_limit = self.totals.context_limit
        self.agent.artifacts = LocalArtifactStore(
            self.workspace, artifact_root=layout.project_artifact_dir(self.workspace, session_id)
        )
        if messages and messages[0].role is Role.SYSTEM:
            messages[0] = self.agent.opening_messages()[0]
        self._goal_message = None
        if self.goal:
            expected = session_goal_text(self.goal)
            matched = next(
                (m for m in self.messages if m.role is Role.USER and m.content == expected),
                None,
            )
            if matched is not None:
                self._goal_message = matched
            else:
                self._place_goal_message()
        self.stopped_for_limit = bool(payload.get("stopped_for_limit"))
        self.interrupted = bool(payload.get("interrupted")) or inserted > 0
        last = messages[-1] if messages else None
        if last is not None and last.role is Role.ASSISTANT and not last.tool_calls:
            self.interrupted = False
        self._last_autosave = None
        self.agent.session_allowed_tools.clear()
        self._analysis_turns, self._analysis_open = [], False
        return True

    def replay_messages(
        self, *, limit: int | None = None, max_chars: int | None = None
    ) -> list[dict[str, Any]]:
        """Build v2 ``session_loaded.messages`` replay items from ``self.messages``."""
        items, _ = self.replay_items_with_omitted(limit=limit, max_chars=max_chars)
        return items

    def replay_items_with_omitted(
        self, *, limit: int | None = 40, max_chars: int | None = 1500
    ) -> tuple[list[dict[str, Any]], int]:
        """Build v2 replay items plus the count of omitted earlier items."""
        calls_by_id = {c.id: c for m in self.messages for c in m.tool_calls}
        items: list[dict[str, Any]] = []

        def clip(text: str) -> str:
            if max_chars is None or len(text) <= max_chars:
                return text
            return text[: max(0, max_chars - 3)] + "..."

        for msg in self.messages:
            if msg.role is Role.SYSTEM or msg.tool_observation:
                continue
            if msg.role is Role.USER:
                text = msg.content or ""
                if (
                    msg is self._goal_message
                    or not text
                    or text.startswith(_PROJECT_INSTRUCTION_PREFIX)
                ):
                    continue
                items.append({"role": "user", "text": clip(text)})
            elif msg.role is Role.ASSISTANT:
                if msg.reasoning:
                    items.append({"role": "thought", "text": clip(msg.reasoning)})
                if msg.content:
                    items.append({"role": "assistant", "text": clip(msg.content)})
            elif msg.role is Role.TOOL:
                call = calls_by_id.get(msg.tool_call_id or "")
                name = call.name if call else "tool"
                summary = tool_summary(call) if call else name
                raw, ok, text, diff = msg.content or "", True, msg.content or "", None
                try:
                    res = ToolResult.model_validate_json(raw)
                    text, diff = res.text, res.diff
                    if res.summary:
                        summary = res.summary
                except Exception:
                    try:
                        err = json.loads(raw)
                        if isinstance(err, dict) and "error" in err:
                            ok, text = False, str(err.get("error") or "")
                    except ValueError:
                        pass
                items.append(
                    {
                        "role": "tool",
                        "text": clip(text),
                        "name": name,
                        "summary": summary,
                        "ok": ok,
                        "diff": clip(diff) if diff is not None else None,
                    }
                )
        if limit is not None and len(items) > limit:
            omitted = len(items) - limit
            return items[-limit:], omitted
        return items, 0

    def save_finished_skill(
        self,
        name: str,
        *,
        description: str | None = None,
        replace: bool = False,
        scope: str = "project",
    ) -> SavedSkill:
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

    def default_skill_name(self) -> str:
        if self._analysis_turns and (words := self._analysis_turns[0].request.strip().split()[:4]):
            try:
                return skill_slug(" ".join(words))
            except SkillSaveError:
                pass
        return "saved-analysis"

    def preview_saved_skill(self, name: str, *, scope: str = "project") -> SkillPreview:
        if scope == "user" and (
            self.workspace is None or not load_home_skills_enabled(self.workspace)
        ):
            raise SkillSaveError(
                "home skills are off; set ATLAS_HOME_SKILLS or home_skills = true in ~/.atlas/config.toml"
            )
        if self.workspace is None:
            raise SkillSaveError("project scope needs a workspace")
        slug = skill_slug(name)
        preview = preview_skill(self.workspace, slug, scope=scope, home=self.home)
        self._skill_previews[(slug, scope)] = preview.content_sha256
        return preview

    def enable_saved_skill(
        self, name: str, expected_sha256: str, *, scope: str = "project"
    ) -> SavedSkill:
        if scope == "user" and (
            self.workspace is None or not load_home_skills_enabled(self.workspace)
        ):
            raise SkillSaveError(
                "home skills are off; set ATLAS_HOME_SKILLS or home_skills = true in ~/.atlas/config.toml"
            )
        if self.workspace is None:
            raise SkillSaveError("project scope needs a workspace")
        slug = skill_slug(name)
        if (stored := self._skill_previews.get((slug, scope))) is None:
            raise SkillSaveError("enable needs a preview")
        token = expected_sha256.strip().lower()
        if token != stored and not (len(token) == 8 and stored.startswith(token)):
            raise SkillSaveError("skill hash does not match the preview")
        saved = enable_skill(self.workspace, slug, stored, scope=scope, home=self.home)
        self._skill_previews.pop((saved.name, scope), None)
        self.skill_catalog_note = _add_enabled_skill(self, saved.name, scope)
        return saved


def _add_enabled_skill(session: AgentSession, name: str, scope: str) -> str:
    from atlas.agent.skills import UseSkillTool, load_skill_directory, user_skill_load_parts

    restart = "Restart Atlas to load this skill."
    if (workspace := session.workspace) is None:
        return restart
    if scope == "user":
        if session.home is None or not load_home_skills_enabled(workspace):
            return restart
        base, relative, label = user_skill_load_parts(session.home, name)
    else:
        base, relative, label = workspace, f"{layout.ATLAS_SKILLS_RELATIVE}/{name}", ""
    try:
        skill = load_skill_directory(
            base,
            relative,
            label_prefix=label,
            artifact_root=layout.project_artifacts_dir(workspace),
        )
    except OSError:
        return restart
    if skill is None:
        return restart
    skills = sorted(
        [*[s for s in session.agent.skills if s.name != skill.name], skill], key=lambda s: s.name
    )
    session.agent.skills = skills
    session.agent.tools.replace(UseSkillTool(skills))
    if session.messages and session.messages[0].role == Role.SYSTEM:
        system = f"{session.agent.system_prompt()}\n\n{session.agent.environment_context()}"
        session.messages[0] = ChatMessage(role=Role.SYSTEM, content=system)
    return "The skill catalog was reloaded."


@contextlib.contextmanager
def session_bound(session: AgentSession) -> Iterator[None]:
    """Bind the session credential and secret maps for the duration of the block."""
    layout.bind_session_credentials(session.credential_keys)
    layout.bind_session_secrets(session.secrets)
    layout.quiet_http_client_logs()
    try:
        yield
    finally:
        layout.unbind_session_credentials()
        layout.unbind_session_secrets()


def status_fields(session: AgentSession) -> dict[str, Any]:
    return session.status_snapshot()


def step_event(step: AgentStep) -> dict[str, Any]:
    from atlas.agent.rpc import step_event as _step

    return _step(step)


def done_event(session: AgentSession, run: AgentRun) -> dict[str, Any]:
    from atlas.agent.rpc import done_event as _done

    return _done(session, run)


_redact_value = redact_value
_step_event = step_event
_status_fields = status_fields
_done_event = done_event


def resolve_model(workspace: Path, cli_model: str | None) -> str:
    if cli_model is not None and cli_model.strip():
        return cli_model.strip()
    if env_model := os.environ.get("ATLAS_MODEL", "").strip():
        return env_model
    return load_model(workspace)


def _remember_session(workspace: Path, model_id: str) -> str:
    layout.ensure_project_layout(workspace)
    session_id = uuid.uuid4().hex[:12]
    session_dir = layout.project_session_dir(workspace, session_id)
    session_dir.mkdir(parents=True, exist_ok=True)
    layout.project_artifact_dir(workspace, session_id).mkdir(parents=True, exist_ok=True)
    (session_dir / "meta.json").write_text(
        json.dumps({"id": session_id, "model": model_id, "created": datetime.now(UTC).isoformat()}),
        encoding="utf-8",
    )
    return session_id


class SessionNotFound(LookupError):
    """A requested saved session does not exist (``str(exc)`` names the id)."""


@dataclass
class SessionBundle:
    """An assembled session plus the resources its caller must release."""

    session: AgentSession
    model: ToolCapableModel
    api_key: str
    session_id: str
    workspace: Path
    home: Path
    owned_model: bool = True

    def close(self) -> None:
        """Close the model when this bundle built it; safe to call twice."""
        if not self.owned_model:
            return
        self.owned_model = False
        close = getattr(self.model, "close", None)
        if callable(close):
            close()


def _check_resume(workspace: Path, resume: str) -> None:
    """Raise SessionNotFound unless ``resume`` names a saved session; writes nothing."""
    if not resume or resume in {".", ".."} or "/" in resume or "\\" in resume:
        raise SessionNotFound(f"no saved session {resume}")
    if not (layout.project_session_dir(workspace, resume) / "state.json").is_file():
        raise SessionNotFound(f"no saved session {resume}")


def build_session(
    workspace: Path,
    *,
    model_id: str | None = None,
    approval: Literal["allow", "ask", "deny"] | None = None,
    max_tool_calls: int | None = None,
    max_spend_usd: float | None = None,
    resume: str | None = None,
    write_config: bool = True,
    home: Path | None = None,
    model: ToolCapableModel | None = None,
) -> SessionBundle:
    """Assemble an :class:`AgentSession` without serving it.

    Explicit arguments win over ``agent.toml``. A ``resume`` id is validated before
    anything is written, and the session is bound to its workspace before its saved
    state loads. An injected ``model`` is not closed by the bundle. Never serves.
    """
    workspace = workspace.resolve()
    home = home or Path.home()
    if resume is not None:
        _check_resume(workspace, resume)
    if write_config:
        ensure_agent_config(workspace)
    layout.ensure_user_layout(home)
    api_key = ""
    if model is None:
        api_key = (
            layout.resolve_openrouter_key(
                session_key=None,
                env_key=os.environ.get(layout.OPENROUTER_API_KEY),
                home=home,
            )
            or ""
        )
    resolved_model = resolve_model(workspace, model_id)
    session_id = resume or _remember_session(workspace, resolved_model)
    artifact_root = layout.project_artifact_dir(workspace, session_id)
    store = LocalArtifactStore(workspace, artifact_root=artifact_root)
    skill_home = home if load_home_skills_enabled(workspace) else None
    skills = load_skills(store, home=skill_home)
    registry = default_registry()
    register_skill_tool(registry, skills)
    owned = model is None
    active: ToolCapableModel
    if model is None:
        active = OpenRouterModel(
            ModelConfig(
                model=resolved_model,
                api_key=api_key,
                reasoning_effort=load_reasoning_effort(workspace),
            )
        )
    else:
        active = model
    try:
        agent = Agent(
            active,
            registry,
            store,
            max_tool_calls=(
                max_tool_calls if max_tool_calls is not None else load_max_tool_calls(workspace)
            ),
            max_tokens=load_optional_tokens(workspace),
            max_spend_usd=(
                max_spend_usd if max_spend_usd is not None else load_spend_limit(workspace)
            ),
            read_only=load_read_only(workspace),
            approval=approval or load_approval(workspace),
            skills=skills,
        )
        from atlas.agent.tools._subagent import SubagentTool

        def build_child(child_store: LocalArtifactStore, allow_writes: bool, budget: int) -> Agent:
            child_reg = default_registry()
            register_skill_tool(child_reg, skills)
            return Agent(
                active,
                child_reg,
                child_store,
                max_tool_calls=budget,
                max_tokens=agent.max_tokens,
                max_spend_usd=None,
                read_only=agent.read_only or not allow_writes,
                approval=agent.approval,
                on_permission=agent.on_permission,
                skills=skills,
                context_limit=agent.context_limit,
            )

        registry.register(SubagentTool(build_child))
        session = AgentSession(agent)
        session.session_id = session_id
        session.home = home
        session.workspace = workspace
        session.active_key = api_key or None
        session.requires_api_key = owned
        list_models = getattr(active, "list_models", None)
        if list_models is not None:
            session.fetch_models = list_models
        if resume is not None:
            if not session.load_state(resume):
                raise SessionNotFound(f"no saved session {resume}")
            session.loaded_at_launch = True
    except BaseException:
        if owned:
            close = getattr(active, "close", None)
            if callable(close):
                close()
        raise
    return SessionBundle(
        session=session,
        model=active,
        api_key=api_key,
        session_id=session_id,
        workspace=workspace,
        home=home,
        owned_model=owned,
    )


def serve(*args: Any, **kwargs: Any) -> int:
    from atlas.agent.rpc import serve as _serve

    return _serve(*args, **kwargs)


def main(argv: Sequence[str] | None = None) -> int:
    """Console entry for ``python -m atlas.agent`` / ``python -m atlas.agent.session``."""
    parser = argparse.ArgumentParser(prog="atlas.agent.session")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--approval",
        default=None,
        choices=("allow", "ask", "deny"),
        help="Per-call approval for mutating tools (default: agent.toml, else allow).",
    )
    parser.add_argument(
        "--resume",
        default=None,
        metavar="SESSION_ID",
        help="Reopen a saved session from .atlas/sessions/<id>/state.json",
    )
    args = parser.parse_args(argv)

    # Never the workspace: a workspace .env must not inject environment.
    repo_root = _find_repo_root(Path.cwd())
    if repo_root is not None:
        from dotenv import load_dotenv

        load_dotenv(repo_root / ".env", override=False)

    try:
        bundle = build_session(
            Path(args.workspace),
            model_id=args.model,
            approval=args.approval,
            resume=args.resume,
        )
    except SessionNotFound as exc:
        print(f"atlas: {exc}", file=sys.stderr)
        return 2
    except (ValueError, ValidationError, OSError) as exc:
        if args.resume is None:
            raise
        print(f"atlas: cannot resume {args.resume}: {exc}", file=sys.stderr)
        return 2
    try:
        return serve(
            bundle.session,
            sys.stdin,
            sys.stdout,
            redact=bundle.api_key or None,
            resumed=bundle.session.loaded_at_launch,
        )
    finally:
        if bundle.session.session_id:
            discard_unused_session(bundle.workspace, bundle.session.session_id)
        bundle.close()


if __name__ == "__main__":
    raise SystemExit(main())
