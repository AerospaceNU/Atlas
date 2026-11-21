from __future__ import annotations

import datetime
import difflib
import hashlib
import json
import logging
import platform
import re
import secrets
import sys
import time
import tomllib
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from atlas.agent.artifacts import ArtifactError, LocalArtifactStore
from atlas.agent.audit import record as audit_event
from atlas.agent.checkpoints import snapshot
from atlas.agent.contracts import (
    ChatMessage,
    ModelCall,
    ModelImage,
    ModelResponse,
    Role,
    ToolCall,
    ToolCapableModel,
    ToolRegistry,
    ToolResult,
    UnknownToolError,
)
from atlas.agent.images import ingest_image, ingest_images, is_model_image
from atlas.agent.instructions import load_project_instructions
from atlas.agent.layout import project_agent_config_path, project_atlas_root
from atlas.agent.metrics import observe_model_call, output_tokens_per_second
from atlas.agent.model import validate_reasoning_effort
from atlas.agent.skills import (
    Skill,
    render_system_prompt,
)
from atlas.agent.skills import (
    load_home_skills_enabled as load_home_skills_enabled,
)
from atlas.agent.system_prompt import load_system_prompt

_LOGGER = logging.getLogger(__name__)

DEFAULT_MAX_TOOL_CALLS = 256
DEFAULT_MODEL = "deepseek/deepseek-v4.1-flash"
MAX_TOOL_RESULT_CHARS = 20_000
CHARS_PER_TOKEN = 4
MAX_CONTEXT_TOKENS = 1_000_000
COMPACT_PREVIEW_CHARS = 500
KEEP_RECENT_MESSAGES = 6
_COMPACT_TARGET_FRACTION = 0.8


_STOPPED_RESPONSE = "Stopped after reaching the configured tool-use limit."
_STOPPED_BUDGET = "Stopped after reaching the configured token or spend budget."
_STOPPED_CANCELLED = "Stopped at your request."
_STOPPED_CONTEXT = (
    "The current request and required instructions exceed the context limit. "
    "Shorten the request or start a new conversation."
)
_PREVIEW_LIMIT = 4000
_DIGEST_HEADER = "Context digest (earlier turns)"
_DIGEST_LIMIT_CHARS = 3000

_MAX_EMPTY_REPLY_NUDGES = 2
_MAX_INTENT_NUDGES = 1
_INTENT_MAX_CHARS = 400
_INTENT_PREFIXES = (
    "let me",
    "let's",
    "i'll",
    "i will",
    "i'm going to",
    "i am going to",
    "now i",
    "next, i",
    "next i",
)
_NUDGE_INTENT = (
    "You described a next step but did not call a tool. If the work is not finished, make "
    "that tool call now. If it is finished, give your final answer."
)
_MAX_OVERFLOW_RETRIES = 3
_OVERFLOW_SHRINK = 0.5
_NUDGE_LENGTH = (
    "Your last reply was cut off at the output limit before it contained any text or tool "
    "call. Continue the task in smaller steps: call a tool, or give your final answer."
)
_NUDGE_EMPTY = (
    "Your last reply was empty: no text and no tool call. Continue the task: call a tool, "
    "or give your final answer."
)
_DIGEST_EDIT_TOOLS = frozenset({"edit_file", "patch_file", "write_file", "apply_patch"})
_NOTE_CHARS = 200
_DIGEST_CAPS = {"edited": 24, "read": 24, "commands": 12, "errors": 12, "notes": 8, "plan": 12}

# Tool calls in one assistant turn are independent, so they run concurrently.
# The cap keeps a model that emits twenty calls from opening twenty shells.
_MAX_PARALLEL_TOOLS = 4

# Tools that change the workspace. A read-only session refuses them, and each
# one is checkpointed before it runs so /undo can put the file back.
MUTATING_TOOLS = frozenset(
    {"apply_patch", "bash", "edit_file", "git_commit", "patch_file", "write_file"}
)

# Satellite tools that become mutating only for specific arguments. Approval-gated, not
# undoable (regenerable analysis/ data).
_PATH_WRITING_TOOLS = frozenset({"render_map", "export_geotiff"})


def _truthy(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return value is True or (type(value) is int and value == 1)


def _writes_to_path(call: ToolCall) -> bool:
    path = call.arguments.get("path")
    return call.name in _PATH_WRITING_TOOLS and isinstance(path, str) and bool(path.strip())


_MAX_VALIDATION_CHARS = 1500
_REPEAT_NOTE_AT = 3
_MIN_BUDGET_WARNING_CALLS = 20
_TOOL_OUTPUT_DIR = "tool-output"
_MAX_SPILL_ID_CHARS = 64
_MAX_ENV_PATH_CHARS = 300

ErrorKind = Literal[
    "validation",
    "unknown_tool",
    "tool_failure",
    "limit",
    "cancelled",
    "budget",
    "denied",
]


class AgentStep(BaseModel):
    """One tool action recorded in an autonomous run."""

    call: ToolCall
    result: ToolResult | None = None
    error: str | None = None
    error_kind: ErrorKind | None = None
    notes: list[str] = Field(default_factory=list)


class PermissionRequest(BaseModel):
    """One mutating action waiting for a person's approval.

    Attributes:
        id: The tool-call id, reused as the approval id.
        tool: Registered tool name.
        summary: One line a person can decide on.
        arguments: The model's arguments, for display only.
        preview: Unified diff or command text, truncated for the transcript.
    """

    id: str
    tool: str
    summary: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    preview: str = ""


class ContextCompaction(BaseModel):
    """Changes to model context, excluding the original tool results in the run log."""

    occurred: bool = False
    tokens_reclaimed: int = 0
    tool_results_shortened: int = 0
    messages_dropped: int = 0
    digest_inserted: bool = False
    digest_chars: int = 0


class AgentRun(BaseModel):
    """Final response plus a transparent record of every tool action."""

    response: str
    steps: list[AgentStep] = Field(default_factory=list)
    stopped_for_limit: bool = False
    stopped_for_budget: bool = False
    stopped_for_context: bool = False
    cancelled: bool = False
    calls: list[ModelCall] = Field(default_factory=list)
    compaction: ContextCompaction = Field(default_factory=ContextCompaction)
    empty_replies: int = 0
    intent_nudges: int = 0
    error: str | None = None


def _announces_next_step(text: str) -> bool:
    """Return whether a short text-only reply announces work instead of finishing it."""
    stripped = text.strip()
    if not stripped or len(stripped) > _INTENT_MAX_CHARS:
        return False
    return stripped.lower().startswith(_INTENT_PREFIXES)


def _load_agent_config(root: Path) -> dict[str, object]:
    """Return ``.atlas/agent.toml`` under ``root``, or an empty mapping if absent.

    Malformed TOML raises ``ValueError`` (``tomllib.TOMLDecodeError``).
    """
    config_path = project_agent_config_path(root)
    if not config_path.is_file():
        return {}
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("agent.toml must be a table")
    return data


def load_optional_limit(
    root: Path, key: str, *, kind: type[int] | type[float]
) -> int | float | None:
    """Read an optional numeric budget from ``.atlas/agent.toml``.

    Args:
        root: Workspace root that holds ``.atlas/agent.toml``.
        key: Budget key, such as ``max_tokens`` or ``max_spend_usd``.
        kind: ``int`` or ``float``; a value of the wrong type is refused.

    Returns:
        The configured value, or None when the key is absent.

    Raises:
        ValueError: The key is present but not a positive number of ``kind``.
    """
    data = _load_agent_config(root)
    if key not in data:
        return None
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{key} must be a positive number")
    if kind is int and not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    if value <= 0:
        raise ValueError(f"{key} must be greater than zero")
    return value


def load_approval(root: Path) -> Literal["allow", "ask", "deny"]:
    """Return the approval mode from ``.atlas/agent.toml`` (default ``allow``)."""
    data = _load_agent_config(root)
    value = data.get("approval", "allow")
    if value == "allow":
        return "allow"
    if value == "ask":
        return "ask"
    if value == "deny":
        return "deny"
    raise ValueError("approval must be allow, ask, or deny")


def load_read_only(root: Path) -> bool:
    """Return whether ``.atlas/agent.toml`` marks this workspace read-only."""
    data = _load_agent_config(root)
    value = data.get("read_only", False)
    if not isinstance(value, bool):
        raise ValueError("read_only must be true or false")
    return value


def load_optional_tokens(root: Path) -> int | None:
    """Return ``max_tokens`` from ``.atlas/agent.toml``, or None when unset."""
    value = load_optional_limit(root, "max_tokens", kind=int)
    return None if value is None else int(value)


def load_spend_limit(root: Path) -> float | None:
    """Return ``max_spend_usd`` from ``.atlas/agent.toml``, or None when unset."""
    value = load_optional_limit(root, "max_spend_usd", kind=float)
    return None if value is None else float(value)


def load_max_tool_calls(root: Path) -> int:
    """Read ``.atlas/agent.toml`` under ``root``; missing key means the default.

    A present key must be an int (not a bool) greater than or equal to 1.
    Anything else, including malformed TOML, raises ``ValueError``.
    """
    data = _load_agent_config(root)
    if "max_tool_calls" not in data:
        return DEFAULT_MAX_TOOL_CALLS
    value = data["max_tool_calls"]
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("max_tool_calls must be an integer >= 1")
    return value


def load_reasoning_effort(root: Path) -> str | None:
    """Return ``reasoning_effort`` from ``.atlas/agent.toml``, or None when unset.

    Raises:
        ValueError: The key is present but not an OpenRouter effort level.
    """
    data = _load_agent_config(root)
    if "reasoning_effort" not in data:
        return None
    value = data["reasoning_effort"]
    if not isinstance(value, str):
        raise ValueError("reasoning_effort must be a string")
    return validate_reasoning_effort(value)


def ensure_agent_config(root: Path) -> Path:
    """Create ``.atlas/agent.toml`` with defaults when the file is missing.

    An existing file is left unchanged, including a file that omits a key.
    """
    config_dir = project_atlas_root(root)
    config_path = project_agent_config_path(root)
    if config_path.is_file():
        return config_path
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                "# OpenRouter model id. Change this to switch the agent TUI.",
                f'model = "{DEFAULT_MODEL}"',
                "",
                "# Tool-call budget for one turn. Must be an integer >= 1.",
                f"max_tool_calls = {DEFAULT_MAX_TOOL_CALLS}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def load_model(root: Path) -> str:
    """Read ``model`` from ``.atlas/agent.toml``; missing key means the default.

    A present key must be a non-empty string. The value is an OpenRouter model id.
    """
    data = _load_agent_config(root)
    if "model" not in data:
        return DEFAULT_MODEL
    value = data["model"]
    if not isinstance(value, str) or not value.strip():
        raise ValueError("model must be a non-empty string")
    return value.strip()


class Agent:
    """Run a tools-capable model against a fixed local artifact workspace.

    The agent spends a bounded budget of tool calls. Only tools registered on
    the supplied registry can be invoked; a model cannot invent a capability by
    naming it in a response.
    """

    def __init__(
        self,
        model: ToolCapableModel,
        tools: ToolRegistry,
        artifacts: LocalArtifactStore,
        *,
        max_tool_calls: int | None = None,
        max_tokens: int | None = None,
        max_spend_usd: float | None = None,
        read_only: bool = False,
        approval: Literal["allow", "ask", "deny"] = "allow",
        on_permission: Callable[[PermissionRequest], bool] | None = None,
        skills: Sequence[Skill] | None = None,
    ) -> None:
        if max_tool_calls is None:
            max_tool_calls = load_max_tool_calls(artifacts.root)
        elif isinstance(max_tool_calls, bool) or max_tool_calls < 1:
            raise ValueError("max_tool_calls must be an integer >= 1")
        self.model = model
        self.tools = tools
        self.artifacts = artifacts
        if max_tokens is not None and (isinstance(max_tokens, bool) or max_tokens < 1):
            raise ValueError("max_tokens must be an integer >= 1")
        if max_spend_usd is not None and max_spend_usd <= 0:
            raise ValueError("max_spend_usd must be greater than zero")
        self.max_tool_calls = max_tool_calls
        self.max_tokens = max_tokens
        self.max_spend_usd = max_spend_usd
        self.read_only = read_only
        self.approval = approval
        self.on_permission = on_permission
        self.skills = list(skills or [])
        self.instruction_nonce = secrets.token_hex(8)
        self.partial_run: AgentRun | None = None
        # Applied to the saved copy of an oversized tool result (never to the visible head/tail).
        self.redact_output: Callable[[str], str] | None = None
        self._empty_replies = 0
        self._intent_nudges = 0

    def system_prompt(self) -> str:
        """Return the package system prompt, plus the skill catalog when skills are loaded."""
        return render_system_prompt(load_system_prompt(), self.skills)

    def environment_context(self) -> str:
        """Return host facts the model cannot infer: directory, shell, platform, date.

        Contains no file contents and no git data.
        """
        store = self.artifacts
        root = _clean_path(store.root)
        shell_root = _clean_path(store.read_root or store.root)
        return "\n".join(
            [
                "Environment",
                f"- Working directory: {root} (relative tool paths resolve here)",
                f"- Shell: bash runs each command in a new shell starting in {shell_root}; "
                "cd and exported variables do not carry over to the next call",
                f"- Platform: {platform.system()} {platform.release()} ({sys.platform})",
                f"- Date: {_today()}",
            ]
        )

    def opening_messages(self) -> list[ChatMessage]:
        """Return the trusted system prompt and any project instructions.

        The first message is :meth:`system_prompt` followed by
        :meth:`environment_context` and nothing else. When the
        project root has a readable instruction file, the next message is that
        file as delimited project guidance with role ``user``. The wrapper tags
        use :attr:`instruction_nonce`. A missing or unreadable file leaves the
        list as the system message alone. Loading the file does not change
        ``tools`` or the store roots.

        Returns:
            The messages that start a run or a session.
        """
        system = f"{self.system_prompt()}\n\n{self.environment_context()}"
        messages = [ChatMessage(role=Role.SYSTEM, content=system)]
        instructions = load_project_instructions(self.artifacts, nonce=self.instruction_nonce)
        if instructions is not None:
            messages.append(ChatMessage(role=Role.USER, content=instructions))
        return messages

    def run(self, request: str, *, images: Sequence[str] = ()) -> AgentRun:
        """Answer ``request``, making bounded, registry-approved tool calls.

        ``images`` are paths the store can resolve, including absolute paths.
        Each one is read through the store and sent with the user message.
        Project instructions, when present, are inserted after the system
        prompt and before ``request``.
        """
        user = ChatMessage(role=Role.USER, content=request)
        if images:
            user.images = ingest_images(self.artifacts, list(images))
        messages = [*self.opening_messages(), user]
        return self.advance(messages)

    def advance(
        self,
        messages: list[ChatMessage],
        *,
        on_step: Callable[[AgentStep], None] | None = None,
        on_phase: Callable[[str, str | None], None] | None = None,
        on_thought: Callable[[str, float | None], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> AgentRun:
        """Continue ``messages`` in place while the tool-call budget lasts.

        The caller owns the history: no user or system message is appended here.
        Every tool call that fits in the budget runs, even after one fails. A
        call past the budget is recorded as a ``limit`` step and still gets a
        tool message so every ``tool_call_id`` is answered.

        ``should_cancel`` is polled between rounds and before a tool batch. When
        it returns true the turn stops without another model call, and every
        pending ``tool_call_id`` still gets a ``cancelled`` tool message so the
        history stays valid for the next turn.

        ``max_tokens`` and ``max_spend_usd`` stop the turn with
        :data:`_STOPPED_BUDGET` once the running total is at or past the limit.

        An empty model reply (no text, no tool call) is answered with a short
        nudge, at most :data:`_MAX_EMPTY_REPLY_NUDGES` times in a row (a reply with
        tool calls resets the count), and never spends tool budget. A text-only
        reply that announces a next step after at least one tool call is nudged
        once per call (:func:`_announces_next_step`), also without spending budget. A model exception is re-raised unchanged after the
        work so far is stored on :attr:`partial_run`. A provider refusal for
        context length compacts the history and retries.
        """
        self.partial_run = None
        self._empty_replies = 0
        self._intent_nudges = 0
        run = self._advance_turn(
            messages,
            on_step=on_step,
            on_phase=on_phase,
            on_thought=on_thought,
            should_cancel=should_cancel,
        )
        run.empty_replies = self._empty_replies
        run.intent_nudges = self._intent_nudges
        return run

    def _advance_turn(
        self,
        messages: list[ChatMessage],
        *,
        on_step: Callable[[AgentStep], None] | None,
        on_phase: Callable[[str, str | None], None] | None,
        on_thought: Callable[[str, float | None], None] | None,
        should_cancel: Callable[[], bool] | None,
    ) -> AgentRun:
        """Run the loop for :meth:`advance`."""
        nudges = 0
        steps: list[AgentStep] = []
        calls: list[ModelCall] = []
        executed = 0
        budget_warned = False
        repeats: dict[tuple[str, str], tuple[int, str]] = {}
        compaction = ContextCompaction()
        current_user = _current_user_message(messages)

        def emit(phase: str, name: str | None = None) -> None:
            if on_phase is not None:
                on_phase(phase, name)

        while executed < self.max_tool_calls:
            if should_cancel is not None and should_cancel():
                return AgentRun(
                    response=_STOPPED_CANCELLED,
                    steps=steps,
                    cancelled=True,
                    calls=calls,
                    compaction=compaction,
                )
            config = getattr(self.model, "config", None)
            reasoning = True if config is None else bool(getattr(config, "reasoning", True))
            emit("thinking" if reasoning else "working")
            if _estimate_tokens(messages) > MAX_CONTEXT_TOKENS:
                change = _compact(messages, protected_user=current_user, root=self.artifacts.root)
                _merge_compaction(compaction, change)
                if _estimate_tokens(messages) > MAX_CONTEXT_TOKENS:
                    return AgentRun(
                        response=_STOPPED_CONTEXT,
                        steps=steps,
                        calls=calls,
                        stopped_for_context=True,
                        compaction=compaction,
                    )
            completed = self._complete(messages, current_user, steps, calls, compaction)
            if isinstance(completed, AgentRun):
                return completed
            response = completed
            observed = observe_model_call(response)
            if observed is not None:
                calls.append(observed)
            empty = not response.tool_calls and not (response.content or "").strip()
            if empty:
                self._empty_replies += 1
            elif response.tool_calls:
                nudges = 0  # the cap counts consecutive empty replies only
            discard = empty and nudges < _MAX_EMPTY_REPLY_NUDGES
            if not discard and not empty:
                messages.append(
                    ChatMessage(
                        role=Role.ASSISTANT,
                        content=response.content,
                        tool_calls=response.tool_calls,
                        reasoning=response.reasoning,
                        reasoning_details=response.reasoning_details,
                    )
                )
            try:
                if response.reasoning and on_thought is not None:
                    speed = output_tokens_per_second(
                        response.usage.completion_tokens,
                        response.timing,
                    )
                    on_thought(response.reasoning, speed)
                if discard:
                    if self._over_budget(calls):
                        return AgentRun(
                            response=_STOPPED_BUDGET,
                            steps=steps,
                            stopped_for_budget=True,
                            calls=calls,
                            compaction=compaction,
                        )
                    nudges += 1
                    nudge = _NUDGE_LENGTH if response.finish_reason == "length" else _NUDGE_EMPTY
                    messages.append(
                        ChatMessage(role=Role.USER, content=nudge, tool_observation=True)
                    )
                    continue
                if (
                    not response.tool_calls
                    and executed >= 1
                    and self._intent_nudges < _MAX_INTENT_NUDGES
                    and _announces_next_step(response.content or "")
                ):
                    if self._over_budget(calls):
                        return AgentRun(
                            response=_STOPPED_BUDGET,
                            steps=steps,
                            stopped_for_budget=True,
                            calls=calls,
                            compaction=compaction,
                        )
                    self._intent_nudges += 1
                    messages.append(
                        ChatMessage(role=Role.USER, content=_NUDGE_INTENT, tool_observation=True)
                    )
                    continue
                if not response.tool_calls:
                    return AgentRun(
                        response=response.content or "",
                        steps=steps,
                        calls=calls,
                        compaction=compaction,
                    )
                if should_cancel is not None and should_cancel():
                    run = self._stop_cancelled(messages, steps, calls, response.tool_calls)
                    run.compaction = compaction
                    return run
                if self._over_budget(calls):
                    self._answer_pending(
                        messages,
                        steps,
                        response.tool_calls,
                        error="token or spend budget reached",
                        error_kind="budget",
                    )
                    return AgentRun(
                        response=_STOPPED_BUDGET,
                        steps=steps,
                        stopped_for_budget=True,
                        calls=calls,
                        compaction=compaction,
                    )

                pending: list[ToolCall] = []
                limited: list[ToolCall] = []
                for call in response.tool_calls:
                    if executed < self.max_tool_calls:
                        pending.append(call)
                        executed += 1
                    else:
                        limited.append(call)

                warning: str | None = None
                if (
                    pending
                    and not budget_warned
                    and self.max_tool_calls >= _MIN_BUDGET_WARNING_CALLS
                    and executed >= self.max_tool_calls - max(2, self.max_tool_calls // 10)
                ):
                    budget_warned = True
                    warning = (
                        f"tool budget: {executed} of {self.max_tool_calls} calls used this turn, "
                        f"{self.max_tool_calls - executed} left. Finish the current change, verify "
                        "it, and give your final answer before the budget runs out."
                    )
                self._tool_phase(
                    messages,
                    steps,
                    response,
                    pending,
                    limited,
                    repeats=repeats,
                    warning=warning,
                    emit=emit,
                    on_step=on_step,
                )
            except BaseException as exc:
                self._close_interrupted_round(
                    messages, steps, calls, compaction, response.tool_calls, exc
                )
                raise
            if executed >= self.max_tool_calls or limited:
                return AgentRun(
                    response=_STOPPED_RESPONSE,
                    steps=steps,
                    stopped_for_limit=True,
                    calls=calls,
                    compaction=compaction,
                )
        return AgentRun(
            response=_STOPPED_RESPONSE,
            steps=steps,
            stopped_for_limit=True,
            calls=calls,
            compaction=compaction,
        )

    def _tool_phase(
        self,
        messages: list[ChatMessage],
        steps: list[AgentStep],
        response: ModelResponse,
        pending: Sequence[ToolCall],
        limited: Sequence[ToolCall],
        *,
        repeats: dict[tuple[str, str], tuple[int, str]],
        warning: str | None,
        emit: Callable[[str, str | None], None],
        on_step: Callable[[AgentStep], None] | None,
    ) -> None:
        """Run one round's tools and append a tool message for every call, in order."""
        for call in (*pending, *limited):
            emit("tool_calling", call.name)
        argument_errors = response.argument_errors
        results = self._run_round(pending, argument_errors, emit)
        for call, step in zip(pending, results, strict=True):
            if call.id not in argument_errors:
                self._add_notes(step, repeats)
        if warning is not None and results:
            results[-1].notes.append(warning)
        attached: list[ModelImage] = []
        image_errors: list[str] = []
        for call, step in zip(pending, results, strict=True):
            images, errors = _ingest_tool_images(step, self.artifacts)
            if errors and step.error is None:
                step.error = "\n".join(errors)
                step.error_kind = "tool_failure"
            steps.append(step)
            messages.append(self._tool_chat_message(call, step))
            attached.extend(images)
            image_errors.extend(errors)
            if on_step is not None:
                on_step(step)
        for call in limited:
            step = AgentStep(call=call, error="tool call limit reached", error_kind="limit")
            steps.append(step)
            messages.append(self._tool_chat_message(call, step))
            if on_step is not None:
                on_step(step)
        if attached or image_errors:
            messages.append(
                ChatMessage(
                    role=Role.USER,
                    content="\n".join(image_errors) if image_errors else None,
                    images=attached,
                    tool_observation=True,
                )
            )

    def _tool_chat_message(self, call: ToolCall, step: AgentStep) -> ChatMessage:
        """Build the tool message for a step, with saved output redacted when configured."""
        return ChatMessage(
            role=Role.TOOL,
            tool_call_id=call.id,
            content=_tool_message(step, self.artifacts, self.redact_output),
        )

    def _close_interrupted_round(
        self,
        messages: list[ChatMessage],
        steps: list[AgentStep],
        calls: list[ModelCall],
        compaction: ContextCompaction,
        tool_calls: Sequence[ToolCall],
        exc: BaseException,
    ) -> None:
        """Answer every tool call still unanswered after an exception; never swallows it.

        The caller re-raises ``exc``. History stays a valid call/result sequence.
        """
        answered = {m.tool_call_id for m in messages if m.role is Role.TOOL}
        missing = [call for call in tool_calls if call.id not in answered]
        self._answer_pending(
            messages,
            steps,
            missing,
            error=f"interrupted: {type(exc).__name__}",
            error_kind="cancelled",
        )
        if isinstance(exc, Exception):
            self._store_partial(steps, calls, compaction, exc)

    def _store_partial(
        self,
        steps: list[AgentStep],
        calls: list[ModelCall],
        compaction: ContextCompaction,
        exc: BaseException,
    ) -> None:
        """Record the work done before ``exc`` on :attr:`partial_run`."""
        self.partial_run = AgentRun(
            response="",
            steps=steps,
            calls=calls,
            compaction=compaction,
            empty_replies=self._empty_replies,
            intent_nudges=self._intent_nudges,
            error=f"{type(exc).__name__}: {exc}",
        )

    def _complete(
        self,
        messages: list[ChatMessage],
        current_user: ChatMessage | None,
        steps: list[AgentStep],
        calls: list[ModelCall],
        compaction: ContextCompaction,
    ) -> ModelResponse | AgentRun:
        """Call the model once, compacting and retrying on a context-length refusal.

        The shrink target is a fraction of the current estimate; no provider window is
        read and ``MAX_CONTEXT_TOKENS`` is never changed. Any other exception is stored
        as :attr:`partial_run` and re-raised unchanged (a ``BaseException`` too).
        """
        retries = 0
        while True:
            try:
                return self.model.complete(messages, self.tools.definitions)
            except BaseException as exc:
                if not (isinstance(exc, Exception) and getattr(exc, "is_context_overflow", False)):
                    self._store_partial(steps, calls, compaction, exc)
                    raise
                change = ContextCompaction()
                if retries < _MAX_OVERFLOW_RETRIES:
                    retries += 1
                    limit = max(1, int(_estimate_tokens(messages) * _OVERFLOW_SHRINK))
                    change = _compact(
                        messages,
                        protected_user=current_user,
                        root=self.artifacts.root,
                        limit=limit,
                    )
                    _merge_compaction(compaction, change)
                if not change.occurred:
                    return AgentRun(
                        response=_STOPPED_CONTEXT,
                        steps=steps,
                        calls=calls,
                        stopped_for_context=True,
                        compaction=compaction,
                    )

    def _over_budget(self, calls: Sequence[ModelCall]) -> bool:
        """Return true once the running usage reaches a configured budget."""
        if self.max_tokens is not None:
            total = sum(call.usage.prompt_tokens + call.usage.completion_tokens for call in calls)
            if total >= self.max_tokens:
                return True
        if self.max_spend_usd is not None:
            spent = 0.0
            for call in calls:
                if call.usage.cost is not None:
                    spent += call.usage.cost
            if spent >= self.max_spend_usd:
                return True
        return False

    def _execute_all(
        self,
        calls: Sequence[ToolCall],
        *,
        on_phase: Callable[[str, str | None], None] | None = None,
    ) -> list[AgentStep]:
        """Run one round's tool calls, returning them in request order.

        Read-only calls run concurrently. As soon as one call in the round can
        change the workspace, the whole batch runs in order: two writes to one
        file, or an edit beside a patch, must not interleave, and a checkpoint
        taken in parallel with the write it protects is not a checkpoint.
        """
        serial = any(_is_mutating(call) for call in calls) or len(calls) <= 1
        if on_phase is not None:
            for call in calls:
                on_phase("tool_executing", call.name)
        if serial:
            return [self._execute(call) for call in calls]
        with ThreadPoolExecutor(max_workers=min(_MAX_PARALLEL_TOOLS, len(calls))) as pool:
            return list(pool.map(self._execute, calls))

    def _stop_cancelled(
        self,
        messages: list[ChatMessage],
        steps: list[AgentStep],
        calls: list[ModelCall],
        pending: Sequence[ToolCall],
    ) -> AgentRun:
        """Answer every unanswered tool call after a cancel, then stop."""
        self._answer_pending(
            messages, steps, pending, error="cancelled by the user", error_kind="cancelled"
        )
        return AgentRun(response=_STOPPED_CANCELLED, steps=steps, cancelled=True, calls=calls)

    def _answer_pending(
        self,
        messages: list[ChatMessage],
        steps: list[AgentStep],
        pending: Sequence[ToolCall],
        *,
        error: str,
        error_kind: ErrorKind,
    ) -> None:
        """Close unexecuted call/result pairs when a turn stops."""
        for call in pending:
            step = AgentStep(call=call, error=error, error_kind=error_kind)
            steps.append(step)
            messages.append(self._tool_chat_message(call, step))

    def _run_round(
        self,
        pending: Sequence[ToolCall],
        argument_errors: dict[str, str],
        emit: Callable[[str, str | None], None],
    ) -> list[AgentStep]:
        """Execute a round, recording unparseable calls as validation steps unexecuted."""
        runnable = [call for call in pending if call.id not in argument_errors]
        ran = iter(self._execute_all(runnable, on_phase=emit) if runnable else [])
        steps: list[AgentStep] = []
        for call in pending:
            if call.id in argument_errors:
                steps.append(_argument_error_step(call, argument_errors[call.id]))
            else:
                steps.append(next(ran))
        return steps

    def _add_notes(self, step: AgentStep, repeats: dict[tuple[str, str], tuple[int, str]]) -> None:
        """Attach harness feedback to a step; never changes its result or raises."""
        call = step.call
        unknown = self._unknown_argument_note(call)
        if unknown is not None:
            step.notes.append(unknown)
        try:
            key = (call.name, json.dumps(call.arguments, sort_keys=True, default=str))
            text = step.result.text if step.result is not None else (step.error or "")
            digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
            previous = repeats.get(key)
            count = previous[0] + 1 if previous is not None and previous[1] == digest else 1
            repeats[key] = (count, digest)
        except Exception:
            return
        if count >= _REPEAT_NOTE_AT:
            step.notes.append(
                f"this exact call has returned the same result {count} times; repeating it "
                "will not help. Change the arguments or try a different approach."
            )

    def _unknown_argument_note(self, call: ToolCall) -> str | None:
        """Name arguments the tool's input model does not declare, or None."""
        try:
            tool = self.tools.get(call.name)
            if tool is None:
                return None
            fields = tool.input_model.model_fields
            accepted = set(fields)
            for info in fields.values():
                if info.alias:
                    accepted.add(info.alias)
                if isinstance(info.validation_alias, str):
                    accepted.add(info.validation_alias)
            unknown = sorted(str(name) for name in call.arguments if name not in accepted)
            if not unknown:
                return None
            return (
                f"ignored unknown argument(s): {', '.join(unknown)}; "
                f"valid arguments: {', '.join(fields)}"
            )
        except Exception:
            return None

    def _validation_message(self, call: ToolCall, exc: ValidationError) -> str:
        """Render a validation failure without pydantic URLs or echoed input."""
        lines = [f"Invalid arguments for {call.name}:"]
        for error in exc.errors(include_url=False, include_context=False, include_input=False):
            location = ".".join(str(part) for part in error["loc"]) or "(arguments)"
            lines.append(f"- {location}: {error['msg']}")
        try:
            tool = self.tools.get(call.name)
            if tool is not None:
                parts = []
                for name, info in tool.input_model.model_fields.items():
                    if info.is_required():
                        parts.append(f"{name} (required)")
                    else:
                        default = info.get_default(call_default_factory=True)
                        parts.append(f"{name} (default {default!r})")
                lines.append("Expected arguments: " + ", ".join(parts))
        except Exception:
            pass
        text = "\n".join(lines)
        if len(text) > _MAX_VALIDATION_CHARS:
            text = text[: _MAX_VALIDATION_CHARS - 3] + "..."
        return text

    def _checkpoint(self, call: ToolCall) -> None:
        """Snapshot a file a mutating tool is about to change."""
        # Satellite overwrite/path writes are directories or rasters, regenerable, so they
        # are skipped explicitly rather than snapshotted and logged.
        if call.name not in MUTATING_TOOLS or call.name in {"run_tests", "git_commit"}:
            return
        if call.name == "apply_patch":
            paths = [] if self._patch_is_dry_run(call) else _patch_paths(call.arguments)
        else:
            path = call.arguments.get("path")
            paths = [path] if isinstance(path, str) and path else []
        for path in paths:
            try:
                snapshot(self.artifacts, path)
            except Exception:
                _LOGGER.exception("Could not checkpoint %s before %s", path, call.name)

    def _patch_is_dry_run(self, call: ToolCall) -> bool:
        """Return whether apply_patch will not write, using the tool's own coercion.

        Validation failure also means nothing is written, so no snapshot is needed.
        """
        tool = self.tools.get("apply_patch")
        if tool is None:
            return True
        try:
            request = tool.input_model.model_validate(call.arguments)
        except ValidationError:
            return True
        return getattr(request, "dry_run", False) is True

    def _execute(self, call: ToolCall) -> AgentStep:
        decision = self._approve(call)
        if decision is not None:
            return AgentStep(call=call, error=decision, error_kind="denied")
        self._checkpoint(call)
        started = time.perf_counter()
        try:
            result = self.tools.execute(call.name, call.arguments, self.artifacts)
        except Exception as exc:
            if isinstance(exc, UnknownToolError):
                error = self._unknown_tool_message(call.name)
            elif isinstance(exc, ValidationError):
                error = self._validation_message(call, exc)
            else:
                error = f"{type(exc).__name__}: {exc}"
            audit_event(
                self.artifacts,
                "tool",
                tool=call.name,
                outcome="error",
                error_kind=_kind(exc),
                duration_ms=round((time.perf_counter() - started) * 1000, 1),
            )
            return AgentStep(call=call, error=error, error_kind=_kind(exc))
        audit_event(
            self.artifacts,
            "tool",
            tool=call.name,
            outcome="ok",
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return AgentStep(call=call, result=result)

    def _unknown_tool_message(self, name: str) -> str:
        """Name the registered tools and the closest match; nothing is aliased."""
        names = sorted(self.tools.names)
        text = f"Unknown tool '{name}'."
        close = difflib.get_close_matches(name, names, n=1, cutoff=0.6)
        if close:
            text += f" Did you mean '{close[0]}'?"
        return f"{text} Available tools: {', '.join(names)}."

    def _approve(self, call: ToolCall) -> str | None:
        """Return a refusal message, or None when the call may run.

        A read-only session refuses every mutating tool. Otherwise ``deny``
        refuses them, ``allow`` runs them, and ``ask`` delegates to
        ``on_permission``. An ``ask`` session with no approver is treated as
        deny with an explicit message, never as allow.
        """
        if not _is_mutating(call):
            return None
        if self.read_only:
            return f"{call.name} is refused: this session is read-only"
        if self.approval == "allow":
            return None
        if self.approval == "deny":
            return f"{call.name} is refused: approval mode is deny"
        if self.on_permission is None:
            return f"{call.name} needs approval, but no approver is attached"
        request = PermissionRequest(
            id=call.id,
            tool=call.name,
            summary=_summary(call),
            arguments=call.arguments,
            preview=self._preview(call),
        )
        if self.on_permission(request):
            return None
        return f"{call.name} was not approved"

    def _preview(self, call: ToolCall) -> str:
        """Return a short, human-readable preview of what the call would change."""
        if call.name == "bash":
            command = call.arguments.get("command")
            return str(command or "")[:_PREVIEW_LIMIT]
        if call.name == "run_tests":
            return _argv_text(call.arguments)[:_PREVIEW_LIMIT]
        if call.name == "git_commit":
            return str(call.arguments.get("message") or "")[:_PREVIEW_LIMIT]
        if call.name not in MUTATING_TOOLS:
            if _truthy(call.arguments.get("overwrite")):
                target = call.arguments.get("name") or call.arguments.get("path") or "?"
                return f"replaces existing output: {target}"
            if _writes_to_path(call):
                return f"writes outside analysis/: {call.arguments.get('path')}"
        patch = call.arguments.get("patch")
        if isinstance(patch, str):
            return patch[:_PREVIEW_LIMIT]
        path = call.arguments.get("path")
        if not isinstance(path, str) or not path:
            return ""
        try:
            old = self.artifacts.resolve_read(path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError, ArtifactError):
            old = ""
        new = _preview_text(old, call.arguments)
        if new is None:
            return ""
        diff = list(
            difflib.unified_diff(
                old.splitlines(keepends=True),
                new.splitlines(keepends=True),
                fromfile=path,
                tofile=path,
            )
        )
        return "".join(diff)[:_PREVIEW_LIMIT]


def _summary(call: ToolCall) -> str:
    """Return a one-line description of a mutating call for the approver."""
    if call.name == "bash":
        return f"run: {str(call.arguments.get('command') or '')[:200]}"
    if call.name == "run_tests":
        return f"run: {_argv_text(call.arguments)[:200]}"
    if call.name == "git_commit":
        return f"git commit: {str(call.arguments.get('message') or '')[:200]}"
    path = call.arguments.get("path")
    if call.name not in MUTATING_TOOLS:
        if _truthy(call.arguments.get("overwrite")):
            name = call.arguments.get("name")
            return f"{call.name} overwrite=true {name or path or '?'}"[:200]
        if _writes_to_path(call):
            return f"{call.name} write {path}"
    return f"{call.name} {path}" if path else call.name


def _argv_text(arguments: dict[str, Any]) -> str:
    """Return a run_tests argv as one display line."""
    command = arguments.get("command")
    if isinstance(command, list):
        return " ".join(str(part) for part in command)
    return str(command or "")


def _is_mutating(call: ToolCall) -> bool:
    """Return whether a call can change the workspace or repository.

    ``run_tests`` is only mutating with an explicit ``command``, which runs any argv. Any call
    with a truthy ``overwrite``, and ``render_map``/``export_geotiff`` with a non-empty
    ``path``, also count; ``get_satellite_data`` downloads stay ungated.
    """
    if call.name in MUTATING_TOOLS:
        return True
    if _truthy(call.arguments.get("overwrite")):
        return True
    if _writes_to_path(call):
        return True
    return call.name == "run_tests" and bool(call.arguments.get("command"))


def _patch_paths(arguments: dict[str, Any]) -> list[str]:
    """Return the files named by the ``---``/``+++`` headers of an apply_patch call."""
    patch = arguments.get("patch")
    if not isinstance(patch, str):
        return []
    paths: list[str] = []
    for line in patch.splitlines():
        candidate = _patch_line_path(line)
        if candidate and candidate != "/dev/null" and candidate not in paths:
            paths.append(candidate)
    return paths


def _patch_line_path(line: str) -> str:
    """Return the path a patch header or envelope line names, or an empty string."""
    for prefix in _PATCH_ENVELOPE_PREFIXES:
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    if not line.startswith(("+++ ", "--- ")):
        return ""
    candidate = line[4:].split("\t", 1)[0].strip().strip('"')
    if candidate.startswith(("a/", "b/")):
        candidate = candidate[2:]
    return candidate


_PATCH_ENVELOPE_PREFIXES = ("*** Update File:", "*** Add File:")


def _argument_error_step(call: ToolCall, reason: str) -> AgentStep:
    """Record a call whose arguments could not be parsed; it is never executed."""
    return AgentStep(
        call=call,
        error=(
            f"Invalid arguments for {call.name}: {reason} Resend the call with complete, valid "
            "JSON arguments; split a large write into several smaller calls."
        ),
        error_kind="validation",
    )


def _today() -> str:
    """Return the local date as YYYY-MM-DD; tests monkeypatch this."""
    return datetime.date.today().isoformat()


def _clean_path(path: Path) -> str:
    """Return a path safe to place in a prompt: no control characters, capped length."""
    text = "".join("?" if ord(ch) < 32 or ord(ch) == 127 else ch for ch in str(path))
    return text[:_MAX_ENV_PATH_CHARS]


def _preview_text(old: str, arguments: dict[str, Any]) -> str | None:
    """Return the file text this call would produce, or None when unknown."""
    if "content" in arguments and isinstance(arguments["content"], str):
        return arguments["content"]
    edits = arguments.get("edits")
    if isinstance(edits, list):
        text = old
        for edit in edits:
            if not isinstance(edit, dict):
                return None
            find = edit.get("old_string")
            replace = edit.get("new_string")
            if not isinstance(find, str) or not isinstance(replace, str):
                return None
            text = text.replace(find, replace, -1 if edit.get("replace_all") else 1)
        return text
    find = arguments.get("old_string")
    replace = arguments.get("new_string")
    if isinstance(find, str) and isinstance(replace, str):
        return old.replace(find, replace, 1)
    return None


def _kind(exc: Exception) -> ErrorKind:
    if isinstance(exc, UnknownToolError):
        return "unknown_tool"
    if isinstance(exc, ValidationError):
        return "validation"
    return "tool_failure"


@dataclass
class _Digest:
    """What the model still needs from tool exchanges that left the context."""

    dropped: int = 0
    edited: list[str] = field(default_factory=list)
    read: list[str] = field(default_factory=list)
    commands: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    plan: list[str] = field(default_factory=list)


_DIGEST_SECTIONS = (
    ("edited", "Edited:"),
    ("read", "Read:"),
    ("commands", "Commands:"),
    ("errors", "Errors:"),
    ("notes", "Notes:"),
    ("plan", "Plan:"),
)


def _merge_compaction(total: ContextCompaction, change: ContextCompaction) -> None:
    """Fold one compaction pass into the running totals for a turn."""
    total.occurred |= change.occurred
    total.tokens_reclaimed += change.tokens_reclaimed
    total.tool_results_shortened += change.tool_results_shortened
    total.messages_dropped += change.messages_dropped
    total.digest_inserted |= change.digest_inserted
    total.digest_chars = max(total.digest_chars, change.digest_chars)


def _is_digest(message: ChatMessage) -> bool:
    """Return whether ``message`` is the rolling context digest."""
    return message.role is Role.SYSTEM and (message.content or "").startswith(_DIGEST_HEADER)


def _compact(
    messages: list[ChatMessage],
    *,
    protected_user: ChatMessage | None = None,
    root: Path | None = None,
    limit: int | None = None,
) -> ContextCompaction:
    """Shorten observations, then replace dropped tool exchanges with a digest.

    Keep the opening instructions, every system message, the current user
    request, and the last six messages. A tool-call batch intersecting those
    messages is kept whole. Dropped exchanges are folded into one protected
    system digest so later turns keep edited and read files, commands, errors,
    assistant notes, and the plan. Every user request that is not a tool
    observation is protected. ``limit`` is the token trigger; None reads the
    module constant ``MAX_CONTEXT_TOKENS`` at call time. Work toward 80% of it.
    """
    if limit is None:
        limit = MAX_CONTEXT_TOKENS
    sizes = [_message_chars(message) for message in messages]
    total = sum(sizes)
    before = total // CHARS_PER_TOKEN
    result = ContextCompaction()
    if before <= limit:
        return result

    if protected_user is None:
        protected_user = _current_user_message(messages)
    groups = _context_groups(messages)
    prefix_end = next(
        (index for index, message in enumerate(messages) if message.role is Role.ASSISTANT),
        len(messages),
    )
    recent_start = max(0, len(messages) - KEEP_RECENT_MESSAGES)
    digest_index = next(
        (index for index, message in enumerate(messages) if _is_digest(message)), None
    )
    protected = {
        index
        for group in groups
        if any(
            index < prefix_end
            or index >= recent_start
            or messages[index].role is Role.SYSTEM
            or messages[index] is protected_user
            or (messages[index].role is Role.USER and not messages[index].tool_observation)
            for index in group
        )
        for index in group
    }
    if digest_index is not None:
        protected.add(digest_index)
    target = int(limit * _COMPACT_TARGET_FRACTION) * CHARS_PER_TOKEN

    for index, message in enumerate(messages):
        if total <= target:
            break
        if index in protected or message.role is not Role.TOOL:
            continue
        content = _compact_tool_content(message.content or "")
        shorter = message.model_copy(update={"content": content})
        size = _message_chars(shorter)
        if size >= sizes[index]:
            continue
        messages[index] = shorter
        total -= sizes[index] - size
        sizes[index] = size
        result.tool_results_shortened += 1

    dropped_indexes: list[int] = []
    for group in groups:
        if total <= target:
            break
        if any(index in protected for index in group):
            continue
        dropped_indexes.extend(group)
        total -= sum(sizes[index] for index in group)
    if dropped_indexes:
        digest = _collect_digest(messages, dropped_indexes, root)
        if digest_index is not None:
            digest = _merge_digest(_parse_digest(messages[digest_index].content or ""), digest)
        _trim_digest(digest)
        content = _render_digest(digest)
        dropped_ids = {id(messages[index]) for index in dropped_indexes}
        digest_message = ChatMessage(role=Role.SYSTEM, content=content)
        if digest_index is not None:
            messages[digest_index] = digest_message
        else:
            messages.insert(min(dropped_indexes), digest_message)
        messages[:] = [message for message in messages if id(message) not in dropped_ids]
        result.messages_dropped = len(dropped_indexes)
        result.digest_inserted = True
        result.digest_chars = len(content)
    final_total = sum(_message_chars(message) for message in messages)
    result.tokens_reclaimed = before - final_total // CHARS_PER_TOKEN
    result.occurred = bool(result.tool_results_shortened or dropped_indexes)
    return result


def _collect_digest(
    messages: list[ChatMessage], indexes: Sequence[int], root: Path | None
) -> _Digest:
    """Summarise the dropped messages without re-reading their full text."""
    calls = {call.id: call for message in messages for call in message.tool_calls}
    failed_calls = _failed_call_ids(messages)
    digest = _Digest(dropped=len(indexes))
    edited: list[str] = []
    read: list[str] = []
    commands: list[str] = []
    errors: list[str] = []
    notes: list[str] = []
    for index in indexes:
        message = messages[index]
        for tool_call in message.tool_calls:
            if tool_call.name not in _DIGEST_EDIT_TOOLS:
                _add_argument_paths(tool_call.arguments, read)
            elif tool_call.id not in failed_calls:
                _add_argument_paths(tool_call.arguments, edited)
        if message.role is Role.ASSISTANT:
            note = _note_line(message.content)
            if note:
                notes.append(note)
        if message.role is not Role.TOOL:
            continue
        current_call = calls.get(message.tool_call_id or "")
        name = current_call.name if current_call is not None else "tool"
        content = message.content or ""
        try:
            result = ToolResult.model_validate_json(content)
        except ValidationError:
            error = _json_error(content)
            if error is not None:
                kind, detail = error
                errors.append(f"{kind}: {detail}")
                read.extend(_saved_paths(_error_text(content)))
            continue
        read.extend(result.artifacts)
        read.extend(_saved_paths(result.text))
        command = _command_summary(current_call, result.text, name)
        if command:
            commands.append(command)
    if root is not None:
        digest.plan = [_one_line(line) for line in _plan_lines(root)[-_DIGEST_CAPS["plan"] :]]
    edited, read, commands, errors, notes = (
        [_one_line(value) for value in values] for values in (edited, read, commands, errors, notes)
    )
    digest.edited = _unique(edited)[-_DIGEST_CAPS["edited"] :]
    digest.read = [path for path in _unique(read) if path not in digest.edited]
    digest.read = digest.read[-_DIGEST_CAPS["read"] :]
    digest.commands = _unique(commands)[-_DIGEST_CAPS["commands"] :]
    digest.errors = _unique(errors)[-_DIGEST_CAPS["errors"] :]
    digest.notes = _unique(notes)[-_DIGEST_CAPS["notes"] :]
    return digest


def _saved_paths(text: str) -> list[str]:
    """Return the paths of saved full-output copies a tool message points to."""
    return [match.group(1).strip() for match in _SAVED_PATH_RE.finditer(text)]


def _error_text(content: str) -> str:
    """Return the full error string of a JSON error payload, or an empty string."""
    try:
        payload = json.loads(content)
    except ValueError:
        return ""
    error = payload.get("error") if isinstance(payload, dict) else None
    return error if isinstance(error, str) else ""


def _failed_call_ids(messages: Sequence[ChatMessage]) -> set[str]:
    """Return ids of tool calls whose matching tool message is an error payload."""
    return {
        message.tool_call_id
        for message in messages
        if message.role is Role.TOOL
        and message.tool_call_id
        and _json_error(message.content or "") is not None
    }


def _note_line(content: str | None) -> str:
    """Return the first non-empty line of assistant text, capped, or an empty string."""
    for line in (content or "").splitlines():
        if line.strip():
            return line.strip()[:_NOTE_CHARS]
    return ""


def _add_argument_paths(arguments: dict[str, Any], files: list[str]) -> None:
    """Collect file paths mentioned by one tool call."""
    path = arguments.get("path")
    if isinstance(path, str) and path.strip():
        files.append(path.strip())
    files.extend(_patch_paths(arguments))
    targets = arguments.get("files")
    if isinstance(targets, list):
        for target in targets:
            if isinstance(target, dict) and isinstance(target.get("path"), str):
                files.append(target["path"])


def _json_error(content: str) -> tuple[str, str] | None:
    """Return the error kind and first line from a tool error payload."""
    try:
        payload = json.loads(content)
    except ValueError:
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("error"), str):
        return None
    kind = payload.get("error_kind")
    label = kind if isinstance(kind, str) else "error"
    return label, payload["error"].splitlines()[0][:160] if payload["error"] else ""


def _command_summary(call: ToolCall | None, text: str, name: str) -> str:
    """Return ``command -> exit N`` for a tool result that ran something."""
    command = ""
    if call is not None:
        raw = call.arguments.get("command")
        if isinstance(raw, str):
            command = raw
        elif isinstance(raw, list) and all(isinstance(part, str) for part in raw):
            command = " ".join(raw)
    exit_code: str | None = None
    for line in text.splitlines():
        if not command and line.startswith("command: "):
            command = line[len("command: ") :].strip()
        if line.startswith("exit_code: "):
            exit_code = line[len("exit_code: ") :].strip()
            break
    if exit_code is None:
        return ""
    label = (command or name).replace("\n", " ").strip()
    return f"{label[:120]} -> exit {exit_code}"


def _plan_lines(root: Path) -> list[str]:
    """Read the session plan, if one exists, as compact status lines."""
    try:
        data = json.loads((root / ".atlas" / "plan.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    items = data.get("items") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []
    marks = {"completed": "x", "in_progress": "~", "pending": " "}
    lines: list[str] = []
    for item in items[:12]:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        status = item.get("status")
        mark = marks.get(status, "?") if isinstance(status, str) else "?"
        if isinstance(text, str):
            lines.append(f"[{mark}] {text[:80]}")
    return lines


def _unique(values: Sequence[str]) -> list[str]:
    """Return values without duplicates, each at the position of its last occurrence."""
    seen: set[str] = set()
    kept: list[str] = []
    for value in reversed(values):
        if value not in seen:
            seen.add(value)
            kept.append(value)
    kept.reverse()
    return kept


def _one_line(value: str) -> str:
    """Replace every line break (as ``str.splitlines`` sees them) with a space."""
    return " ".join(value.splitlines()) if value else value


def _merge_digest(existing: _Digest, new: _Digest) -> _Digest:
    """Roll a new digest into the existing one; newest entries win the cap.

    The plan is a snapshot: the new plan replaces the old one when it has any.
    """
    merged = _Digest(dropped=existing.dropped + new.dropped)
    for name, limit in _DIGEST_CAPS.items():
        if name == "plan":
            merged.plan = list(new.plan or existing.plan)
            continue
        combined = _unique([*getattr(existing, name), *getattr(new, name)])
        setattr(merged, name, combined[-limit:])
    merged.read = [path for path in merged.read if path not in merged.edited]
    return merged


def _render_digest(digest: _Digest) -> str:
    """Render the digest as one compact system message, one entry per line."""
    lines = [f"{_DIGEST_HEADER}: {digest.dropped} messages compacted"]
    for name, title in _DIGEST_SECTIONS:
        entries: list[str] = getattr(digest, name)
        if entries:
            lines.append(title)
            lines.extend(f"  {entry}" for entry in entries)
    return "\n".join(lines)


def _trim_digest(digest: _Digest) -> None:
    """Drop the oldest entry of the largest section until the message fits its cap."""
    while len(_render_digest(digest)) > _DIGEST_LIMIT_CHARS:
        groups: list[list[str]] = [getattr(digest, name) for name, _ in _DIGEST_SECTIONS]
        largest = max(groups, key=len)
        if not largest:
            return
        largest.pop(0)


def _parse_digest(text: str) -> _Digest:
    """Read a digest already in the history so it can be rolled forward.

    The legacy single line ``Files: a, b`` is read as the Read section.
    """
    digest = _Digest()
    titles = {title: name for name, title in _DIGEST_SECTIONS}
    section: list[str] | None = None
    for line in text.splitlines():
        if line.startswith(_DIGEST_HEADER):
            match = re.search(r"(\d+) messages compacted", line)
            if match is not None:
                digest.dropped = int(match.group(1))
            section = None
        elif line.startswith("Files: "):
            digest.read = [
                part.strip() for part in line[len("Files: ") :].split(",") if part.strip()
            ]
            section = None
        elif line in titles:
            section = getattr(digest, titles[line])
        elif section is not None and line.startswith("  "):
            section.append(line[2:])
    return digest


def _context_groups(messages: list[ChatMessage]) -> list[range]:
    """Group assistant tool calls with all their results, including parallel calls."""
    groups: list[range] = []
    start = 0
    while start < len(messages):
        message = messages[start]
        end = start + 1
        if message.role is Role.ASSISTANT and message.tool_calls:
            pending = {call.id for call in message.tool_calls}
            while end < len(messages) and pending:
                following = messages[end]
                if following.role is Role.TOOL:
                    pending.discard(following.tool_call_id or "")
                end += 1
            if end < len(messages) and messages[end].tool_observation:
                end += 1
        groups.append(range(start, end))
        start = end
    return groups


def _current_user_message(messages: list[ChatMessage]) -> ChatMessage | None:
    return next(
        (m for m in reversed(messages) if m.role is Role.USER and not m.tool_observation), None
    )


def _compact_tool_content(content: str) -> str:
    """Shorten only result text, preserving JSON, artifacts, and error fields."""
    try:
        result = ToolResult.model_validate_json(content)
    except ValidationError:
        try:
            error = json.loads(content)
        except ValueError:
            return _compact_text(content)
        if isinstance(error, dict) and isinstance(error.get("error"), str):
            error["error"] = _compact_text(error["error"])
            return json.dumps(error)
        return content
    return result.model_copy(update={"text": _compact_text(result.text)}).model_dump_json()


# The marker runs to the end of its line, so a saved path containing "]" still matches.
_COMPACT_MARKER_RE = re.compile(r"\n\.\.\.\[\d+ characters compacted[^\n]*\]\.\.\.\n")
_SAVED_POINTER_RE = re.compile(r"full output saved to (.+?): read it with read_file")
_SAVED_PATH_RE = re.compile(r"full output saved to (.+?)(?:: read it with read_file|\]\.\.\.)")


def _compact_text(text: str) -> str:
    """Keep the head and tail of long text, eliding the middle with a marker.

    The marker makes repeated compaction idempotent, so a second pass cannot
    shave the tail that the first pass preserved.
    """
    if len(text) <= COMPACT_PREVIEW_CHARS or _COMPACT_MARKER_RE.search(text):
        return text
    head = COMPACT_PREVIEW_CHARS // 2
    tail = COMPACT_PREVIEW_CHARS - head
    removed = len(text) - head - tail
    saved = _SAVED_POINTER_RE.search(text)
    pointer = f"; full output saved to {saved.group(1)}" if saved is not None else ""
    return f"{text[:head]}\n...[{removed} characters compacted{pointer}]...\n{text[-tail:]}"


def _message_chars(message: ChatMessage) -> int:
    return len(message.model_dump_json(exclude={"images", "tool_observation"}))


def _estimate_tokens(messages: list[ChatMessage]) -> int:
    return sum(_message_chars(message) for message in messages) // CHARS_PER_TOKEN


_SERIALISE_FAILURE = json.dumps(
    {"error": "tool result could not be serialised", "error_kind": "tool_failure"}
)


def _scrub(text: str) -> str:
    """Replace code points that cannot be encoded as UTF-8 (lone surrogates)."""
    return text.encode("utf-8", "replace").decode("utf-8")


def _tool_message(
    step: AgentStep,
    store: LocalArtifactStore | None = None,
    redact: Callable[[str], str] | None = None,
) -> str:
    """Render a step for the model: bounded text, then harness notes. Never raises.

    The stored step is never changed. Oversized text or error keeps its head and tail
    and, given a ``store``, a full copy on disk (passed through ``redact`` first).
    Model-visible text is scrubbed of unencodable code points. Notes come last so
    truncation cannot cut them.
    """
    try:
        return _render_tool_message(step, store, redact)
    except Exception:
        return _SERIALISE_FAILURE


def _render_tool_message(
    step: AgentStep, store: LocalArtifactStore | None, redact: Callable[[str], str] | None
) -> str:
    suffix = "".join(f"\n[note: {_scrub(note)}]" for note in step.notes)
    if suffix:
        suffix = "\n" + suffix
    if step.result is not None:
        text = _bounded_text(_scrub(step.result.text), step.call.id, store, redact) + suffix
        artifacts = [_scrub(path) for path in step.result.artifacts]
        update = {"text": text, "artifacts": artifacts}
        return step.result.model_copy(update=update).model_dump_json()
    error = step.error
    if error is not None:
        error = _bounded_text(_scrub(error), step.call.id, store, redact) + suffix
    return json.dumps({"error": error, "error_kind": step.error_kind})


def _bounded_text(
    text: str,
    call_id: str,
    store: LocalArtifactStore | None,
    redact: Callable[[str], str] | None = None,
) -> str:
    """Keep the head and tail of text over :data:`MAX_TOOL_RESULT_CHARS`."""
    if len(text) <= MAX_TOOL_RESULT_CHARS:
        return text
    head = MAX_TOOL_RESULT_CHARS // 2
    tail = MAX_TOOL_RESULT_CHARS - head
    cut = len(text) - MAX_TOOL_RESULT_CHARS
    saved = _spill_output(store, call_id, text, redact) if store is not None else None
    if saved is None:
        marker = f"...[truncated {cut} characters]..."
    else:
        marker = (
            f"...[truncated {cut} characters; full output saved to {saved}: "
            "read it with read_file (offset/limit) or search it with grep]..."
        )
    return f"{text[:head]}\n{marker}\n{text[-tail:]}"


def _spill_output(
    store: LocalArtifactStore,
    call_id: str,
    text: str,
    redact: Callable[[str], str] | None = None,
) -> str | None:
    """Save the full text of an oversized result; None when the save fails.

    ``redact`` runs over the text before it is written. A failed save never fails the turn.
    """
    if redact is not None:
        try:
            text = redact(text)
        except Exception:
            # Text that could not be redacted is never written.
            return None
    digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:12]
    stem = re.sub(r"[^A-Za-z0-9_-]", "", call_id)[:_MAX_SPILL_ID_CHARS]
    name = f"{stem}-{digest}" if stem else digest
    if store.artifact_root != store.root:
        directory = store.artifact_root / _TOOL_OUTPUT_DIR
    else:
        directory = store.root / ".atlas" / _TOOL_OUTPUT_DIR
    path = directory / f"{name}.txt"
    try:
        directory.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8", errors="replace"))
    except OSError:
        return None
    return store.relative(path)


def _ingest_tool_images(
    step: AgentStep, store: LocalArtifactStore
) -> tuple[list[ModelImage], list[str]]:
    """Read image artifacts for a following user message.

    An over-cap, unreadable, or decompression-bomb image is an error string.
    Other artifacts are left as the tool's text result.
    """
    if step.result is None:
        return [], []
    images: list[ModelImage] = []
    errors: list[str] = []
    for path in step.result.artifacts:
        if not is_model_image(path):
            continue
        try:
            images.append(ingest_image(store, path))
        except (OSError, ValueError) as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    return images, errors
