from __future__ import annotations

import difflib
import json
import logging
import re
import secrets
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
    Role,
    ToolCall,
    ToolCapableModel,
    ToolRegistry,
    ToolResult,
)
from atlas.agent.images import ingest_image, ingest_images, is_model_image
from atlas.agent.instructions import load_project_instructions
from atlas.agent.layout import project_agent_config_path, project_atlas_root
from atlas.agent.metrics import observe_model_call, output_tokens_per_second
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
_DIGEST_LIMIT_CHARS = 2000

# Tool calls in one assistant turn are independent, so they run concurrently.
# The cap keeps a model that emits twenty calls from opening twenty shells.
_MAX_PARALLEL_TOOLS = 4

# Tools that change the workspace. A read-only session refuses them, and each
# one is checkpointed before it runs so /undo can put the file back.
MUTATING_TOOLS = frozenset({"apply_patch", "bash_tool", "edit_file", "patch_file", "write_file"})

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

    def system_prompt(self) -> str:
        """Return the package system prompt, plus the skill catalog when skills are loaded."""
        return render_system_prompt(load_system_prompt(), self.skills)

    def opening_messages(self) -> list[ChatMessage]:
        """Return the trusted system prompt and any project instructions.

        The first message is :meth:`system_prompt` and nothing else. When the
        project root has a readable instruction file, the next message is that
        file as delimited project guidance with role ``user``. The wrapper tags
        use :attr:`instruction_nonce`. A missing or unreadable file leaves the
        list as the system message alone. Loading the file does not change
        ``tools`` or the store roots.

        Returns:
            The messages that start a run or a session.
        """
        messages = [ChatMessage(role=Role.SYSTEM, content=self.system_prompt())]
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
        """
        steps: list[AgentStep] = []
        calls: list[ModelCall] = []
        executed = 0
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
                compaction.occurred |= change.occurred
                compaction.tokens_reclaimed += change.tokens_reclaimed
                compaction.tool_results_shortened += change.tool_results_shortened
                compaction.messages_dropped += change.messages_dropped
                compaction.digest_inserted |= change.digest_inserted
                compaction.digest_chars = max(compaction.digest_chars, change.digest_chars)
                if _estimate_tokens(messages) > MAX_CONTEXT_TOKENS:
                    return AgentRun(
                        response=_STOPPED_CONTEXT,
                        steps=steps,
                        calls=calls,
                        stopped_for_context=True,
                        compaction=compaction,
                    )
            response = self.model.complete(messages, self.tools.definitions)
            observed = observe_model_call(response)
            if observed is not None:
                calls.append(observed)
            messages.append(
                ChatMessage(
                    role=Role.ASSISTANT,
                    content=response.content,
                    tool_calls=response.tool_calls,
                    reasoning=response.reasoning,
                    reasoning_details=response.reasoning_details,
                )
            )
            if response.reasoning and on_thought is not None:
                speed = output_tokens_per_second(
                    response.usage.completion_tokens,
                    response.timing,
                )
                on_thought(response.reasoning, speed)
            if not response.tool_calls:
                return AgentRun(
                    response=response.content or "", steps=steps, calls=calls, compaction=compaction
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

            for call in (*pending, *limited):
                emit("tool_calling", call.name)
            results = self._execute_all(pending, on_phase=emit) if pending else []

            attached: list[ModelImage] = []
            image_errors: list[str] = []
            for call, step in zip(pending, results, strict=True):
                images, errors = _ingest_tool_images(step, self.artifacts)
                if errors and step.error is None:
                    step.error = "\n".join(errors)
                    step.error_kind = "tool_failure"
                steps.append(step)
                messages.append(
                    ChatMessage(
                        role=Role.TOOL,
                        tool_call_id=call.id,
                        content=_tool_message(step),
                    )
                )
                attached.extend(images)
                image_errors.extend(errors)
                if on_step is not None:
                    on_step(step)
            for call in limited:
                step = AgentStep(call=call, error="tool call limit reached", error_kind="limit")
                steps.append(step)
                messages.append(
                    ChatMessage(
                        role=Role.TOOL,
                        tool_call_id=call.id,
                        content=_tool_message(step),
                    )
                )
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
        serial = any(call.name in MUTATING_TOOLS for call in calls) or len(calls) <= 1
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
            messages.append(
                ChatMessage(
                    role=Role.TOOL,
                    tool_call_id=call.id,
                    content=_tool_message(step),
                )
            )

    def _checkpoint(self, call: ToolCall) -> None:
        """Snapshot a file a mutating tool is about to change."""
        if call.name not in MUTATING_TOOLS:
            return
        path = call.arguments.get("path")
        if not isinstance(path, str) or not path:
            return
        try:
            snapshot(self.artifacts, path)
        except Exception:
            _LOGGER.exception("Could not checkpoint %s before %s", path, call.name)

    def _execute(self, call: ToolCall) -> AgentStep:
        decision = self._approve(call)
        if decision is not None:
            return AgentStep(call=call, error=decision, error_kind="denied")
        self._checkpoint(call)
        started = time.perf_counter()
        try:
            result = self.tools.execute(call.name, call.arguments, self.artifacts)
        except Exception as exc:
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

    def _approve(self, call: ToolCall) -> str | None:
        """Return a refusal message, or None when the call may run.

        A read-only session refuses every mutating tool. Otherwise ``deny``
        refuses them, ``allow`` runs them, and ``ask`` delegates to
        ``on_permission``. An ``ask`` session with no approver is treated as
        deny with an explicit message, never as allow.
        """
        if call.name not in MUTATING_TOOLS:
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
        if call.name == "bash_tool":
            command = call.arguments.get("command")
            return str(command or "")[:_PREVIEW_LIMIT]
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
    if call.name == "bash_tool":
        return f"run: {str(call.arguments.get('command') or '')[:200]}"
    path = call.arguments.get("path")
    return f"{call.name} {path}" if path else call.name


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
    if isinstance(exc, KeyError):
        return "unknown_tool"
    if isinstance(exc, ValidationError):
        return "validation"
    return "tool_failure"


@dataclass
class _Digest:
    """What the model still needs from tool exchanges that left the context."""

    dropped: int = 0
    files: list[str] = field(default_factory=list)
    commands: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    plan: list[str] = field(default_factory=list)


def _is_digest(message: ChatMessage) -> bool:
    """Return whether ``message`` is the rolling context digest."""
    return message.role is Role.SYSTEM and (message.content or "").startswith(_DIGEST_HEADER)


def _compact(
    messages: list[ChatMessage],
    *,
    protected_user: ChatMessage | None = None,
    root: Path | None = None,
) -> ContextCompaction:
    """Shorten observations, then replace dropped tool exchanges with a digest.

    Keep the opening instructions, every system message, the current user
    request, and the last six messages. A tool-call batch intersecting those
    messages is kept whole. Dropped exchanges are folded into one protected
    system digest so later turns keep the files, commands, errors, and plan.
    Work toward 80% of the trigger to leave headroom.
    """
    sizes = [_message_chars(message) for message in messages]
    total = sum(sizes)
    before = total // CHARS_PER_TOKEN
    result = ContextCompaction()
    if before <= MAX_CONTEXT_TOKENS:
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
            for index in group
        )
        for index in group
    }
    if digest_index is not None:
        protected.add(digest_index)
    target = int(MAX_CONTEXT_TOKENS * _COMPACT_TARGET_FRACTION) * CHARS_PER_TOKEN

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
    digest = _Digest(dropped=len(indexes))
    files: list[str] = []
    commands: list[str] = []
    errors: list[str] = []
    for index in indexes:
        message = messages[index]
        for tool_call in message.tool_calls:
            _add_argument_paths(tool_call.arguments, files)
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
            continue
        for artifact in result.artifacts:
            files.append(artifact)
        command = _command_summary(current_call, result.text, name)
        if command:
            commands.append(command)
    if root is not None:
        digest.plan = _plan_lines(root)
    digest.files = _unique(files)[:24]
    digest.commands = _unique(commands)[-12:]
    digest.errors = _unique(errors)[-12:]
    return digest


def _add_argument_paths(arguments: dict[str, Any], files: list[str]) -> None:
    """Collect file paths mentioned by one tool call."""
    path = arguments.get("path")
    if isinstance(path, str) and path.strip():
        files.append(path.strip())
    patch = arguments.get("patch")
    if isinstance(patch, str):
        for line in patch.splitlines():
            if not line.startswith("+++ "):
                continue
            candidate = line[4:].split("\t", 1)[0].strip().strip('"')
            if candidate.startswith(("a/", "b/")):
                candidate = candidate[2:]
            if candidate and candidate != "/dev/null":
                files.append(candidate)
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
    """Return values in first-seen order without duplicates."""
    seen: list[str] = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return seen


def _merge_digest(existing: _Digest, new: _Digest) -> _Digest:
    """Roll a new digest into the one already in the history."""
    merged = _Digest(dropped=existing.dropped + new.dropped)
    limits = {"files": 24, "commands": 12, "errors": 12, "plan": 12}
    for name, limit in limits.items():
        combined = _unique([*getattr(existing, name), *getattr(new, name)])
        setattr(merged, name, combined[-limit:])
    return merged


def _render_digest(digest: _Digest) -> str:
    """Render the digest as a compact system message."""
    lines = [f"{_DIGEST_HEADER}: {digest.dropped} messages compacted"]
    if digest.files:
        lines.append("Files: " + ", ".join(digest.files))
    if digest.commands:
        lines.append("Commands:")
        lines.extend(f"  {command}" for command in digest.commands)
    if digest.errors:
        lines.append("Errors:")
        lines.extend(f"  {error}" for error in digest.errors)
    if digest.plan:
        lines.append("Plan:")
        lines.extend(f"  {item}" for item in digest.plan)
    return "\n".join(lines)


def _trim_digest(digest: _Digest) -> None:
    """Drop digest entries until the rendered message fits its hard cap."""
    while len(_render_digest(digest)) > _DIGEST_LIMIT_CHARS:
        groups = [digest.files, digest.commands, digest.errors, digest.plan]
        largest = max(groups, key=len)
        if not largest:
            return
        largest.pop()


def _parse_digest(text: str) -> _Digest:
    """Read a digest already in the history so it can be rolled forward."""
    digest = _Digest()
    section: list[str] | None = None
    for line in text.splitlines():
        if line.startswith(_DIGEST_HEADER):
            match = re.search(r"(\d+) messages compacted", line)
            if match is not None:
                digest.dropped = int(match.group(1))
        elif line.startswith("Files: "):
            digest.files = [
                part.strip() for part in line[len("Files: ") :].split(",") if part.strip()
            ]
        elif line == "Commands:":
            section = digest.commands
        elif line == "Errors:":
            section = digest.errors
        elif line == "Plan:":
            section = digest.plan
        elif section is not None and line.startswith("  "):
            section.append(line.strip())
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


_COMPACT_MARKER_RE = re.compile(r"\n\.\.\.\[\d+ characters compacted\]\.\.\.\n")


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
    return f"{text[:head]}\n...[{removed} characters compacted]...\n{text[-tail:]}"


def _message_chars(message: ChatMessage) -> int:
    return len(message.model_dump_json(exclude={"images", "tool_observation"}))


def _estimate_tokens(messages: list[ChatMessage]) -> int:
    return sum(_message_chars(message) for message in messages) // CHARS_PER_TOKEN


def _tool_message(step: AgentStep) -> str:
    if step.result is not None:
        result = step.result
        if len(result.text) > MAX_TOOL_RESULT_CHARS:
            cut = len(result.text) - MAX_TOOL_RESULT_CHARS
            marker = f"...[truncated {cut} characters]"
            result = result.model_copy(
                update={"text": result.text[:MAX_TOOL_RESULT_CHARS] + marker}
            )
        return result.model_dump_json()
    return json.dumps({"error": step.error, "error_kind": step.error_kind})


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
