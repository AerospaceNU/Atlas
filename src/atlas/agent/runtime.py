from __future__ import annotations

import difflib
import json
import logging
import secrets
import time
import tomllib
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
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

_STOPPED_RESPONSE = "Stopped after reaching the configured tool-use limit."
_STOPPED_BUDGET = "Stopped after reaching the configured token or spend budget."
_STOPPED_CANCELLED = "Stopped at your request."
_PREVIEW_LIMIT = 4000

# Tool calls in one assistant turn are independent, so they run concurrently.
# The cap keeps a model that emits twenty calls from opening twenty shells.
_MAX_PARALLEL_TOOLS = 4

# Tools that change the workspace. A read-only session refuses them, and each
# one is checkpointed before it runs so /undo can put the file back.
MUTATING_TOOLS = frozenset({"bash_tool", "edit_file", "patch_file", "write_file"})

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


class AgentRun(BaseModel):
    """Final response plus a transparent record of every tool action."""

    response: str
    steps: list[AgentStep] = Field(default_factory=list)
    stopped_for_limit: bool = False
    stopped_for_budget: bool = False
    cancelled: bool = False
    calls: list[ModelCall] = Field(default_factory=list)


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

        def emit(phase: str, name: str | None = None) -> None:
            if on_phase is not None:
                on_phase(phase, name)

        while executed < self.max_tool_calls:
            if should_cancel is not None and should_cancel():
                return AgentRun(
                    response=_STOPPED_CANCELLED, steps=steps, cancelled=True, calls=calls
                )
            config = getattr(self.model, "config", None)
            reasoning = True if config is None else bool(getattr(config, "reasoning", True))
            emit("thinking" if reasoning else "working")
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
                return AgentRun(response=response.content or "", steps=steps, calls=calls)
            if should_cancel is not None and should_cancel():
                return self._stop_cancelled(messages, steps, calls, response.tool_calls)
            if self._over_budget(calls):
                return AgentRun(
                    response=_STOPPED_BUDGET,
                    steps=steps,
                    stopped_for_budget=True,
                    calls=calls,
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
                    )
                )
            if executed >= self.max_tool_calls or limited:
                return AgentRun(
                    response=_STOPPED_RESPONSE,
                    steps=steps,
                    stopped_for_limit=True,
                    calls=calls,
                )
        return AgentRun(
            response=_STOPPED_RESPONSE,
            steps=steps,
            stopped_for_limit=True,
            calls=calls,
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
        for call in pending:
            step = AgentStep(call=call, error="cancelled by the user", error_kind="cancelled")
            steps.append(step)
            messages.append(
                ChatMessage(
                    role=Role.TOOL,
                    tool_call_id=call.id,
                    content=_tool_message(step),
                )
            )
        return AgentRun(response=_STOPPED_CANCELLED, steps=steps, cancelled=True, calls=calls)

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


def _tool_message(step: AgentStep) -> str:
    if step.result is not None:
        return step.result.model_dump_json()
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
