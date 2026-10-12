from __future__ import annotations

import json
import logging
import secrets
import tomllib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from atlas.agent.artifacts import LocalArtifactStore, SandboxDenied, record_denial
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
from atlas.agent.metrics import observe_model_call, tokens_per_second
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

ErrorKind = Literal["validation", "unknown_tool", "tool_failure", "limit"]


class AgentStep(BaseModel):
    """One tool action recorded in an autonomous run."""

    call: ToolCall
    result: ToolResult | None = None
    error: str | None = None
    error_kind: ErrorKind | None = None


class AgentRun(BaseModel):
    """Final response plus a transparent record of every tool action."""

    response: str
    steps: list[AgentStep] = Field(default_factory=list)
    stopped_for_limit: bool = False
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
        skills: Sequence[Skill] | None = None,
    ) -> None:
        if max_tool_calls is None:
            max_tool_calls = load_max_tool_calls(artifacts.root)
        elif isinstance(max_tool_calls, bool) or max_tool_calls < 1:
            raise ValueError("max_tool_calls must be an integer >= 1")
        self.model = model
        self.tools = tools
        self.artifacts = artifacts
        self.max_tool_calls = max_tool_calls
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
        ``tools`` or either sandbox root.

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

        ``images`` are paths in the write sandbox or the project workspace.
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
    ) -> AgentRun:
        """Continue ``messages`` in place while the tool-call budget lasts.

        The caller owns the history: no user or system message is appended here.
        Every tool call that fits in the budget runs in order, even after one
        fails. A call past the budget is recorded as a ``limit`` step and still
        gets a tool message so every ``tool_call_id`` is answered.
        """
        steps: list[AgentStep] = []
        calls: list[ModelCall] = []
        executed = 0

        def emit(phase: str, name: str | None = None) -> None:
            if on_phase is not None:
                on_phase(phase, name)

        while executed < self.max_tool_calls:
            # Reasoning stays on unless the model config turns it off. The
            # request is one blocking call, so this is the whole wait, not a
            # separate stream of thought tokens.
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
            # The call is blocking, so the thought arrives with the completion,
            # ahead of this round's tool steps and ahead of the final answer.
            if response.reasoning and on_thought is not None:
                speed = None
                if response.timing is not None:
                    speed = tokens_per_second(
                        response.usage.completion_tokens,
                        response.timing.elapsed_seconds,
                    )
                on_thought(response.reasoning, speed)
            if not response.tool_calls:
                return AgentRun(response=response.content or "", steps=steps, calls=calls)
            hit_limit = False
            attached: list[ModelImage] = []
            image_errors: list[str] = []
            for call in response.tool_calls:
                emit("tool_calling", call.name)
                if executed < self.max_tool_calls:
                    emit("tool_executing", call.name)
                    step = self._execute(call)
                    executed += 1
                else:
                    hit_limit = True
                    step = AgentStep(call=call, error="tool call limit reached", error_kind="limit")
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
            if attached or image_errors:
                messages.append(
                    ChatMessage(
                        role=Role.USER,
                        content="\n".join(image_errors) if image_errors else None,
                        images=attached,
                    )
                )
            if executed >= self.max_tool_calls or hit_limit:
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

    def _execute(self, call: ToolCall) -> AgentStep:
        try:
            result = self.tools.execute(call.name, call.arguments, self.artifacts)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, SandboxDenied):
                try:
                    record_denial(self.artifacts, call.name, error)
                except Exception:
                    _LOGGER.exception("Could not record a sandbox denial for %s", call.name)
            return AgentStep(call=call, error=error, error_kind=_kind(exc))
        return AgentStep(call=call, result=result)


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
