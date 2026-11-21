"""Headless single-turn runner for harness2-style evaluation.

``run_headless`` replaces the interactive JSON session with one
non-interactive agent turn. It writes a :class:`atlas.agent.trace.TraceRecord`
and returns the :class:`atlas.agent.runtime.AgentRun`, so a benchmark adapter
scores the same record that produced the patch.

The harness override is file-based on purpose: harness2 improves a *tree* of
files and materialises it per rollout. Pass ``harness_dir`` pointing at that
tree and :func:`atlas.harness_eval.substrate.install_harness` maps every
component into the run: the always-on files become the system prompt, skills
and scripts are installed in the workspace, and ``sub_agents`` become named
roles for ``spawn_agent``. With no tree the package prompt stays in force.
"""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import (
    ChatMessage,
    Role,
    ToolCapableModel,
    ToolRegistry,
    default_registry,
)
from atlas.agent.model import ModelConfig, OpenRouterModel
from atlas.agent.runtime import Agent, AgentRun, AgentStep
from atlas.agent.skills import Skill, load_skills, register_skill_tool, render_system_prompt
from atlas.agent.tools._list_tools import ListToolsTool
from atlas.agent.tools._subagent import SubagentTool
from atlas.agent.trace import write_trace
from atlas.harness_eval.substrate import InstalledHarness, SubagentRole, install_harness

TOOL_PROFILES = ("full", "coding", "minimal")

# The "minimal" profile: a shell plus file read and edit, the shape DeepSeek reports
# working best for its V4.1 models (bash + str_replace editor).
MINIMAL_TOOLS = frozenset({"bash", "read_file", "edit_file", "write_file"})

# Tools the "coding" profile withholds: the satellite data tool, the model-as-tool
# plugins (see atlas.models), and web_fetch. The set is explicit so a new tool is
# never dropped by accident; extend it when a remote-sensing tool is added.
NON_CODING_TOOLS = frozenset(
    {
        "get_satellite_data",
        "spectral_indices",
        "rgb_change",
        "burn_scar",
        "flood_mask",
        "segment_landcover",
        "mask_clouds",
        "web_fetch",
    }
)

_ROLE_LINE = "agent:"
_NOTE_CHARS = 4000


@dataclass
class HeadlessRun:
    """One non-interactive run and its on-disk trace.

    ``notes`` maps a tool-call id to the assistant text that preceded it, so
    a rendered trajectory can show why a call was made.
    """

    run: AgentRun
    trace_path: Path
    workspace: Path
    harness: InstalledHarness = field(default_factory=InstalledHarness)
    notes: dict[str, str] = field(default_factory=dict)
    wall_seconds: float = 0.0


class HarnessAgent(Agent):
    """An agent whose base prompt comes from a harness tree, not package data."""

    def __init__(self, *args: Any, base_prompt: str | None = None, **kwargs: Any) -> None:
        """Build the agent; ``base_prompt=None`` keeps the package prompt."""
        super().__init__(*args, **kwargs)
        self.base_prompt = base_prompt

    def system_prompt(self) -> str:
        """Return the harness prompt plus the skill catalog."""
        if self.base_prompt is None:
            return super().system_prompt()
        return render_system_prompt(self.base_prompt, self.skills)


class _RoleRunner:
    """Run a delegated task, applying a named role when the task asks for one.

    A task whose first line is ``agent: <name>`` runs under that role's
    ``AGENT.md`` body. Any other task runs as a plain child.
    """

    def __init__(self, agent: HarnessAgent, roles: Sequence[SubagentRole]) -> None:
        self._agent = agent
        self._roles = {role.name: role for role in roles}

    def run(self, task: str) -> AgentRun:
        first, _, rest = task.lstrip().partition("\n")
        if first.lower().startswith(_ROLE_LINE):
            role = self._roles.get(first[len(_ROLE_LINE) :].strip().strip("`"))
            if role is not None and rest.strip():
                base = self._agent.system_prompt()
                self._agent.base_prompt = f"{base}\n\n## Your role: {role.name}\n\n{role.body}"
                # The catalog is already in ``base``; do not render it twice.
                self._agent.skills = []
                task = rest
        return self._agent.run(task)


def profile_registry(tool_profile: str = "full") -> ToolRegistry:
    """Return the registry for a headless tool profile.

    "full" is :func:`default_registry`. "coding" is the same registry minus
    :data:`NON_CODING_TOOLS` and any tool whose ``domain`` is "satellite". "minimal" keeps only
    :data:`MINIMAL_TOOLS` (also minus satellite-domain tools). Only registered tools ever run; this narrows the
    allow-list and never aliases a name.

    Raises:
        ValueError: ``tool_profile`` is not in :data:`TOOL_PROFILES`.
    """
    if tool_profile not in TOOL_PROFILES:
        raise ValueError(f"unknown tool profile {tool_profile!r}; use one of {TOOL_PROFILES}")
    full = default_registry()
    if tool_profile == "full":
        return full
    registry = ToolRegistry()
    for name in full.names:
        tool = full.get(name)
        if tool is None:
            continue
        if tool_profile == "minimal" and name not in MINIMAL_TOOLS:
            continue
        if tool_profile == "coding" and name in NON_CODING_TOOLS:
            continue
        if tool_profile in {"coding", "minimal"} and getattr(tool, "domain", "") == "satellite":
            continue
        registry.register(tool)
    if tool_profile == "minimal":
        return registry
    # list_tools closes over the full registry; rebind it to the narrowed one.
    registry.replace(ListToolsTool(lambda: registry.definitions))
    return registry


def _build_agent(
    *,
    store: LocalArtifactStore,
    model: ToolCapableModel,
    harness: InstalledHarness,
    max_tool_calls: int | None,
    read_only: bool,
    approval: str,
    tool_profile: str = "full",
) -> HarnessAgent:
    """Assemble an agent the way ``session.main`` does, without stdin/stdout."""
    skills: list[Skill] = load_skills(store)
    registry = profile_registry(tool_profile)
    register_skill_tool(registry, skills)
    agent = HarnessAgent(
        model,
        registry,
        store,
        max_tool_calls=max_tool_calls,
        read_only=read_only,
        approval=approval,
        skills=skills,
        base_prompt=harness.system_prompt,
    )

    def _build_child(
        child_store: LocalArtifactStore, allow_writes: bool, budget: int
    ) -> _RoleRunner:
        # The child registry has no spawn_agent, so delegation cannot recurse.
        child_registry = profile_registry(tool_profile)
        register_skill_tool(child_registry, skills)
        child = HarnessAgent(
            model,
            child_registry,
            child_store,
            max_tool_calls=budget,
            read_only=agent.read_only or not allow_writes,
            approval=agent.approval,
            on_permission=agent.on_permission,
            skills=skills,
            base_prompt=harness.system_prompt,
        )
        return _RoleRunner(child, harness.roles)

    registry.register(SubagentTool(_build_child))
    return agent


def run_headless(
    *,
    workspace: Path,
    prompt: str,
    model_id: str,
    api_key: str,
    harness_dir: Path | None = None,
    artifact_root: Path | None = None,
    trace_path: Path | None = None,
    max_tool_calls: int | None = None,
    reasoning_effort: str | None = None,
    provider_order: tuple[str, ...] = (),
    read_only: bool = False,
    approval: str = "allow",
    model: ToolCapableModel | None = None,
    deadline_seconds: float | None = None,
    on_step: Callable[[AgentStep], None] | None = None,
    tool_profile: str = "full",
    temperature: float | None = None,
    top_p: float | None = None,
) -> HeadlessRun:
    """Run ``prompt`` once in ``workspace`` and persist a trace record.

    Args:
        workspace: Working tree the tools see. Must exist.
        prompt: The user message (task instruction, no grader hints).
        model_id: OpenRouter model id when ``model`` is None.
        api_key: OpenRouter key when ``model`` is None.
        harness_dir: Harness tree to install. ``None`` uses the package prompt
            and installs nothing.
        artifact_root: Where tool outputs land. Defaults to ``workspace``.
        trace_path: Where to write :class:`TraceRecord`. Defaults to
            ``workspace/.atlas/eval-trace.json``.
        max_tool_calls: Tool budget. ``None`` loads ``.atlas/agent.toml``.
        reasoning_effort: OpenRouter reasoning effort for ``model_id``; ``None`` leaves it unset.
        provider_order: OpenRouter providers to pin, in order, with no fallback;
            empty keeps default routing.
        read_only: Refuse mutating tools when true.
        approval: Approval mode (``allow`` for eval; ``ask``/``deny`` keep edits out).
        model: Injected model (tests). When set, ``model_id``/``api_key`` are ignored.
        deadline_seconds: Wall-clock allowance. Once it has passed the turn
            stops at the next round with ``cancelled=True``. It does not
            interrupt a tool call or a model request that is in flight.
        on_step: Called after each tool action, so a caller can keep a
            partial record of a run that is later killed.
        tool_profile: ``full`` keeps every tool; ``coding`` drops the satellite and
            model tools and web_fetch, for the agent and its spawned children.
        temperature: Sampling temperature sent to the provider; ``None`` leaves it unset.
        top_p: Nucleus sampling value; ``None`` leaves it unset.

    Returns:
        The run plus the trace location.

    Raises:
        FileNotFoundError: ``workspace`` does not exist.
        ValueError: ``prompt`` is empty.
    """
    if not workspace.is_dir():
        raise FileNotFoundError(f"workspace not found: {workspace}")
    if not prompt.strip():
        raise ValueError("prompt must be non-empty")
    profile_registry(tool_profile)  # validates the profile before any side effects
    harness = install_harness(harness_dir, workspace)
    store = LocalArtifactStore(workspace, artifact_root=artifact_root or workspace)
    active_model: ToolCapableModel = (
        model
        if model is not None
        else OpenRouterModel(
            ModelConfig(
                model=model_id,
                api_key=api_key,
                reasoning_effort=reasoning_effort,
                provider_order=provider_order,
                temperature=temperature,
                top_p=top_p,
            )
        )
    )
    owns_model = model is None
    started = time.monotonic()
    try:
        agent = _build_agent(
            store=store,
            model=active_model,
            harness=harness,
            max_tool_calls=max_tool_calls,
            read_only=read_only,
            approval=approval,
            tool_profile=tool_profile,
        )
        messages = [*agent.opening_messages(), ChatMessage(role=Role.USER, content=prompt)]
        opening = len(messages)
        out = trace_path or (workspace / ".atlas" / "eval-trace.json")
        try:
            run = agent.advance(
                messages,
                on_step=on_step,
                should_cancel=(
                    None
                    if deadline_seconds is None
                    else lambda: time.monotonic() - started > deadline_seconds
                ),
            )
        except Exception:
            if agent.partial_run is not None:
                write_trace(out, agent.partial_run)
            raise
        write_trace(out, run)
        return HeadlessRun(
            run=run,
            trace_path=out,
            workspace=workspace,
            harness=harness,
            notes=_assistant_notes(messages[opening:]),
            wall_seconds=round(time.monotonic() - started, 1),
        )
    finally:
        close = getattr(active_model, "close", None)
        if owns_model and callable(close):
            close()


def _assistant_notes(messages: Sequence[ChatMessage]) -> dict[str, str]:
    """Map the first tool-call id of each assistant message to its text.

    A reasoning model often leaves ``content`` empty beside a tool call, so
    its reasoning stands in, capped so one long thought cannot crowd a
    trajectory.
    """
    notes: dict[str, str] = {}
    for message in messages:
        if message.role is not Role.ASSISTANT or not message.tool_calls:
            continue
        text = (message.content or message.reasoning or "").strip()
        if text:
            notes[message.tool_calls[0].id] = text[:_NOTE_CHARS]
    return notes


def collect_patch(workspace: Path) -> str:
    """Return ``git diff`` for a workspace, or empty when not a git tree."""
    if not (workspace / ".git").exists():
        return ""
    proc = subprocess.run(
        ["git", "-C", str(workspace), "diff", "--no-color"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return proc.stdout


def render_step(step: AgentStep, *, obs_trunc: int = 50_000) -> str:
    """Render one tool action and its observation or error."""
    call = step.call
    head_line = f"[ACTION] {call.name}({call.arguments})"
    if step.result is None:
        return f"{head_line}\n\n[ERROR] {step.error or 'failed'}"
    text = step.result.text
    if len(text) > obs_trunc:
        head = obs_trunc // 2
        tail = obs_trunc - head
        text = f"{text[:head]}\n...[{len(text) - head - tail} chars truncated]...\n{text[-tail:]}"
    return f"{head_line}\n\n[OBSERVATION]\n{text}"


def render_trajectory(
    run: AgentRun, *, obs_trunc: int = 50_000, notes: dict[str, str] | None = None
) -> str:
    """Render an :class:`AgentRun` into harness2's shared grammar.

    Each step becomes ``[ACTION] <tool>(<args>)`` then the tool text as
    ``[OBSERVATION]``, truncated head-and-tail at ``obs_trunc`` chars so a
    long run does not flood the improver. Errors use ``[ERROR]``. ``notes``
    (tool-call id to assistant text) adds an ``[ASSISTANT]`` block before the
    call it explains. The final response is appended as ``[FINAL]``.
    """
    parts: list[str] = []
    for step in run.steps:
        note = (notes or {}).get(step.call.id)
        if note:
            parts.append(f"[ASSISTANT]\n{note}")
        parts.append(render_step(step, obs_trunc=obs_trunc))
    if run.response.strip():
        parts.append(f"[FINAL]\n{run.response.strip()}")
    for stopped, reason in (
        (run.stopped_for_limit, "stopped: tool-call budget reached"),
        (run.stopped_for_budget, "stopped: token or spend budget reached"),
        (run.stopped_for_context, "stopped: context limit reached"),
        (run.cancelled, "stopped: wall-clock limit reached"),
    ):
        if stopped:
            parts.append(f"[ERROR] {reason}")
    return "\n\n".join(parts) + "\n"
