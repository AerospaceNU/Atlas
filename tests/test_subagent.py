from __future__ import annotations

from pathlib import Path
from typing import Any

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import (
    ChatMessage,
    ModelResponse,
    ToolCall,
    ToolDefinition,
    ToolRegistry,
)
from atlas.agent.runtime import Agent
from atlas.agent.tools._subagent import SubagentTool
from atlas.agent.tools.read_file import ReadFileTool
from atlas.agent.tools.write_file import WriteTool


class _ScriptedModel:
    """Answer from a shared queue so parent and child calls interleave."""

    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = responses
        self.prompts: list[str] = []

    def complete(self, messages: list[ChatMessage], tools: list[ToolDefinition]) -> ModelResponse:
        last = next((m.content for m in reversed(messages) if m.role == "user" and m.content), "")
        self.prompts.append(last or "")
        return self.responses.pop(0)


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(ReadFileTool())
    registry.register(WriteTool())
    return registry


def test_subagent_returns_its_answer_and_keeps_parent_context_small(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("answer is 42\n", encoding="utf-8")
    model = _ScriptedModel(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="s1",
                        name="spawn_agent",
                        arguments={"task": "read notes.txt and report the number"},
                    )
                ]
            ),
            ModelResponse(content="answer is 42"),
            ModelResponse(content="the subagent says 42"),
        ]
    )
    child_registry = _registry()

    def build_child(store: LocalArtifactStore, allow_writes: bool, budget: int) -> Agent:
        return Agent(
            model,
            child_registry,
            store,
            max_tool_calls=budget,
            read_only=not allow_writes,
        )

    parent_registry = _registry()
    parent_registry.register(SubagentTool(build_child))
    agent = Agent(model, parent_registry, LocalArtifactStore(tmp_path))

    run = agent.run("find the number")

    assert run.response == "the subagent says 42"
    assert "answer is 42" in (run.steps[0].result.text if run.steps[0].result else "")
    assert "subagent" in (run.steps[0].result.text if run.steps[0].result else "")
    # The parent never saw the child's tool round; only the summary came back.
    assert len(run.steps) == 1
    assert model.prompts[1] == "read notes.txt and report the number"


def test_child_cannot_spawn_another_child(tmp_path: Path) -> None:
    model = _ScriptedModel(
        [
            ModelResponse(
                tool_calls=[ToolCall(id="s1", name="spawn_agent", arguments={"task": "recurse"})]
            ),
            ModelResponse(
                tool_calls=[ToolCall(id="s2", name="spawn_agent", arguments={"task": "again"})]
            ),
            ModelResponse(content="stopped"),
            ModelResponse(content="parent done"),
        ]
    )

    def build_child(store: LocalArtifactStore, allow_writes: bool, budget: int) -> Agent:
        return Agent(model, _registry(), store, max_tool_calls=budget)

    registry = _registry()
    registry.register(SubagentTool(build_child))
    agent = Agent(model, registry, LocalArtifactStore(tmp_path))

    run = agent.run("delegate")

    child_report = run.steps[0].result.text if run.steps[0].result else ""
    assert "stopped" in child_report
    assert "spawn_agent" not in [d.name for d in _registry().definitions]
    assert run.response == "parent done"


def test_child_writes_are_refused_by_default(tmp_path: Path) -> None:
    model = _ScriptedModel(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(id="s1", name="spawn_agent", arguments={"task": "create a file"})
                ]
            ),
            ModelResponse(
                tool_calls=[
                    ToolCall(id="w", name="write_file", arguments={"path": "x.py", "content": "y"})
                ]
            ),
            ModelResponse(content="could not write"),
            ModelResponse(content="done"),
        ]
    )

    def build_child(store: LocalArtifactStore, allow_writes: bool, budget: int) -> Agent:
        return Agent(model, _registry(), store, max_tool_calls=budget, read_only=not allow_writes)

    registry = _registry()
    registry.register(SubagentTool(build_child))
    agent = Agent(model, registry, LocalArtifactStore(tmp_path))

    agent.run("delegate a write")

    assert not (tmp_path / "x.py").exists()


def test_child_artifacts_live_under_subagents(tmp_path: Path) -> None:
    seen: list[Path] = []

    def build_child(store: LocalArtifactStore, allow_writes: bool, budget: int) -> Any:
        seen.append(store.artifact_root)
        return Agent(_ScriptedModel([ModelResponse(content="ok")]), _registry(), store)

    registry = _registry()
    registry.register(SubagentTool(build_child))
    parent_store = LocalArtifactStore(
        tmp_path, artifact_root=tmp_path / ".atlas" / "artifacts" / "s"
    )
    agent = Agent(
        _ScriptedModel(
            [
                ModelResponse(
                    tool_calls=[ToolCall(id="s1", name="spawn_agent", arguments={"task": "x"})]
                ),
                ModelResponse(content="done"),
            ]
        ),
        registry,
        parent_store,
    )

    agent.run("go")

    assert seen and seen[0].parent.name == "subagents"


class _Run:
    def __init__(self, **flags: bool) -> None:
        self.steps: list[Any] = []
        self.response = "partial answer"
        for name, value in flags.items():
            setattr(self, name, value)


class _FakeAgent:
    def __init__(self, run: Any) -> None:
        self._run = run

    def run(self, task: str) -> Any:
        return self._run


def _header(tmp_path: Path, run: Any) -> str:
    tool = SubagentTool(lambda *_args: _FakeAgent(run))
    return tool.run({"task": "x"}, LocalArtifactStore(tmp_path)).text.split("\n", 1)[0]


def test_allow_writes_description_names_real_tools() -> None:
    from atlas.agent.tools._subagent import SubagentInput

    description = SubagentInput.model_fields["allow_writes"].description or ""
    assert "apply_patch and bash" in description
    assert "bash_tool" not in description


def test_report_header_marks_early_stops(tmp_path: Path) -> None:
    assert _header(tmp_path, _Run(stopped_for_limit=True)).endswith(
        "writes=refused; stopped: tool-call budget reached):"
    )
    assert "; stopped: token or spend budget reached" in _header(
        tmp_path, _Run(stopped_for_budget=True)
    )
    assert "; stopped: context limit" in _header(tmp_path, _Run(stopped_for_context=True))


def test_report_header_is_unchanged_without_stop_flags(tmp_path: Path) -> None:
    header = _header(tmp_path, _Run())

    assert "stopped" not in header
    assert header.endswith("(0 tool calls, writes=refused):")
