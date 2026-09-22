from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, Field

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import (
    ChatMessage,
    ModelResponse,
    Role,
    Tool,
    ToolCall,
    ToolDefinition,
    ToolRegistry,
    ToolResult,
)
from atlas.agent.runtime import (
    DEFAULT_MAX_TOOL_CALLS,
    DEFAULT_MODEL,
    Agent,
    ensure_agent_config,
    load_max_tool_calls,
    load_model,
)
from atlas.agent.session import AgentSession, main, resolve_model, serve


class ScriptedModel:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = responses
        self.requests: list[tuple[list[ChatMessage], list[ToolDefinition]]] = []

    def complete(self, messages: list[ChatMessage], tools: list[ToolDefinition]) -> ModelResponse:
        self.requests.append((list(messages), list(tools)))
        return self.responses.pop(0)


class EchoInput(BaseModel):
    value: int = Field(ge=0)


class EchoTool(Tool):
    name = "echo"

    description = "Echo a non-negative integer."

    @property
    def input_model(self) -> type[BaseModel]:
        return EchoInput

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        request = EchoInput.model_validate(arguments)
        return ToolResult(text=f"echo:{request.value}")


class BoomTool(Tool):
    name = "boom"

    description = "Always raises."

    @property
    def input_model(self) -> type[BaseModel]:
        return EchoInput

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        raise RuntimeError("tool exploded")


def _session(model: ScriptedModel, *, max_tool_calls: int = 256, tmp_path: Path) -> AgentSession:
    agent = Agent(
        model, ToolRegistry(), LocalArtifactStore(tmp_path), max_tool_calls=max_tool_calls
    )
    return AgentSession(agent)


def test_second_turn_sees_full_history(tmp_path: Path) -> None:
    model = ScriptedModel(
        [
            ModelResponse(content="first answer"),
            ModelResponse(content="second answer"),
        ]
    )
    session = _session(model, tmp_path=tmp_path)

    session.turn("first question")
    session.turn("second question")

    messages = model.requests[1][0]
    roles = [message.role for message in messages]
    assert roles == ["system", "user", "assistant", "user"]
    assert messages[0].content is not None
    assert messages[0].content.startswith("You are Atlas")
    assert messages[1].content == "first question"
    assert messages[2].content == "first answer"
    assert messages[3].content == "second question"
    assert roles.count("system") == 1


def test_stopped_for_limit_and_resume_budget(tmp_path: Path) -> None:
    model = ScriptedModel(
        [
            ModelResponse(tool_calls=[ToolCall(id="a", name="echo", arguments={"value": 1})]),
            ModelResponse(content="finished"),
        ]
    )
    session = _session(model, max_tool_calls=1, tmp_path=tmp_path)

    first = session.turn("please use a tool")

    assert first.stopped_for_limit is True
    assert session.stopped_for_limit is True

    resumed = session.resume()

    assert resumed.stopped_for_limit is False
    assert resumed.response == "finished"
    assert session.stopped_for_limit is False
    assert len(model.requests) == 2


def test_resume_before_limit_raises(tmp_path: Path) -> None:
    model = ScriptedModel([ModelResponse(content="done")])
    session = _session(model, tmp_path=tmp_path)
    session.turn("hello")

    with pytest.raises(RuntimeError, match="not stopped"):
        session.resume()


def test_batch_beyond_the_budget_records_the_rest_as_limit(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(id="1", name="echo", arguments={"value": 1}),
                    ToolCall(id="2", name="echo", arguments={"value": 2}),
                ]
            ),
        ]
    )
    agent = Agent(model, registry, LocalArtifactStore(tmp_path), max_tool_calls=1)
    messages = [
        ChatMessage(role=Role.SYSTEM, content="system"),
        ChatMessage(role=Role.USER, content="echo twice"),
    ]

    result = agent.advance(messages)

    assert result.stopped_for_limit is True
    assert len(model.requests) == 1
    assert [step.error_kind for step in result.steps] == [None, "limit"]
    assert result.steps[0].result is not None
    assert result.steps[0].result.text == "echo:1"
    assert result.steps[1].result is None
    assert result.steps[1].error == "tool call limit reached"
    tool_messages = [message for message in messages if message.role == Role.TOOL]
    assert [message.tool_call_id for message in tool_messages] == ["1", "2"]


def test_missing_config_uses_the_default_limit(tmp_path: Path) -> None:
    assert load_max_tool_calls(tmp_path) == 256
    agent = Agent(ScriptedModel([]), ToolRegistry(), LocalArtifactStore(tmp_path))
    assert agent.max_tool_calls == 256


def test_workspace_config_sets_the_limit(tmp_path: Path) -> None:
    config = tmp_path / ".atlas"
    config.mkdir()
    (config / "agent.toml").write_text("max_tool_calls = 4\n", encoding="utf-8")

    agent = Agent(ScriptedModel([]), ToolRegistry(), LocalArtifactStore(tmp_path))

    assert agent.max_tool_calls == 4


def test_explicit_limit_overrides_the_config(tmp_path: Path) -> None:
    config = tmp_path / ".atlas"
    config.mkdir()
    (config / "agent.toml").write_text("max_tool_calls = 4\n", encoding="utf-8")

    agent = Agent(ScriptedModel([]), ToolRegistry(), LocalArtifactStore(tmp_path), max_tool_calls=9)

    assert agent.max_tool_calls == 9


@pytest.mark.parametrize("content", ["max_tool_calls = 0", "max_tool_calls = -3"])
def test_config_rejects_values_below_one(tmp_path: Path, content: str) -> None:
    config = tmp_path / ".atlas"
    config.mkdir()
    (config / "agent.toml").write_text(content + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="integer >= 1"):
        load_max_tool_calls(tmp_path)


def test_config_rejects_a_bool(tmp_path: Path) -> None:
    config = tmp_path / ".atlas"
    config.mkdir()
    (config / "agent.toml").write_text("max_tool_calls = true\n", encoding="utf-8")

    with pytest.raises(ValueError, match="integer >= 1"):
        load_max_tool_calls(tmp_path)


def test_config_rejects_a_float(tmp_path: Path) -> None:
    config = tmp_path / ".atlas"
    config.mkdir()
    (config / "agent.toml").write_text("max_tool_calls = 1.5\n", encoding="utf-8")

    with pytest.raises(ValueError, match="integer >= 1"):
        load_max_tool_calls(tmp_path)


def test_config_rejects_a_string(tmp_path: Path) -> None:
    config = tmp_path / ".atlas"
    config.mkdir()
    (config / "agent.toml").write_text('max_tool_calls = "many"\n', encoding="utf-8")

    with pytest.raises(ValueError, match="integer >= 1"):
        load_max_tool_calls(tmp_path)


def test_config_rejects_malformed_toml(tmp_path: Path) -> None:
    config = tmp_path / ".atlas"
    config.mkdir()
    (config / "agent.toml").write_text("max_tool_calls = \n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_max_tool_calls(tmp_path)


def test_ensure_agent_config_writes_defaults_once(tmp_path: Path) -> None:
    path = ensure_agent_config(tmp_path)

    text = path.read_text(encoding="utf-8")
    assert f'model = "{DEFAULT_MODEL}"' in text
    assert f"max_tool_calls = {DEFAULT_MAX_TOOL_CALLS}" in text
    assert load_model(tmp_path) == DEFAULT_MODEL
    assert load_max_tool_calls(tmp_path) == DEFAULT_MAX_TOOL_CALLS

    path.write_text('model = "kept/model"\n', encoding="utf-8")
    assert ensure_agent_config(tmp_path) == path
    assert path.read_text(encoding="utf-8") == 'model = "kept/model"\n'


def test_missing_config_uses_the_default_model(tmp_path: Path) -> None:
    assert load_model(tmp_path) == DEFAULT_MODEL
    assert DEFAULT_MODEL == "google/gemini-3.8-flash"


def test_workspace_config_sets_the_model(tmp_path: Path) -> None:
    config = tmp_path / ".atlas"
    config.mkdir()
    (config / "agent.toml").write_text(
        'model = "google/gemini-3.8-flash"\nmax_tool_calls = 4\n',
        encoding="utf-8",
    )

    assert load_model(tmp_path) == "google/gemini-3.8-flash"
    assert load_max_tool_calls(tmp_path) == 4


@pytest.mark.parametrize(
    "content",
    ["model = 1", "model = true", 'model = ""', 'model = "   "'],
)
def test_config_rejects_a_bad_model(tmp_path: Path, content: str) -> None:
    config = tmp_path / ".atlas"
    config.mkdir()
    (config / "agent.toml").write_text(content + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="non-empty string"):
        load_model(tmp_path)


def test_resolve_model_prefers_cli_then_env_then_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / ".atlas"
    config.mkdir()
    (config / "agent.toml").write_text('model = "from/toml"\n', encoding="utf-8")
    monkeypatch.setenv("ATLAS_MODEL", "from/env")

    assert resolve_model(tmp_path, "from/cli") == "from/cli"
    assert resolve_model(tmp_path, None) == "from/env"
    monkeypatch.delenv("ATLAS_MODEL")
    assert resolve_model(tmp_path, None) == "from/toml"


def test_agent_rejects_a_zero_limit(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="integer >= 1"):
        Agent(ScriptedModel([]), ToolRegistry(), LocalArtifactStore(tmp_path), max_tool_calls=0)


def test_tool_failures_get_error_kinds(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())
    registry.register(BoomTool())
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(id="1", name="echo", arguments={"value": -1}),
                    ToolCall(id="2", name="boom", arguments={"value": 1}),
                    ToolCall(id="3", name="missing_tool", arguments={}),
                    ToolCall(id="4", name="echo", arguments={"value": 7}),
                ]
            ),
            ModelResponse(content="done"),
        ]
    )
    agent = Agent(model, registry, LocalArtifactStore(tmp_path))

    result = agent.run("do things")

    kinds = [step.error_kind for step in result.steps]
    assert kinds == ["validation", "tool_failure", "unknown_tool", None]
    assert result.steps[0].error is not None
    assert result.steps[0].error.startswith("ValidationError")
    assert result.steps[1].error == "RuntimeError: tool exploded"
    assert result.steps[2].error == "KeyError: 'Unknown tool: missing_tool'"
    assert result.steps[3].result is not None
    assert result.steps[3].result.text == "echo:7"


class _Recorder:
    def __init__(self) -> None:
        self.events: list[str] = []

    def __call__(self, step: Any) -> None:
        self.events.append(step.call.name)


def test_on_step_called_once_per_step(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())
    model = ScriptedModel(
        [
            ModelResponse(tool_calls=[ToolCall(id="1", name="echo", arguments={"value": 3})]),
            ModelResponse(content="done"),
        ]
    )
    agent = Agent(model, registry, LocalArtifactStore(tmp_path))
    recorder = _Recorder()

    agent.advance(
        [
            ChatMessage(role="system", content="system"),
            ChatMessage(role="user", content="hi"),
        ],
        on_step=recorder,
    )

    assert recorder.events == ["echo"]


def _run_protocol(
    lines: list[str], session: AgentSession, *, redact: str | None = None
) -> list[dict[str, Any]]:
    stdin = io.StringIO("".join(f"{line}\n" for line in lines))
    stdout = io.StringIO()
    code = serve(session, stdin, stdout, redact=redact)
    assert code == 0
    return [json.loads(line) for line in stdout.getvalue().splitlines()]


def test_protocol_ready_step_done_and_second_turn(tmp_path: Path) -> None:
    model = ScriptedModel(
        [
            ModelResponse(tool_calls=[ToolCall(id="1", name="echo", arguments={"value": 4})]),
            ModelResponse(content="first done"),
            ModelResponse(content="second done"),
        ]
    )
    session = _session(model, tmp_path=tmp_path)
    session.agent.tools.register(EchoTool())

    events = _run_protocol(
        ['{"type":"user","text":"one"}', '{"type":"user","text":"two"}'], session
    )

    assert events[0]["type"] == "ready"
    assert any(tool["name"] == "echo" for tool in events[0]["tools"])
    assert events[1]["type"] == "step"
    assert events[1]["name"] == "echo"
    assert events[1]["ok"] is True
    assert events[1]["text"] == "echo:4"
    assert events[2]["type"] == "done"
    assert events[2]["response"] == "first done"
    assert events[3]["type"] == "done"
    assert events[3]["response"] == "second done"


def test_protocol_invalid_json_continues(tmp_path: Path) -> None:
    model = ScriptedModel([ModelResponse(content="ok")])
    session = _session(model, tmp_path=tmp_path)

    events = _run_protocol(
        ["not json", '{"type":"user","text":"hi"}', '{"type":"quit"}'],
        session,
    )

    assert events[1]["type"] == "error"
    assert events[2]["type"] == "done"
    assert events[0]["type"] == "ready"


def test_protocol_unknown_type_is_recoverable(tmp_path: Path) -> None:
    model = ScriptedModel([])
    session = _session(model, tmp_path=tmp_path)

    events = _run_protocol(['{"type":"nope"}'], session)

    assert events[1]["type"] == "error"
    assert "Unknown message type" in events[1]["message"]


def test_protocol_resume_before_limit_is_error(tmp_path: Path) -> None:
    model = ScriptedModel([ModelResponse(content="ok")])
    session = _session(model, tmp_path=tmp_path)

    events = _run_protocol(['{"type":"user","text":"hi"}', '{"type":"resume"}'], session)

    assert events[1]["type"] == "done"
    assert events[2]["type"] == "error"
    assert "not stopped" in events[2]["message"]


def test_protocol_redacts_api_key(tmp_path: Path) -> None:
    api_key = "sk-or-v1-secret-value"
    model = ScriptedModel([ModelResponse(content=f"the key is {api_key}")])
    session = _session(model, tmp_path=tmp_path)

    events = _run_protocol(['{"type":"user","text":"hi"}'], session, redact=api_key)

    assert api_key not in json.dumps(events)
    assert "***" in events[1]["response"]


def test_main_without_key_returns_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)

    code = main(["--workspace", str(workspace)])

    captured = capsys.readouterr()
    assert code == 1
    assert (workspace / ".atlas" / "agent.toml").is_file()
    assert load_model(workspace) == DEFAULT_MODEL
    events = [json.loads(line) for line in captured.out.splitlines()]
    assert events[0]["type"] == "error"
    assert events[0]["message"] == "OPENROUTER_API_KEY is not set"
    assert all(event["type"] != "ready" for event in events)
