from __future__ import annotations

import io
import json
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, Field

from atlas.agent import session as session_module
from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import (
    CallTiming,
    CatalogModel,
    ChatMessage,
    ModelResponse,
    Role,
    TokenUsage,
    Tool,
    ToolCall,
    ToolDefinition,
    ToolRegistry,
    ToolResult,
)
from atlas.agent.layout import (
    ensure_project_layout,
    ensure_user_layout,
    resolve_openrouter_key,
    write_user_key,
)
from atlas.agent.model import (
    MISSING_KEY_MESSAGE,
    OPENROUTER_MODELS_URL,
    ModelConfig,
    OpenRouterHTTPError,
    OpenRouterModel,
)
from atlas.agent.runtime import (
    DEFAULT_MAX_TOOL_CALLS,
    DEFAULT_MODEL,
    Agent,
    ensure_agent_config,
    load_max_tool_calls,
    load_model,
)
from atlas.agent.session import AgentSession, main, resolve_model, serve, session_goal_text


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


def test_start_fresh_clears_history_for_the_next_turn(tmp_path: Path) -> None:
    model = ScriptedModel(
        [
            ModelResponse(content="first answer"),
            ModelResponse(content="second answer"),
        ]
    )
    session = _session(model, tmp_path=tmp_path)

    session.turn("first question")
    session.start_fresh()
    session.turn("second question")

    messages = model.requests[1][0]
    assert [message.role for message in messages] == ["system", "user"]
    assert messages[0].content is not None
    assert messages[0].content.startswith("You are Atlas")
    assert messages[1].content == "second question"
    assert session.stopped_for_limit is False


def test_start_fresh_keeps_model_and_zeroes_totals(tmp_path: Path) -> None:
    model = ScriptedModel([ModelResponse(content="hi")])
    session = _session(model, tmp_path=tmp_path)
    session.model_id = "keep/model"
    session.totals.model = "keep/model"
    session.totals.context_limit = 111
    session.totals.total_tokens = 42
    session.totals.cost = 0.05

    events = _run_protocol(['{"type":"user","text":"hi"}', '{"type":"new"}'], session)

    ready = events[-1]
    assert ready["type"] == "ready"
    assert ready["total_tokens"] == 0
    assert ready["prompt_tokens"] == 0
    assert ready["completion_tokens"] == 0
    assert ready["spend"] == 0
    assert ready["estimated"] is False
    assert "spend $0" in ready["status_line"]
    assert "spend unknown" not in ready["status_line"]
    assert ready["tokens_per_second"] is None
    assert ready["recent"] == []
    assert ready["model"] == "keep/model"
    assert ready["context_limit"] == 111
    assert session.model_id == "keep/model"


def test_start_fresh_moves_writes_but_keeps_reads(tmp_path: Path) -> None:
    model = ScriptedModel([ModelResponse(content="ok")])
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    read_root = tmp_path / "source"
    read_root.mkdir()
    agent = Agent(
        model,
        ToolRegistry(),
        LocalArtifactStore(tmp_path / "artifacts", read_root=read_root),
    )
    session = AgentSession(agent)
    session.workspace = workspace

    previous = session.agent.artifacts.root
    session.start_fresh()

    artifacts = session.agent.artifacts
    assert artifacts.root != previous
    assert artifacts.root.parent == (workspace / ".atlas" / "artifacts").resolve()
    assert artifacts.read_root == read_root


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
    assert DEFAULT_MODEL == "deepseek/deepseek-v4.1-flash"


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
    steps = [event for event in events if event["type"] == "step"]
    dones = [event for event in events if event["type"] == "done"]
    phases = [event["phase"] for event in events if event["type"] == "phase"]
    assert steps[0]["name"] == "echo"
    assert steps[0]["ok"] is True
    assert steps[0]["text"] == "echo:4"
    assert dones[0]["response"] == "first done"
    assert dones[1]["response"] == "second done"
    assert phases[:4] == ["thinking", "tool_calling", "tool_executing", "thinking"]
    assert events[events.index(steps[0]) - 1]["phase"] == "tool_executing"


def test_protocol_emits_thoughts_before_steps_and_the_answer(tmp_path: Path) -> None:
    model = ScriptedModel(
        [
            ModelResponse(
                reasoning="check the file",
                usage=TokenUsage(completion_tokens=8),
                timing=CallTiming(
                    http_status=200,
                    elapsed_seconds=0.5,
                    time_to_first_token_seconds=0.1,
                ),
                tool_calls=[ToolCall(id="1", name="echo", arguments={"value": 4})],
            ),
            ModelResponse(reasoning="that was enough", content="done"),
        ]
    )
    session = _session(model, tmp_path=tmp_path)
    session.agent.tools.register(EchoTool())

    events = _run_protocol(['{"type":"user","text":"one"}'], session)

    thoughts = [event for event in events if event["type"] == "thought"]
    steps = [event for event in events if event["type"] == "step"]
    dones = [event for event in events if event["type"] == "done"]
    assert [event["text"] for event in thoughts] == ["check the file", "that was enough"]
    assert thoughts[0]["tokens_per_second"] == 20.0
    assert "tokens_per_second" not in thoughts[1]
    assert events.index(thoughts[0]) < events.index(steps[0])
    assert events.index(steps[0]) < events.index(thoughts[1])
    assert events.index(thoughts[1]) < events.index(dones[0])
    assert dones[0]["response"] == "done"
    assistant = next(message for message in model.requests[1][0] if message.role == "assistant")
    assert assistant.reasoning == "check the file"


def test_protocol_invalid_json_continues(tmp_path: Path) -> None:
    model = ScriptedModel([ModelResponse(content="ok")])
    session = _session(model, tmp_path=tmp_path)

    events = _run_protocol(
        ["not json", '{"type":"user","text":"hi"}', '{"type":"quit"}'],
        session,
    )

    assert events[0]["type"] == "ready"
    assert events[1]["type"] == "error"
    assert any(event["type"] == "done" for event in events)


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

    dones = [event for event in events if event["type"] == "done"]
    errors = [event for event in events if event["type"] == "error"]
    assert dones[0]["response"] == "ok"
    assert "not stopped" in errors[0]["message"]


def test_protocol_redacts_api_key(tmp_path: Path) -> None:
    api_key = "sk-or-v1-secret-value"
    model = ScriptedModel([ModelResponse(content=f"the key is {api_key}")])
    session = _session(model, tmp_path=tmp_path)

    events = _run_protocol(['{"type":"user","text":"hi"}'], session, redact=api_key)

    assert api_key not in json.dumps(events)
    done = next(event for event in events if event["type"] == "done")
    assert "***" in done["response"]


def test_main_without_key_prompts_instead_of_a_stack_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr("atlas.agent.session.Path.home", lambda: home)
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO('{"type":"user","text":"hi"}\n{"type":"quit"}\n'),
    )

    code = main(["--workspace", str(workspace)])

    captured = capsys.readouterr()
    assert code == 0
    assert (workspace / ".atlas" / "agent.toml").is_file()
    assert load_model(workspace) == DEFAULT_MODEL
    assert "Traceback" not in captured.out
    assert "httpx" not in captured.out
    events = [json.loads(line) for line in captured.out.splitlines()]
    assert events[0]["type"] == "ready"
    assert events[0]["key_set"] is False
    assert "context 0/unknown" in events[0]["status_line"]
    assert events[1]["type"] == "error"
    assert events[1]["message"] == MISSING_KEY_MESSAGE


class _QueueResponse:
    def __init__(
        self,
        payload: dict[str, Any],
        *,
        status_code: int,
        elapsed_seconds: float,
        time_to_first_token_seconds: float | None = None,
    ) -> None:
        self.payload = payload
        self.status_code = status_code
        self.elapsed_seconds = elapsed_seconds
        self.time_to_first_token_seconds = time_to_first_token_seconds

    def json(self) -> dict[str, Any]:
        return self.payload


class _QueueClient:
    def __init__(
        self,
        payloads: list[dict[str, Any]],
        *,
        status_code: int = 200,
        elapsed_seconds: float = 0.5,
        time_to_first_token_seconds: float | None = None,
    ) -> None:
        self.payloads = list(payloads)
        self.status_code = status_code
        self.elapsed_seconds = elapsed_seconds
        self.time_to_first_token_seconds = time_to_first_token_seconds

    def post(self, _url: str, **_kwargs: Any) -> _QueueResponse:
        return _QueueResponse(
            self.payloads.pop(0),
            status_code=self.status_code,
            elapsed_seconds=self.elapsed_seconds,
            time_to_first_token_seconds=self.time_to_first_token_seconds,
        )

    def close(self) -> None:
        return None


def test_turn_accumulates_usage_cost_status_and_speed(tmp_path: Path) -> None:
    client = _QueueClient(
        [
            {
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                    "cost": 0.01,
                },
                "choices": [{"message": {"content": "one", "tool_calls": []}}],
            },
            {
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 6,
                    "total_tokens": 26,
                    "cost": 0.03,
                },
                "choices": [{"message": {"content": "two", "tool_calls": []}}],
            },
        ],
        time_to_first_token_seconds=0.1,
    )
    model = OpenRouterModel(ModelConfig(model="example/model", api_key="test-key"), client)
    session = AgentSession(Agent(model, ToolRegistry(), LocalArtifactStore(tmp_path)))

    events = _run_protocol(
        ['{"type":"user","text":"one"}', '{"type":"user","text":"two"}'],
        session,
        redact="test-key",
    )

    done = [event for event in events if event["type"] == "done"]
    assert done[0]["prompt_tokens"] == 10
    assert done[0]["completion_tokens"] == 2
    assert done[0]["total_tokens"] == 12
    assert done[0]["spend"] == 0.01
    assert done[0]["cached_tokens"] == 0
    assert done[0]["reasoning_tokens"] == 0
    assert done[0]["tokens_per_second"] == 5.0
    assert "last 5.0 tok/s" in done[0]["status_line"]
    assert done[0]["http_status"] == 200
    assert done[0]["elapsed_seconds"] == 0.5
    assert done[0]["time_to_first_token_seconds"] == 0.1
    assert done[0]["recent"] == ["200 0.50s"]
    assert "http" not in done[0]["status_line"]
    assert "latency" not in done[0]["status_line"]
    assert "ttft" not in done[0]["status_line"]
    assert "recent" not in done[0]["status_line"]
    assert done[0]["context_used"] == 10
    assert done[0]["context_limit"] is None
    assert "context 10/unknown" in done[0]["status_line"]
    assert done[1]["prompt_tokens"] == 30
    assert done[1]["completion_tokens"] == 8
    assert done[1]["total_tokens"] == 38
    assert done[1]["spend"] == pytest.approx(0.04)
    assert done[1]["context_used"] == 20
    assert done[1]["tokens_per_second"] == 15.0
    line = done[1]["status_line"]
    assert "tokens 38 (prompt 30 completion 8)" in line
    assert "last 15.0 tok/s" in line
    assert "http" not in line
    assert "latency" not in line
    assert done[1]["recent"] == ["200 0.50s", "200 0.50s"]
    assert "ttft" not in line
    assert "recent" not in line
    assert "test-key" not in json.dumps(events)


def test_http_error_keeps_status_and_short_body(tmp_path: Path) -> None:
    client = _QueueClient(
        [{"error": {"message": "slow down"}}],
        status_code=429,
        elapsed_seconds=0.25,
    )
    model = OpenRouterModel(ModelConfig(model="example/model", api_key="test-key"), client)
    session = AgentSession(Agent(model, ToolRegistry(), LocalArtifactStore(tmp_path)))

    events = _run_protocol(['{"type":"user","text":"hi"}'], session, redact="test-key")

    error = next(event for event in events if event["type"] == "error")
    assert error["http_status"] == 429
    assert error["elapsed_seconds"] == 0.25
    assert error["error_body"] == "slow down"
    assert "http" not in error["status_line"]
    assert "latency" not in error["status_line"]
    assert any(event.get("phase") == "thinking" for event in events)
    dumped = json.dumps(events)
    assert "Traceback" not in dumped
    assert "test-key" not in dumped


def test_model_fetch_failure_keeps_the_selected_model(tmp_path: Path) -> None:
    session = _session(ScriptedModel([]), tmp_path=tmp_path)
    session.model_id = "keep-me"
    session.totals.model = "keep-me"
    session.totals.context_limit = 111

    def fail() -> list[CatalogModel]:
        raise OpenRouterHTTPError(500, "down", 0.3)

    session.fetch_models = fail
    events = _run_protocol(['{"type":"models"}'], session)

    assert session.model_id == "keep-me"
    assert session.totals.context_limit == 111
    assert events[1]["type"] == "models"
    assert events[1]["ok"] is False
    assert "500" in events[1]["message"]
    assert "down" in events[1]["message"]


def test_select_model_updates_context_limit_and_rejects_unknown(tmp_path: Path) -> None:
    class Configurable:
        def __init__(self) -> None:
            self.config = ModelConfig(model="a", api_key="test-key")

        def complete(
            self, messages: list[ChatMessage], tools: list[ToolDefinition]
        ) -> ModelResponse:
            return ModelResponse(content=self.config.model)

    model = Configurable()
    session = AgentSession(Agent(model, ToolRegistry(), LocalArtifactStore(tmp_path)))
    session.catalog = [
        CatalogModel(id="a", context_length=111, prompt_price="0.1", completion_price="0.2"),
        CatalogModel(id="b", context_length=222, prompt_price="0.3", completion_price="0.4"),
    ]
    session.model_id = "a"
    session.totals.model = "a"
    session.totals.context_limit = 111
    session.totals.context_used = 10

    missing = _run_protocol(['{"type":"select_model","id":"missing"}'], session)
    assert session.model_id == "a"
    assert session.totals.context_limit == 111
    assert model.config.model == "a"
    assert missing[1]["type"] == "error"

    selected = _run_protocol(['{"type":"select_model","id":"b"}'], session)
    assert session.model_id == "b"
    assert model.config.model == "b"
    assert session.totals.context_limit == 222
    assert session.totals.context_used == 10
    assert selected[1]["type"] == "model"
    assert selected[1]["context_length"] == 222
    assert "context 10/222" in selected[1]["status_line"]


def test_catalog_fills_the_active_model_context_on_the_dashboard(tmp_path: Path) -> None:
    session = _session(ScriptedModel([]), tmp_path=tmp_path)
    session.model_id = "google/gemini-3.8-flash"
    session.totals.model = "google/gemini-3.8-flash"
    session.totals.context_used = 355
    session.totals.prompt_tokens = 355
    session.totals.completion_tokens = 25
    session.totals.total_tokens = 380
    session.fetch_models = lambda: [
        CatalogModel(id="google/gemini-3.8-flash", context_length=1_048_576),
        CatalogModel(id="other/model", context_length=8_192),
    ]

    events = _run_protocol(['{"type":"models"}'], session)

    assert session.totals.context_limit == 1_048_576
    line = events[1]["status_line"]
    assert "context 355/1.05M" in line
    assert "tokens 380 (prompt 355 completion 25)" in line


def test_session_key_overrides_the_user_key_and_is_not_stored_in_the_project(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project"
    project.mkdir()

    class Configurable:
        def __init__(self) -> None:
            self.config = ModelConfig(model="example/model", api_key="")

        def complete(
            self, messages: list[ChatMessage], tools: list[ToolDefinition]
        ) -> ModelResponse:
            return ModelResponse(content="ok")

    model = Configurable()
    session = AgentSession(Agent(model, ToolRegistry(), LocalArtifactStore(project)))
    session.home = home
    session.workspace = project
    session.requires_api_key = True
    events = _run_protocol(
        [
            '{"type":"set_key","scope":"user","key":"user-key"}',
            '{"type":"set_key","scope":"session","key":"session-key"}',
            '{"type":"user","text":"hi"}',
        ],
        session,
    )

    assert model.config.api_key == "session-key"
    assert (
        resolve_openrouter_key(session_key=session.session_key, env_key="env-key", home=home)
        == "session-key"
    )
    assert resolve_openrouter_key(session_key=None, env_key="env-key", home=home) == "env-key"
    assert resolve_openrouter_key(session_key="  ", env_key=None, home=home) == "user-key"
    assert resolve_openrouter_key(session_key=None, env_key=None, home=home) == "user-key"
    dumped = json.dumps(events)
    assert "user-key" not in dumped
    assert "session-key" not in dumped
    assert events[1]["type"] == "key"
    assert events[1]["scope"] == "user"
    assert events[2]["scope"] == "session"
    for path in (project / ".atlas").rglob("*"):
        if path.is_file():
            assert "user-key" not in path.read_text(encoding="utf-8")
            assert "session-key" not in path.read_text(encoding="utf-8")
    user_key = home / ".atlas" / "keys" / "openrouter_api_key"
    assert user_key.is_file()
    assert user_key.read_text(encoding="utf-8").strip() == "user-key"


class _HeaderResponse:
    def __init__(
        self,
        payload: dict[str, Any],
        *,
        status_code: int,
        elapsed_seconds: float,
        time_to_first_token_seconds: float | None = None,
    ) -> None:
        self.payload = payload
        self.status_code = status_code
        self.elapsed_seconds = elapsed_seconds
        self.time_to_first_token_seconds = time_to_first_token_seconds

    def json(self) -> dict[str, Any]:
        return self.payload


class _HeaderClient:
    """Records OpenRouter headers and returns one canned catalog or completion."""

    def __init__(
        self,
        payload: dict[str, Any],
        *,
        status_code: int = 200,
        elapsed_seconds: float = 0.0,
    ) -> None:
        self.payload = payload
        self.status_code = status_code
        self.elapsed_seconds = elapsed_seconds
        self.gets: list[dict[str, Any]] = []
        self.posts: list[dict[str, Any]] = []

    def get(self, url: str, **kwargs: Any) -> _HeaderResponse:
        self.gets.append({"url": url, **kwargs})
        return _HeaderResponse(
            self.payload,
            status_code=self.status_code,
            elapsed_seconds=self.elapsed_seconds,
        )

    def post(self, url: str, **kwargs: Any) -> _HeaderResponse:
        self.posts.append({"url": url, **kwargs})
        return _HeaderResponse(
            {
                "usage": {
                    "prompt_tokens": 4,
                    "completion_tokens": 2,
                    "total_tokens": 6,
                    "cost": 0.01,
                },
                "choices": [{"message": {"content": "ok", "tool_calls": []}}],
            },
            status_code=200,
            elapsed_seconds=0.5,
            time_to_first_token_seconds=0.1,
        )

    def close(self) -> None:
        return None


def test_catalog_fetch_failure_uses_openrouter_and_keeps_the_model(tmp_path: Path) -> None:
    client = _HeaderClient(
        {"error": {"message": "down"}},
        status_code=500,
        elapsed_seconds=0.3,
    )
    model = OpenRouterModel(ModelConfig(model="keep-me", api_key="catalog-key"), client)
    session = AgentSession(Agent(model, ToolRegistry(), LocalArtifactStore(tmp_path)))
    session.model_id = "keep-me"
    session.totals.model = "keep-me"
    session.totals.context_limit = 111
    session.fetch_models = model.list_models

    events = _run_protocol(['{"type":"models"}'], session)

    assert session.model_id == "keep-me"
    assert model.config.model == "keep-me"
    assert session.totals.context_limit == 111
    assert events[1]["type"] == "models"
    assert events[1]["ok"] is False
    assert "500" in events[1]["message"]
    assert "down" in events[1]["message"]
    assert client.gets[0]["url"] == OPENROUTER_MODELS_URL
    assert client.gets[0]["headers"]["Authorization"] == "Bearer catalog-key"
    assert client.posts == []
    assert "catalog-key" not in json.dumps(events)


def test_models_command_returns_the_live_catalog_without_changing_the_model(
    tmp_path: Path,
) -> None:
    client = _HeaderClient(
        {
            "data": [
                {
                    "id": "google/gemini-3.8-flash",
                    "context_length": 128000,
                    "pricing": {"prompt": "0.1", "completion": "0.4"},
                    "supported_parameters": ["tools"],
                },
                {"id": "  "},
                {"name": "skipped"},
            ]
        }
    )
    model = OpenRouterModel(ModelConfig(model="keep-me", api_key="catalog-key"), client)
    session = AgentSession(Agent(model, ToolRegistry(), LocalArtifactStore(tmp_path)))
    session.model_id = "keep-me"
    session.totals.model = "keep-me"
    session.totals.context_limit = 111
    session.fetch_models = model.list_models

    events = _run_protocol(['{"type":"models"}'], session)

    assert session.model_id == "keep-me"
    assert model.config.model == "keep-me"
    assert client.gets[0]["url"] == OPENROUTER_MODELS_URL
    assert client.gets[0]["headers"]["Authorization"] == "Bearer catalog-key"
    assert client.gets[0]["follow_redirects"] is False
    assert client.posts == []
    catalog = events[1]
    assert catalog["type"] == "models"
    assert catalog["ok"] is True
    assert catalog["models"] == [
        {
            "id": "google/gemini-3.8-flash",
            "name": None,
            "description": None,
            "context_length": 128000,
            "prompt_price": "0.1",
            "completion_price": "0.4",
            "cache_read_price": None,
            "supports_tools": True,
        },
        {
            "id": "keep-me",
            "name": None,
            "description": None,
            "context_length": None,
            "prompt_price": None,
            "completion_price": None,
            "cache_read_price": None,
            "supports_tools": True,
        },
    ]
    assert session.totals.context_limit == 111
    assert "catalog-key" not in json.dumps(events)


def test_later_page_failure_keeps_rows_in_the_session(tmp_path: Path) -> None:
    first = OPENROUTER_MODELS_URL
    second = "https://openrouter.ai/api/v1/models?offset=1"

    class Pages:
        def __init__(self) -> None:
            self.urls: list[str] = []

        def get(self, url: str, **kwargs: Any) -> _HeaderResponse:
            self.urls.append(url)
            assert kwargs["follow_redirects"] is False
            if url == first:
                return _HeaderResponse(
                    {
                        "data": [
                            {
                                "id": "alpha/one",
                                "name": "Alpha\x1b[31m!\x1b[0m",
                                "description": "coast\x00",
                                "supported_parameters": ["tools"],
                            }
                        ],
                        "links": {"next": "/api/v1/models?offset=1"},
                    },
                    status_code=200,
                    elapsed_seconds=0.1,
                )
            return _HeaderResponse(
                {"error": {"message": "down"}},
                status_code=500,
                elapsed_seconds=0.2,
            )

        def close(self) -> None:
            return None

    client = Pages()
    model = OpenRouterModel(ModelConfig(model="keep-me", api_key="catalog-key"), client)
    session = AgentSession(Agent(model, ToolRegistry(), LocalArtifactStore(tmp_path)))
    session.model_id = "keep-me"
    session.totals.model = "keep-me"
    session.fetch_models = model.list_models

    events = _run_protocol(['{"type":"models"}'], session)

    assert client.urls == [first, second]
    assert session.model_id == "keep-me"
    assert [item.id for item in session.catalog] == ["alpha/one", "keep-me"]
    assert session.catalog[0].name == "Alpha!"
    catalog = events[1]
    assert catalog["ok"] is False
    assert "500" in catalog["message"]
    assert catalog["models"][0]["id"] == "alpha/one"
    assert catalog["models"][0]["name"] == "Alpha!"
    assert catalog["models"][1]["id"] == "keep-me"
    assert "catalog-key" not in json.dumps(events)


def test_session_catalog_marks_non_tool_models_and_keeps_the_selected_one(
    tmp_path: Path,
) -> None:
    class Client:
        def get(self, url: str, **kwargs: Any) -> _HeaderResponse:
            assert url == OPENROUTER_MODELS_URL
            assert kwargs["follow_redirects"] is False
            return _HeaderResponse(
                {
                    "data": [
                        {
                            "id": "plain/text",
                            "name": "Plain",
                            "supported_parameters": ["temperature"],
                        },
                        {"id": "other/model", "supported_parameters": ["tools"]},
                    ]
                },
                status_code=200,
                elapsed_seconds=0.1,
            )

        def close(self) -> None:
            return None

    model = OpenRouterModel(ModelConfig(model="keep-me", api_key="catalog-key"), Client())
    session = AgentSession(Agent(model, ToolRegistry(), LocalArtifactStore(tmp_path)))
    session.model_id = "keep-me"
    session.totals.model = "keep-me"
    session.catalog = [
        CatalogModel(id="keep-me", name="Current", context_length=111, supports_tools=True)
    ]
    session.fetch_models = model.list_models

    events = _run_protocol(['{"type":"models"}'], session)

    assert session.model_id == "keep-me"
    rows = events[1]["models"]
    assert [(row["id"], row["supports_tools"]) for row in rows] == [
        ("plain/text", False),
        ("other/model", True),
        ("keep-me", True),
    ]
    assert rows[0]["name"] == "Plain"
    assert rows[2]["name"] == "Current"
    assert rows[2]["context_length"] == 111
    assert [item.id for item in session.catalog] == ["plain/text", "other/model", "keep-me"]
    assert "catalog-key" not in json.dumps(events)


class _HeldCatalogLock:
    """Sets an event before acquiring the real catalog lock."""

    def __init__(self, inner: threading.Lock, waiting: threading.Event) -> None:
        self._inner = inner
        self._waiting = waiting

    def __enter__(self) -> _HeldCatalogLock:
        self._waiting.set()
        self._inner.acquire()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._inner.release()


def test_catalog_publish_waits_for_the_catalog_lock(tmp_path: Path) -> None:
    session = _session(ScriptedModel([]), tmp_path=tmp_path)
    session.model_id = "keep-me"
    inner = session._catalog_lock
    inner.acquire()
    waiting = threading.Event()
    session._catalog_lock = _HeldCatalogLock(inner, waiting)
    stdout = io.StringIO()

    def publish() -> None:
        session_module._publish_catalog(
            session,
            stdout,
            session_module.CatalogFetch(
                models=[CatalogModel(id="plain/text", supports_tools=False)]
            ),
        )

    worker = threading.Thread(target=publish)
    worker.start()
    assert waiting.wait(timeout=2)
    assert worker.is_alive()
    assert session.catalog == []
    inner.release()
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert [item.id for item in session.catalog] == ["plain/text", "keep-me"]
    assert session.catalog[0].supports_tools is False


def test_catalog_paging_does_not_block_the_next_session_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = threading.Event()
    release = threading.Event()
    saw_new = threading.Event()

    class PausingStdin:
        def __init__(self) -> None:
            self._lines = ['{"type":"models"}\n', '{"type":"new"}\n']
            self._index = 0

        def __iter__(self) -> PausingStdin:
            return self

        def __next__(self) -> str:
            if self._index >= len(self._lines):
                raise StopIteration
            if self._index == 1:
                assert started.wait(timeout=2)
            line = self._lines[self._index]
            self._index += 1
            return line

    def fetch() -> list[CatalogModel]:
        started.set()
        assert release.wait(timeout=2)
        return [CatalogModel(id="vendor/model", supports_tools=True)]

    original = session_module._handle_new

    def wrapped(active: AgentSession, stdout: Any) -> None:
        assert started.is_set()
        saw_new.set()
        release.set()
        original(active, stdout)

    monkeypatch.setattr(session_module, "_handle_new", wrapped)
    session = _session(ScriptedModel([]), tmp_path=tmp_path)
    session.fetch_models = fetch
    stdout = io.StringIO()
    code = serve(session, PausingStdin(), stdout)
    assert code == 0
    assert saw_new.is_set()
    events = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert any(event["type"] == "ready" for event in events)
    models = [event for event in events if event["type"] == "models"]
    assert models[0]["ok"] is True
    assert models[0]["models"][0]["id"] == "vendor/model"


def test_session_key_is_what_the_next_completion_sends(tmp_path: Path) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project"
    project.mkdir()
    client = _HeaderClient({"data": []})
    model = OpenRouterModel(ModelConfig(model="example/model", api_key=""), client)
    session = AgentSession(Agent(model, ToolRegistry(), LocalArtifactStore(project)))
    session.home = home
    session.workspace = project
    session.requires_api_key = True

    events = _run_protocol(
        [
            '{"type":"set_key","scope":"user","key":"user-key"}',
            '{"type":"set_key","scope":"session","key":"session-key"}',
            '{"type":"set_key","scope":"user","key":"later-user-key"}',
            '{"type":"user","text":"hi"}',
        ],
        session,
    )

    assert client.posts[0]["headers"]["Authorization"] == "Bearer session-key"
    assert model.config.api_key == "session-key"
    assert session.session_key == "session-key"
    done = [event for event in events if event["type"] == "done"]
    assert done[0]["tokens_per_second"] == 5.0
    assert "last 5.0 tok/s" in done[0]["status_line"]
    dumped = json.dumps(events)
    assert "user-key" not in dumped
    assert "session-key" not in dumped
    assert "later-user-key" not in dumped
    user_key = home / ".atlas" / "keys" / "openrouter_api_key"
    assert user_key.read_text(encoding="utf-8").strip() == "later-user-key"
    assert "session-key" not in user_key.read_text(encoding="utf-8")
    for path in (project / ".atlas").rglob("*"):
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            assert "session-key" not in text
            assert "user-key" not in text
            assert "later-user-key" not in text


def test_protocol_remove_confirms_and_stays_inside_the_root(tmp_path: Path) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project"
    ensure_project_layout(project)
    ensure_user_layout(home)
    session_dir = project / ".atlas" / "sessions" / "s1"
    artifact = project / ".atlas" / "artifacts" / "s1"
    session_dir.mkdir()
    artifact.mkdir()
    (artifact / "chip.txt").write_text("chip", encoding="utf-8")
    outside = tmp_path / "secret.txt"
    outside.write_text("nope", encoding="utf-8")
    key = write_user_key("user-key", home)
    weight = home / ".atlas" / "weights" / "model.json"
    weight.write_text("{}", encoding="utf-8")
    data = project / ".atlas" / "data" / "keep.txt"
    data.write_text("table", encoding="utf-8")

    session = _session(ScriptedModel([]), tmp_path=project)
    session.home = home
    session.workspace = project

    denied = _run_protocol(
        [
            '{"type":"remove","scope":"project","kind":"sessions","name":"s1"}',
        ],
        session,
    )
    assert denied[1]["type"] == "error"
    assert "confirmation required" in denied[1]["message"]
    assert session_dir.is_dir()
    assert artifact.is_dir()

    escaped = _run_protocol(
        [
            '{"type":"remove","scope":"project","kind":"sessions",'
            '"name":"../secret.txt","confirm":true}',
        ],
        session,
    )
    assert escaped[1]["type"] == "error"
    assert "escapes" in escaped[1]["message"]
    assert outside.is_file()
    assert key.is_file()
    assert weight.is_file()

    removed = _run_protocol(
        [
            '{"type":"remove","scope":"project","kind":"sessions","name":"s1","confirm":true}',
        ],
        session,
    )
    assert removed[1]["type"] == "removed"
    assert removed[1]["scope"] == "project"
    assert removed[1]["name"] == "s1"
    assert not session_dir.exists()
    assert not artifact.exists()
    assert data.is_file()
    assert key.read_text(encoding="utf-8").strip() == "user-key"
    assert weight.is_file()
    assert "user-key" not in json.dumps(removed)


def test_omitted_provider_cost_stays_unset(tmp_path: Path) -> None:
    client = _QueueClient(
        [
            {
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
                "choices": [{"message": {"content": "ok", "tool_calls": []}}],
            }
        ],
        elapsed_seconds=0.25,
        time_to_first_token_seconds=0.05,
    )
    model = OpenRouterModel(ModelConfig(model="example/model", api_key="test-key"), client)
    session = AgentSession(Agent(model, ToolRegistry(), LocalArtifactStore(tmp_path)))

    events = _run_protocol(['{"type":"user","text":"hi"}'], session, redact="test-key")

    done = next(event for event in events if event["type"] == "done")
    assert done["spend"] is None
    assert done["prompt_tokens"] == 3
    assert done["completion_tokens"] == 1
    assert done["total_tokens"] == 4
    assert done["tokens_per_second"] == 5.0
    assert done["time_to_first_token_seconds"] == 0.05
    assert done["recent"] == ["200 0.25s"]
    assert "spend unknown" in done["status_line"]
    assert "spend $0" not in done["status_line"]
    assert "last 5.0 tok/s" in done["status_line"]
    assert "ttft" not in done["status_line"]
    assert "recent" not in done["status_line"]
    assert "test-key" not in json.dumps(events)


def test_omitted_cost_uses_catalog_prices_and_token_details(tmp_path: Path) -> None:
    client = _QueueClient(
        [
            {
                "usage": {
                    "prompt_tokens": 1000,
                    "completion_tokens": 100,
                    "total_tokens": 1100,
                    "prompt_tokens_details": {"cached_tokens": 800},
                    "completion_tokens_details": {"reasoning_tokens": 40},
                },
                "choices": [{"message": {"content": "ok", "tool_calls": []}}],
            }
        ],
        elapsed_seconds=0.5,
        time_to_first_token_seconds=0.1,
    )
    model = OpenRouterModel(ModelConfig(model="priced/model", api_key="test-key"), client)
    session = AgentSession(Agent(model, ToolRegistry(), LocalArtifactStore(tmp_path)))
    session.catalog = [
        CatalogModel(
            id="priced/model",
            prompt_price="0.000002",
            completion_price="0.00001",
        )
    ]

    events = _run_protocol(['{"type":"user","text":"hi"}'], session, redact="test-key")

    done = next(event for event in events if event["type"] == "done")
    assert done["cached_tokens"] == 800
    assert done["reasoning_tokens"] == 40
    assert done["spend"] == pytest.approx(0.003)
    assert done["estimated"] is True
    assert "spend ~$0.003" in done["status_line"]
    assert "spend $0.003" not in done["status_line"]
    assert "spend unknown" not in done["status_line"]
    assert "cached 800" in done["status_line"]
    assert "reasoning 40" in done["status_line"]
    assert done["tokens_per_second"] == 250.0
    assert "last 250.0 tok/s" in done["status_line"]
    assert "test-key" not in json.dumps(events)


def test_turn_after_select_uses_the_catalog_context_limit(tmp_path: Path) -> None:
    client = _QueueClient(
        [
            {
                "usage": {
                    "prompt_tokens": 9,
                    "completion_tokens": 1,
                    "total_tokens": 10,
                    "cost": 0.02,
                },
                "choices": [{"message": {"content": "changed", "tool_calls": []}}],
            }
        ],
        elapsed_seconds=0.5,
    )
    model = OpenRouterModel(ModelConfig(model="a", api_key="test-key"), client)
    session = AgentSession(Agent(model, ToolRegistry(), LocalArtifactStore(tmp_path)))
    session.catalog = [
        CatalogModel(id="a", context_length=111, prompt_price="0.1", completion_price="0.2"),
        CatalogModel(id="b", context_length=222, prompt_price="0.3", completion_price="0.4"),
    ]
    session.model_id = "a"
    session.totals.model = "a"
    session.totals.context_limit = 111

    events = _run_protocol(
        ['{"type":"select_model","id":"b"}', '{"type":"user","text":"look"}'],
        session,
        redact="test-key",
    )

    selected = next(event for event in events if event["type"] == "model")
    done = next(event for event in events if event["type"] == "done")
    assert selected["id"] == "b"
    assert selected["context_length"] == 222
    assert model.config.model == "b"
    assert done["context_used"] == 9
    assert done["context_limit"] == 222
    assert "context 9/222" in done["status_line"]
    assert "128000" not in done["status_line"]
    assert "test-key" not in json.dumps(events)


def test_main_accepts_the_environment_key_without_printing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-session-key")
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr("atlas.agent.session.Path.home", lambda: home)
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"type":"quit"}\n'))

    code = main(["--workspace", str(workspace)])

    captured = capsys.readouterr()
    assert code == 0
    events = [json.loads(line) for line in captured.out.splitlines()]
    assert events[0]["type"] == "ready"
    assert events[0]["key_set"] is True
    assert "context 0/unknown" in events[0]["status_line"]
    assert "env-session-key" not in captured.out
    assert "env-session-key" not in captured.err
    assert not (home / ".atlas" / "keys" / "openrouter_api_key").exists()
    assert (home / ".atlas" / "weights").is_dir()
    assert (home / ".atlas" / "defaults").is_dir()
    for path in (workspace / ".atlas").rglob("*"):
        if path.is_file():
            assert "env-session-key" not in path.read_text(encoding="utf-8")


def test_protocol_lists_the_chosen_root_and_names_a_weight(tmp_path: Path) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project"
    ensure_project_layout(project)
    ensure_user_layout(home)
    (project / ".atlas" / "sessions" / "s1").mkdir()
    (project / ".atlas" / "artifacts" / "s1").mkdir()
    (home / ".atlas" / "sessions" / "user-only").mkdir(parents=True)
    weight = home / ".atlas" / "weights" / "model.json"
    weight.write_text("{}", encoding="utf-8")
    key = write_user_key("user-key", home)
    session = _session(ScriptedModel([]), tmp_path=project)
    session.home = home
    session.workspace = project

    listed = _run_protocol(
        [
            '{"type":"list","scope":"project","kind":"sessions"}',
            '{"type":"list","scope":"project","kind":"artifacts"}',
            '{"type":"list","scope":"user","kind":"sessions"}',
            '{"type":"list","scope":"project","kind":"keys"}',
        ],
        session,
    )

    assert listed[1]["type"] == "listing"
    assert listed[1]["names"] == ["s1"]
    assert listed[2]["names"] == ["s1"]
    assert listed[3]["scope"] == "user"
    assert listed[3]["names"] == ["user-only"]
    assert listed[4]["type"] == "error"
    assert "sessions" in listed[4]["message"]

    removed = _run_protocol(
        ['{"type":"remove","scope":"user","kind":"weights","name":"model.json","confirm":true}'],
        session,
    )
    assert removed[1]["type"] == "removed"
    assert removed[1]["scope"] == "user"
    assert not weight.exists()
    assert key.is_file()
    assert (project / ".atlas" / "sessions" / "s1").is_dir()
    assert (project / ".atlas" / "data").is_dir()
    assert "user-key" not in json.dumps(listed + removed)


class _ProtocolIn:
    """Blocking stdin so a test can answer after ``main`` creates the session."""

    def __init__(self) -> None:
        self._lines: queue.Queue[str] = queue.Queue()

    def push(self, line: str) -> None:
        self._lines.put(line)

    def __iter__(self) -> _ProtocolIn:
        return self

    def __next__(self) -> str:
        return self._lines.get()


class _CollectOut:
    def __init__(self) -> None:
        self._chunks: list[str] = []
        self._lock = threading.Lock()

    def write(self, text: str) -> int:
        with self._lock:
            self._chunks.append(text)
        return len(text)

    def flush(self) -> None:
        return None

    def text(self) -> str:
        with self._lock:
            return "".join(self._chunks)


class _ScriptResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.status_code = 200
        self.elapsed_seconds = 0.2

    def json(self) -> dict[str, Any]:
        return self.payload


class _ScriptClient:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self.posts = 0

    def post(self, _url: str, **_kwargs: Any) -> _ScriptResponse:
        self.posts += 1
        if self.posts == 1:
            payload: dict[str, Any] = {
                "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "write_file",
                                        "arguments": json.dumps(
                                            {"path": "note.txt", "content": "from-main"}
                                        ),
                                    },
                                }
                            ],
                        }
                    }
                ],
            }
        else:
            payload = {
                "usage": {"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10},
                "choices": [{"message": {"content": "wrote it", "tool_calls": []}}],
            }
        return _ScriptResponse(payload)

    def close(self) -> None:
        return None


def _wait_until(predicate: Any, detail: str) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError(detail)


def test_main_writes_tools_into_the_session_artifact_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-session-key")
    monkeypatch.setattr("atlas.agent.model.httpx.Client", _ScriptClient)
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    weight = home / ".atlas" / "weights" / "keep.json"
    weight.parent.mkdir(parents=True)
    weight.write_text("{}", encoding="utf-8")
    data = workspace / ".atlas" / "data" / "keep.txt"
    data.parent.mkdir(parents=True)
    data.write_text("table", encoding="utf-8")
    other_note = workspace / ".atlas" / "artifacts" / "other" / "keep.txt"
    other_session = workspace / ".atlas" / "sessions" / "other"
    other_note.parent.mkdir(parents=True)
    other_session.mkdir(parents=True)
    other_note.write_text("keep", encoding="utf-8")
    (other_session / "meta.json").write_text('{"id":"other"}\n', encoding="utf-8")
    monkeypatch.setattr("atlas.agent.session.Path.home", lambda: home)
    monkeypatch.chdir(workspace)
    protocol = _ProtocolIn()
    stdout = _CollectOut()
    monkeypatch.setattr(sys, "stdin", protocol)
    monkeypatch.setattr(sys, "stdout", stdout)
    holder: dict[str, object] = {}

    def run() -> None:
        try:
            holder["code"] = main(["--workspace", str(workspace)])
        except Exception as exc:  # pragma: no cover - failure path for the assertion
            holder["error"] = exc

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        sessions = workspace / ".atlas" / "sessions"
        _wait_until(
            lambda: any(path.parent.name != "other" for path in sessions.glob("*/meta.json")),
            f"main did not create a session: {stdout.text()}",
        )
        session_id = next(
            path.parent.name for path in sessions.glob("*/meta.json") if path.parent.name != "other"
        )
        note = workspace / ".atlas" / "artifacts" / session_id / "note.txt"
        artifact_dir = note.parent

        protocol.push('{"type":"user","text":"save a note"}')
        _wait_until(lambda: note.is_file(), f"tool did not write {note}: {stdout.text()}")
        assert note.read_text(encoding="utf-8") == "from-main"
        assert not (workspace / "note.txt").exists()
        assert str(note.resolve()) not in stdout.text()

        protocol.push(
            json.dumps(
                {
                    "type": "remove",
                    "scope": "project",
                    "kind": "sessions",
                    "name": session_id,
                    "confirm": True,
                }
            )
        )
        session_dir = sessions / session_id
        _wait_until(
            lambda: not artifact_dir.exists() and not session_dir.exists(),
            f"session remove left files behind: {stdout.text()}",
        )
        assert other_note.read_text(encoding="utf-8") == "keep"
        assert other_session.is_dir()
        assert data.is_file()
        assert weight.is_file()
    finally:
        protocol.push('{"type":"quit"}')
    thread.join(timeout=5)

    assert holder.get("error") is None
    assert holder.get("code") == 0
    assert not thread.is_alive()
    assert data.read_text(encoding="utf-8") == "table"
    assert weight.read_text(encoding="utf-8") == "{}"
    assert "env-session-key" not in stdout.text()
    events = [json.loads(line) for line in stdout.text().splitlines()]
    steps = [event for event in events if event["type"] == "step"]
    assert steps[0]["name"] == "write_file"
    assert steps[0]["artifacts"] == ["note.txt"]


class _TwoCallClient:
    """One completion that asks for two writes, then a plain answer."""

    posts = 0

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        return None

    def post(self, _url: str, **_kwargs: Any) -> _ScriptResponse:
        type(self).posts += 1
        if type(self).posts == 1:
            payload: dict[str, Any] = {
                "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "c1",
                                    "type": "function",
                                    "function": {
                                        "name": "write_file",
                                        "arguments": json.dumps(
                                            {"path": "note.txt", "content": "one"}
                                        ),
                                    },
                                },
                                {
                                    "id": "c2",
                                    "type": "function",
                                    "function": {
                                        "name": "write_file",
                                        "arguments": json.dumps(
                                            {"path": "other.txt", "content": "two"}
                                        ),
                                    },
                                },
                            ],
                        }
                    }
                ],
            }
        else:
            payload = {
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
                "choices": [{"message": {"content": "done", "tool_calls": []}}],
            }
        return _ScriptResponse(payload)

    def close(self) -> None:
        return None


def test_main_uses_the_workspace_tool_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-session-key")
    _TwoCallClient.posts = 0
    monkeypatch.setattr("atlas.agent.model.httpx.Client", _TwoCallClient)
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = workspace / ".atlas"
    config.mkdir()
    (config / "agent.toml").write_text("max_tool_calls = 1\n", encoding="utf-8")
    monkeypatch.setattr("atlas.agent.session.Path.home", lambda: home)
    monkeypatch.chdir(workspace)
    protocol = _ProtocolIn()
    stdout = _CollectOut()
    monkeypatch.setattr(sys, "stdin", protocol)
    monkeypatch.setattr(sys, "stdout", stdout)
    holder: dict[str, object] = {}

    def run() -> None:
        try:
            holder["code"] = main(["--workspace", str(workspace)])
        except Exception as exc:  # pragma: no cover - failure path for the assertion
            holder["error"] = exc

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        sessions = workspace / ".atlas" / "sessions"
        _wait_until(
            lambda: any(sessions.glob("*/meta.json")),
            f"main did not create a session: {stdout.text()}",
        )
        session_id = next(path.parent.name for path in sessions.glob("*/meta.json"))
        note = workspace / ".atlas" / "artifacts" / session_id / "note.txt"
        other = workspace / ".atlas" / "artifacts" / session_id / "other.txt"
        protocol.push('{"type":"user","text":"write two files"}')
        _wait_until(
            lambda: "stopped_for_limit" in stdout.text(),
            f"turn did not finish: {stdout.text()}",
        )
        assert note.read_text(encoding="utf-8") == "one"
        assert not other.exists()
        assert _TwoCallClient.posts == 1
    finally:
        protocol.push('{"type":"quit"}')
    thread.join(timeout=5)

    assert holder.get("error") is None
    assert holder.get("code") == 0
    events = [json.loads(line) for line in stdout.text().splitlines()]
    steps = [event for event in events if event["type"] == "step"]
    done = [event for event in events if event["type"] == "done"]
    assert [step["name"] for step in steps] == ["write_file", "write_file"]
    assert steps[0]["ok"] is True
    assert steps[0]["error_kind"] is None
    assert steps[1]["ok"] is False
    assert steps[1]["error_kind"] == "limit"
    assert done[0]["stopped_for_limit"] is True
    assert "env-session-key" not in stdout.text()


def test_goal_is_session_context_on_later_turns(tmp_path: Path) -> None:
    model = ScriptedModel(
        [
            ModelResponse(content="first answer"),
            ModelResponse(content="second answer"),
        ]
    )
    read_root = tmp_path / "project"
    read_root.mkdir()
    write_root = tmp_path / "writes"
    registry = ToolRegistry()
    registry.register(EchoTool())
    agent = Agent(
        model,
        registry,
        LocalArtifactStore(write_root, read_root=read_root),
    )
    session = _session_from(agent)
    tool_names = [tool.name for tool in session.agent.tools.definitions]
    root = session.agent.artifacts.root
    mounted_read = session.agent.artifacts.read_root

    session.set_goal("  map  the flood  ")
    session.turn("what first?")
    session.turn("what next?")

    assert session.goal == "map  the flood"
    assert tool_names == ["echo"]
    assert [tool.name for tool in session.agent.tools.definitions] == tool_names
    assert "goal" not in tool_names
    assert session.agent.artifacts.root == root
    assert session.agent.artifacts.read_root == mounted_read
    first = model.requests[0][0]
    second = model.requests[1][0]
    assert first[0].role == Role.SYSTEM
    assert "map  the flood" not in (first[0].content or "")
    assert first[1].content == session_goal_text("map  the flood")
    assert first[2].content == "what first?"
    assert second[1].content == session_goal_text("map  the flood")
    assert second[-1].content == "what next?"
    assert [tool.name for tool in model.requests[1][1]] == ["echo"]


def test_replacing_a_goal_keeps_one_block(tmp_path: Path) -> None:
    model = ScriptedModel([ModelResponse(content="ok")])
    session = _session(model, tmp_path=tmp_path)
    session.set_goal("first goal")
    session.set_goal("map the flood")
    session.turn("go")

    contents = [message.content for message in model.requests[0][0]]
    assert contents.count(session_goal_text("map the flood")) == 1
    assert session_goal_text("first goal") not in contents


def test_clearing_a_goal_removes_it_from_later_turns(tmp_path: Path) -> None:
    model = ScriptedModel([ModelResponse(content="ok")])
    session = _session(model, tmp_path=tmp_path)
    session.set_goal("map the flood")
    session.clear_goal()
    session.turn("go")

    assert session.goal == ""
    assert all("map the flood" not in (message.content or "") for message in model.requests[0][0])
    assert all(
        message.content != session_goal_text("map the flood") for message in session.messages
    )


def test_goal_follows_project_instructions(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("Prefer metric units.\n", encoding="utf-8")
    model = ScriptedModel([ModelResponse(content="ok")])
    agent = Agent(model, ToolRegistry(), LocalArtifactStore(tmp_path))
    session = AgentSession(agent)
    session.set_goal("map the flood")
    session.turn("how wide is the river?")

    messages = model.requests[0][0]
    assert messages[0].content is not None
    assert messages[0].content.startswith("You are Atlas")
    assert "map the flood" not in messages[0].content
    assert "Prefer metric units." in (messages[1].content or "")
    assert messages[2].content == session_goal_text("map the flood")
    assert messages[3].content == "how wide is the river?"


def test_new_clears_the_goal_with_the_conversation(tmp_path: Path) -> None:
    model = ScriptedModel(
        [
            ModelResponse(content="before"),
            ModelResponse(content="after"),
        ]
    )
    session = _session(model, tmp_path=tmp_path)
    session.set_goal("map the flood")
    session.turn("before")

    events = _run_protocol(['{"type":"new"}', '{"type":"goal"}'], session)

    assert session.goal == ""
    shown = events[-1]
    assert shown["type"] == "goal"
    assert shown["text"] == ""
    assert shown["message"] == "none is set"
    assert shown["cleared"] is False
    session.turn("after")
    assert all("map the flood" not in (message.content or "") for message in model.requests[-1][0])


def test_goal_protocol_sets_shows_and_clears(tmp_path: Path) -> None:
    model = ScriptedModel([ModelResponse(content="ok")])
    session = _session(model, tmp_path=tmp_path)

    empty = _run_protocol(['{"type":"goal"}'], session)
    assert empty[-1] == {
        "type": "goal",
        "text": "",
        "cleared": False,
        "message": "none is set",
    }

    set_goal = _run_protocol(['{"type":"goal","text":"map  the flood"}'], session)
    assert set_goal[-1]["type"] == "goal"
    assert set_goal[-1]["text"] == "map  the flood"
    assert set_goal[-1]["message"] == "map  the flood"
    assert set_goal[-1]["cleared"] is False

    shown = _run_protocol(['{"type":"goal"}'], session)
    assert shown[-1]["text"] == "map  the flood"
    assert shown[-1]["message"] == "map  the flood"

    cleared = _run_protocol(['{"type":"goal","clear":true}'], session)
    assert cleared[-1] == {
        "type": "goal",
        "text": "",
        "cleared": True,
        "message": "goal cleared",
    }
    assert session.goal == ""


def test_goal_protocol_rejects_extra_fields_without_changing_the_goal(tmp_path: Path) -> None:
    model = ScriptedModel([])
    session = _session(model, tmp_path=tmp_path)
    session.set_goal("map the flood")

    extra = _run_protocol(['{"type":"goal","clear":true,"text":"something else"}'], session)
    blank = _run_protocol(['{"type":"goal","text":"   "}'], session)
    flag = _run_protocol(['{"type":"goal","clear":false}'], session)

    assert session.goal == "map the flood"
    assert extra[-1] == {"type": "error", "message": "usage: /goal [text] | /goal clear"}
    assert blank[-1]["type"] == "error"
    assert flag[-1]["type"] == "error"
    assert session.messages[1].content == session_goal_text("map the flood")


def test_user_turn_with_goal_wording_survives_set_and_clear(tmp_path: Path) -> None:
    """A user turn that starts like a goal stays through ``/goal`` set and clear.

    The goal used to be found by matching any user message that started with
    ``Session goal:``. Setting or clearing the goal then overwrote or deleted
    that turn. The session now keeps the goal message it inserted.
    """
    lookalike = "Session goal:\nkeep this user turn"
    model = ScriptedModel(
        [
            ModelResponse(content="noted"),
            ModelResponse(content="still here"),
        ]
    )
    session = _session(model, tmp_path=tmp_path)
    session.turn(lookalike)
    original = session.messages[1]
    assert original.content == lookalike

    cleared = _run_protocol(['{"type":"goal","clear":true}'], session)
    assert cleared[-1]["type"] == "goal"
    assert cleared[-1]["cleared"] is True
    assert session.goal == ""
    assert any(message is original for message in session.messages)
    assert original.content == lookalike

    set_goal = _run_protocol([json.dumps({"type": "goal", "text": "map the flood"})], session)
    assert set_goal[-1]["type"] == "goal"
    assert set_goal[-1]["text"] == "map the flood"
    assert any(message is original for message in session.messages)
    assert original.content == lookalike
    goal_message = next(
        message
        for message in session.messages
        if message.content == session_goal_text("map the flood")
    )
    assert goal_message is not original

    replaced = _run_protocol([json.dumps({"type": "goal", "text": "map the coast"})], session)
    assert replaced[-1]["text"] == "map the coast"
    assert any(message is goal_message for message in session.messages)
    assert goal_message.content == session_goal_text("map the coast")
    assert any(message is original for message in session.messages)
    assert original.content == lookalike
    assert session_goal_text("map the flood") not in [
        message.content for message in session.messages
    ]

    removed = _run_protocol(['{"type":"goal","clear":true}'], session)
    assert removed[-1]["cleared"] is True
    assert session.goal == ""
    assert all(message is not goal_message for message in session.messages)
    assert any(message is original for message in session.messages)
    assert original.content == lookalike

    collision = session_goal_text("map the flood")
    session.turn(collision)
    collision_message = next(
        message for message in session.messages if message.content == collision
    )
    session.set_goal("map the flood")
    assert sum(message.content == collision for message in session.messages) == 2
    assert any(message is collision_message for message in session.messages)
    assert collision_message.content == collision
    session.clear_goal()
    assert any(message is collision_message for message in session.messages)
    assert collision_message.content == collision
    assert sum(message.content == collision for message in session.messages) == 1


def test_goal_longer_than_2000_characters_is_rejected(tmp_path: Path) -> None:
    model = ScriptedModel([])
    session = _session(model, tmp_path=tmp_path)
    session.set_goal("keep this")
    kept = session.messages[1].content

    too_long = "x" * 2001
    with pytest.raises(ValueError, match="at most 2000 characters"):
        session.set_goal(too_long)
    assert session.goal == "keep this"
    assert session.messages[1].content == kept

    events = _run_protocol([json.dumps({"type": "goal", "text": too_long})], session)
    assert events[-1] == {
        "type": "error",
        "message": "goal text must be at most 2000 characters",
    }
    assert session.goal == "keep this"
    assert session.messages[1].content == kept

    exact = "y" * 2000
    accepted = _run_protocol([json.dumps({"type": "goal", "text": exact})], session)
    assert accepted[-1]["type"] == "goal"
    assert accepted[-1]["text"] == exact
    assert session.goal == exact
    assert session.messages[1].content == session_goal_text(exact)

    session.set_goal(f"  {'z' * 2000}  ")
    assert session.goal == "z" * 2000


def test_goal_set_mid_conversation_is_its_own_user_message(tmp_path: Path) -> None:
    """A goal set after a turn stays a separate user message beside that turn.

    The goal is wrapped in ``<session_goal>`` tags and inserted after the
    opening context. The earlier user turn is still a user message, so the
    model sees those two user messages in a row. Atlas does not merge them.
    """
    model = ScriptedModel(
        [
            ModelResponse(content="first answer"),
            ModelResponse(content="second answer"),
        ]
    )
    session = _session(model, tmp_path=tmp_path)
    session.turn("what first?")
    session.set_goal("map the flood")

    assert [message.role for message in session.messages] == [
        "system",
        "user",
        "user",
        "assistant",
    ]
    assert session.messages[1].content == "<session_goal>\nmap the flood\n</session_goal>"
    assert session.messages[1].content == session_goal_text("map the flood")
    assert session.messages[2].content == "what first?"
    assert session.messages[2].role == "user"
    assert session.messages[1] is not session.messages[2]

    session.turn("what next?")
    sent = model.requests[1][0]
    assert [message.role for message in sent[:5]] == [
        "system",
        "user",
        "user",
        "assistant",
        "user",
    ]
    assert sent[1].content == session_goal_text("map the flood")
    assert sent[2].content == "what first?"
    assert sent[3].content == "first answer"
    assert sent[4].content == "what next?"


def _session_from(agent: Agent) -> AgentSession:
    return AgentSession(agent)
