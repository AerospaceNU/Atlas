from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, Field

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import (
    CatalogModel,
    ChatMessage,
    ModelResponse,
    Role,
    Tool,
    ToolCall,
    ToolDefinition,
    ToolRegistry,
    ToolResult,
)
from atlas.agent.layout import resolve_openrouter_key
from atlas.agent.model import (
    MISSING_KEY_MESSAGE,
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
        self, payload: dict[str, Any], *, status_code: int, elapsed_seconds: float
    ) -> None:
        self.payload = payload
        self.status_code = status_code
        self.elapsed_seconds = elapsed_seconds

    def json(self) -> dict[str, Any]:
        return self.payload


class _QueueClient:
    def __init__(
        self,
        payloads: list[dict[str, Any]],
        *,
        status_code: int = 200,
        elapsed_seconds: float = 0.5,
    ) -> None:
        self.payloads = list(payloads)
        self.status_code = status_code
        self.elapsed_seconds = elapsed_seconds

    def post(self, _url: str, **_kwargs: Any) -> _QueueResponse:
        return _QueueResponse(
            self.payloads.pop(0),
            status_code=self.status_code,
            elapsed_seconds=self.elapsed_seconds,
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
        ]
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
    assert done[0]["tokens_per_second"] == 4.0
    assert done[0]["http_status"] == 200
    assert done[0]["elapsed_seconds"] == 0.5
    assert done[0]["context_used"] == 10
    assert done[0]["context_limit"] is None
    assert "context 10/unknown" in done[0]["status_line"]
    assert done[1]["prompt_tokens"] == 30
    assert done[1]["completion_tokens"] == 8
    assert done[1]["total_tokens"] == 38
    assert done[1]["spend"] == pytest.approx(0.04)
    assert done[1]["context_used"] == 20
    assert done[1]["tokens_per_second"] == 12.0
    line = done[1]["status_line"]
    assert "prompt 30" in line
    assert "completion 8" in line
    assert "tok/s" in line
    assert "http 200" in line
    assert "latency 0.50s" in line
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

    assert events[1]["type"] == "error"
    assert events[1]["http_status"] == 429
    assert events[1]["elapsed_seconds"] == 0.25
    assert events[1]["error_body"] == "slow down"
    assert "http 429" in events[1]["status_line"]
    assert "latency 0.25s" in events[1]["status_line"]
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
