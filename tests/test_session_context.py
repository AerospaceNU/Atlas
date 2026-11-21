from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from atlas.agent import session as session_module
from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import (
    ChatMessage,
    ModelResponse,
    Role,
    TokenUsage,
    ToolCall,
    ToolDefinition,
    ToolRegistry,
    default_registry,
)
from atlas.agent.layout import project_session_dir
from atlas.agent.runtime import Agent
from atlas.agent.session import AgentSession, serve


class FlakyModel:
    """Play scripted responses; an `Exception` item is raised instead."""

    def __init__(self, responses: list[ModelResponse | Exception]) -> None:
        self.responses = responses

    def complete(self, messages: list[ChatMessage], tools: list[ToolDefinition]) -> ModelResponse:
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _usage(prompt: int = 100, completion: int = 10) -> TokenUsage:
    return TokenUsage(
        prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion
    )


def _read(call_id: str = "c1") -> ModelResponse:
    return ModelResponse(
        tool_calls=[ToolCall(id=call_id, name="read_file", arguments={"path": "a.txt"})],
        usage=_usage(),
    )


def _session(
    tmp_path: Path, responses: list[ModelResponse | Exception], *, read_only: bool = False
) -> AgentSession:
    registry = default_registry()
    agent = Agent(
        FlakyModel(responses), registry, LocalArtifactStore(tmp_path), read_only=read_only
    )
    session = AgentSession(agent)
    session.workspace = tmp_path
    session.session_id = "abc123"
    return session


def _drive(session: AgentSession, *lines: dict[str, Any]) -> list[dict[str, Any]]:
    stdin = io.StringIO("".join(json.dumps(line) + "\n" for line in lines))
    stdout = io.StringIO()
    serve(session, stdin, stdout)
    return [json.loads(line) for line in stdout.getvalue().splitlines()]


def _long_history(session: AgentSession, rounds: int = 40) -> None:
    for index in range(rounds):
        call_id = f"h{index}"
        session.messages.append(ChatMessage(role=Role.USER, content=f"real request {index}"))
        session.messages.append(
            ChatMessage(
                role=Role.ASSISTANT,
                content="",
                tool_calls=[ToolCall(id=call_id, name="read_file", arguments={"path": "a.txt"})],
            )
        )
        session.messages.append(
            ChatMessage(role=Role.TOOL, tool_call_id=call_id, name="read_file", content="x " * 3000)
        )
        session.messages.append(ChatMessage(role=Role.ASSISTANT, content=f"done {index}"))


def test_compact_shortens_history_and_keeps_protected_messages(tmp_path: Path) -> None:
    session = _session(tmp_path, [])
    session.set_goal("find the fire")
    _long_history(session)
    first_two = list(session.messages[:2])
    goal = session._goal_message
    before = session_module._estimate_tokens(session.messages)

    compaction = session.compact()

    after = session_module._estimate_tokens(session.messages)
    assert compaction.occurred
    assert after < before
    assert session.messages[0] is first_two[0]
    assert session.messages[1] is first_two[1]
    assert any(message is goal for message in session.messages)
    contents = [message.content for message in session.messages]
    for index in range(40):
        assert f"real request {index}" in contents
    assert session._goal_index() is not None


def test_compact_does_not_touch_the_context_constant(tmp_path: Path) -> None:
    from atlas.agent import runtime

    session = _session(tmp_path, [])
    _long_history(session)
    session.compact()
    assert runtime.MAX_CONTEXT_TOKENS == 1_000_000


def test_compact_message_reports_and_saves(tmp_path: Path) -> None:
    session = _session(tmp_path, [])
    _long_history(session)

    events = _drive(session, {"type": "compact"}, {"type": "quit"})

    compacted = next(event for event in events if event["type"] == "compacted")
    assert compacted["compaction"]["occurred"] is True
    assert "status_line" in compacted and "context_used" in compacted
    state = project_session_dir(tmp_path, "abc123") / "state.json"
    assert state.is_file()


def test_unknown_type_is_still_a_recoverable_error(tmp_path: Path) -> None:
    session = _session(tmp_path, [])
    events = _drive(session, {"type": "frobnicate"}, {"type": "quit"})
    assert events[-1]["type"] == "error"
    assert "Unknown message type" in events[-1]["message"]


def test_failed_turn_keeps_usage_and_history_and_next_turn_works(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    session = _session(
        tmp_path,
        [_read(), RuntimeError("provider exploded"), ModelResponse(content="ok", usage=_usage())],
    )

    events = _drive(
        session,
        {"type": "user", "text": "first"},
        {"type": "user", "text": "second"},
        {"type": "quit"},
    )

    kinds = [event["type"] for event in events]
    assert "error" in kinds
    error = next(event for event in events if event["type"] == "error")
    assert "provider exploded" in error["message"]
    assert kinds[-1] == "done"
    assert events[-1]["response"] == "ok"
    assert session.totals.prompt_tokens >= 200
    state = json.loads(
        (project_session_dir(tmp_path, "abc123") / "state.json").read_text(encoding="utf-8")
    )
    texts = [message["content"] for message in state["messages"]]
    assert "first" in texts and "second" in texts


def test_failed_turn_usage_is_counted_before_reraise(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    session = _session(tmp_path, [_read(), RuntimeError("boom")])
    with pytest.raises(RuntimeError, match="boom"):
        session.turn("go")
    assert session.totals.prompt_tokens == 100
    assert session.messages[-1].role in (Role.TOOL, Role.ASSISTANT)


def test_step_events_carry_notes_and_done_carries_empty_replies(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    session = _session(tmp_path, [_read(), ModelResponse(content="fin", usage=_usage())])
    events = _drive(session, {"type": "user", "text": "go"}, {"type": "quit"})
    step = next(event for event in events if event["type"] == "step")
    assert "notes" in step and isinstance(step["notes"], list)
    done = next(event for event in events if event["type"] == "done")
    assert done["empty_replies"] == 0


def test_read_only_session_cannot_create_a_writing_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    workspace = tmp_path / "ws"
    (workspace / ".atlas").mkdir(parents=True)
    (workspace / ".atlas" / "agent.toml").write_text("read_only = true\n", encoding="utf-8")
    monkeypatch.setattr("atlas.agent.session.Path.home", lambda: home)
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    captured: dict[str, AgentSession] = {}

    def fake_serve(session: AgentSession, *args: Any, **kwargs: Any) -> int:
        captured["session"] = session
        return 0

    monkeypatch.setattr(session_module, "serve", fake_serve)
    session_module.main(["--workspace", str(workspace)])

    tool = captured["session"].agent.tools.get("spawn_agent")
    assert tool is not None
    child = tool._make_agent(LocalArtifactStore(workspace), True, 3)  # type: ignore[attr-defined]
    assert child.read_only is True
    assert isinstance(ToolRegistry(), ToolRegistry)


def test_http_error_after_tool_round_saves_history(tmp_path: Path) -> None:
    from atlas.agent.model import OpenRouterHTTPError

    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    session = _session(tmp_path, [_read(), OpenRouterHTTPError(429, "slow down", 0.1)])
    events = _drive(session, {"type": "user", "text": "first"}, {"type": "quit"})
    assert any(event["type"] == "error" for event in events)
    state = json.loads(
        (project_session_dir(tmp_path, "abc123") / "state.json").read_text(encoding="utf-8")
    )
    roles = [message["role"] for message in state["messages"]]
    assert "tool" in roles
    assert session.totals.prompt_tokens == 100


def test_compact_updates_the_context_size_in_the_totals(tmp_path: Path) -> None:
    session = _session(tmp_path, [])
    _long_history(session)
    session.totals.context_used = 999_999
    session.totals.prompt_tokens = 7

    compaction = session.compact()

    assert compaction.occurred
    assert session.totals.context_used == session_module._estimate_tokens(session.messages)
    assert session.totals.prompt_tokens == 7


def test_saved_tool_output_is_redacted_with_the_live_secret_list(tmp_path: Path) -> None:
    from atlas.agent.contracts import ToolResult
    from atlas.agent.runtime import AgentStep, _tool_message

    session = _session(tmp_path, [])
    secret = "sk-late-key-9999"
    session.secrets.append(secret)  # added after construction, as set_key does
    assert session.agent.redact_output is not None
    step = AgentStep(
        call=ToolCall(id="b1", name="bash", arguments={}),
        result=ToolResult(text=secret + " " + "z" * 50_000),
    )

    _tool_message(step, session.agent.artifacts, session.agent.redact_output)

    saved = list((tmp_path / ".atlas").rglob("*.txt"))
    assert saved and all(secret not in p.read_text(encoding="utf-8") for p in saved)
