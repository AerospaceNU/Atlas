from __future__ import annotations

import json
import queue
import threading
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
from atlas.agent.runtime import Agent, PermissionRequest
from atlas.agent.session import AgentSession, serve
from atlas.agent.tools.write_file import WriteTool


class ScriptedModel:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = responses

    def complete(self, messages: list[ChatMessage], tools: list[ToolDefinition]) -> ModelResponse:
        return self.responses.pop(0)


def write_turn(name: str) -> list[ModelResponse]:
    return [
        ModelResponse(
            tool_calls=[
                ToolCall(id=name, name="write_file", arguments={"path": name, "content": "ok"})
            ]
        ),
        ModelResponse(content="done"),
    ]


def make_session(tmp_path: Path, responses: list[ModelResponse]) -> AgentSession:
    registry = ToolRegistry()
    registry.register(WriteTool())
    return AgentSession(
        Agent(ScriptedModel(responses), registry, LocalArtifactStore(tmp_path), approval="ask")
    )


class ProtocolInput:
    def __init__(self) -> None:
        self.lines: queue.Queue[str | None] = queue.Queue()

    def send(self, **message: Any) -> None:
        self.lines.put(json.dumps(message) + "\n")

    def __iter__(self) -> ProtocolInput:
        return self

    def __next__(self) -> str:
        line = self.lines.get()
        if line is None:
            raise StopIteration
        return line


class ProtocolOutput:
    def __init__(self) -> None:
        self.events: queue.Queue[dict[str, Any]] = queue.Queue()
        self.seen: list[dict[str, Any]] = []

    def write(self, text: str) -> int:
        if text.strip():
            self.events.put(json.loads(text))
        return len(text)

    def flush(self) -> None:
        pass

    def until(self, kind: str) -> dict[str, Any]:
        while True:
            event = self.events.get(timeout=3)
            self.seen.append(event)
            assert event["type"] != "error", event
            if event["type"] == kind:
                return event


def test_yolo_releases_pending_call_skips_prompts_and_can_be_turned_off(tmp_path: Path) -> None:
    session = make_session(
        tmp_path, write_turn("first.txt") + write_turn("second.txt") + write_turn("third.txt")
    )
    stdin, stdout = ProtocolInput(), ProtocolOutput()
    thread = threading.Thread(target=serve, args=(session, stdin, stdout), daemon=True)
    thread.start()
    try:
        assert stdout.until("ready")["yolo_enabled"] is False
        stdin.send(type="user", text="first")
        assert stdout.until("permission_request")["id"] == "first.txt"
        assert not (tmp_path / "first.txt").exists()

        stdin.send(type="yolo")
        assert stdout.until("yolo")["enabled"] is True
        stdout.until("done")
        assert (tmp_path / "first.txt").read_text() == "ok"

        stdin.send(type="user", text="second")
        stdout.until("done")
        assert (tmp_path / "second.txt").read_text() == "ok"
        assert [item["id"] for item in stdout.seen if item["type"] == "permission_request"] == [
            "first.txt"
        ]

        stdin.send(type="yolo")
        assert stdout.until("yolo")["enabled"] is False
        stdin.send(type="user", text="third")
        assert stdout.until("permission_request")["id"] == "third.txt"
        stdin.send(type="permission", id="third.txt", allow=False)
        stdout.until("done")
        assert not (tmp_path / "third.txt").exists()
        (tmp_path / "events.json").write_text(json.dumps(stdout.seen, indent=2))
    finally:
        session.cancel_event.set()
        for request_id in list(session.permission_slots):
            session.resolve_permission(request_id, False)
        stdin.send(type="quit")
        stdin.lines.put(None)
        thread.join(timeout=3)
    assert not thread.is_alive()
    assert not make_session(tmp_path, []).yolo_enabled


def test_yolo_toggle_before_a_permission_callback_does_not_wait(tmp_path: Path) -> None:
    session = make_session(tmp_path, [])
    events: list[dict[str, Any]] = []
    session.toggle_yolo(events.append)
    assert session.ask_permission(
        PermissionRequest(id="late", tool="write_file", summary="write a file", arguments={}),
        events.append,
    )
    assert events == [{"type": "yolo", "enabled": True}]
    assert not session.permission_slots


def test_yolo_keeps_read_only_but_allows_writes_outside_the_workspace(tmp_path: Path) -> None:
    session = make_session(tmp_path, write_turn("blocked.txt"))
    session.agent.read_only = True
    session.toggle_yolo(lambda event: None)
    run = session.turn("write")
    assert run.steps[0].error_kind == "denied"
    assert not (tmp_path / "blocked.txt").exists()

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session = make_session(workspace, write_turn("../outside.txt"))
    session.toggle_yolo(lambda event: None)
    run = session.turn("write outside")
    assert run.steps[0].error is None
    assert (tmp_path / "outside.txt").read_text(encoding="utf-8") == "ok"
