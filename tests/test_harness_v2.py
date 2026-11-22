"""Regression and contract tests for the v2 Python agent harness overhaul."""

from __future__ import annotations

import io
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import BaseModel, Field

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.checkpoints import restore_latest, snapshot, snapshot_many
from atlas.agent.contracts import (
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
    current_cancel_event,
    emit_tool_output,
)
from atlas.agent.instructions import neutralize_untrusted_tags, wrap_untrusted
from atlas.agent.model import (
    ModelConfig,
    OpenRouterHTTPError,
    OpenRouterModel,
    StreamCancelledError,
)
from atlas.agent.redaction import redact_text, redact_url, redact_value
from atlas.agent.rpc import serve
from atlas.agent.runtime import (
    MAX_TOOL_RESULT_CHARS,
    Agent,
    AgentStep,
    PermissionDecision,
    PermissionRequest,
    _is_mutating,
    _tool_message,
)
from atlas.agent.session import AgentSession, main
from atlas.agent.tools._subagent import SubagentTool
from atlas.agent.tools.apply_patch import ApplyPatchTool
from atlas.agent.tools.write_file import WriteTool
from atlas.cli import tui as cli_tui


class _QueueIn:
    def __init__(self) -> None:
        self._q: queue.Queue[str | None] = queue.Queue()

    def push(self, line: str) -> None:
        self._q.put(line + "\n")

    def close(self) -> None:
        self._q.put(None)

    def __iter__(self) -> _QueueIn:
        return self

    def __next__(self) -> str:
        item = self._q.get()
        if item is None:
            raise StopIteration
        return item


class _CollectOut:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._lines: list[str] = []

    def write(self, text: str) -> int:
        with self._lock:
            self._lines.append(text)
        return len(text)

    def flush(self) -> None:
        return None

    def events(self) -> list[dict[str, Any]]:
        with self._lock:
            raw = "".join(self._lines)
        return [json.loads(line) for line in raw.splitlines() if line.strip()]


def _wait(predicate: Any, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("timed out waiting for condition")


class _ScriptedModel:
    def __init__(self, responses: list[ModelResponse | Exception]) -> None:
        self.responses = list(responses)
        self.requests: list[list[ChatMessage]] = []
        self.config = ModelConfig(model="scripted/model", api_key="sk-or-test")

    def complete(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition],
        *,
        on_content_delta: Any = None,
        on_thought_delta: Any = None,
        should_cancel: Any = None,
    ) -> ModelResponse:
        self.requests.append(list(messages))
        nxt = self.responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        if nxt.reasoning and on_thought_delta is not None:
            on_thought_delta(nxt.reasoning)
        if nxt.content and on_content_delta is not None:
            on_content_delta(nxt.content)
        return nxt


class _NoopInput(BaseModel):
    value: str = ""


class _MutatingStubTool(Tool):
    input_model = _NoopInput

    def __init__(self, tool_name: str) -> None:
        self.name = tool_name
        self.description = f"Stub tool {tool_name}"
        self.calls: list[dict[str, Any]] = []

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        self.calls.append(dict(arguments))
        return ToolResult(text=f"ran {self.name}")


class _StreamingTool(Tool):
    name = "stream_echo"
    description = "Emits incremental tool output and honors cancellation."
    input_model = _NoopInput

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        emit_tool_output("chunk-1\n")
        emit_tool_output("chunk-2\n")
        cancelled = bool(current_cancel_event() and current_cancel_event().is_set())
        return ToolResult(text=f"done cancelled={cancelled}", summary="streamed chunks")


# ---------------------------------------------------------------------------
# 1. --resume workspace ordering, rebinding, no duplicate session, replay
# ---------------------------------------------------------------------------


def test_cli_resume_rebinds_state_skips_new_session_and_replays_transcript(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    home = tmp_path / "home"
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-resumekey1234567890")
    monkeypatch.setattr("atlas.agent.session.Path.home", lambda: home)

    seed_model = _ScriptedModel(
        [
            ModelResponse(
                reasoning="checking note",
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="write_file",
                        arguments={"path": "note.txt", "content": "saved"},
                    )
                ],
            ),
            ModelResponse(content="All done."),
        ]
    )
    reg = ToolRegistry()
    reg.register(WriteTool())
    seed_session = AgentSession(Agent(seed_model, reg, LocalArtifactStore(workspace)))
    seed_session.workspace = workspace
    seed_session.session_id = "orig1234"
    seed_session.model_id = "vendor/saved-model"
    seed_session.totals.model = "vendor/saved-model"
    seed_session.set_goal("finish flood map")
    seed_session.active_skills = ["flood"]
    seed_session.turn("write note")
    seed_session.save_state()

    sessions_before = sorted(p.name for p in (workspace / ".atlas" / "sessions").iterdir())
    assert sessions_before == ["orig1234"]

    monkeypatch.setattr(sys, "stdin", io.StringIO('{"type":"quit"}\n'))
    code = main(["--workspace", str(workspace), "--resume", "orig1234"])
    assert code == 0

    sessions_after = sorted(p.name for p in (workspace / ".atlas" / "sessions").iterdir())
    assert sessions_after == ["orig1234"]

    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert events[0]["type"] == "ready"
    assert events[0]["session_id"] == "orig1234"
    assert events[0]["model"] == "vendor/saved-model"
    assert events[0]["goal"] == "finish flood map"
    assert events[0]["active_skills"] == ["flood"]

    assert events[1]["type"] == "session_loaded"
    assert events[1]["session_id"] == "orig1234"
    assert events[1]["model"] == "vendor/saved-model"
    assert events[1]["goal"] == "finish flood map"
    assert events[1]["active_skills"] == ["flood"]
    roles = [m["role"] for m in events[1]["messages"]]
    assert roles == ["user", "thought", "tool", "assistant"]
    assert events[1]["messages"][0]["text"] == "write note"
    assert events[1]["messages"][1]["text"] == "checking note"
    assert events[1]["messages"][2]["name"] == "write_file"
    assert events[1]["messages"][2]["ok"] is True
    assert events[1]["messages"][3]["text"] == "All done."


# ---------------------------------------------------------------------------
# 2. cancel immediately wakes blocked permission_request (no 300s hang)
# ---------------------------------------------------------------------------


def test_cancel_wakes_blocked_permission_prompt_immediately(tmp_path: Path) -> None:
    model = _ScriptedModel(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="p1",
                        name="write_file",
                        arguments={"path": "out.txt", "content": "nope"},
                    )
                ]
            ),
            ModelResponse(content="should not continue"),
        ]
    )
    reg = ToolRegistry()
    reg.register(WriteTool())
    session = AgentSession(Agent(model, reg, LocalArtifactStore(tmp_path), approval="ask"))
    stdin, stdout = _QueueIn(), _CollectOut()
    t = threading.Thread(target=serve, args=(session, stdin, stdout), daemon=True)
    t.start()
    started = time.monotonic()
    try:
        _wait(lambda: any(e["type"] == "ready" for e in stdout.events()))
        stdin.push('{"type":"user","text":"write out.txt"}')
        _wait(lambda: any(e["type"] == "permission_request" for e in stdout.events()))
        stdin.push('{"type":"cancel"}')
        _wait(lambda: any(e["type"] == "done" for e in stdout.events()))
    finally:
        stdin.push('{"type":"quit"}')
        t.join(timeout=2.0)

    elapsed = time.monotonic() - started
    assert elapsed < 2.0
    assert not (tmp_path / "out.txt").exists()
    done = next(e for e in stdout.events() if e["type"] == "done")
    assert done["cancelled"] is True


# ---------------------------------------------------------------------------
# 3. permission allow_session & deny with reason
# ---------------------------------------------------------------------------


def test_permission_allow_session_and_deny_with_reason(tmp_path: Path) -> None:
    prompts: list[PermissionRequest] = []

    def approver(req: PermissionRequest) -> PermissionDecision:
        prompts.append(req)
        if req.arguments.get("path") == "blocked.txt":
            return PermissionDecision(decision="deny", reason="protected artifact")
        return PermissionDecision(decision="allow_session")

    model = _ScriptedModel(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="write_file",
                        arguments={"path": "blocked.txt", "content": "x"},
                    )
                ]
            ),
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="c2",
                        name="write_file",
                        arguments={"path": "a.txt", "content": "one"},
                    )
                ]
            ),
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="c3",
                        name="write_file",
                        arguments={"path": "b.txt", "content": "two"},
                    )
                ]
            ),
            ModelResponse(content="done"),
        ]
    )
    reg = ToolRegistry()
    reg.register(WriteTool())
    session = AgentSession(
        Agent(model, reg, LocalArtifactStore(tmp_path), approval="ask", on_permission=approver)
    )

    run = session.turn("write files")
    assert run.response == "done"
    assert len(prompts) == 2
    assert prompts[0].preview_kind == "diff"
    assert run.steps[0].error_kind == "denied"
    assert "protected artifact" in (run.steps[0].error or "")
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "one"
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "two"
    assert "write_file" in session.agent.session_allowed_tools

    session.start_fresh()
    assert session.agent.session_allowed_tools == set()


# ---------------------------------------------------------------------------
# 4. Expanded _is_mutating(call) governance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "arguments", "expected"),
    [
        ("apply_patch", {"patch": ""}, True),
        ("bash", {"command": "ls"}, True),
        ("edit_file", {"path": "a.py"}, True),
        ("patch_file", {"path": "a.py"}, True),
        ("write_file", {"path": "a.py"}, True),
        ("git_commit", {"message": "msg"}, True),
        ("run_tests", {"command": ["pytest"]}, True),
        ("run_tests", {}, False),
        ("spawn_agent", {"task": "fix", "allow_writes": True}, True),
        ("spawn_agent", {"task": "inspect", "allow_writes": False}, False),
        ("render_map", {"path": "out.png"}, True),
        ("render_map", {}, False),
        ("spectral_indices", {"overwrite": True}, True),
        ("spectral_indices", {"overwrite": False}, False),
        ("read_file", {"path": "a.py"}, False),
    ],
)
def test_is_mutating_gating_matrix(name: str, arguments: dict[str, Any], expected: bool) -> None:
    assert _is_mutating(ToolCall(id="t1", name=name, arguments=arguments)) is expected


def test_read_only_blocks_mutating_variants_and_allows_readonly_variants(tmp_path: Path) -> None:
    reg = ToolRegistry()
    run_tests_stub = _MutatingStubTool("run_tests")
    git_commit_stub = _MutatingStubTool("git_commit")
    reg.register(run_tests_stub)
    reg.register(git_commit_stub)
    model = _ScriptedModel(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(id="c1", name="git_commit", arguments={"message": "wip"}),
                    ToolCall(id="c2", name="run_tests", arguments={"command": ["rm", "-rf"]}),
                    ToolCall(id="c3", name="run_tests", arguments={}),
                ]
            ),
            ModelResponse(content="finished"),
        ]
    )
    agent = Agent(model, reg, LocalArtifactStore(tmp_path), read_only=True)
    run = agent.run("check")

    assert run.steps[0].error_kind == "denied"
    assert run.steps[1].error_kind == "denied"
    assert run.steps[2].result is not None
    assert git_commit_stub.calls == []
    assert run_tests_stub.calls == [{}]


# ---------------------------------------------------------------------------
# 5. Multi-file apply_patch checkpointing & checkpoints store.resolve()
# ---------------------------------------------------------------------------


def test_multifile_apply_patch_checkpoints_and_restores_atomically(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("alpha\n", encoding="utf-8")
    patch = (
        "--- a/a.txt\n"
        "+++ b/a.txt\n"
        "@@ -1 +1 @@\n"
        "-alpha\n"
        "+ALPHA\n"
        "--- /dev/null\n"
        "+++ b/b.txt\n"
        "@@ -0,0 +1 @@\n"
        "+brand new\n"
    )
    reg = ToolRegistry()
    reg.register(ApplyPatchTool())
    model = _ScriptedModel(
        [
            ModelResponse(
                tool_calls=[ToolCall(id="p1", name="apply_patch", arguments={"patch": patch})]
            ),
            ModelResponse(content="patched"),
        ]
    )
    store = LocalArtifactStore(tmp_path)
    run = Agent(model, reg, store).run("patch both")
    assert run.steps[0].result is not None
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "ALPHA\n"
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "brand new\n"

    first = restore_latest(store)
    second = restore_latest(store)
    assert {first, second} == {"a.txt", "b.txt"}
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "alpha\n"
    assert not (tmp_path / "b.txt").exists()


def test_snapshot_many_restores_multiple_files_atomically(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("alpha\n", encoding="utf-8")
    store = LocalArtifactStore(tmp_path)
    snapshot_many(store, ["a.txt", "b.txt"])
    (tmp_path / "a.txt").write_text("ALPHA\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("brand new\n", encoding="utf-8")

    restored = restore_latest(store)
    assert restored is not None
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "alpha\n"
    assert not (tmp_path / "b.txt").exists()


def test_checkpoint_snapshot_uses_write_root_not_read_root(tmp_path: Path) -> None:
    read_dir = tmp_path / "read"
    write_dir = tmp_path / "write"
    read_dir.mkdir()
    write_dir.mkdir()
    (read_dir / "target.txt").write_text("from-read", encoding="utf-8")
    (write_dir / "target.txt").write_text("from-write", encoding="utf-8")

    store = LocalArtifactStore(write_dir, read_root=read_dir)
    snapshot(store, "target.txt")
    (write_dir / "target.txt").write_text("mutated", encoding="utf-8")

    assert restore_latest(store) is not None
    assert (write_dir / "target.txt").read_text(encoding="utf-8") == "from-write"


# ---------------------------------------------------------------------------
# 6. Subagent shared budget, cancel propagation, and call folding
# ---------------------------------------------------------------------------


def test_subagent_folds_calls_and_enforces_shared_spend_budget(tmp_path: Path) -> None:
    child_model = _ScriptedModel(
        [
            ModelResponse(
                content="child done",
                usage=TokenUsage(
                    prompt_tokens=100, completion_tokens=50, total_tokens=150, cost=0.04
                ),
            )
        ]
    )
    parent_model = _ScriptedModel(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="sub1",
                        name="spawn_agent",
                        arguments={"task": "inspect workspace", "allow_writes": False},
                    )
                ],
                usage=TokenUsage(
                    prompt_tokens=200, completion_tokens=20, total_tokens=220, cost=0.02
                ),
            ),
            ModelResponse(content="should stop for budget before calling parent again"),
        ]
    )

    def make_child(child_store: LocalArtifactStore, allow_writes: bool, budget: int) -> Agent:
        return Agent(
            child_model,
            ToolRegistry(),
            child_store,
            max_tool_calls=budget,
            read_only=not allow_writes,
        )

    reg = ToolRegistry()
    reg.register(SubagentTool(make_child))
    session = AgentSession(
        Agent(parent_model, reg, LocalArtifactStore(tmp_path), max_spend_usd=0.05)
    )

    run = session.turn("delegate")
    assert run.stopped_for_budget is True
    assert len(parent_model.requests) == 1
    assert len(run.calls) == 2
    assert session.totals.total_tokens == 370
    assert session.status_snapshot()["spend"] == pytest.approx(0.06)


def test_subagent_inherits_cancel_event(tmp_path: Path) -> None:
    cancel = threading.Event()
    child_model = _ScriptedModel([ModelResponse(content="never called")])

    def make_child(child_store: LocalArtifactStore, allow_writes: bool, budget: int) -> Agent:
        cancel.set()
        return Agent(
            child_model, ToolRegistry(), child_store, max_tool_calls=budget, read_only=True
        )

    reg = ToolRegistry()
    reg.register(SubagentTool(make_child))
    parent_model = _ScriptedModel(
        [
            ModelResponse(
                tool_calls=[ToolCall(id="sub1", name="spawn_agent", arguments={"task": "work"})]
            ),
            ModelResponse(content="done"),
        ]
    )
    agent = Agent(parent_model, reg, LocalArtifactStore(tmp_path))
    run = agent.advance(
        [*agent.opening_messages(), ChatMessage(role=Role.USER, content="go")],
        cancel_event=cancel,
    )
    assert child_model.requests == []
    assert run.cancelled is True


# ---------------------------------------------------------------------------
# 7. Malformed tool-call JSON feeds validation error back to model
# ---------------------------------------------------------------------------


def test_malformed_tool_call_json_recovers_via_validation_error(tmp_path: Path) -> None:
    responses = [
        httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "bad1",
                                    "type": "function",
                                    "function": {"name": "write_file", "arguments": '{"path": "x'},
                                }
                            ],
                        }
                    }
                ]
            },
        ),
        httpx.Response(
            200,
            json={"choices": [{"message": {"content": "recovered", "tool_calls": []}}]},
        ),
    ]
    transport = httpx.MockTransport(lambda _req: responses.pop(0))
    client = httpx.Client(transport=transport)
    try:
        model = OpenRouterModel(ModelConfig(model="m", api_key="sk-or-test"), client)
        reg = ToolRegistry()
        reg.register(WriteTool())
        run = Agent(model, reg, LocalArtifactStore(tmp_path)).run("try writing")
    finally:
        client.close()

    assert run.response == "recovered"
    assert run.steps[0].error_kind == "validation"
    assert "arguments were not valid JSON" in (run.steps[0].error or "")


# ---------------------------------------------------------------------------
# 8. Mid-turn HTTP error resets analysis state and lets the next turn succeed
# ---------------------------------------------------------------------------


def test_mid_turn_http_error_resets_analysis_state_cleanly(tmp_path: Path) -> None:
    model = _ScriptedModel(
        [
            OpenRouterHTTPError(500, "upstream failure", 0.2),
            ModelResponse(content="clean second turn"),
        ]
    )
    session = AgentSession(Agent(model, ToolRegistry(), LocalArtifactStore(tmp_path)))

    with pytest.raises(OpenRouterHTTPError):
        session.turn("broken turn")

    assert session._analysis_open is False

    run = session.turn("healthy turn")
    assert run.response == "clean second turn"


# ---------------------------------------------------------------------------
# 9. Real SSE streaming via httpx-sse, Retry-After, and stream cancellation
# ---------------------------------------------------------------------------


def test_real_httpx_sse_streaming_emits_deltas_and_honors_retry_after(tmp_path: Path) -> None:
    sse_body = (
        'data: {"choices":[{"delta":{"reasoning":"plan "}}]}\n\n'
        'data: {"choices":[{"delta":{"reasoning":"step"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
        'data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":4,"total_tokens":16}}\n\n'
        "data: [DONE]\n\n"
    )
    calls = 0

    def handler(_req: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "0.02"}, text="rate limited")
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=sse_body.encode("utf-8"),
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        model = OpenRouterModel(ModelConfig(model="stream/m", api_key="sk-or-test"), client)
        session = AgentSession(Agent(model, ToolRegistry(), LocalArtifactStore(tmp_path)))
        out = io.StringIO()
        serve(session, io.StringIO('{"type":"user","text":"hi"}\n'), out)
    finally:
        client.close()

    assert calls == 2
    events = [json.loads(line) for line in out.getvalue().splitlines()]
    assert [e["text"] for e in events if e["type"] == "thought_delta"] == ["plan ", "step"]
    assert [e["text"] for e in events if e["type"] == "content_delta"] == ["Hel", "lo"]
    assert any(e["type"] == "status" and e["status"]["prompt_tokens"] == 12 for e in events)
    done = next(e for e in events if e["type"] == "done")
    assert done["response"] == "Hello"
    assert done["thoughts"] == "plan step"


def test_stream_cancellation_raises_and_stops_promptly() -> None:
    cancel = False
    sse_body = (
        'data: {"choices":[{"delta":{"content":"part1"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"part2"}}]}\n\n'
        "data: [DONE]\n\n"
    )

    def on_delta(_chunk: str) -> None:
        nonlocal cancel
        cancel = True

    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda _req: httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=sse_body.encode()
            )
        )
    )
    try:
        model = OpenRouterModel(ModelConfig(model="stream/m", api_key="sk-or-test"), client)
        with pytest.raises(StreamCancelledError):
            model.complete(
                [ChatMessage(role=Role.USER, content="hi")],
                [],
                on_content_delta=on_delta,
                should_cancel=lambda: cancel,
            )
    finally:
        client.close()


# ---------------------------------------------------------------------------
# 10. Non-blocking RPC, tool_start / tool_output_delta, and fast quit
# ---------------------------------------------------------------------------


def test_nonblocking_rpc_handles_yolo_during_turn_and_tool_streaming(tmp_path: Path) -> None:
    gate = threading.Event()
    entered = threading.Event()

    class _SlowModel:
        config = ModelConfig(model="slow/m", api_key="sk-or-test")

        def __init__(self) -> None:
            self.n = 0

        def complete(
            self, messages: list[ChatMessage], tools: list[ToolDefinition]
        ) -> ModelResponse:
            self.n += 1
            if self.n == 1:
                entered.set()
                gate.wait(timeout=3.0)
                return ModelResponse(
                    tool_calls=[ToolCall(id="s1", name="stream_echo", arguments={})]
                )
            return ModelResponse(content="all streamed")

    reg = ToolRegistry()
    reg.register(_StreamingTool())
    session = AgentSession(Agent(_SlowModel(), reg, LocalArtifactStore(tmp_path), approval="ask"))
    stdin, stdout = _QueueIn(), _CollectOut()
    t = threading.Thread(target=serve, args=(session, stdin, stdout), daemon=True)
    t.start()
    try:
        _wait(lambda: any(e["type"] == "ready" for e in stdout.events()))
        stdin.push('{"type":"user","text":"go"}')
        assert entered.wait(timeout=2.0)

        stdin.push('{"type":"yolo","enabled":true}')
        _wait(lambda: any(e["type"] == "yolo" and e["enabled"] is True for e in stdout.events()))

        gate.set()
        _wait(lambda: any(e["type"] == "done" for e in stdout.events()))
    finally:
        stdin.push('{"type":"quit"}')
        t.join(timeout=2.0)

    events = stdout.events()
    assert any(e["type"] == "tool_start" and e["name"] == "stream_echo" for e in events)
    chunks = [e["chunk"] for e in events if e["type"] == "tool_output_delta"]
    assert chunks == ["chunk-1\n", "chunk-2\n"]


def test_quit_does_not_block_on_slow_catalog_fetch(tmp_path: Path) -> None:
    started = threading.Event()
    release = threading.Event()

    def slow_fetch() -> list[CatalogModel]:
        started.set()
        release.wait(timeout=10.0)
        return []

    session = AgentSession(Agent(_ScriptedModel([]), ToolRegistry(), LocalArtifactStore(tmp_path)))
    session.fetch_models = slow_fetch
    stdin, stdout = _QueueIn(), _CollectOut()
    t = threading.Thread(target=serve, args=(session, stdin, stdout), daemon=True)
    t.start()
    try:
        _wait(lambda: any(e["type"] == "ready" for e in stdout.events()))
        stdin.push('{"type":"models"}')
        assert started.wait(timeout=2.0)
        t0 = time.monotonic()
        stdin.push('{"type":"quit"}')
        t.join(timeout=2.0)
        assert not t.is_alive()
        assert time.monotonic() - t0 < 1.0
    finally:
        release.set()


# ---------------------------------------------------------------------------
# 11. Overflow compact-and-retry, /compact RPC, and >20 KB spill keeping stderr
# ---------------------------------------------------------------------------


def test_overflow_error_triggers_compact_and_retry(tmp_path: Path) -> None:
    model = _ScriptedModel(
        [
            OpenRouterHTTPError(400, "This model's maximum context length is 4096 tokens", 0.1),
            ModelResponse(content="succeeded after compaction"),
        ]
    )
    agent = Agent(model, ToolRegistry(), LocalArtifactStore(tmp_path))
    messages = [
        ChatMessage(role=Role.SYSTEM, content="sys"),
        ChatMessage(role=Role.USER, content="start"),
    ]
    for i in range(10):
        tc = ToolCall(id=f"c{i}", name="echo", arguments={})
        messages.extend(
            [
                ChatMessage(role=Role.ASSISTANT, tool_calls=[tc]),
                ChatMessage(
                    role=Role.TOOL,
                    tool_call_id=tc.id,
                    content=ToolResult(text="x" * 2000).model_dump_json(),
                ),
            ]
        )
    messages.append(ChatMessage(role=Role.USER, content="final question"))

    run = agent.advance(messages)
    assert run.response == "succeeded after compaction"
    assert run.compaction.occurred is True
    assert len(model.requests) == 2


def test_tool_output_spill_saves_file_and_preserves_stderr(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    big_stdout = "OUTLINE\n" * 4000
    stderr_section = "\nstderr:\nTraceback: critical failure on line 42\n"
    raw = big_stdout + stderr_section
    assert len(raw) > MAX_TOOL_RESULT_CHARS

    step = AgentStep(
        call=ToolCall(id="call/99", name="bash_tool", arguments={}), result=ToolResult(text=raw)
    )
    encoded = _tool_message(step, store)
    visible = ToolResult.model_validate_json(encoded)

    assert "Traceback: critical failure on line 42" in visible.text
    spilled_files = list((tmp_path / ".atlas" / "tool-output").glob("call99-*.txt"))
    assert len(spilled_files) == 1
    assert f".atlas/tool-output/{spilled_files[0].name}" in visible.text
    assert spilled_files[0].read_text(encoding="utf-8") == raw

    # Also verify stderr-before-stdout ordering preserves stderr in head.
    stderr_first = "exit_code: 1\nstderr:\nFATAL: bad flag\nstdout:\n" + big_stdout
    step2 = AgentStep(
        call=ToolCall(id="call_100", name="bash", arguments={}),
        result=ToolResult(text=stderr_first),
    )
    visible2 = ToolResult.model_validate_json(_tool_message(step2, store))
    assert "FATAL: bad flag" in visible2.text
    spilled_files2 = list((tmp_path / ".atlas" / "tool-output").glob("call_100-*.txt"))
    assert len(spilled_files2) == 1
    assert f".atlas/tool-output/{spilled_files2[0].name}" in visible2.text


def test_rpc_compact_command_emits_compacted_with_status(tmp_path: Path) -> None:
    session = AgentSession(Agent(_ScriptedModel([]), ToolRegistry(), LocalArtifactStore(tmp_path)))
    for i in range(10):
        tc = ToolCall(id=f"c{i}", name="echo", arguments={})
        session.messages.extend(
            [
                ChatMessage(role=Role.ASSISTANT, tool_calls=[tc]),
                ChatMessage(
                    role=Role.TOOL,
                    tool_call_id=tc.id,
                    content=ToolResult(text="z" * 3000).model_dump_json(),
                ),
            ]
        )
    out = io.StringIO()
    serve(session, io.StringIO('{"type":"compact"}\n'), out)
    events = [json.loads(line) for line in out.getvalue().splitlines()]
    compacted = next(e for e in events if e["type"] == "compacted")
    assert compacted["compaction"]["occurred"] is True
    assert compacted["compaction"]["tokens_reclaimed"] > 0
    assert "context_used" in compacted["status"]


# ---------------------------------------------------------------------------
# 12. ToolRegistry caching & JSON-schema noise stripping
# ---------------------------------------------------------------------------


class _NestedItem(BaseModel):
    """Nested model docstring that should be stripped."""

    label: str = Field(description="Keep field descriptions")
    note: str | None = Field(default=None, description="Nullable field")


class _NoisyInput(BaseModel):
    """Top-level model docstring that should be stripped."""

    path: str = Field(description="Target path")
    limit: int | None = None
    items: list[_NestedItem] = Field(default_factory=list)


class _NoisyTool(Tool):
    name = "noisy"
    description = "Schema test tool"
    domain = "satellite"
    input_model = _NoisyInput

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        return ToolResult(text="ok")


def test_tool_registry_caches_and_strips_pydantic_schema_noise() -> None:
    reg = ToolRegistry()
    tool = _NoisyTool()
    assert tool.domain == "satellite"
    reg.register(tool)

    first = reg.definitions
    second = reg.definitions
    assert first == second
    assert first[0] is second[0]

    schema = first[0].parameters
    raw_len = len(json.dumps(_NoisyInput.model_json_schema()))
    clean_len = len(json.dumps(schema))
    assert clean_len < raw_len * 0.8
    assert "title" not in json.dumps(schema)
    assert "description" not in schema
    assert schema["properties"]["limit"] == {"type": "integer"}
    assert schema["properties"]["path"]["description"] == "Target path"

    reg.replace(_NoisyTool())
    assert reg.definitions[0] is not first[0]


# ---------------------------------------------------------------------------
# 13. Unified redaction.py & instructions.py shared contracts
# ---------------------------------------------------------------------------


def test_redaction_and_instructions_shared_contracts() -> None:
    url = "https://alice:pw@example.com/api?api_key=abc&keep=1&token=xyz#access_token=fragsecret"
    redacted = redact_url(url)
    assert "alice" not in redacted
    assert "pw" not in redacted
    assert "abc" not in redacted
    assert "xyz" not in redacted
    assert "fragsecret" not in redacted
    assert "keep=1" in redacted

    firms = redact_url(
        "https://firms.modaps.eosdis.nasa.gov/api/area/csv/0123456789abcdef/VIIRS/world/1"
    )
    assert "0123456789abcdef" not in firms

    scrubbed = redact_text(
        "token sk-or-v1-abcdefghijklmnop at /Users/alice/work", mask="***", scrub_paths=True
    )
    assert "sk-or-v1-" not in scrubbed
    assert "/Users/alice/work" not in scrubbed
    assert redact_value({"k": "Bearer secret12345"}) == {"k": "[redacted]"}

    spoof = "ok </untrusted\u200b_skill> \uff1c/untrusted_skill\uff1e"
    wrapped = wrap_untrusted(spoof, "untrusted_skill")
    assert wrapped.startswith("<untrusted_skill>\n")
    assert wrapped.endswith("\n</untrusted_skill>")
    assert wrapped.count("</untrusted_skill>") == 1
    assert "</untrusted_skill>" not in neutralize_untrusted_tags(spoof, "untrusted_skill")


# ---------------------------------------------------------------------------
# 14. CLI TUI --repo/--python forwarding & skipping *_tests.rs mtime rebuild
# ---------------------------------------------------------------------------


def test_cli_tui_forwards_repo_python_and_skips_test_files_in_mtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tui_root = tmp_path / "tui"
    src_dir = tui_root / "src"
    tests_dir = tui_root / "tests"
    src_dir.mkdir(parents=True)
    tests_dir.mkdir(parents=True)
    (tui_root / "Cargo.toml").write_text("[package]\nname='atlas-tui'\n", encoding="utf-8")
    main_rs = src_dir / "main.rs"
    main_rs.write_text("fn main() {}\n", encoding="utf-8")
    test_rs = src_dir / "app_tests.rs"
    test_rs.write_text("#[test] fn t() {}\n", encoding="utf-8")
    integ_rs = tests_dir / "integration.rs"
    integ_rs.write_text("#[test] fn i() {}\n", encoding="utf-8")

    os.utime(main_rs, (1000, 1000))
    os.utime(tui_root / "Cargo.toml", (1000, 1000))
    os.utime(test_rs, (5000, 5000))
    os.utime(integ_rs, (6000, 6000))

    assert cli_tui._source_mtime(tmp_path) == 1000.0

    fake_bin = tui_root / "target" / "release" / "atlas-tui"
    fake_bin.parent.mkdir(parents=True)
    fake_bin.write_text("", encoding="utf-8")
    os.utime(fake_bin, (2000, 2000))

    captured: dict[str, Any] = {}

    class _ExecIntercepted(BaseException):
        pass

    def fake_execv(path: str, argv: list[str]) -> None:
        captured["path"] = path
        captured["argv"] = argv
        captured["env_python"] = os.environ.get("ATLAS_PYTHON")
        raise _ExecIntercepted

    monkeypatch.delenv("ATLAS_PYTHON", raising=False)
    monkeypatch.setattr(cli_tui, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(cli_tui, "ensure_tui_binary", lambda: fake_bin)
    monkeypatch.setattr(cli_tui.os, "execv", fake_execv)

    with pytest.raises(_ExecIntercepted):
        cli_tui.launch_tui(["--workspace", str(tmp_path / "other")])

    assert captured["path"] == str(fake_bin)
    assert "--repo" in captured["argv"]
    assert str(tmp_path) in captured["argv"]
    assert "--python" in captured["argv"]
    assert sys.executable in captured["argv"]
    assert captured["env_python"] == sys.executable


def test_cli_lazy_imports_skip_heavy_subcommands_for_tui_invocation() -> None:
    import subprocess

    code = (
        "import sys, atlas.cli, atlas.cli.tui\n"
        "atlas.cli.tui.launch_tui = lambda argv: 0\n"
        "assert atlas.cli.main(['--workspace', '.']) == 0\n"
        "for mod in ('atlas.cli.data', 'atlas.cli.evals', 'atlas.cli.indices', 'numpy', 'scipy'):\n"
        "    assert mod not in sys.modules, f'eagerly imported {mod}'\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr


# ---------------------------------------------------------------------------
# 16. Unified v2 + 0.17.0 RPC extensions (sessions list, load last, bash streaming)
# ---------------------------------------------------------------------------


def test_rpc_list_sessions_and_load_last_emit_metadata_and_flags(tmp_path: Path) -> None:
    first = AgentSession(
        Agent(
            _ScriptedModel([ModelResponse(content="first done")]),
            ToolRegistry(),
            LocalArtifactStore(tmp_path),
        )
    )
    first.workspace = tmp_path
    first.session_id = "aaaa1111bbbb"
    first.model_id = "vendor/first"
    first.turn("inspect coastline")
    first.save_state()

    second = AgentSession(Agent(_ScriptedModel([]), ToolRegistry(), LocalArtifactStore(tmp_path)))
    second.workspace = tmp_path
    second.session_id = "cccc2222dddd"
    second.model_id = "vendor/second"

    out = io.StringIO()
    serve(
        second,
        io.StringIO(
            '{"type":"list","kind":"sessions"}\n'
            '{"type":"load_session","id":"last"}\n'
            '{"type":"quit"}\n'
        ),
        out,
    )
    events = [json.loads(line) for line in out.getvalue().splitlines()]
    listing = next(e for e in events if e["type"] == "list")
    assert listing["kind"] == "sessions"
    assert "aaaa1111bbbb" in listing["items"]
    assert listing["sessions"][0]["id"] == "aaaa1111bbbb"
    assert listing["sessions"][0]["prompt"] == "inspect coastline"

    loaded = next(e for e in events if e["type"] == "session_loaded")
    assert loaded["session_id"] == "aaaa1111bbbb"
    assert loaded["stopped_for_limit"] is False
    assert loaded["interrupted"] is False
    assert loaded["omitted"] == 0


def test_rpc_bash_streams_tool_output_delta_and_step_done_v2_017_fields(tmp_path: Path) -> None:
    from atlas.agent.tools.BashTool import BashTool

    reg = ToolRegistry()
    reg.register(BashTool())
    model = _ScriptedModel(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="b1",
                        name="bash",
                        arguments={
                            "command": "printf 'streamed-line-1\\n'; sleep 0.15; printf 'streamed-line-2\\n'"
                        },
                    )
                ]
            ),
            ModelResponse(content="finished bash"),
        ]
    )
    session = AgentSession(Agent(model, reg, LocalArtifactStore(tmp_path)))
    out = io.StringIO()
    serve(session, io.StringIO('{"type":"user","text":"run bash"}\n'), out)
    events = [json.loads(line) for line in out.getvalue().splitlines()]

    deltas = [e for e in events if e["type"] == "tool_output_delta" and e["call_id"] == "b1"]
    combined = "".join(e["chunk"] for e in deltas)
    assert "streamed-line-1" in combined and "streamed-line-2" in combined

    step = next(e for e in events if e["type"] == "step")
    assert step["name"] == "bash"
    assert step["ok"] is True
    assert step["artifacts"] == []
    assert step["notes"] == []
    assert step["error_kind"] is None

    done = next(e for e in events if e["type"] == "done")
    assert done["response"] == "finished bash"
    assert done["compaction"]["occurred"] is False
    assert done["empty_replies"] == 0
    assert done["intent_nudges"] == 0
