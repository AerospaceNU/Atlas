from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from atlas.agent import runtime
from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import (
    ChatMessage,
    ModelResponse,
    Role,
    ToolCall,
    ToolDefinition,
    ToolRegistry,
    ToolResult,
)
from atlas.agent.runtime import Agent


class _Overflow(Exception):
    is_context_overflow = True


class _Plain(Exception):
    pass


class FlakyModel:
    def __init__(self, script: list[ModelResponse | Exception]) -> None:
        self.script = script
        self.requests: list[list[ChatMessage]] = []

    def complete(self, messages: list[ChatMessage], tools: list[ToolDefinition]) -> ModelResponse:
        self.requests.append(list(messages))
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _exchange(
    index: int,
    name: str = "run_tests",
    args: dict[str, Any] | None = None,
    text: str | None = None,
    artifacts: list[str] | None = None,
) -> list[ChatMessage]:
    call = ToolCall(id=f"c{index}", name=name, arguments=args or {"command": ["pytest"]})
    result = ToolResult(
        text=text or f"command: pytest\nexit_code: 1\n{'x' * 4000}",
        artifacts=artifacts or [f"out{index}.txt"],
    )
    return [
        ChatMessage(role=Role.ASSISTANT, tool_calls=[call]),
        ChatMessage(role=Role.TOOL, tool_call_id=call.id, content=result.model_dump_json()),
    ]


def _history(count: int = 30) -> list[ChatMessage]:
    messages = [
        ChatMessage(role=Role.SYSTEM, content="system"),
        ChatMessage(role=Role.USER, content="first request"),
    ]
    for index in range(count):
        messages.extend(_exchange(index))
    messages.append(ChatMessage(role=Role.USER, content="latest request"))
    return messages


def _digest(messages: list[ChatMessage]) -> str:
    found = [m for m in messages if runtime._is_digest(m)]
    assert len(found) == 1
    return found[0].content or ""


def _agent(tmp_path: Path, model: FlakyModel) -> Agent:
    return Agent(model, ToolRegistry(), LocalArtifactStore(tmp_path))


def test_overflow_compacts_and_the_run_completes(tmp_path: Path) -> None:
    model = FlakyModel([_Overflow("too long"), ModelResponse(content="done")])
    messages = _history()
    before = runtime._estimate_tokens(messages)

    run = _agent(tmp_path, model).advance(messages)

    assert run.response == "done"
    assert not run.stopped_for_context
    assert run.compaction.occurred
    assert runtime._estimate_tokens(messages) < before
    assert len(model.requests) == 2
    assert runtime.MAX_CONTEXT_TOKENS == 1_000_000


def test_overflow_with_nothing_to_compact_stops_without_another_call(tmp_path: Path) -> None:
    model = FlakyModel([_Overflow("too long"), ModelResponse(content="never")])
    messages = [
        ChatMessage(role=Role.SYSTEM, content="system"),
        ChatMessage(role=Role.USER, content="request"),
    ]

    run = _agent(tmp_path, model).advance(messages)

    assert run.stopped_for_context
    assert len(model.requests) == 1


def test_overflow_retries_are_bounded(tmp_path: Path) -> None:
    model = FlakyModel([_Overflow("x") for _ in range(10)])
    messages = _history(120)

    run = _agent(tmp_path, model).advance(messages)

    assert run.stopped_for_context
    assert len(model.requests) == runtime._MAX_OVERFLOW_RETRIES + 1


def test_exception_without_the_attribute_propagates(tmp_path: Path) -> None:
    model = FlakyModel([_Plain("nope")])
    agent = _agent(tmp_path, model)
    with pytest.raises(_Plain):
        agent.advance(_history(3))
    assert agent.partial_run is not None
    assert agent.partial_run.error == "_Plain: nope"


def test_compact_limit_works_far_below_the_constant() -> None:
    messages = _history(30)
    assert runtime._estimate_tokens(messages) < runtime.MAX_CONTEXT_TOKENS
    assert not runtime._compact(messages).occurred

    result = runtime._compact(messages, limit=5000)

    assert result.occurred
    assert runtime.MAX_CONTEXT_TOKENS == 1_000_000


def test_earlier_user_requests_survive_a_drop() -> None:
    messages = [
        ChatMessage(role=Role.SYSTEM, content="system"),
        ChatMessage(role=Role.USER, content="first request"),
    ]
    for index in range(10):
        messages.extend(_exchange(index))
    messages.append(ChatMessage(role=Role.USER, content="second request"))
    for index in range(10, 20):
        messages.extend(_exchange(index))
    messages.append(ChatMessage(role=Role.USER, content="third request"))

    runtime._compact(messages, limit=3000)

    contents = [m.content for m in messages if m.role is Role.USER]
    assert contents == ["first request", "second request", "third request"]


def test_digest_has_edited_read_and_notes() -> None:
    messages = [
        ChatMessage(role=Role.SYSTEM, content="system"),
        ChatMessage(role=Role.USER, content="request"),
        ChatMessage(role=Role.ASSISTANT, content="I will fix the parser.\nSecond line."),
    ]
    messages.extend(_exchange(0, "read_files", {"path": "src/a.py"}, "a", []))
    messages.extend(_exchange(1, "edit_file", {"path": "src/a.py"}, "ok", []))
    messages.extend(_exchange(2, "write_file", {"path": "src/b.py"}, "ok", []))
    messages.extend(_exchange(3, "read_files", {"path": "src/c.py"}, "c" * 6000, []))
    for index in range(4, 12):
        messages.extend(_exchange(index))
    messages.append(ChatMessage(role=Role.USER, content="go on"))

    runtime._compact(messages, limit=2500)

    body = _digest(messages)
    sections = [line for line in body.splitlines() if not line.startswith("  ")]
    assert sections[1:4] == ["Edited:", "Read:", "Commands:"]
    assert "  src/a.py" in body.split("Read:")[0]
    assert "  src/a.py" not in body.split("Read:")[1]
    assert "  src/b.py" in body
    assert "  src/c.py" in body.split("Read:")[1]
    assert "Notes:\n  I will fix the parser." in body
    assert "Second line" not in body
    assert "  pytest -> exit 1" in body


def test_legacy_files_line_is_parsed_and_rolled_forward() -> None:
    legacy = f"{runtime._DIGEST_HEADER}: 4 messages compacted\nFiles: old/a.py, old/b.py"
    parsed = runtime._parse_digest(legacy)
    assert parsed.read == ["old/a.py", "old/b.py"]
    assert parsed.dropped == 4

    messages = [
        ChatMessage(role=Role.SYSTEM, content="system"),
        ChatMessage(role=Role.USER, content="request"),
        ChatMessage(role=Role.SYSTEM, content=legacy),
    ]
    for index in range(10):
        messages.extend(_exchange(index))
    messages.append(ChatMessage(role=Role.USER, content="go on"))

    runtime._compact(messages, limit=2500)

    body = _digest(messages)
    assert "  old/a.py" in body
    assert "Files:" not in body
    assert int(body.split(": ", 1)[1].split(" ", 1)[0]) > 4


def test_a_path_with_a_comma_round_trips() -> None:
    digest = runtime._Digest(dropped=2, edited=["dir/a, b.py"], read=["x,y.txt"])
    parsed = runtime._parse_digest(runtime._render_digest(digest))
    assert parsed.edited == ["dir/a, b.py"]
    assert parsed.read == ["x,y.txt"]


def test_trim_removes_the_oldest_entry_of_the_largest_section() -> None:
    digest = runtime._Digest(
        dropped=1,
        read=[f"file_{n:03d}_{'p' * 150}.py" for n in range(24)],
        commands=["cmd -> exit 0"],
    )
    runtime._trim_digest(digest)
    assert len(runtime._render_digest(digest)) <= runtime._DIGEST_LIMIT_CHARS
    assert digest.read[-1].startswith("file_023")
    assert not digest.read[0].startswith("file_000")
    assert digest.commands == ["cmd -> exit 0"]
