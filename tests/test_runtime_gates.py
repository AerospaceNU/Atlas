"""Mutation gates, apply_patch checkpoints, and the environment block."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from atlas.agent import runtime
from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.checkpoints import restore_latest
from atlas.agent.contracts import (
    ChatMessage,
    ModelResponse,
    ToolCall,
    ToolDefinition,
    ToolRegistry,
)
from atlas.agent.runtime import Agent, PermissionRequest
from atlas.agent.system_prompt import load_system_prompt
from atlas.agent.tools.apply_patch import ApplyPatchTool
from atlas.agent.tools.git import GitCommitTool
from atlas.agent.tools.run_tests import RunTestsTool


class _Scripted:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = responses

    def complete(self, messages: list[ChatMessage], tools: list[ToolDefinition]) -> ModelResponse:
        return self.responses.pop(0)


def _agent(tmp_path: Path, calls: list[ToolCall], **kwargs: Any) -> Agent:
    registry = ToolRegistry()
    for tool in (ApplyPatchTool(), GitCommitTool(), RunTestsTool()):
        registry.register(tool)
    model = _Scripted([ModelResponse(tool_calls=calls), ModelResponse(content="done")])
    return Agent(model, registry, LocalArtifactStore(tmp_path), **kwargs)


def test_git_commit_refused_when_read_only(tmp_path: Path) -> None:
    call = ToolCall(id="1", name="git_commit", arguments={"message": "m"})
    step = _agent(tmp_path, [call], read_only=True).run("go").steps[0]
    assert step.error_kind == "denied"
    assert "read-only" in (step.error or "")


def test_run_tests_with_command_is_gated(tmp_path: Path) -> None:
    call = ToolCall(id="1", name="run_tests", arguments={"command": ["echo", "hi"]})
    step = _agent(tmp_path, [call], read_only=True).run("go").steps[0]
    assert step.error_kind == "denied"

    seen: list[PermissionRequest] = []

    def deny(request: PermissionRequest) -> bool:
        seen.append(request)
        return False

    step = _agent(tmp_path, [call], approval="ask", on_permission=deny).run("go").steps[0]
    assert step.error_kind == "denied"
    assert seen[0].summary == "run: echo hi"
    assert seen[0].preview == "echo hi"


def test_git_commit_summary_shows_the_message(tmp_path: Path) -> None:
    seen: list[PermissionRequest] = []

    def deny(request: PermissionRequest) -> bool:
        seen.append(request)
        return False

    call = ToolCall(id="1", name="git_commit", arguments={"message": "fix bug"})
    _agent(tmp_path, [call], approval="ask", on_permission=deny).run("go")
    assert seen[0].summary == "git commit: fix bug"
    assert seen[0].preview == "fix bug"


def test_run_tests_without_command_is_not_gated(tmp_path: Path) -> None:
    call = ToolCall(id="1", name="run_tests", arguments={})
    step = _agent(tmp_path, [call], read_only=True).run("go").steps[0]
    assert step.error_kind != "denied"
    assert not runtime._is_mutating(call)


_PATCH = """--- a/one.txt
+++ b/one.txt
@@ -1 +1 @@
-one
+ONE
--- a/two.txt
+++ b/two.txt
@@ -1 +1 @@
-two
+TWO
"""


def test_apply_patch_is_checkpointed_for_every_file(tmp_path: Path) -> None:
    (tmp_path / "one.txt").write_text("one\n", encoding="utf-8")
    (tmp_path / "two.txt").write_text("two\n", encoding="utf-8")
    call = ToolCall(id="1", name="apply_patch", arguments={"patch": _PATCH})
    agent = _agent(tmp_path, [call])
    step = agent.run("go").steps[0]
    assert step.error is None
    assert (tmp_path / "one.txt").read_text(encoding="utf-8") == "ONE\n"
    restored = restore_latest(agent.artifacts)
    assert restored in {"one.txt", "two.txt"}
    assert (tmp_path / restored).read_text(encoding="utf-8") in {"one\n", "two\n"}
    assert restore_latest(agent.artifacts) in {"one.txt", "two.txt"}


def test_dry_run_takes_no_checkpoint(tmp_path: Path) -> None:
    (tmp_path / "one.txt").write_text("one\n", encoding="utf-8")
    patch = "--- a/one.txt\n+++ b/one.txt\n@@ -1 +1 @@\n-one\n+ONE\n"
    call = ToolCall(id="1", name="apply_patch", arguments={"patch": patch, "dry_run": True})
    agent = _agent(tmp_path, [call])
    agent.run("go")
    assert restore_latest(agent.artifacts) is None
    assert (tmp_path / "one.txt").read_text(encoding="utf-8") == "one\n"


def test_environment_context_in_first_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runtime, "_today", lambda: "2031-02-03")
    agent = _agent(tmp_path, [])
    context = agent.environment_context()
    lines = context.splitlines()
    assert lines[0] == "Environment"
    assert lines[1].startswith(f"- Working directory: {agent.artifacts.root} (relative")
    assert lines[2].startswith("- Shell: bash runs each command in a new shell")
    assert lines[3].startswith("- Platform: ")
    assert lines[4] == "- Date: 2031-02-03"
    first = agent.opening_messages()[0].content
    assert first == f"{agent.system_prompt()}\n\n{context}"
    assert agent.system_prompt() == load_system_prompt()


def test_environment_paths_are_sanitized(tmp_path: Path) -> None:
    assert runtime._clean_path(Path("a\nb")) == "a?b"
    assert len(runtime._clean_path(Path("x" * 500))) == 300
