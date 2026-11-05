from __future__ import annotations

import json
from pathlib import Path

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.checkpoints import restore_latest
from atlas.agent.contracts import (
    ChatMessage,
    ModelResponse,
    ToolCall,
    ToolDefinition,
    ToolRegistry,
)
from atlas.agent.runtime import Agent
from atlas.agent.tools.write_file import WriteTool


class _ScriptedModel:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = responses

    def complete(self, messages: list[ChatMessage], tools: list[ToolDefinition]) -> ModelResponse:
        return self.responses.pop(0)


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(WriteTool())
    return registry


def test_write_is_checkpointed_and_undo_restores(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    (tmp_path / "app.py").write_text("old\n", encoding="utf-8")
    from atlas.agent.checkpoints import snapshot

    snapshot(store, "app.py")
    WriteTool().run({"path": "app.py", "content": "new\n"}, store)

    assert (tmp_path / "app.py").read_text(encoding="utf-8") == "new\n"
    assert restore_latest(store) == "app.py"
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == "old\n"
    assert restore_latest(store) is None


def test_undo_removes_a_file_that_did_not_exist(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    from atlas.agent.checkpoints import snapshot

    snapshot(store, "created.py")
    WriteTool().run({"path": "created.py", "content": "x\n"}, store)

    assert restore_latest(store) == "created.py"
    assert not (tmp_path / "created.py").exists()


def test_read_only_session_refuses_a_write(tmp_path: Path) -> None:
    model = _ScriptedModel(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(id="a", name="write_file", arguments={"path": "x.py", "content": "y"})
                ]
            ),
            ModelResponse(content="done"),
        ]
    )
    agent = Agent(model, _registry(), LocalArtifactStore(tmp_path), read_only=True)

    run = agent.run("write it")

    assert run.steps[0].error_kind == "denied"
    assert "read-only" in (run.steps[0].error or "")
    assert not (tmp_path / "x.py").exists()


def test_agent_checkpoints_before_write(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("old\n", encoding="utf-8")
    model = _ScriptedModel(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="a", name="write_file", arguments={"path": "app.py", "content": "new\n"}
                    )
                ]
            ),
            ModelResponse(content="done"),
        ]
    )
    agent = Agent(model, _registry(), LocalArtifactStore(tmp_path))

    agent.run("write it")

    checkpoint_root = tmp_path / ".atlas" / "checkpoints"
    assert checkpoint_root.is_dir()
    assert any(checkpoint_root.rglob("app.py"))
    assert json.loads(next(checkpoint_root.rglob("manifest.json")).read_text())["path"] == "app.py"
