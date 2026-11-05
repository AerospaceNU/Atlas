from __future__ import annotations

import json
from pathlib import Path

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import (
    ChatMessage,
    ModelImage,
    ModelResponse,
    Role,
    ToolDefinition,
    ToolRegistry,
)
from atlas.agent.runtime import Agent, _current_user_message
from atlas.agent.session import AgentSession


class _ScriptedModel:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = responses

    def complete(self, messages: list[ChatMessage], tools: list[ToolDefinition]) -> ModelResponse:
        return self.responses.pop(0)


def _session(tmp_path: Path) -> AgentSession:
    model = _ScriptedModel([ModelResponse(content="first answer"), ModelResponse(content="second")])
    agent = Agent(model, ToolRegistry(), LocalArtifactStore(tmp_path))  # type: ignore[arg-type]
    agent_session = AgentSession(agent)
    agent_session.workspace = tmp_path
    agent_session.session_id = "abc123"
    agent_session.model_id = "vendor/model"
    agent_session.totals.model = "vendor/model"
    return agent_session


def test_save_state_writes_messages_and_totals(tmp_path: Path) -> None:
    session = _session(tmp_path)
    session.turn("hello")

    session.save_state()

    path = tmp_path / ".atlas" / "sessions" / "abc123" / "state.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["id"] == "abc123"
    assert payload["model"] == "vendor/model"
    chat = [item for item in payload["messages"] if item["role"] != "system"]
    assert [item["role"] for item in chat] == ["user", "assistant"]
    assert chat[0]["content"] == "hello"
    assert all(item["images"] == [] for item in payload["messages"])


def test_saved_state_redacts_known_secrets(tmp_path: Path) -> None:
    session = _session(tmp_path)
    session.secrets.append("sk-secret-value")
    session.messages.append(ChatMessage(role="user", content="key sk-secret-value"))

    session.save_state()

    text = (project_state(session)).read_text(encoding="utf-8")
    assert "sk-secret-value" not in text
    assert "[redacted]" in text


def test_load_state_restores_history_and_totals(tmp_path: Path) -> None:
    session = _session(tmp_path)
    session.turn("hello")
    session.save_state()

    reopened = _session(tmp_path)
    assert reopened.load_state(session.session_id or "") is True

    assert [message.role for message in loaded_roles(reopened)] == ["user", "assistant"]
    assert reopened.messages[1].content == "hello"
    assert reopened.messages[2].content == "first answer"
    assert reopened.model_id == "vendor/model"
    assert reopened.totals.model == "vendor/model"
    assert reopened.agent.artifacts.root == tmp_path.resolve()
    assert (
        reopened.agent.artifacts.artifact_root
        == (tmp_path / ".atlas" / "artifacts" / reopened.session_id).resolve()
    )


def test_load_state_missing_file_leaves_session_untouched(tmp_path: Path) -> None:
    session = _session(tmp_path)

    assert session.load_state("nope") is False
    assert [m for m in session.messages if m.role != Role.SYSTEM] == []


def test_load_state_keeps_tool_observations_distinct_from_user_requests(tmp_path: Path) -> None:
    session = _session(tmp_path)
    session.messages.extend(
        [
            ChatMessage(role=Role.USER, content="Inspect the map."),
            ChatMessage(
                role=Role.USER,
                images=[ModelImage(path="map.png", media_type="image/png", data="AA==")],
                tool_observation=True,
            ),
        ]
    )
    session.save_state()
    reopened = _session(tmp_path)

    assert reopened.load_state("abc123")
    assert reopened.messages[-1].tool_observation
    assert reopened.messages[-1].images == []
    current = _current_user_message(reopened.messages)
    assert current is not None
    assert current.content == "Inspect the map."


def project_state(session: AgentSession) -> Path:
    assert session.workspace is not None
    return session.workspace / ".atlas" / "sessions" / (session.session_id or "") / "state.json"


def loaded_roles(session: AgentSession) -> list[ChatMessage]:
    return [message for message in session.messages if message.role != Role.SYSTEM]
