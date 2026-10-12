"""Path helpers and the project versus user ``.atlas`` defaults."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import ChatMessage, ModelResponse, ToolDefinition, ToolRegistry
from atlas.agent.layout import (
    ensure_project_layout,
    ensure_user_layout,
    project_artifact_dir,
    project_artifacts_dir,
    project_atlas_root,
    project_data_dir,
    project_session_dir,
    project_sessions_dir,
    project_transcript_path,
    remove_entry,
    user_atlas_root,
    user_defaults_dir,
    user_keys_dir,
    user_weights_dir,
)
from atlas.agent.runtime import Agent
from atlas.agent.session import AgentSession, _remember_session, serve
from atlas.agent.session import _append_transcript as append_transcript
from atlas.agent.session import _transcript as transcript_var
from atlas.cli.data import add_data_parser
from atlas.models.base import weights_root


class _ScriptedModel:
    def __init__(self, replies: list[str]) -> None:
        self._replies = replies

    def complete(self, messages: list[ChatMessage], tools: list[ToolDefinition]) -> ModelResponse:
        return ModelResponse(content=self._replies.pop(0))


def test_project_helpers_stay_under_the_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    assert project_atlas_root(workspace) == workspace / ".atlas"
    assert project_sessions_dir(workspace) == workspace / ".atlas" / "sessions"
    assert project_artifacts_dir(workspace) == workspace / ".atlas" / "artifacts"
    assert project_data_dir(workspace) == workspace / ".atlas" / "data"
    assert project_session_dir(workspace, "s1") == workspace / ".atlas" / "sessions" / "s1"
    assert project_artifact_dir(workspace, "s1") == workspace / ".atlas" / "artifacts" / "s1"
    assert project_transcript_path(workspace, "s1") == (
        workspace / ".atlas" / "sessions" / "s1" / "transcript.jsonl"
    )
    assert project_data_dir() == Path(".atlas/data")
    assert not project_data_dir().is_absolute()


def test_user_helpers_stay_under_the_home_directory(tmp_path: Path) -> None:
    home = tmp_path / "home"
    assert user_atlas_root(home) == home / ".atlas"
    assert user_keys_dir(home) == home / ".atlas" / "keys"
    assert user_weights_dir(home) == home / ".atlas" / "weights"
    assert user_defaults_dir(home) == home / ".atlas" / "defaults"


def test_ensure_project_layout_keeps_existing_data(tmp_path: Path) -> None:
    data = project_data_dir(tmp_path) / "keep.txt"
    data.parent.mkdir(parents=True)
    data.write_text("table", encoding="utf-8")

    root = ensure_project_layout(tmp_path)

    assert root == project_atlas_root(tmp_path)
    assert project_sessions_dir(tmp_path).is_dir()
    assert project_artifacts_dir(tmp_path).is_dir()
    assert data.read_text(encoding="utf-8") == "table"


def test_ensure_user_layout_keeps_existing_weights(tmp_path: Path) -> None:
    weight = user_weights_dir(tmp_path) / "model.json"
    weight.parent.mkdir(parents=True)
    weight.write_text("{}", encoding="utf-8")

    root = ensure_user_layout(tmp_path)

    assert root == user_atlas_root(tmp_path)
    assert user_keys_dir(tmp_path).is_dir()
    assert user_defaults_dir(tmp_path).is_dir()
    assert weight.read_text(encoding="utf-8") == "{}"


def test_data_cli_defaults_to_the_project_data_dir() -> None:
    parser = argparse.ArgumentParser()
    add_data_parser(parser.add_subparsers(dest="group"))

    args = parser.parse_args(["data"])

    assert args.out == project_data_dir()
    assert args.out == Path(".atlas/data")


def test_weights_root_prefers_env_then_checkout_then_user_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = tmp_path / "repo"
    checkout.mkdir()
    home = tmp_path / "home"
    monkeypatch.delenv("ATLAS_WEIGHTS_DIR", raising=False)
    monkeypatch.setattr("atlas.models.base.repo_root", lambda: checkout)
    monkeypatch.setattr("atlas.agent.layout.Path.home", lambda: home)

    assert weights_root() == (checkout / "weights").resolve()

    monkeypatch.setattr("atlas.models.base.repo_root", lambda: None)
    assert weights_root() == user_weights_dir().resolve()
    assert weights_root() == (home / ".atlas" / "weights").resolve()

    custom = tmp_path / "custom"
    monkeypatch.setenv("ATLAS_WEIGHTS_DIR", str(custom))
    assert weights_root() == custom.resolve()


def test_session_transcript_is_redacted_and_moves_with_new(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    home = tmp_path / "home"
    secret = "super-secret-value"
    session = AgentSession(
        Agent(
            _ScriptedModel(["first", "second"]),
            ToolRegistry(),
            LocalArtifactStore(tmp_path / "old"),
        )
    )
    session.workspace = workspace
    session.home = home
    session.secrets.append(secret)
    session_id = _remember_session(workspace, "example/model")
    session.session_id = session_id
    session.transcript_path = project_transcript_path(workspace, session_id)

    events = _serve(
        session,
        [
            '{"type":"user","text":"first-marker super-secret-value"}',
            '{"type":"new"}',
            '{"type":"user","text":"second-marker"}',
        ],
    )

    transcripts = sorted((workspace / ".atlas" / "sessions").glob("*/transcript.jsonl"))
    assert len(transcripts) == 2
    by_text = {path.read_text(encoding="utf-8"): path for path in transcripts}
    first = next(text for text in by_text if "first-marker" in text)
    second = next(text for text in by_text if "second-marker" in text)
    assert "second-marker" not in first
    assert "first-marker" not in second
    assert secret not in first
    assert secret not in second
    assert "first-marker ***" in first
    assert '"type": "ready"' in first
    assert '"type": "ready"' in second
    assert by_text[first].parent == project_session_dir(workspace, session_id)
    assert (by_text[first].parent / "meta.json").is_file()
    assert events[0]["type"] == "ready"
    removed = project_session_dir(workspace, session_id)
    remove_entry(project_atlas_root(workspace), "sessions", session_id, confirm=True)
    assert not removed.exists()
    assert not project_artifact_dir(workspace, session_id).exists()


def test_transcript_append_refuses_a_symlink(tmp_path: Path) -> None:
    path = project_transcript_path(tmp_path, "s1")
    path.parent.mkdir(parents=True)
    outside = tmp_path / "outside.txt"
    outside.write_text("nope", encoding="utf-8")
    path.symlink_to(outside)
    token = transcript_var.set(path)
    try:
        with pytest.raises(OSError, match="symlink"):
            append_transcript("hello")
    finally:
        transcript_var.reset(token)
    assert outside.read_text(encoding="utf-8") == "nope"


def _serve(session: AgentSession, lines: list[str]) -> list[dict[str, object]]:
    import io

    stdin = io.StringIO("".join(f"{line}\n" for line in lines))
    stdout = io.StringIO()
    assert serve(session, stdin, stdout) == 0
    return [json.loads(line) for line in stdout.getvalue().splitlines()]
