from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, Field

from atlas.agent.analysis_skill import (
    AnalysisStep,
    AnalysisTurn,
    FinishedAnalysis,
    SkillSaveError,
    save_skill,
)
from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import (
    ChatMessage,
    ModelResponse,
    Tool,
    ToolCall,
    ToolDefinition,
    ToolRegistry,
    ToolResult,
)
from atlas.agent.runtime import Agent
from atlas.agent.session import AgentSession, serve
from atlas.agent.skills import Skill


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
        return ToolResult(text=f"echo:{request.value}", artifacts=["notes/echo.txt"])


class NoteInput(BaseModel):
    note: str


class NoteTool(Tool):
    name = "note"

    description = "Record a note."

    @property
    def input_model(self) -> type[BaseModel]:
        return NoteInput

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        request = NoteInput.model_validate(arguments)
        return ToolResult(text=f"noted {request.note}")


class BoomTool(Tool):
    name = "boom"

    description = "Always raises."

    @property
    def input_model(self) -> type[BaseModel]:
        return EchoInput

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        raise RuntimeError("tool exploded")


def _session(model: ScriptedModel, tmp_path: Path, *, max_tool_calls: int = 256) -> AgentSession:
    project = tmp_path / "project"
    project.mkdir(exist_ok=True)
    agent = Agent(
        model,
        ToolRegistry(),
        LocalArtifactStore(tmp_path / "artifacts", read_root=project),
        max_tool_calls=max_tool_calls,
    )
    session = AgentSession(agent)
    session.workspace = project
    session.home = tmp_path / "home"
    return session


def test_save_skill_writes_the_finished_procedure(tmp_path: Path) -> None:
    model = ScriptedModel(
        [
            ModelResponse(tool_calls=[ToolCall(id="1", name="echo", arguments={"value": 4})]),
            ModelResponse(content="The coast is clear."),
        ]
    )
    session = _session(model, tmp_path)
    session.agent.tools.register(EchoTool())
    session.turn("compare the coast")

    saved = session.save_finished_skill("coast-check")

    assert session.workspace is not None
    path = session.workspace / ".atlas" / "skills-drafts" / "coast-check" / "SKILL.md"
    assert saved.path == ".atlas/skills-drafts/coast-check/SKILL.md"
    assert "replace=True" in saved.note
    assert "author_skill" in saved.note
    assert saved.enabled is False
    assert saved.description == "compare the coast"
    text = path.read_text(encoding="utf-8")
    assert not (session.workspace / ".atlas" / "skills").exists()
    assert not (session.workspace / ".agents" / "skills").exists()
    assert 'name: "coast-check"' in text
    assert '1. `echo` {"value": 4} -> echo:4' in text
    assert "artifacts: notes/echo.txt" in text
    assert "Result: The coast is clear." in text
    assert "Enable it with `/enable-skill coast-check`." in text
    assert str(path.resolve()) not in text


def test_multi_turn_analysis_keeps_every_finished_turn(tmp_path: Path) -> None:
    model = ScriptedModel(
        [
            ModelResponse(tool_calls=[ToolCall(id="1", name="echo", arguments={"value": 1})]),
            ModelResponse(content="first look"),
            ModelResponse(tool_calls=[ToolCall(id="2", name="echo", arguments={"value": 2})]),
            ModelResponse(content="second look"),
        ]
    )
    session = _session(model, tmp_path)
    session.agent.tools.register(EchoTool())
    session.turn("look once")
    session.turn("look again")

    session.save_finished_skill("two-looks")
    text = (tmp_path / "project" / ".atlas" / "skills-drafts" / "two-looks" / "SKILL.md").read_text(
        encoding="utf-8"
    )
    assert "## Turn 1" in text
    assert "## Turn 2" in text
    assert "Request: look once" in text
    assert "Request: look again" in text
    assert '{"value": 1}' in text
    assert '{"value": 2}' in text
    assert "Result: second look" in text


def test_steps_before_the_limit_are_kept_when_the_turn_finishes(tmp_path: Path) -> None:
    model = ScriptedModel(
        [
            ModelResponse(tool_calls=[ToolCall(id="1", name="echo", arguments={"value": 1})]),
            ModelResponse(tool_calls=[ToolCall(id="2", name="echo", arguments={"value": 2})]),
            ModelResponse(content="both steps done"),
        ]
    )
    session = _session(model, tmp_path, max_tool_calls=1)
    session.agent.tools.register(EchoTool())
    first = session.turn("compare the coast")
    assert first.stopped_for_limit
    with pytest.raises(SkillSaveError, match="analysis is not finished"):
        session.save_finished_skill("coast-check")

    second = session.resume()
    assert second.stopped_for_limit
    with pytest.raises(SkillSaveError, match="analysis is not finished"):
        session.save_finished_skill("coast-check")

    finished = session.resume()
    assert finished.response == "both steps done"
    assert not finished.stopped_for_limit
    session.save_finished_skill("coast-check")
    text = (
        tmp_path / "project" / ".atlas" / "skills-drafts" / "coast-check" / "SKILL.md"
    ).read_text(encoding="utf-8")
    assert '{"value": 1}' in text
    assert '{"value": 2}' in text
    assert text.count("## Turn") == 1
    assert "Result: both steps done" in text


def test_refuses_a_chat_with_no_successful_tool_step(tmp_path: Path) -> None:
    model = ScriptedModel([ModelResponse(content="just chatting")])
    session = _session(model, tmp_path)
    session.turn("hello")
    with pytest.raises(SkillSaveError, match="no successful tool step"):
        session.save_finished_skill("chat")


def test_refuses_a_blank_final_response(tmp_path: Path) -> None:
    model = ScriptedModel(
        [
            ModelResponse(tool_calls=[ToolCall(id="1", name="echo", arguments={"value": 1})]),
            ModelResponse(content="   "),
        ]
    )
    session = _session(model, tmp_path)
    session.agent.tools.register(EchoTool())
    session.turn("compare the coast")
    with pytest.raises(SkillSaveError, match="no final response"):
        session.save_finished_skill("coast-check")


def test_failed_tool_steps_are_left_out_of_the_procedure(tmp_path: Path) -> None:
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(id="1", name="boom", arguments={"value": 1}),
                    ToolCall(id="2", name="echo", arguments={"value": 3}),
                ]
            ),
            ModelResponse(content="recovered"),
        ]
    )
    session = _session(model, tmp_path)
    session.agent.tools.register(BoomTool())
    session.agent.tools.register(EchoTool())
    session.turn("compare the coast")
    session.save_finished_skill("coast-check")
    text = (
        tmp_path / "project" / ".atlas" / "skills-drafts" / "coast-check" / "SKILL.md"
    ).read_text(encoding="utf-8")
    assert "boom" not in text
    assert "`echo`" in text
    assert "Result: recovered" in text


def test_refuses_when_nothing_has_finished(tmp_path: Path) -> None:
    session = _session(ScriptedModel([]), tmp_path)
    with pytest.raises(SkillSaveError, match="no finished analysis to save"):
        session.save_finished_skill("coast-check")


def test_new_conversation_drops_the_finished_analysis(tmp_path: Path) -> None:
    model = ScriptedModel(
        [
            ModelResponse(tool_calls=[ToolCall(id="1", name="echo", arguments={"value": 1})]),
            ModelResponse(content="done"),
        ]
    )
    session = _session(model, tmp_path)
    session.agent.tools.register(EchoTool())
    session.turn("compare the coast")
    session.start_fresh()
    with pytest.raises(SkillSaveError, match="no finished analysis to save"):
        session.save_finished_skill("coast-check")


def test_skill_name_must_be_a_slug(tmp_path: Path) -> None:
    analysis = FinishedAnalysis(
        turns=[
            AnalysisTurn(
                request="compare the coast",
                response="done",
                steps=[AnalysisStep(name="echo", arguments={"value": 1}, text="echo:1")],
            )
        ]
    )
    root = tmp_path / ".atlas"
    for name in ("../outside", "Coast", "a/b", "", "has space", "-leading", "trailing-"):
        with pytest.raises(SkillSaveError):
            save_skill(tmp_path, analysis, name)
    assert not (tmp_path / "outside").exists()
    assert list(root.rglob("SKILL.md")) == []


def test_existing_skill_is_kept_unless_replace(tmp_path: Path) -> None:
    model = ScriptedModel(
        [
            ModelResponse(tool_calls=[ToolCall(id="1", name="echo", arguments={"value": 1})]),
            ModelResponse(content="first"),
            ModelResponse(tool_calls=[ToolCall(id="2", name="echo", arguments={"value": 2})]),
            ModelResponse(content="second"),
        ]
    )
    session = _session(model, tmp_path)
    session.agent.tools.register(EchoTool())
    session.turn("first pass")
    session.save_finished_skill("coast-check")
    session.turn("second pass")
    with pytest.raises(SkillSaveError, match="already exists"):
        session.save_finished_skill("coast-check")
    text = (
        tmp_path / "project" / ".atlas" / "skills-drafts" / "coast-check" / "SKILL.md"
    ).read_text(encoding="utf-8")
    assert "Result: first" in text
    assert "second pass" not in text

    session.save_finished_skill("coast-check", replace=True)
    replaced = (
        tmp_path / "project" / ".atlas" / "skills-drafts" / "coast-check" / "SKILL.md"
    ).read_text(encoding="utf-8")
    assert "Request: second pass" in replaced
    assert "Result: second" in replaced


def test_replace_overwrites_a_draft_written_by_the_tool(tmp_path: Path) -> None:
    from atlas.agent.skill_author import SkillDraft, author_skill, project_skill_drafts_root

    project = tmp_path / "project"
    project.mkdir()
    author_skill(
        project_skill_drafts_root(project),
        SkillDraft(name="coast-check", description="from the tool", body="tool body\n"),
    )
    model = ScriptedModel(
        [
            ModelResponse(tool_calls=[ToolCall(id="1", name="echo", arguments={"value": 1})]),
            ModelResponse(content="from the analysis"),
        ]
    )
    session = _session(model, tmp_path)
    session.agent.tools.register(EchoTool())
    session.turn("compare the coast")
    with pytest.raises(SkillSaveError, match="already exists"):
        session.save_finished_skill("coast-check")
    saved = session.save_finished_skill("coast-check", replace=True)
    assert "replace=True" in saved.note
    assert "author_skill" in saved.note
    text = (project / ".atlas" / "skills-drafts" / "coast-check" / "SKILL.md").read_text(
        encoding="utf-8"
    )
    assert "from the tool" not in text
    assert "tool body" not in text
    assert "Result: from the analysis" in text


def test_user_scope_writes_under_the_home_atlas_root(tmp_path: Path) -> None:
    model = ScriptedModel(
        [
            ModelResponse(tool_calls=[ToolCall(id="1", name="echo", arguments={"value": 1})]),
            ModelResponse(content="done"),
        ]
    )
    session = _session(model, tmp_path)
    session.agent.tools.register(EchoTool())
    session.turn("compare the coast")
    saved = session.save_finished_skill("coast-check", scope="user", description='say "hi"')
    home_skill = tmp_path / "home" / ".atlas" / "skills-drafts" / "coast-check" / "SKILL.md"
    assert home_skill.is_file()
    assert saved.path == "~/.atlas/skills-drafts/coast-check/SKILL.md"
    assert "replace=True" in saved.note
    assert saved.description == 'say "hi"'
    assert saved.enabled is False
    assert 'description: "say \\"hi\\""' in home_skill.read_text(encoding="utf-8")
    assert not (tmp_path / "home" / ".atlas" / "skills").exists()
    assert not (tmp_path / "home" / ".agents" / "skills").exists()
    assert not (tmp_path / "project" / ".atlas" / "skills").exists()


def test_secrets_are_redacted_in_the_skill_file(tmp_path: Path) -> None:
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=[ToolCall(id="1", name="note", arguments={"note": "token super-secret"})]
            ),
            ModelResponse(content="the token is super-secret"),
        ]
    )
    session = _session(model, tmp_path)
    session.agent.tools.register(NoteTool())
    session.secrets.append("super-secret")
    session.turn("use super-secret on the coast")
    session.save_finished_skill("coast-check")
    text = (
        tmp_path / "project" / ".atlas" / "skills-drafts" / "coast-check" / "SKILL.md"
    ).read_text(encoding="utf-8")
    assert "super-secret" not in text
    assert "use *** on the coast" in text
    assert "token ***" in text
    assert "the token is ***" in text


def test_symlink_skills_directory_is_refused(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    atlas = tmp_path / ".atlas"
    atlas.mkdir()
    (atlas / "skills-drafts").symlink_to(outside, target_is_directory=True)
    analysis = FinishedAnalysis(
        turns=[
            AnalysisTurn(
                request="compare the coast",
                response="done",
                steps=[AnalysisStep(name="echo", text="echo:1")],
            )
        ]
    )
    with pytest.raises(SkillSaveError, match="symlink"):
        save_skill(tmp_path, analysis, "coast-check")
    assert list(outside.rglob("*")) == []


def test_description_falls_back_to_the_request_and_truncates(tmp_path: Path) -> None:
    request = "compare " + ("the coast " * 40)
    analysis = FinishedAnalysis(
        turns=[
            AnalysisTurn(
                request=request,
                response="done",
                steps=[AnalysisStep(name="echo", text="echo:1")],
            )
        ]
    )
    saved = save_skill(tmp_path, analysis, "coast-check")
    assert len(saved.description) <= 200
    assert saved.description.endswith("…")
    text = (tmp_path / ".atlas" / "skills-drafts" / "coast-check" / "SKILL.md").read_text(
        encoding="utf-8"
    )
    assert saved.description in text


def test_protocol_save_skill_reports_the_relative_path(tmp_path: Path) -> None:
    model = ScriptedModel(
        [
            ModelResponse(tool_calls=[ToolCall(id="1", name="echo", arguments={"value": 4})]),
            ModelResponse(content="The coast is clear."),
        ]
    )
    session = _session(model, tmp_path)
    session.agent.tools.register(EchoTool())
    events = _serve(
        [
            '{"type":"user","text":"compare the coast"}',
            '{"type":"save_skill","name":"coast-check","scope":"project"}',
            '{"type":"save_skill","name":"coast-check"}',
            '{"type":"save_skill"}',
        ],
        session,
    )
    skill = next(event for event in events if event["type"] == "saved_skill")
    assert skill["name"] == "coast-check"
    assert skill["scope"] == "project"
    assert skill["enabled"] is False
    assert skill["path"] == ".atlas/skills-drafts/coast-check/SKILL.md"
    assert "replace=True" in skill["note"]
    assert "author_skill" in skill["note"]
    assert skill["enable"] == "/enable-skill coast-check"
    errors = [event for event in events if event["type"] == "error"]
    assert any("already exists" in event["message"] for event in errors)
    assert any("needs a name" in event["message"] for event in errors)
    assert (
        tmp_path / "project" / ".atlas" / "skills-drafts" / "coast-check" / "SKILL.md"
    ).is_file()
    assert not (tmp_path / "project" / ".atlas" / "skills").exists()


def test_new_turn_keeps_steps_from_a_limited_turn(tmp_path: Path) -> None:
    model = ScriptedModel(
        [
            ModelResponse(tool_calls=[ToolCall(id="1", name="echo", arguments={"value": 1})]),
            ModelResponse(tool_calls=[ToolCall(id="2", name="echo", arguments={"value": 2})]),
            ModelResponse(content="finished the second request"),
        ]
    )
    session = _session(model, tmp_path, max_tool_calls=1)
    session.agent.tools.register(EchoTool())
    stopped = session.turn("first request")
    assert stopped.stopped_for_limit
    with pytest.raises(SkillSaveError, match="analysis is not finished"):
        session.save_finished_skill("kept-steps")

    finished = session.turn("second request")
    assert finished.stopped_for_limit
    done = session.resume()
    assert done.response == "finished the second request"
    assert not done.stopped_for_limit

    session.save_finished_skill("kept-steps")
    text = (
        tmp_path / "project" / ".atlas" / "skills-drafts" / "kept-steps" / "SKILL.md"
    ).read_text(encoding="utf-8")
    assert "Request: first request" in text
    assert "Request: second request" in text
    assert '{"value": 1}' in text
    assert '{"value": 2}' in text
    assert "Result: finished the second request" in text


def test_redact_scrubs_absolute_paths_and_key_shaped_strings(tmp_path: Path) -> None:
    openrouter = "sk-or-v1-" + ("ab" * 32)
    weight = "ab" * 32
    scene_id = "c" * 32
    scene = "LC08_L1TP_044034_20200101_20200101_01_T1"
    url = "https://example.com/coast/scene.png"
    secret_url = "https://user:s3cret@example.com/a?api_key=abc&scene=LC08&token=zzz"
    windows = "C:\\Atlas\\secret.txt"
    file_url = "file:///home/ubuntu/secret.txt"
    jwt = (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        ".eyJzdWIiOiIxMjM0NTY3ODkwIn0"
        ".dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
    )
    analysis = FinishedAnalysis(
        turns=[
            AnalysisTurn(
                request=(
                    f"read /home/ubuntu/project/scene.png and {url} "
                    f"via host:/srv/data PATH=/usr/bin:/home/alice out:/home/bob/scene.png "
                    f"{file_url} {secret_url}"
                ),
                response=f"token {openrouter} lives at {windows} jwt {jwt}",
                steps=[
                    AnalysisStep(
                        name="note",
                        arguments={
                            "path": "/var/atlas/scene.png",
                            "note": "super-secret",
                            "meta": {
                                "weights": windows,
                                "sha256": weight,
                                "scene_id": scene_id,
                                "scene": scene,
                                "url": url,
                                "file": file_url,
                                "fetch": secret_url,
                            },
                        },
                        text="ok",
                        artifacts=["notes/echo.txt"],
                    )
                ],
            )
        ]
    )
    save_skill(tmp_path, analysis, "coast-check", secrets=["super-secret"])
    text = (tmp_path / ".atlas" / "skills-drafts" / "coast-check" / "SKILL.md").read_text(
        encoding="utf-8"
    )
    assert "/home/ubuntu/project/scene.png" not in text
    assert "/var/atlas/scene.png" not in text
    assert "C:\\Atlas\\secret.txt" not in text
    assert "C:\\\\Atlas\\\\secret.txt" not in text
    assert "secret.txt" not in text
    assert openrouter not in text
    assert "super-secret" not in text
    assert "https:***" not in text
    assert url in text
    assert "https://example.com/a?scene=LC08" in text
    assert "user:s3cret" not in text
    assert "api_key" not in text
    assert "s3cret" not in text
    assert file_url not in text
    assert "file://" not in text
    assert "/srv/data" not in text
    assert "/usr/bin" not in text
    assert "/home/alice" not in text
    assert "/home/bob/scene.png" not in text
    assert jwt not in text
    assert weight in text
    assert scene_id in text
    assert scene in text
    assert '"weights": "***"' in text
    assert "notes/echo.txt" in text
    assert "`note`" in text
    assert "***" in text


def test_enable_skill_moves_the_draft_into_atlas_skills(tmp_path: Path) -> None:
    model = ScriptedModel(
        [
            ModelResponse(tool_calls=[ToolCall(id="1", name="echo", arguments={"value": 4})]),
            ModelResponse(content="The coast is clear."),
        ]
    )
    session = _session(model, tmp_path)
    session.agent.tools.register(EchoTool())
    session.turn("compare the coast")
    session.save_finished_skill("coast-check")
    draft = tmp_path / "project" / ".atlas" / "skills-drafts" / "coast-check"
    assert draft.is_dir()

    enabled = session.enable_saved_skill("coast-check")
    assert enabled.enabled is True
    assert enabled.path == ".atlas/skills/coast-check/SKILL.md"
    assert not draft.exists()
    skill = tmp_path / "project" / ".atlas" / "skills" / "coast-check" / "SKILL.md"
    text = skill.read_text(encoding="utf-8")
    assert skill.is_file()
    assert "echo:4" in text
    assert "This skill is a draft" not in text
    assert "coast-check" in session.messages[0].content
    assert any(item.name == "coast-check" for item in session.agent.skills)
    assert not (tmp_path / "project" / ".agents" / "skills").exists()

    with pytest.raises(SkillSaveError, match="not found"):
        session.enable_saved_skill("coast-check")


def test_start_fresh_keeps_the_skill_catalog(tmp_path: Path) -> None:
    session = _session(ScriptedModel([]), tmp_path)
    session.agent.skills = [
        Skill(
            name="coast-check",
            description="compare the coast",
            directory=".atlas/skills/coast-check",
            base=tmp_path,
            relative_dir=".atlas/skills/coast-check",
            content_sha256="ab" * 32,
            artifact_root=None,
        )
    ]
    session.start_fresh()
    prompt = session.messages[0].content
    assert prompt == session.agent.system_prompt()
    assert "coast-check" in prompt
    assert "compare the coast" in prompt


def test_protocol_enable_skill_reports_the_skills_path(tmp_path: Path) -> None:
    import hashlib

    model = ScriptedModel(
        [
            ModelResponse(tool_calls=[ToolCall(id="1", name="echo", arguments={"value": 4})]),
            ModelResponse(content="The coast is clear."),
        ]
    )
    session = _session(model, tmp_path)
    session.agent.tools.register(EchoTool())
    preview_events = _serve(
        [
            '{"type":"user","text":"compare the coast"}',
            '{"type":"save_skill","name":"coast-check"}',
            '{"type":"enable_skill","name":"coast-check","scope":"project"}',
        ],
        session,
    )
    draft = tmp_path / "project" / ".atlas" / "skills-drafts" / "coast-check"
    assert draft.is_dir()
    preview = next(event for event in preview_events if event["type"] == "enable_preview")
    assert preview["description"] == "compare the coast"
    assert preview["path"] == ".atlas/skills-drafts/coast-check/SKILL.md"
    assert len(preview["content_sha256"]) == 64
    assert "echo:4" in preview["preview"]
    assert preview["confirm"] == "/enable-skill coast-check confirm"
    assert not any(event["type"] == "enabled_skill" for event in preview_events)

    events = _serve(
        [
            '{"type":"enable_skill","name":"coast-check","scope":"project","confirm":true}',
            '{"type":"enable_skill"}',
        ],
        session,
    )
    enabled = next(event for event in events if event["type"] == "enabled_skill")
    assert enabled["name"] == "coast-check"
    assert enabled["description"] == "compare the coast"
    assert enabled["path"] == ".atlas/skills/coast-check/SKILL.md"
    assert enabled["enabled"] is True
    assert enabled["catalog"] == "The skill catalog was reloaded."
    published = tmp_path / "project" / ".atlas" / "skills" / "coast-check" / "SKILL.md"
    assert hashlib.sha256(published.read_bytes()).hexdigest() == preview["content_sha256"]
    assert not draft.exists()
    assert "use_skill" in session.agent.tools._tools
    assert any("needs a name" in event["message"] for event in events if event["type"] == "error")


def test_enable_refuses_a_name_that_does_not_match_the_folder(tmp_path: Path) -> None:
    analysis = _finished("compare the coast")
    save_skill(tmp_path, analysis, "coast-check")
    skill = tmp_path / ".atlas" / "skills-drafts" / "coast-check" / "SKILL.md"
    skill.write_text(
        skill.read_text(encoding="utf-8").replace('name: "coast-check"', 'name: "other-name"'),
        encoding="utf-8",
    )
    with pytest.raises(SkillSaveError, match="name must match"):
        from atlas.agent.analysis_skill import enable_skill

        enable_skill(tmp_path, "coast-check")
    assert skill.is_file()
    assert not (tmp_path / ".atlas" / "skills" / "coast-check").exists()


def test_enable_reads_a_folded_description(tmp_path: Path) -> None:
    from atlas.agent.analysis_skill import enable_skill

    save_skill(tmp_path, _finished("compare the coast"), "coast-check")
    skill = tmp_path / ".atlas" / "skills-drafts" / "coast-check" / "SKILL.md"
    skill.write_text(
        skill.read_text(encoding="utf-8").replace(
            'description: "compare the coast"',
            "description: >\n  compare the coast\n  from orbit",
        ),
        encoding="utf-8",
    )
    enabled = enable_skill(tmp_path, "coast-check")
    assert enabled.description == "compare the coast from orbit"


def test_user_scope_enable_refuses_while_home_skills_are_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from atlas.agent.skills import load_home_skills_enabled

    monkeypatch.setenv("ATLAS_HOME_SKILLS", "0")
    model = ScriptedModel(
        [
            ModelResponse(tool_calls=[ToolCall(id="1", name="echo", arguments={"value": 1})]),
            ModelResponse(content="done"),
        ]
    )
    session = _session(model, tmp_path)
    session.agent.tools.register(EchoTool())
    session.turn("compare the coast")
    session.save_finished_skill("coast-check", scope="user")
    draft = tmp_path / "home" / ".atlas" / "skills-drafts" / "coast-check" / "SKILL.md"
    assert draft.is_file()
    assert session.workspace is not None
    assert load_home_skills_enabled(session.workspace) is False
    with pytest.raises(SkillSaveError, match="home skills are off"):
        session.preview_saved_skill("coast-check", scope="user")
    with pytest.raises(SkillSaveError, match="home skills are off"):
        session.enable_saved_skill("coast-check", scope="user")
    assert draft.is_file()
    assert not (tmp_path / "home" / ".atlas" / "skills" / "coast-check").exists()


def test_user_scope_enable_loads_when_home_skills_are_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from atlas.agent.skills import load_home_skills_enabled

    monkeypatch.setenv("ATLAS_HOME_SKILLS", "1")
    model = ScriptedModel(
        [
            ModelResponse(tool_calls=[ToolCall(id="1", name="echo", arguments={"value": 1})]),
            ModelResponse(content="done"),
        ]
    )
    session = _session(model, tmp_path)
    session.agent.tools.register(EchoTool())
    session.turn("compare the coast")
    session.save_finished_skill("coast-check", scope="user")
    assert session.workspace is not None
    assert load_home_skills_enabled(session.workspace) is True
    preview = session.preview_saved_skill("coast-check", scope="user")
    assert preview.description == "compare the coast"
    assert len(preview.content_sha256) == 64
    assert (tmp_path / "home" / ".atlas" / "skills-drafts" / "coast-check").is_dir()
    enabled = session.enable_saved_skill("coast-check", scope="user")
    assert enabled.enabled is True
    assert enabled.path == "~/.atlas/skills/coast-check/SKILL.md"
    events = _serve(
        ['{"type":"enable_skill","name":"missing","scope":"user"}'],
        session,
    )
    assert any(skill.name == "coast-check" for skill in session.agent.skills)
    assert "coast-check" in session.messages[0].content
    assert not any(event["type"] == "enabled_skill" for event in events)


def test_enable_oserror_stays_inside_the_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = ScriptedModel(
        [
            ModelResponse(tool_calls=[ToolCall(id="1", name="echo", arguments={"value": 1})]),
            ModelResponse(content="done"),
        ]
    )
    session = _session(model, tmp_path)
    session.agent.tools.register(EchoTool())
    session.turn("compare the coast")
    session.save_finished_skill("coast-check")

    def explode(*_args: object, **_kwargs: object) -> None:
        raise OSError(13, "Permission denied", "/home/secret/atlas")

    monkeypatch.setattr("atlas.agent.analysis_skill._rename_exclusive", explode)
    events = _serve(
        ['{"type":"enable_skill","name":"coast-check","scope":"project","confirm":true}'],
        session,
    )
    errors = [event for event in events if event["type"] == "error"]
    assert errors
    assert errors[-1]["message"] == "could not enable skill"
    assert "/home/secret" not in errors[-1]["message"]
    assert (tmp_path / "project" / ".atlas" / "skills-drafts" / "coast-check").is_dir()


def test_enable_does_not_replace_an_existing_skill(tmp_path: Path) -> None:
    from atlas.agent.analysis_skill import enable_skill

    save_skill(tmp_path, _finished("compare the coast"), "coast-check")
    destination = tmp_path / ".atlas" / "skills" / "coast-check"
    destination.mkdir(parents=True)
    marker = destination / "SKILL.md"
    marker.write_text("keep me\n", encoding="utf-8")
    with pytest.raises(SkillSaveError, match="already exists"):
        enable_skill(tmp_path, "coast-check")
    assert marker.read_text(encoding="utf-8") == "keep me\n"
    assert (tmp_path / ".atlas" / "skills-drafts" / "coast-check" / "SKILL.md").is_file()


def _finished(request: str) -> FinishedAnalysis:
    return FinishedAnalysis(
        turns=[
            AnalysisTurn(
                request=request,
                response="done",
                steps=[AnalysisStep(name="echo", arguments={"value": 1}, text="echo:1")],
            )
        ]
    )


def _serve(lines: list[str], session: AgentSession) -> list[dict[str, Any]]:
    stdin = io.StringIO("".join(f"{line}\n" for line in lines))
    stdout = io.StringIO()
    assert serve(session, stdin, stdout) == 0
    return [json.loads(line) for line in stdout.getvalue().splitlines()]
