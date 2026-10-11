from __future__ import annotations

import errno
import hashlib
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
    enable_skill,
    preview_skill,
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
from atlas.agent.skills import Skill, register_skill_tool


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
    assert "Request: first request" not in text
    assert '{"value": 1}' not in text
    assert "Request: second request" in text
    assert '{"value": 2}' in text
    assert "Result: finished the second request" in text
    assert text.count("## Turn") == 1


def test_redact_scrubs_absolute_paths_and_key_shaped_strings(tmp_path: Path) -> None:
    openrouter = "sk-or-v1-" + ("ab" * 32)
    weight = "ab" * 32
    scene_id = "c" * 32
    scene = "LC08_L1TP_044034_20200101_20200101_01_T1"
    url = "https://example.com/coast/scene.png"
    secret_url = "https://user:s3cret@example.com/a?api_key=abc&scene=LC08&token=zzz"
    signed_url = (
        "https://bucket.example/scene.png?X-Amz-Signature=sigvalue"
        "&X-Amz-Credential=credvalue&X-Amz-Security-Token=tokvalue"
        "&X-Goog-Signature=googvalue&key=keyvalue&auth=authvalue&code=codevalue"
        "&scene=LC08"
    )
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
                    f"{file_url} {secret_url} {signed_url} "
                    "#access_token=fragsecret https://example.com/cb#access_token=urlfrag"
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
    assert "sigvalue" not in text
    assert "credvalue" not in text
    assert "tokvalue" not in text
    assert "googvalue" not in text
    assert "keyvalue" not in text
    assert "authvalue" not in text
    assert "codevalue" not in text
    assert "X-Amz-Signature" not in text
    assert "scene=LC08" in text
    assert "fragsecret" not in text
    assert "urlfrag" not in text
    assert "#access_token=***" in text
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

    preview = session.preview_saved_skill("coast-check")
    assert preview.body_characters == len(preview.body)
    assert preview.body_lines == len(preview.body.splitlines())
    assert "echo:4" in preview.body
    assert preview.confirm.endswith(preview.content_sha256[:8])
    enabled = session.enable_saved_skill("coast-check", preview.content_sha256)
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

    with pytest.raises(SkillSaveError, match="enable needs a preview"):
        session.enable_saved_skill("coast-check", preview.content_sha256)
    workspace = session.workspace
    assert workspace is not None
    with pytest.raises(SkillSaveError, match="not found"):
        enable_skill(workspace, "coast-check", preview.content_sha256)


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
    assert preview["body"]
    assert preview["body_characters"] == len(preview["body"])
    assert preview["body_lines"] == len(preview["body"].splitlines())
    assert "echo:4" in preview["body"]
    assert (
        preview["confirm"] == f"/enable-skill coast-check confirm {preview['content_sha256'][:8]}"
    )
    assert not any(event["type"] == "enabled_skill" for event in preview_events)

    events = _serve(
        [
            json.dumps(
                {
                    "type": "enable_skill",
                    "name": "coast-check",
                    "scope": "project",
                    "confirm": True,
                    "expected_sha256": preview["content_sha256"],
                }
            ),
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
    from atlas.agent.analysis_skill import _manifest_sha256

    assert _manifest_sha256(published.parent) == preview["content_sha256"]
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
        enable_skill(tmp_path, "coast-check", "0" * 64)
    assert skill.is_file()
    assert not (tmp_path / ".atlas" / "skills" / "coast-check").exists()


def test_enable_reads_a_folded_description(tmp_path: Path) -> None:
    save_skill(tmp_path, _finished("compare the coast"), "coast-check")
    skill = tmp_path / ".atlas" / "skills-drafts" / "coast-check" / "SKILL.md"
    skill.write_text(
        skill.read_text(encoding="utf-8").replace(
            'description: "compare the coast"',
            "description: >\n  compare the coast\n  from orbit",
        ),
        encoding="utf-8",
    )
    preview = preview_skill(tmp_path, "coast-check")
    assert preview.description == "compare the coast from orbit"
    enabled = enable_skill(tmp_path, "coast-check", preview.content_sha256)
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
    with pytest.raises(SkillSaveError, match="home skills are off") as refused:
        session.preview_saved_skill("coast-check", scope="user")
    assert "ATLAS_HOME_SKILLS" in str(refused.value)
    assert "~/.atlas/config.toml" in str(refused.value)
    with pytest.raises(SkillSaveError, match="home skills are off"):
        session.enable_saved_skill("coast-check", "ab" * 32, scope="user")
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
    enabled = session.enable_saved_skill("coast-check", preview.content_sha256, scope="user")
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

    preview_events = _serve(
        ['{"type":"enable_skill","name":"coast-check","scope":"project"}'],
        session,
    )
    digest = next(
        event["content_sha256"] for event in preview_events if event["type"] == "enable_preview"
    )
    monkeypatch.setattr("atlas.agent.analysis_skill._rename_exclusive", explode)
    events = _serve(
        [
            json.dumps(
                {
                    "type": "enable_skill",
                    "name": "coast-check",
                    "scope": "project",
                    "confirm": True,
                    "expected_sha256": digest,
                }
            )
        ],
        session,
    )
    errors = [event for event in events if event["type"] == "error"]
    assert errors
    assert errors[-1]["message"] == "could not enable skill"
    assert "/home/secret" not in errors[-1]["message"]
    assert (tmp_path / "project" / ".atlas" / "skills-drafts" / "coast-check").is_dir()


def test_enable_does_not_replace_an_existing_skill(tmp_path: Path) -> None:
    save_skill(tmp_path, _finished("compare the coast"), "coast-check")
    destination = tmp_path / ".atlas" / "skills" / "coast-check"
    destination.mkdir(parents=True)
    marker = destination / "SKILL.md"
    marker.write_text("keep me\n", encoding="utf-8")
    with pytest.raises(SkillSaveError, match="already exists"):
        enable_skill(tmp_path, "coast-check", preview_skill(tmp_path, "coast-check").content_sha256)
    assert marker.read_text(encoding="utf-8") == "keep me\n"
    assert (tmp_path / ".atlas" / "skills-drafts" / "coast-check" / "SKILL.md").is_file()


def test_confirm_requires_the_preview_hash(tmp_path: Path) -> None:
    session = _saved_session(tmp_path)
    draft = tmp_path / "project" / ".atlas" / "skills-drafts" / "coast-check" / "SKILL.md"
    original = draft.read_text(encoding="utf-8")
    assert "This skill is a draft" in original

    with pytest.raises(SkillSaveError, match="enable needs a preview"):
        session.enable_saved_skill("coast-check", "ab" * 32)
    assert draft.read_text(encoding="utf-8") == original

    preview = session.preview_saved_skill("coast-check")
    with pytest.raises(SkillSaveError, match="skill hash does not match the preview"):
        session.enable_saved_skill("coast-check", "ab" * 32)
    with pytest.raises(SkillSaveError, match="skill hash does not match the preview"):
        session.enable_saved_skill("coast-check", "00000000")
    assert draft.read_text(encoding="utf-8") == original
    assert not (tmp_path / "project" / ".atlas" / "skills" / "coast-check").exists()

    enabled = session.enable_saved_skill("coast-check", preview.content_sha256[:8])
    assert enabled.enabled is True
    assert not draft.exists()
    published = tmp_path / "project" / ".atlas" / "skills" / "coast-check" / "SKILL.md"
    assert published.is_file()
    assert "This skill is a draft" not in published.read_text(encoding="utf-8")


def test_protocol_confirm_without_a_hash_is_refused(tmp_path: Path) -> None:
    session = _saved_session(tmp_path)
    preview_events = _serve(
        ['{"type":"enable_skill","name":"coast-check","confirm":false}'],
        session,
    )
    digest = next(
        event["content_sha256"] for event in preview_events if event["type"] == "enable_preview"
    )
    events = _serve(
        [
            '{"type":"enable_skill","name":"coast-check","confirm":true}',
            json.dumps(
                {
                    "type": "enable_skill",
                    "name": "coast-check",
                    "confirm": True,
                    "expected_sha256": "00000000",
                }
            ),
        ],
        session,
    )
    messages = [event["message"] for event in events if event["type"] == "error"]
    assert messages == [
        "enable needs a preview",
        "skill hash does not match the preview",
    ]
    draft = tmp_path / "project" / ".atlas" / "skills-drafts" / "coast-check"
    assert draft.is_dir()
    assert "This skill is a draft" in (draft / "SKILL.md").read_text(encoding="utf-8")
    confirmed = _serve(
        [
            json.dumps(
                {
                    "type": "enable_skill",
                    "name": "coast-check",
                    "confirm": True,
                    "expected_sha256": digest[:8],
                }
            )
        ],
        session,
    )
    assert any(event["type"] == "enabled_skill" for event in confirmed)
    assert not draft.exists()


def test_preview_shows_the_full_body_and_bundled_resources(tmp_path: Path) -> None:
    save_skill(tmp_path, _finished("compare the coast"), "coast-check")
    draft = tmp_path / ".atlas" / "skills-drafts" / "coast-check"
    skill = draft / "SKILL.md"
    extra = "Keep the full instruction.\n" * 30
    skill.write_text(skill.read_text(encoding="utf-8") + extra, encoding="utf-8")
    (draft / "scripts").mkdir()
    (draft / "scripts" / "run.py").write_text("print(1)\n", encoding="utf-8")
    (draft / "notes").mkdir()
    (draft / "notes" / "readme.txt").write_text("bundled\n", encoding="utf-8")

    preview = preview_skill(tmp_path, "coast-check")
    assert len(preview.body) > 400
    assert extra.strip() in preview.body
    assert "…" not in preview.body
    assert preview.body_characters == len(preview.body)
    assert preview.body_lines == len(preview.body.splitlines())
    assert preview.path == ".atlas/skills-drafts/coast-check/SKILL.md"
    assert [item.path for item in preview.resources] == ["notes/readme.txt", "scripts/run.py"]
    assert "bundled" in preview.resources[0].text
    assert "print(1)" in preview.resources[1].text
    assert preview.resources[0].omitted == ""
    assert preview.confirm.endswith(preview.content_sha256[:8])


def test_enable_strips_the_notice_before_the_move(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    save_skill(tmp_path, _finished("compare the coast"), "coast-check")
    preview = preview_skill(tmp_path, "coast-check")
    source = tmp_path / ".atlas" / "skills-drafts" / "coast-check"

    def stop(draft: Path, _target: Path) -> None:
        text = (draft / "SKILL.md").read_text(encoding="utf-8")
        assert "This skill is a draft" not in text
        raise OSError(1, "stopped")

    monkeypatch.setattr("atlas.agent.analysis_skill._rename_exclusive", stop)
    with pytest.raises(SkillSaveError, match="could not enable skill"):
        enable_skill(tmp_path, "coast-check", preview.content_sha256)
    assert source.is_dir()
    assert not (tmp_path / ".atlas" / "skills" / "coast-check").exists()


def test_enable_rechecks_the_bytes_immediately_before_the_move(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    save_skill(tmp_path, _finished("compare the coast"), "coast-check")
    preview = preview_skill(tmp_path, "coast-check")
    real_read = Path.read_bytes

    def tamper(self: Path) -> bytes:
        data = real_read(self)
        if self.name == "SKILL.md" and b"This skill is a draft" not in data:
            return b"tampered\n"
        return data

    monkeypatch.setattr(Path, "read_bytes", tamper)
    with pytest.raises(SkillSaveError, match="skill hash does not match the preview"):
        enable_skill(tmp_path, "coast-check", preview.content_sha256)
    assert not (tmp_path / ".atlas" / "skills" / "coast-check").exists()
    assert (tmp_path / ".atlas" / "skills-drafts" / "coast-check").is_dir()


def test_enable_claims_the_destination_with_mkdir_when_renameat2_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    save_skill(tmp_path, _finished("compare the coast"), "coast-check")
    preview = preview_skill(tmp_path, "coast-check")
    calls: list[tuple[str, str]] = []
    from atlas.agent import analysis_skill

    original = analysis_skill.atomic_rename

    def spy(source: Path, target: Path) -> None:
        calls.append((source.parent.name, source.name))
        original(source, target)

    monkeypatch.setattr(analysis_skill, "_renameat2", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(analysis_skill, "atomic_rename", spy)
    enabled = enable_skill(tmp_path, "coast-check", preview.content_sha256)
    assert enabled.path == ".atlas/skills/coast-check/SKILL.md"
    assert calls == [(".coast-check.enabling", "SKILL.md")]
    published = tmp_path / ".atlas" / "skills" / "coast-check" / "SKILL.md"
    assert "This skill is a draft" not in published.read_text(encoding="utf-8")
    assert not (tmp_path / ".atlas" / "skills-drafts" / "coast-check").exists()
    assert not (tmp_path / ".atlas" / "skills" / ".coast-check.enabling").exists()


def test_renameat2_exdev_copies_then_claims_the_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    save_skill(tmp_path, _finished("compare the coast"), "coast-check")
    preview = preview_skill(tmp_path, "coast-check")
    calls: list[str] = []
    from atlas.agent import analysis_skill

    original = analysis_skill.atomic_rename

    def spy(source: Path, target: Path) -> None:
        calls.append(source.parent.name)
        original(source, target)

    monkeypatch.setattr(analysis_skill, "_renameat2", lambda *_args, **_kwargs: errno.EXDEV)
    monkeypatch.setattr(analysis_skill, "atomic_rename", spy)
    enable_skill(tmp_path, "coast-check", preview.content_sha256)
    assert calls == [".coast-check.enabling"]
    published = tmp_path / ".atlas" / "skills" / "coast-check" / "SKILL.md"
    assert published.is_file()
    assert "This skill is a draft" not in published.read_text(encoding="utf-8")
    assert not (tmp_path / ".atlas" / "skills-drafts" / "coast-check").exists()
    assert not (tmp_path / ".atlas" / "skills" / ".coast-check.enabling").exists()


def test_fallback_does_not_replace_an_existing_skill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    save_skill(tmp_path, _finished("compare the coast"), "coast-check")
    destination = tmp_path / ".atlas" / "skills" / "coast-check"
    destination.mkdir(parents=True)
    marker = destination / "SKILL.md"
    marker.write_text("keep me\n", encoding="utf-8")
    digest = preview_skill(tmp_path, "coast-check").content_sha256
    monkeypatch.setattr("atlas.agent.analysis_skill._renameat2", lambda *_args, **_kwargs: None)
    with pytest.raises(SkillSaveError, match="already exists"):
        enable_skill(tmp_path, "coast-check", digest)
    monkeypatch.setattr(
        "atlas.agent.analysis_skill._renameat2", lambda *_args, **_kwargs: errno.EXDEV
    )
    with pytest.raises(SkillSaveError, match="already exists"):
        enable_skill(tmp_path, "coast-check", digest)
    assert marker.read_text(encoding="utf-8") == "keep me\n"
    draft = tmp_path / ".atlas" / "skills-drafts" / "coast-check" / "SKILL.md"
    assert draft.is_file()
    assert "This skill is a draft" not in draft.read_text(encoding="utf-8")


def test_enable_keeps_hashes_of_skills_already_loaded(tmp_path: Path) -> None:
    session = _saved_session(tmp_path)
    existing = Skill(
        name="kept-hash",
        description="already loaded",
        directory=".atlas/skills/kept-hash",
        base=tmp_path / "project",
        relative_dir=".atlas/skills/kept-hash",
        content_sha256="cd" * 32,
        artifact_root=None,
    )
    session.agent.skills = [existing]
    register_skill_tool(session.agent.tools, [existing])
    echo = session.agent.tools._tools["echo"]
    preview = session.preview_saved_skill("coast-check")
    session.enable_saved_skill("coast-check", preview.content_sha256[:8])
    assert session.agent.tools._tools["echo"] is echo
    loaded = session.agent.tools._tools["use_skill"]
    assert loaded._by_name["kept-hash"] is existing
    assert existing.content_sha256 == "cd" * 32
    added = loaded._by_name["coast-check"]
    assert added is not existing
    published = tmp_path / "project" / ".atlas" / "skills" / "coast-check" / "SKILL.md"
    assert added.content_sha256 == hashlib.sha256(published.read_bytes()).hexdigest()
    assert added.content_sha256 != preview.content_sha256
    assert [item.name for item in session.agent.skills] == ["coast-check", "kept-hash"]


def test_registry_replace_overwrites_one_name() -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())
    with pytest.raises(ValueError, match="already registered"):
        registry.register(EchoTool())

    class ReplacedEcho(EchoTool):
        description = "replaced echo"

    registry.replace(ReplacedEcho())
    registry.replace(NoteTool())
    assert registry._tools["echo"].description == "replaced echo"
    assert "note" in registry._tools


def test_confirm_hash_covers_every_file_in_the_draft(tmp_path: Path) -> None:
    save_skill(tmp_path, _finished("compare the coast"), "coast-check")
    draft = tmp_path / ".atlas" / "skills-drafts" / "coast-check"
    script = draft / "scripts" / "run.py"
    script.parent.mkdir()
    script.write_text("print(1)\n", encoding="utf-8")
    preview = preview_skill(tmp_path, "coast-check")
    skill_only = hashlib.sha256((draft / "SKILL.md").read_bytes()).hexdigest()
    assert preview.content_sha256 != skill_only
    assert preview.resources[0].path == "scripts/run.py"
    assert preview.resources[0].text == "print(1)\n"
    assert preview.confirm.endswith(preview.content_sha256[:8])

    script.write_text("print(2)\n", encoding="utf-8")
    original = (draft / "SKILL.md").read_text(encoding="utf-8")
    with pytest.raises(SkillSaveError, match="skill hash does not match the preview"):
        enable_skill(tmp_path, "coast-check", preview.content_sha256)
    assert (draft / "SKILL.md").read_text(encoding="utf-8") == original
    assert not (tmp_path / ".atlas" / "skills" / "coast-check").exists()


def test_large_resources_are_counted_without_hiding_the_omission(tmp_path: Path) -> None:
    save_skill(tmp_path, _finished("compare the coast"), "coast-check")
    draft = tmp_path / ".atlas" / "skills-drafts" / "coast-check"
    blob = draft / "notes" / "long.txt"
    blob.parent.mkdir()
    blob.write_text("x" * 2001, encoding="utf-8")
    binary = draft / "data.bin"
    binary.write_bytes(b"\xff\xfe binary")
    preview = preview_skill(tmp_path, "coast-check")
    by_path = {item.path: item for item in preview.resources}
    long = by_path["notes/long.txt"]
    assert long.text == ""
    assert long.characters == 2001
    assert "body not shown" in long.omitted
    assert "2001 characters" in long.omitted
    raw = by_path["data.bin"]
    assert raw.text == ""
    assert "body not shown" in raw.omitted
    assert "bytes" in raw.omitted


def test_mixed_case_name_matches_the_preview(tmp_path: Path) -> None:
    session = _saved_session(tmp_path)
    preview = session.preview_saved_skill("Coast-Check")
    assert preview.name == "coast-check"
    enabled = session.enable_saved_skill("Coast-Check", preview.content_sha256[:8])
    assert enabled.name == "coast-check"
    assert enabled.path == ".atlas/skills/coast-check/SKILL.md"


def test_leftover_enabling_directory_is_recovered(tmp_path: Path) -> None:
    save_skill(tmp_path, _finished("compare the coast"), "coast-check")
    preview = preview_skill(tmp_path, "coast-check")
    skills = tmp_path / ".atlas" / "skills"
    skills.mkdir(parents=True)
    leftover = skills / ".coast-check.enabling"
    leftover.mkdir()
    (leftover / "SKILL.md").write_text("partial\n", encoding="utf-8")
    enable_skill(tmp_path, "coast-check", preview.content_sha256)
    published = skills / "coast-check" / "SKILL.md"
    assert published.is_file()
    assert "partial" not in published.read_text(encoding="utf-8")
    assert "This skill is a draft" not in published.read_text(encoding="utf-8")
    assert not leftover.exists()

    save_skill(tmp_path, _finished("compare the coast"), "other-check")
    other = preview_skill(tmp_path, "other-check")
    blocked = skills / ".other-check.enabling"
    blocked.symlink_to(skills)
    with pytest.raises(SkillSaveError, match=r"\.other-check\.enabling must not be a symlink"):
        enable_skill(tmp_path, "other-check", other.content_sha256)
    assert blocked.is_symlink()
    assert (tmp_path / ".atlas" / "skills-drafts" / "other-check").is_dir()
    assert not (skills / "other-check").exists()


def test_user_scope_error_names_the_home_skills_setting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ATLAS_HOME_SKILLS", raising=False)
    home = tmp_path / "home"
    monkeypatch.setattr("atlas.agent.skills.Path.home", lambda: home)
    session = _saved_session(tmp_path)
    session.save_finished_skill("coast-check", scope="user")
    project = tmp_path / "project"
    (project / ".atlas" / "agent.toml").write_text("home_skills = true\n", encoding="utf-8")
    with pytest.raises(SkillSaveError) as refused:
        session.preview_saved_skill("coast-check", scope="user")
    assert str(refused.value) == (
        "home skills are off; set ATLAS_HOME_SKILLS or home_skills = true in ~/.atlas/config.toml"
    )
    assert "agent.toml" not in str(refused.value)
    config = home / ".atlas" / "config.toml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text("home_skills = true\n", encoding="utf-8")
    preview = session.preview_saved_skill("coast-check", scope="user")
    assert preview.name == "coast-check"


def _saved_session(tmp_path: Path) -> AgentSession:
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
    return session


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
