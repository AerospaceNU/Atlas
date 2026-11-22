"""Opt-in tool that writes one Agent Skill into ``.atlas/skills-drafts``."""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import Tool, ToolResult
from atlas.agent.layout import (
    ATLAS_SKILL_DRAFTS_RELATIVE,
    project_atlas_root,
    project_skill_drafts_root,
)
from atlas.agent.skill_author import (
    SkillAuthorError,
    SkillDraft,
    author_skill,
    verify_contained_dir,
)


class AuthorSkillInput(BaseModel):
    """Arguments for ``author_skill``; unknown fields are rejected."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Skill name. Lowercase letters, digits, and hyphens.")
    description: str = Field(description="What the skill does and when to use it.")
    body: str = Field(description="Markdown instructions stored after the frontmatter.")


class AuthorSkillTool(Tool):
    """Write a draft Agent Skill under ``<workspace>/.atlas/skills-drafts``."""

    name = "author_skill"
    description = (
        "Write a draft Agent Skill (SKILL.md) under .atlas/skills-drafts. "
        "The draft is not loaded and is not added to the tool registry. "
        "A person enables it later with /enable-skill."
    )
    trust: ClassVar[Literal["default", "opt_in"]] = "opt_in"

    @property
    def input_model(self) -> type[BaseModel]:
        return AuthorSkillInput

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        """Validate the draft and write it under the workspace drafts root."""
        request = AuthorSkillInput.model_validate(arguments)
        draft = SkillDraft(name=request.name, description=request.description, body=request.body)
        workspace = store.read_root if store.read_root is not None else store.root
        drafts = project_skill_drafts_root(workspace)
        try:
            root = verify_contained_dir(
                workspace,
                drafts,
                atlas=project_atlas_root(workspace),
                exact_root=project_skill_drafts_root(workspace.expanduser().resolve()),
            )
        except SkillAuthorError as exc:
            raise ValueError("Skills root is not the workspace drafts directory") from exc
        try:
            written = author_skill(root, draft)
        except SkillAuthorError as exc:
            raise ValueError(str(exc)) from exc
        artifact = f"{ATLAS_SKILL_DRAFTS_RELATIVE}/{written.name}/SKILL.md"
        return ToolResult(
            text=(
                f"Wrote draft skill {written.name} at {artifact}. "
                "It is not loaded. A person enables it with "
                f"/enable-skill {written.name}."
            ),
            artifacts=[artifact],
        )
