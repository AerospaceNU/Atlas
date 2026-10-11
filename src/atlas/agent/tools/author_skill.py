"""Opt-in tool that writes one Agent Skill into the artifact workspace.

The skill file is staged for review. It is not loaded into the session and it
is not added to the default tool registry, same as a script proposal.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import Tool, ToolResult
from atlas.agent.skill_author import (
    SkillAuthorError,
    SkillDraft,
    render_skill_md,
    skill_directory_name,
)


class AuthorSkillInput(BaseModel):
    """Arguments for ``author_skill``.

    ``name``, ``description``, and ``body`` are the Agent Skill fields. The
    tool writes ``skills/<name>/SKILL.md`` inside the artifact workspace.
    """

    name: str = Field(description="Skill name. Lowercase letters, digits, and hyphens.")
    description: str = Field(description="What the skill does and when to use it.")
    body: str = Field(description="Markdown instructions stored after the frontmatter.")


class AuthorSkillTool(Tool):
    """Write an Agent Skill into the artifact workspace without registering it.

    The path is resolved through :meth:`LocalArtifactStore.resolve`, so it
    cannot leave the sandbox. The written path is returned in
    ``ToolResult.artifacts``.
    """

    name = "author_skill"
    description = (
        "Write an Agent Skill (SKILL.md) into the artifact workspace for review. "
        "The skill is not loaded and is not added to the tool registry."
    )
    trust: ClassVar[Literal["default", "opt_in"]] = "opt_in"

    @property
    def input_model(self) -> type[BaseModel]:
        return AuthorSkillInput

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        """Validate the draft and write ``skills/<name>/SKILL.md`` in ``store``."""
        request = AuthorSkillInput.model_validate(arguments)
        draft = SkillDraft(name=request.name, description=request.description, body=request.body)
        try:
            name = skill_directory_name(draft)
            rendered = render_skill_md(draft)
        except SkillAuthorError as exc:
            raise ValueError(str(exc)) from exc
        relative = f"skills/{name}/SKILL.md"
        path = store.resolve(relative)
        if path.exists() or path.is_symlink():
            raise ValueError(f"Skill already exists: {relative}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
        artifact = store.relative(path)
        return ToolResult(
            text=f"Wrote skill {name} at {artifact}. It is not loaded or registered.",
            artifacts=[artifact],
        )
