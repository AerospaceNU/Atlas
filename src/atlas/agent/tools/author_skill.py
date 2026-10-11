"""Opt-in tool that writes one Agent Skill into the unscanned drafts folder.

Other tools write only inside the artifact store. This one is a deliberate
sandbox exception: it writes under ``<workspace>/.atlas/skills-drafts``, the
same folder analysis drafts use. That directory is not ``.atlas/skills``, so
the model cannot replace a vetted skill and run it in this session. Every
write goes through :func:`author_skill`, which refuses path escapes, symlinks,
and hardlinks. A ``.atlas`` or ``.atlas/skills-drafts`` symlink that resolves
outside the workspace is refused before that call. Promoting a draft is the
human ``/enable-skill`` command, not this tool.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import Tool, ToolResult
from atlas.agent.skill_author import (
    SkillAuthorError,
    SkillDraft,
    author_skill,
    is_within,
    project_skill_drafts_root,
)


class AuthorSkillInput(BaseModel):
    """Arguments for ``author_skill``.

    ``name``, ``description``, and ``body`` are the Agent Skill fields. The
    tool writes ``<workspace>/.atlas/skills-drafts/<name>`` and does not
    enable or overwrite a skill.
    """

    name: str = Field(description="Skill name. Lowercase letters, digits, and hyphens.")
    description: str = Field(description="What the skill does and when to use it.")
    body: str = Field(description="Markdown instructions stored after the frontmatter.")


class AuthorSkillTool(Tool):
    """Write a draft Agent Skill under ``.atlas/skills-drafts``.

    The session read root is the workspace. ``ToolResult.artifacts`` carries
    a logical path, not a host path. The draft is not loaded.
    """

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
        try:
            root = _drafts_root(store)
            written = author_skill(root, draft)
        except SkillAuthorError as exc:
            raise ValueError(str(exc)) from exc
        artifact = f".atlas/skills-drafts/{written.name}/SKILL.md"
        return ToolResult(
            text=(
                f"Wrote draft skill {written.name} at {artifact}. "
                "It is not loaded. A person enables it with "
                f"/enable-skill {written.name}."
            ),
            artifacts=[artifact],
        )


def _drafts_root(store: LocalArtifactStore) -> Path:
    workspace = store.read_root
    if workspace is None:
        raise ValueError("Draft skills require a workspace read root")
    return _contained_drafts_root(workspace, project_skill_drafts_root(workspace))


def _contained_drafts_root(base: Path, drafts_root: Path) -> Path:
    """Return ``drafts_root`` only when it is ``<workspace>/.atlas/skills-drafts``.

    A symlink at ``skills-drafts`` is refused even when it stays inside the
    workspace, including a link to ``.atlas/skills``. :func:`author_skill`
    would otherwise follow it and write a live skill. The resolved path must
    also be that exact directory.
    """
    root = base.expanduser().resolve()
    atlas = base / ".atlas"
    if atlas.is_symlink() and not is_within(root, atlas):
        raise ValueError("Skills root escapes the sandbox")
    if drafts_root.is_symlink():
        raise ValueError("Skills root escapes the sandbox")
    expected = root / ".atlas" / "skills-drafts"
    try:
        resolved = drafts_root.resolve()
    except OSError as exc:
        raise ValueError("Skills root escapes the sandbox") from exc
    if resolved != expected:
        raise ValueError("Skills root escapes the sandbox")
    return drafts_root
