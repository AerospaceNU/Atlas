"""Opt-in tool that writes one Agent Skill where the loader will look.

Other tools write only inside the artifact store. This one is a deliberate
sandbox exception: it writes under ``<workspace>/.atlas/skills``
(``scope=project``) or ``~/.atlas/skills`` (``scope=user``), because that is
where skills are loaded from. The write goes through :func:`author_skill`,
which refuses path escapes, symlinks, and hardlinks. A ``.atlas`` or
``.atlas/skills`` symlink that resolves outside the workspace or home is
refused before that call. The skill is not loaded and it is not registered.
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
    project_skills_root,
    user_skills_root,
)

_Scope = Literal["project", "user"]


class AuthorSkillInput(BaseModel):
    """Arguments for ``author_skill``.

    ``name``, ``description``, and ``body`` are the Agent Skill fields.
    ``scope=project`` writes ``<workspace>/.atlas/skills``. ``scope=user``
    writes ``~/.atlas/skills``.
    """

    name: str = Field(description="Skill name. Lowercase letters, digits, and hyphens.")
    description: str = Field(description="What the skill does and when to use it.")
    body: str = Field(description="Markdown instructions stored after the frontmatter.")
    scope: _Scope = Field(
        default="project",
        description="project writes the workspace .atlas/skills; user writes ~/.atlas/skills.",
    )
    overwrite: bool = Field(
        default=False, description="Replace an existing skill of the same name."
    )


class AuthorSkillTool(Tool):
    """Write an Agent Skill under ``.atlas/skills`` without registering it.

    The artifact store is not the skills directory. Project scope uses the
    session read root (the workspace). User scope uses the home directory.
    ``ToolResult.artifacts`` carries a logical path, not a host path.
    """

    name = "author_skill"
    description = (
        "Write an Agent Skill (SKILL.md) under .atlas/skills. "
        "scope=project uses the workspace; scope=user uses the home directory. "
        "The skill is not loaded and is not added to the tool registry."
    )
    trust: ClassVar[Literal["default", "opt_in"]] = "opt_in"

    @property
    def input_model(self) -> type[BaseModel]:
        return AuthorSkillInput

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        """Validate the draft and write it under the project or user skills root."""
        request = AuthorSkillInput.model_validate(arguments)
        draft = SkillDraft(name=request.name, description=request.description, body=request.body)
        try:
            root = _skills_root(request.scope, store)
            written = author_skill(root, draft, overwrite=request.overwrite)
        except SkillAuthorError as exc:
            raise ValueError(str(exc)) from exc
        artifact = _artifact_label(request.scope, written.name)
        return ToolResult(
            text=f"Wrote skill {written.name} at {artifact}. It is not loaded or registered.",
            artifacts=[artifact],
        )


def _skills_root(scope: _Scope, store: LocalArtifactStore) -> Path:
    if scope == "user":
        home = Path.home()
        return _contained_skills_root(home, user_skills_root(home))
    workspace = store.read_root
    if workspace is None:
        raise ValueError("Project skills require a workspace read root")
    return _contained_skills_root(workspace, project_skills_root(workspace))


def _contained_skills_root(base: Path, skills_root: Path) -> Path:
    """Return ``skills_root`` unless a symlink would leave ``base``.

    :func:`author_skill` resolves its root and would follow the symlink.
    """
    root = base.expanduser().resolve()
    for candidate in (base / ".atlas", skills_root):
        if not candidate.is_symlink():
            continue
        target = candidate.resolve()
        if target != root and root not in target.parents:
            raise ValueError("Skills root escapes the sandbox")
    return skills_root


def _artifact_label(scope: _Scope, name: str) -> str:
    if scope == "user":
        return f"~/.atlas/skills/{name}/SKILL.md"
    return f".atlas/skills/{name}/SKILL.md"
