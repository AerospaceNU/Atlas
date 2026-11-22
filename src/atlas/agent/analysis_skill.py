"""Save a finished analysis as a draft Agent Skill and promote drafts to live skills."""

from __future__ import annotations

import contextlib
import copy
import errno
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from atlas.agent.layout import (
    SKILL_DRAFTS_DIR_NAME,
    SKILLS_DIR_NAME,
    project_atlas_root,
    user_atlas_root,
    user_skill_drafts_root,
)
from atlas.agent.redaction import redact_text, redact_value
from atlas.agent.runtime import AgentRun
from atlas.agent.skill_author import (
    SkillAuthorError,
    SkillDraft,
    author_skill,
    is_within,
    normalize_skill_name,
    project_skill_drafts_root,
    project_skills_root,
    user_skills_root,
    verify_contained_dir,
)
from atlas.agent.skills import parse_skill_document

_MAX_DESCRIPTION = 200
_MAX_STEP_TEXT = 500
_MAX_RESPONSE = 2000
_SMALL_RESOURCE_CHARACTERS = 2000
_REDACTED = "***"
_REPLACE_NOTE = (
    "replace=True overwrites a draft of the same name, "
    "including one written by the author_skill tool."
)
_DRAFT_NOTICE = re.compile(
    r"^This skill is a draft\. Enable it with `/enable-skill [^`]+`\.\n?", re.MULTILINE
)


class SkillSaveError(ValueError):
    """The analysis cannot be saved or promoted as a skill."""


class AnalysisStep(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    text: str = ""
    artifacts: list[str] = Field(default_factory=list)


class AnalysisTurn(BaseModel):
    request: str
    response: str
    steps: list[AnalysisStep] = Field(default_factory=list)


class FinishedAnalysis(BaseModel):
    turns: list[AnalysisTurn] = Field(default_factory=list)


class SavedSkill(BaseModel):
    name: str
    description: str
    path: str
    enabled: bool = False
    note: str = ""


class ResourcePreview(BaseModel):
    path: str
    sha256: str
    characters: int
    lines: int
    text: str = ""
    omitted: str = ""


class SkillPreview(BaseModel):
    name: str
    description: str
    path: str
    content_sha256: str
    body: str
    body_characters: int
    body_lines: int
    resources: list[ResourcePreview] = Field(default_factory=list)
    confirm: str


@dataclass(frozen=True)
class _LoadedDraft:
    slug: str
    description: str
    body: str
    text: str
    source: Path


def enable_command(name: str) -> str:
    """Return the slash command that promotes a draft skill."""
    return f"/enable-skill {name}"


def successful_steps(run: AgentRun) -> list[AnalysisStep]:
    """Return tool steps in ``run`` that succeeded, in execution order."""
    return [
        AnalysisStep(
            name=step.call.name,
            arguments=copy.deepcopy(step.call.arguments),
            text=step.result.text,
            artifacts=list(step.result.artifacts),
        )
        for step in run.steps
        if step.result is not None
    ]


def logical_skill_path(scope: str, directory: str, name: str) -> str:
    """Return a logical ``.atlas/...`` or ``~/.atlas/...`` path label."""
    prefix = "~/.atlas" if scope == "user" else ".atlas"
    return f"{prefix}/{directory}/{name}/SKILL.md"


def skill_slug(name: str) -> str:
    """Fold case and validate ``name`` as a skill slug."""
    return _skill_slug(unicodedata.normalize("NFKC", name).strip().lower())


def save_skill(
    workspace: Path,
    analysis: FinishedAnalysis,
    name: str,
    *,
    description: str | None = None,
    replace: bool = False,
    secrets: Sequence[str] = (),
    scope: str = "project",
    home: Path | None = None,
) -> SavedSkill:
    """Write a draft skill for ``analysis`` under the shared ``skills-drafts`` root."""
    if not analysis.turns:
        raise SkillSaveError("no finished analysis to save")
    if not analysis.turns[-1].response.strip():
        raise SkillSaveError("analysis has no final response")
    if not any(turn.steps for turn in analysis.turns):
        raise SkillSaveError("finished analysis has no successful tool step")

    slug = _skill_slug(name)
    raw_summary = description if description is not None else analysis.turns[0].request
    summary = _redact_text(" ".join(raw_summary.split()), secrets).replace("---", "-")
    if not summary:
        raise SkillSaveError("skill description is empty")
    if len(summary) > _MAX_DESCRIPTION:
        summary = summary[: _MAX_DESCRIPTION - 1].rstrip() + "…"

    body = render_procedure(analysis, secrets=secrets, enable=enable_command(slug))
    draft = SkillDraft(name=slug, description=summary, body=body)
    try:
        _, drafts_dir = _atlas_subdir(workspace, scope, home, drafts=True)
        written = author_skill(drafts_dir, draft, overwrite=replace)
    except SkillAuthorError as exc:
        raise SkillSaveError("; ".join(exc.errors)) from exc
    return SavedSkill(
        name=written.name,
        description=summary,
        path=logical_skill_path(scope, SKILL_DRAFTS_DIR_NAME, written.name),
        enabled=False,
        note=_REPLACE_NOTE,
    )


def preview_skill(
    workspace: Path,
    name: str,
    *,
    scope: str = "project",
    home: Path | None = None,
) -> SkillPreview:
    """Describe a draft skill and compute its manifest digest without moving it."""
    try:
        loaded = _load_draft(workspace, name, scope=scope, home=home)
    except OSError as exc:
        raise SkillSaveError("could not enable skill") from exc
    published = _DRAFT_NOTICE.sub("", loaded.text, count=1)
    digest = _manifest_sha256(loaded.source, skill_md=published.encode("utf-8"))
    body = _DRAFT_NOTICE.sub("", loaded.body, count=1).strip()
    return SkillPreview(
        name=loaded.slug,
        description=loaded.description,
        path=logical_skill_path(scope, SKILL_DRAFTS_DIR_NAME, loaded.slug),
        content_sha256=digest,
        body=body,
        body_characters=len(body),
        body_lines=len(body.splitlines()),
        resources=_resource_previews(loaded.source),
        confirm=f"{enable_command(loaded.slug)} confirm {digest[:8]}",
    )


def enable_skill(
    workspace: Path,
    name: str,
    expected_sha256: str,
    *,
    scope: str = "project",
    home: Path | None = None,
) -> SavedSkill:
    """Verify ``expected_sha256``, strip the draft notice, and move the draft into ``skills``."""
    try:
        loaded = _load_draft(workspace, name, scope=scope, home=home)
        published = _DRAFT_NOTICE.sub("", loaded.text, count=1)
        expected = expected_sha256.strip().lower()
        if _manifest_sha256(loaded.source, skill_md=published.encode("utf-8")) != expected:
            raise SkillSaveError("skill hash does not match the preview")
        _write_published(loaded.source / "SKILL.md", published)
        if _manifest_sha256(loaded.source) != expected:
            raise SkillSaveError("skill hash does not match the preview")
        base, skills = _atlas_subdir(workspace, scope, home, drafts=False)
        skills.mkdir(parents=True, exist_ok=True)
        destination = skills / loaded.slug
        if destination.is_symlink() or not (
            is_within(base, destination) or is_within(skills.parent, destination)
        ):
            raise SkillSaveError("skill path escapes the atlas root")
        _rename_exclusive(loaded.source, destination)
    except OSError as exc:
        raise SkillSaveError("could not enable skill") from exc
    return SavedSkill(
        name=loaded.slug,
        description=loaded.description,
        path=logical_skill_path(scope, SKILLS_DIR_NAME, loaded.slug),
        enabled=True,
        note=_REPLACE_NOTE,
    )


def render_procedure(
    analysis: FinishedAnalysis,
    *,
    secrets: Sequence[str] = (),
    enable: str = "",
) -> str:
    """Render redacted markdown procedure instructions for ``analysis``."""
    lines = [
        "Replay this procedure when the user asks for the same kind of result. "
        "It was saved from a finished analysis.",
        "",
    ]
    if enable:
        lines.extend([f"This skill is a draft. Enable it with `{enable}`.", ""])
    for index, turn in enumerate(analysis.turns, start=1):
        lines.extend(
            [f"## Turn {index}", "", "Request: " + _redact_text(turn.request.strip(), secrets), ""]
        )
        if turn.steps:
            for step_index, step in enumerate(turn.steps, start=1):
                try:
                    arguments = json.dumps(
                        _redact_data(step.arguments, secrets), sort_keys=True, ensure_ascii=False
                    )
                except TypeError:
                    arguments = json.dumps(
                        {str(k): _redact_text(str(v), secrets) for k, v in step.arguments.items()},
                        sort_keys=True,
                        ensure_ascii=False,
                    )
                outcome = _one_line(_redact_text(step.text, secrets), _MAX_STEP_TEXT)
                lines.append(f"{step_index}. `{step.name}` {arguments} -> {outcome}")
                if step.artifacts:
                    lines.append(
                        "   artifacts: "
                        + ", ".join(_redact_text(item, secrets) for item in step.artifacts)
                    )
            lines.append("")
        lines.extend(
            ["Result: " + _one_line(_redact_text(turn.response, secrets), _MAX_RESPONSE), ""]
        )
    return "\n".join(lines).rstrip() + "\n"


def _skill_slug(name: str) -> str:
    normalized, errors = normalize_skill_name(name)
    if errors or normalized is None:
        raise SkillSaveError("; ".join(errors or ["Field 'name' must be a non-empty string"]))
    return normalized


def _atlas_subdir(
    workspace: Path, scope: str, home: Path | None, *, drafts: bool
) -> tuple[Path, Path]:
    if scope == "user":
        if home is None:
            raise SkillSaveError("user scope needs a home directory")
        base = Path(home).expanduser()
        atlas = user_atlas_root(base)
        target = user_skill_drafts_root(base) if drafts else user_skills_root(base)
    elif scope == "project":
        base = Path(workspace).expanduser()
        atlas = project_atlas_root(base)
        target = project_skill_drafts_root(base) if drafts else project_skills_root(base)
    else:
        raise SkillSaveError("scope must be project or user")
    try:
        verify_contained_dir(base, target, atlas=atlas)
    except SkillAuthorError as exc:
        raise SkillSaveError("; ".join(exc.errors)) from exc
    if target.exists() and not target.is_dir():
        raise SkillSaveError(
            f"{'skills-drafts' if drafts else 'skills'} directory is not a directory"
        )
    return base, target


def _load_draft(workspace: Path, name: str, *, scope: str, home: Path | None) -> _LoadedDraft:
    slug = skill_slug(name)
    base, drafts = _atlas_subdir(workspace, scope, home, drafts=True)
    source = drafts / slug
    atlas = drafts.parent
    if (
        source.is_symlink()
        or not source.is_dir()
        or not (is_within(base, source) or is_within(atlas, source))
    ):
        raise SkillSaveError("draft skill was not found")
    for path in source.rglob("*"):
        if path.is_symlink():
            raise SkillSaveError("draft skill must not contain a symlink")
    skill_file = source / "SKILL.md"
    if (
        skill_file.is_symlink()
        or not skill_file.is_file()
        or not (is_within(base, skill_file) or is_within(atlas, skill_file))
    ):
        raise SkillSaveError("draft skill was not found")
    try:
        text = skill_file.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise SkillSaveError("draft skill is not UTF-8 text") from exc
    try:
        fields, body = parse_skill_document(text)
    except ValueError as exc:
        raise SkillSaveError("draft skill frontmatter is invalid") from exc
    normalized, errors = normalize_skill_name(fields.get("name"))
    if errors or normalized is None or normalized != unicodedata.normalize("NFKC", slug):
        raise SkillSaveError("draft skill name must match its directory")
    description = fields.get("description")
    if not isinstance(description, str) or not description.strip():
        raise SkillSaveError("draft skill is missing a description")
    return _LoadedDraft(slug, description.strip(), body, text, source)


def _bundled_resources(directory: Path) -> list[str]:
    return [
        path.relative_to(directory).as_posix()
        for path in sorted(directory.rglob("*"))
        if not path.is_symlink()
        and path.is_file()
        and path.relative_to(directory).as_posix() != "SKILL.md"
    ]


def _manifest_payload(records: list[dict[str, str]]) -> str:
    return json.dumps(records, ensure_ascii=False, separators=(",", ":"))


def _manifest_sha256(directory: Path, *, skill_md: bytes | None = None) -> str:
    skill_bytes = skill_md if skill_md is not None else (directory / "SKILL.md").read_bytes()
    files = [("SKILL.md", skill_bytes)] + [
        (relative, (directory / relative).read_bytes())
        for relative in _bundled_resources(directory)
    ]
    records = [
        {"path": path, "sha256": hashlib.sha256(data).hexdigest()}
        for path, data in sorted(files, key=lambda item: item[0])
    ]
    return hashlib.sha256(_manifest_payload(records).encode("utf-8")).hexdigest()


def _resource_previews(directory: Path) -> list[ResourcePreview]:
    previews: list[ResourcePreview] = []
    for relative in _bundled_resources(directory):
        data = (directory / relative).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        try:
            decoded = data.decode("utf-8")
        except UnicodeDecodeError:
            characters, lines, text, omitted = (
                len(data),
                0,
                "",
                f"{len(data)} bytes, body not shown",
            )
        else:
            characters, lines = len(decoded), len(decoded.splitlines())
            if characters <= _SMALL_RESOURCE_CHARACTERS:
                text, omitted = decoded, ""
            else:
                text, omitted = "", f"{characters} characters, {lines} lines, body not shown"
        previews.append(
            ResourcePreview(
                path=relative,
                sha256=digest,
                characters=characters,
                lines=lines,
                text=text,
                omitted=omitted,
            )
        )
    return previews


def _write_published(skill_file: Path, published: str) -> None:
    data = published.encode("utf-8")
    if skill_file.read_bytes() == data:
        return
    fd, tmp_name = tempfile.mkstemp(
        dir=skill_file.parent.parent, prefix=f".{skill_file.parent.name}.SKILL.md."
    )
    temporary = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.chmod(temporary, 0o644)
        os.replace(temporary, skill_file)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _rename_exclusive(source: Path, target: Path) -> None:
    if sys.platform == "win32":
        if target.exists() or target.is_symlink():
            raise SkillSaveError(f"Skill already exists: {target.name}")
        shutil.move(str(source), str(target))
        return
    try:
        os.mkdir(target)
    except FileExistsError as exc:
        raise SkillSaveError(f"Skill already exists: {target.name}") from exc
    staged: Path | None = None
    try:
        try:
            os.rename(source, target)
            return
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise
        staged = Path(tempfile.mkdtemp(dir=target.parent, prefix=f".{target.name}.enabling."))
        shutil.copytree(source, staged, symlinks=False, dirs_exist_ok=True)
        os.rename(staged, target)
        staged = None
    except OSError as exc:
        with contextlib.suppress(OSError):
            os.rmdir(target)
        if staged is not None:
            shutil.rmtree(staged, ignore_errors=True)
        if exc.errno in {errno.EEXIST, errno.ENOTEMPTY}:
            raise SkillSaveError(f"Skill already exists: {target.name}") from exc
        raise
    shutil.rmtree(source, ignore_errors=True)


def _one_line(text: str, limit: int) -> str:
    collapsed = " ".join(text.split())
    if not collapsed:
        return "(no output)"
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 1].rstrip() + "…"


def _redact_data(value: Any, secrets: Sequence[str]) -> Any:
    return redact_value(value, secrets, mask=_REDACTED, scrub_paths=True)


def _redact_text(text: str, secrets: Sequence[str]) -> str:
    return redact_text(text, secrets, mask=_REDACTED, scrub_paths=True)
