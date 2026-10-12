"""Save a finished analysis as a draft Agent Skill.

A finished analysis is one or more completed turns. Each turn has a request,
a final response, and the tool steps that succeeded. Saving uses
:func:`atlas.agent.skill_author.author_skill` and writes
``skills-drafts/<name>/SKILL.md`` under a project or user ``.atlas`` root.

That directory is not ``.atlas/skills`` or ``.agents/skills``, so a loader
that scans those trees does not activate the skill. ``/enable-skill <name>``
moves the directory into ``.atlas/skills``.
"""

from __future__ import annotations

import copy
import ctypes
import ctypes.util
import errno
import hashlib
import json
import os
import re
import shutil
import time
import unicodedata
import uuid
from collections.abc import Sequence
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
from atlas.agent.runtime import AgentRun
from atlas.agent.skill_author import (
    SkillAuthorError,
    SkillDraft,
    author_skill,
    is_within,
    normalize_skill_name,
    project_skill_drafts_root,
    project_skills_root,
    skill_directory_name,
    user_skills_root,
)
from atlas.agent.skills import parse_skill_document

_MAX_DESCRIPTION = 200
_MAX_STEP_TEXT = 500
_MAX_RESPONSE = 2000
_SMALL_RESOURCE_CHARACTERS = 2000
_DRAFTS_DIR = SKILL_DRAFTS_DIR_NAME
_SKILLS_DIR = SKILLS_DIR_NAME
_REDACTED = "***"
_RENAME_NOREPLACE = 1
_AT_FDCWD = -100
# A staging directory younger than this, with a live pid or no pid, is in use.
_STALE_ENABLING_SECONDS = 3600
_DRAFT_NOTICE = re.compile(
    r"^This skill is a draft\. Enable it with `/enable-skill [^`]+`\.\n?",
    re.MULTILINE,
)

# http(s) URLs are held aside, then restored after userinfo and secret query
# params are removed. ``file://`` is not held: the whole URL is a path.
_HTTP_URL = re.compile(r"https?://[^\s\"'\\<>]+", re.IGNORECASE)
_FILE_URL = re.compile(r"file://[^\s\"'\\<>]+", re.IGNORECASE)
_USERINFO = re.compile(r"^(https?://)[^/?#\s@]+@", re.IGNORECASE)
# ``(?<!:/)(?<!://)`` keeps the slashes in ``https://`` from matching when a
# URL was not held, and still redacts ``host:/srv`` and ``PATH=/usr/bin:/home``.
_ABS_UNIX = re.compile(r"(?<!:/)(?<!://)(?<![A-Za-z0-9_])/(?:[A-Za-z0-9._+-]+/)*[A-Za-z0-9._+-]+")
_ABS_WINDOWS = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/](?:[^\\/\s\"']+[\\/])*(?:[^\\/\s\"']+)")
# Prefixed credentials and JWTs. Bare hex is left alone so sha256 weight
# hashes and hex scene ids survive.
_KEY_SHAPED = re.compile(
    r"(?i)(?:"
    r"sk-(?:or-v1-|ant-(?:api|admin)\d{2}-|proj-)?[A-Za-z0-9_-]{16,}"
    r"|ghp_[A-Za-z0-9]{20,}"
    r"|github_pat_[A-Za-z0-9_]{20,}"
    r"|AKIA[0-9A-Z]{16}"
    r"|AIza[0-9A-Za-z_-]{20,}"
    r"|xox[baprs]-[A-Za-z0-9-]{10,}"
    r"|Bearer\s+[A-Za-z0-9._-]{8,}"
    r"|eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"
    r")"
)
_REPLACE_NOTE = (
    "replace=True overwrites a draft of the same name, "
    "including one written by the author_skill tool."
)
_SECRET_QUERY_NAMES = frozenset(
    {
        "api_key",
        "apikey",
        "api-key",
        "token",
        "access_token",
        "refresh_token",
        "id_token",
        "password",
        "passwd",
        "secret",
        "client_secret",
        "sig",
        "signature",
        "x-amz-signature",
        "x-amz-credential",
        "x-amz-security-token",
        "x-goog-signature",
        "key",
        "auth",
        "code",
    }
)


class SkillSaveError(ValueError):
    """The analysis cannot be saved as a skill."""


class AnalysisStep(BaseModel):
    """One successful tool call recorded in an analysis."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    text: str = ""
    artifacts: list[str] = Field(default_factory=list)


class AnalysisTurn(BaseModel):
    """One completed turn: the request, the procedure, and the final response."""

    request: str
    response: str
    steps: list[AnalysisStep] = Field(default_factory=list)


class FinishedAnalysis(BaseModel):
    """Turns that each completed with a final response."""

    turns: list[AnalysisTurn] = Field(default_factory=list)


class SavedSkill(BaseModel):
    """A skill file written under an ``.atlas`` root."""

    name: str
    description: str
    path: str
    enabled: bool = False
    note: str = ""


class ResourcePreview(BaseModel):
    """One bundled file shown before a draft is enabled."""

    path: str
    sha256: str
    characters: int
    lines: int
    text: str = ""
    omitted: str = ""


class SkillPreview(BaseModel):
    """What ``/enable-skill`` shows before it moves a draft."""

    name: str
    description: str
    path: str
    content_sha256: str
    body: str
    body_characters: int
    body_lines: int
    resources: list[ResourcePreview] = Field(default_factory=list)
    confirm: str


def enable_command(name: str) -> str:
    """Return the slash command that moves a draft skill into ``.atlas/skills``."""
    return f"/enable-skill {name}"


def successful_steps(run: AgentRun) -> list[AnalysisStep]:
    """Return the tool steps in ``run`` that produced a result.

    Failed calls and calls rejected by the tool-call limit are omitted. The
    procedure a skill replays is the steps that succeeded.

    Args:
        run: One agent advance.

    Returns:
        Successful steps, in order.
    """
    steps: list[AnalysisStep] = []
    for step in run.steps:
        if step.result is None:
            continue
        steps.append(
            AnalysisStep(
                name=step.call.name,
                arguments=copy.deepcopy(step.call.arguments),
                text=step.result.text,
                artifacts=list(step.result.artifacts),
            )
        )
    return steps


def logical_skill_path(scope: str, directory: str, name: str) -> str:
    """Return the same kind of label ``author_skill`` uses for a written skill.

    Project scope is ``.atlas/<directory>/<name>/SKILL.md``. User scope is
    ``~/.atlas/<directory>/<name>/SKILL.md``. Drafts use ``skills-drafts`` so
    the label cannot be mistaken for a loaded ``.atlas/skills`` skill.
    """
    prefix = "~/.atlas" if scope == "user" else ".atlas"
    return f"{prefix}/{directory}/{name}/SKILL.md"


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
    """Write a draft skill for ``analysis`` under the shared drafts root.

    Project drafts use :func:`project_skill_drafts_root`, the same directory
    the ``author_skill`` tool writes. ``replace`` overwrites that draft,
    including one the tool already wrote. The skill stays a draft until
    ``/enable-skill`` moves it into ``.atlas/skills``.

    Args:
        workspace: Project directory that contains ``.atlas``.
        analysis: A finished analysis. An empty analysis, a blank final
            response, or an analysis with no successful tool step is refused.
        name: Skill slug. Lowercase letters, digits, and hyphens.
        description: One-line summary. When omitted, the first request is used.
        replace: Overwrite an existing draft skill of the same name, including
            a draft written by the ``author_skill`` tool.
        secrets: Extra substrings replaced with ``***`` before the file is written.
            Absolute paths and key-shaped strings are scrubbed as well.
        scope: ``project`` or ``user``.
        home: User home for ``scope="user"``.

    Returns:
        The saved skill. ``path`` matches the tool's logical label.
        ``enabled`` is false. ``note`` says that ``replace=True`` overwrites
        a tool draft.

    Raises:
        SkillSaveError: If the name, the analysis, or the destination is not
            safe to write. The message never includes a host path.
    """
    _require_saveable(analysis)
    slug = _skill_slug(name)
    summary = _description(analysis, description, secrets).replace("---", "-")
    body = render_procedure(analysis, secrets=secrets, enable=enable_command(slug))
    draft = SkillDraft(name=slug, description=summary, body=body)
    try:
        written = author_skill(
            _drafts_root(workspace, scope, home),
            draft,
            overwrite=replace,
        )
    except SkillAuthorError as exc:
        raise SkillSaveError("; ".join(exc.errors)) from exc
    written_name = written.name
    return SavedSkill(
        name=written_name,
        description=summary,
        path=logical_skill_path(scope, _DRAFTS_DIR, written_name),
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
    """Describe a draft without moving it.

    The description comes from :func:`parse_skill_document`. ``content_sha256``
    is the SHA-256 of a manifest listing every file that would be published,
    each as its relative path plus the SHA-256 of its bytes. ``SKILL.md`` is
    hashed after the draft notice is removed. ``body`` is the full instruction
    text. Small text resources include their text; a larger or binary file
    reports its size and says the body is not shown.

    Args:
        workspace: Project directory that contains ``.atlas``.
        name: Skill slug previously written by :func:`save_skill`.
        scope: ``project`` or ``user``.
        home: User home for ``scope="user"``.

    Returns:
        The description, full body, manifest hash, resource previews, and
        confirm command. The command ends with the first eight hex characters
        of ``content_sha256``.

    Raises:
        SkillSaveError: If the draft is missing or the loader would skip it.
            The message never includes a host path.
    """
    try:
        loaded = _load_draft(workspace, name, scope=scope, home=home)
    except OSError as exc:
        raise SkillSaveError("could not enable skill") from exc
    published = _published_text(loaded.text)
    digest = _manifest_sha256(loaded.source, skill_md=published.encode("utf-8"))
    body = _published_body(loaded.body)
    return SkillPreview(
        name=loaded.slug,
        description=loaded.description,
        path=logical_skill_path(scope, _DRAFTS_DIR, loaded.slug),
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
    """Move ``skills-drafts/<name>`` into ``.atlas/skills``.

    The draft notice is removed before the directory moves. Immediately
    before that move, the manifest of every file is hashed again and refused
    when it is not ``expected_sha256``.

    Args:
        workspace: Project directory that contains ``.atlas``.
        name: Skill slug previously written by :func:`save_skill`.
        expected_sha256: Manifest digest shown by :func:`preview_skill`.
        scope: ``project`` or ``user``.
        home: User home for ``scope="user"``.

    Returns:
        The enabled skill. ``path`` is the live ``.atlas/skills`` label.
        ``description`` is the parsed frontmatter description.

    Raises:
        SkillSaveError: If the draft is missing, the hash does not match, the
            frontmatter would be skipped by the loader, or the destination is
            not a real directory inside the workspace or home. The message
            never includes a host path.
    """
    try:
        return _enable_skill(
            workspace,
            name,
            expected_sha256,
            scope=scope,
            home=home,
        )
    except OSError as exc:
        raise SkillSaveError("could not enable skill") from exc


def _enable_skill(
    workspace: Path,
    name: str,
    expected_sha256: str,
    *,
    scope: str,
    home: Path | None,
) -> SavedSkill:
    loaded = _load_draft(workspace, name, scope=scope, home=home)
    published = _published_text(loaded.text)
    expected = expected_sha256.strip().lower()
    published_bytes = published.encode("utf-8")
    if _manifest_sha256(loaded.source, skill_md=published_bytes) != expected:
        raise SkillSaveError("skill hash does not match the preview")
    skill_file = loaded.source / "SKILL.md"
    _write_published(skill_file, published)
    if _manifest_sha256(loaded.source) != expected:
        raise SkillSaveError("skill hash does not match the preview")
    skills = _skills_root(workspace, scope, home)
    if not skills.exists():
        skills.mkdir(parents=True)
    destination = skills / loaded.slug
    base = _scope_base(workspace, scope, home)
    if destination.is_symlink() or not _inside_layout(base, destination, skills.parent):
        raise SkillSaveError("skill path escapes the atlas root")
    _rename_exclusive(loaded.source, destination)
    return SavedSkill(
        name=loaded.slug,
        description=loaded.description,
        path=logical_skill_path(scope, _SKILLS_DIR, loaded.slug),
        enabled=True,
        note=_REPLACE_NOTE,
    )


def render_procedure(
    analysis: FinishedAnalysis,
    *,
    secrets: Sequence[str] = (),
    enable: str = "",
) -> str:
    """Render the markdown instructions for ``analysis``, without frontmatter.

    Args:
        analysis: The finished analysis to record.
        secrets: Extra substrings replaced with ``***``. Absolute paths and
            key-shaped strings are scrubbed as well.
        enable: Slash command that moves this draft into ``.atlas/skills``.

    Returns:
        The skill body.
    """
    lines = [
        "Replay this procedure when the user asks for the same kind of result. "
        "It was saved from a finished analysis.",
        "",
    ]
    if enable:
        lines.append(f"This skill is a draft. Enable it with `{enable}`.")
        lines.append("")
    for index, turn in enumerate(analysis.turns, start=1):
        lines.append(f"## Turn {index}")
        lines.append("")
        lines.append("Request: " + _redact_text(turn.request.strip(), secrets))
        lines.append("")
        if turn.steps:
            for step_index, step in enumerate(turn.steps, start=1):
                arguments = _arguments_json(step.arguments, secrets)
                outcome = _one_line(_redact_text(step.text, secrets), _MAX_STEP_TEXT)
                lines.append(f"{step_index}. `{step.name}` {arguments} -> {outcome}")
                if step.artifacts:
                    artifacts = ", ".join(_redact_text(item, secrets) for item in step.artifacts)
                    lines.append("   artifacts: " + artifacts)
            lines.append("")
        lines.append("Result: " + _one_line(_redact_text(turn.response, secrets), _MAX_RESPONSE))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _require_saveable(analysis: FinishedAnalysis) -> None:
    if not analysis.turns:
        raise SkillSaveError("no finished analysis to save")
    if not analysis.turns[-1].response.strip():
        raise SkillSaveError("analysis has no final response")
    if not any(turn.steps for turn in analysis.turns):
        raise SkillSaveError("finished analysis has no successful tool step")


def _description(
    analysis: FinishedAnalysis, description: str | None, secrets: Sequence[str]
) -> str:
    if not analysis.turns and description is None:
        raise SkillSaveError("no finished analysis to save")
    raw = description if description is not None else analysis.turns[0].request
    text = _redact_text(" ".join(raw.split()), secrets)
    if not text:
        raise SkillSaveError("skill description is empty")
    if len(text) > _MAX_DESCRIPTION:
        text = text[: _MAX_DESCRIPTION - 1].rstrip() + "…"
    return text


def _skill_slug(name: str) -> str:
    try:
        return skill_directory_name(SkillDraft(name=name, description="draft", body="draft"))
    except SkillAuthorError as exc:
        raise SkillSaveError("; ".join(exc.errors)) from exc


def skill_slug(name: str) -> str:
    """Return the drafts directory name for ``name``.

    Case is folded first, so ``Coast-Check`` and ``coast-check`` name the
    same skill. Saving a skill still requires the lowercase form.

    Args:
        name: A skill name, possibly with different case.

    Returns:
        The normalized directory name.

    Raises:
        SkillSaveError: If the folded name is not a skill slug.
    """
    folded = unicodedata.normalize("NFKC", name).strip().lower()
    return _skill_slug(folded)


def _scope_base(workspace: Path, scope: str, home: Path | None) -> Path:
    if scope == "user":
        if home is None:
            raise SkillSaveError("user scope needs a home directory")
        return Path(home).expanduser()
    if scope != "project":
        raise SkillSaveError("scope must be project or user")
    return Path(workspace).expanduser()


def _drafts_root(workspace: Path, scope: str, home: Path | None) -> Path:
    """Return the drafts directory the ``author_skill`` tool writes."""
    base = _scope_base(workspace, scope, home)
    if scope == "user":
        atlas = user_atlas_root(base)
        drafts = user_skill_drafts_root(base)
    else:
        atlas = project_atlas_root(base)
        drafts = project_skill_drafts_root(base)
    _refuse_escape(base, drafts, atlas=atlas)
    if drafts.exists() and not drafts.is_dir():
        raise SkillSaveError("skills-drafts directory is not a directory")
    return drafts


def _skills_root(workspace: Path, scope: str, home: Path | None) -> Path:
    base = _scope_base(workspace, scope, home)
    if scope == "user":
        atlas = user_atlas_root(base)
        skills = user_skills_root(base)
    else:
        atlas = project_atlas_root(base)
        skills = project_skills_root(base)
    _refuse_escape(base, skills, atlas=atlas)
    if skills.exists() and not skills.is_dir():
        raise SkillSaveError("skills directory is not a directory")
    return skills


def _refuse_escape(base: Path, path: Path, *, atlas: Path) -> None:
    """Refuse a symlink, then keep ``path`` inside ``base`` or ``atlas``.

    ``author_skill`` follows a directory symlink, so a drafts or skills
    directory that is itself a symlink is refused even when the target stays
    inside ``base``. ``atlas`` is the project or user ``.atlas`` root from the
    layout helpers. ``ATLAS_HOME`` may sit outside ``base``; a path inside
    that root is still accepted.
    """
    for candidate in (atlas, path):
        if candidate.is_symlink():
            raise SkillSaveError(f"{candidate.name} directory must not be a symlink")
    if not _inside_layout(base, path, atlas):
        raise SkillSaveError("skill path escapes the atlas root")


def _inside_layout(base: Path, path: Path, atlas: Path) -> bool:
    """Return whether ``path`` stays inside the scope base or the atlas root."""
    return is_within(base, path) or is_within(atlas, path)


class _LoadedDraft:
    def __init__(self, slug: str, description: str, body: str, text: str, source: Path) -> None:
        self.slug = slug
        self.description = description
        self.body = body
        self.text = text
        self.source = source


def _load_draft(workspace: Path, name: str, *, scope: str, home: Path | None) -> _LoadedDraft:
    slug = skill_slug(name)
    base = _scope_base(workspace, scope, home)
    drafts = _drafts_root(workspace, scope, home)
    source = drafts / slug
    atlas = drafts.parent
    if source.is_symlink() or not source.is_dir() or not _inside_layout(base, source, atlas):
        raise SkillSaveError("draft skill was not found")
    _refuse_symlinks(source)
    skill_file = source / "SKILL.md"
    if (
        skill_file.is_symlink()
        or not skill_file.is_file()
        or not _inside_layout(base, skill_file, atlas)
    ):
        raise SkillSaveError("draft skill was not found")
    try:
        text = skill_file.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise SkillSaveError("draft skill is not UTF-8 text") from exc
    description, body = _validated_description(text, slug)
    return _LoadedDraft(slug, description, body, text, source)


def _published_text(text: str) -> str:
    """Return ``SKILL.md`` with the draft notice removed."""
    return _DRAFT_NOTICE.sub("", text, count=1)


def _published_body(body: str) -> str:
    """Return the instruction body with the draft notice removed."""
    return _published_text(body).strip()


def _bundled_resources(directory: Path) -> list[str]:
    """Return every file under ``directory`` except ``SKILL.md``.

    Names such as ``scripts/.run.enabling`` are real skill files. The enable
    temp lives beside the draft, not inside it, so nothing here is skipped by
    name.
    """
    resources: list[str] = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        relative = path.relative_to(directory).as_posix()
        if relative == "SKILL.md":
            continue
        resources.append(relative)
    return resources


def _manifest_payload(records: list[dict[str, str]]) -> str:
    """Encode path/hash records so spaces and newlines in names stay distinct.

    The text is a JSON list of ``{"path", "sha256"}`` objects. ``json.dumps``
    escapes those characters, so two different names cannot form the same line.
    """
    return json.dumps(records, ensure_ascii=False, separators=(",", ":"))


def _manifest_sha256(directory: Path, *, skill_md: bytes | None = None) -> str:
    """Hash a JSON manifest of every file that enable would publish.

    ``SKILL.md`` uses ``skill_md`` when that is given, so the preview can hash
    the notice-stripped bytes before they are written. Every other file in
    the directory is included.
    """
    skill_bytes = skill_md if skill_md is not None else (directory / "SKILL.md").read_bytes()
    files = [("SKILL.md", skill_bytes)]
    files.extend(
        (relative, (directory / relative).read_bytes())
        for relative in _bundled_resources(directory)
    )
    records = [
        {"path": path, "sha256": hashlib.sha256(data).hexdigest()}
        for path, data in sorted(files, key=lambda item: item[0])
    ]
    return hashlib.sha256(_manifest_payload(records).encode("utf-8")).hexdigest()


def _resource_previews(directory: Path) -> list[ResourcePreview]:
    """Describe bundled files, including the text of small UTF-8 resources."""
    previews: list[ResourcePreview] = []
    for relative in _bundled_resources(directory):
        data = (directory / relative).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            previews.append(
                ResourcePreview(
                    path=relative,
                    sha256=digest,
                    characters=len(data),
                    lines=0,
                    omitted=f"{len(data)} bytes, body not shown",
                )
            )
            continue
        characters = len(text)
        lines = len(text.splitlines())
        if characters <= _SMALL_RESOURCE_CHARACTERS:
            previews.append(
                ResourcePreview(
                    path=relative,
                    sha256=digest,
                    characters=characters,
                    lines=lines,
                    text=text,
                )
            )
            continue
        previews.append(
            ResourcePreview(
                path=relative,
                sha256=digest,
                characters=characters,
                lines=lines,
                omitted=f"{characters} characters, {lines} lines, body not shown",
            )
        )
    return previews


def _write_published(skill_file: Path, published: str) -> None:
    """Replace ``SKILL.md`` without leaving a temp file inside the draft."""
    data = published.encode("utf-8")
    if skill_file.read_bytes() == data:
        return
    temporary = skill_file.parent.parent / f".{skill_file.parent.name}.SKILL.md.enabling"
    temporary.write_bytes(data)
    try:
        os.replace(temporary, skill_file)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


def _refuse_symlinks(directory: Path) -> None:
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise SkillSaveError("draft skill must not contain a symlink")


def _validated_description(text: str, folder: str) -> tuple[str, str]:
    """Return the description and body when the loader would accept this skill."""
    try:
        fields, body = parse_skill_document(text)
    except ValueError as exc:
        raise SkillSaveError("draft skill frontmatter is invalid") from exc
    normalized, errors = normalize_skill_name(fields.get("name"))
    directory_key = unicodedata.normalize("NFKC", folder)
    if errors or normalized is None or normalized != directory_key:
        raise SkillSaveError("draft skill name must match its directory")
    description = fields.get("description")
    if not isinstance(description, str) or not description.strip():
        raise SkillSaveError("draft skill is missing a description")
    return description.strip(), body


def rename_without_replace(source: Path, target: Path) -> None:
    """Move the whole ``source`` directory onto ``target`` in one rename.

    ``renameat2(RENAME_NOREPLACE)`` refuses an existing destination. When that
    call is unavailable or returns ``EXDEV``, ``os.mkdir`` claims ``target``
    and one ``os.rename`` moves ``source`` onto that empty directory. Children
    are not moved one at a time, so a loader cannot observe ``SKILL.md``
    without the rest of the tree.
    """
    result = _renameat2(source, target, _RENAME_NOREPLACE)
    if result == 0:
        return
    if result == errno.EEXIST:
        raise SkillSaveError(f"Skill already exists: {target.name}")
    if result not in {None, errno.EXDEV}:
        raise SkillSaveError("could not enable skill")
    try:
        os.mkdir(target)
    except FileExistsError as exc:
        raise SkillSaveError(f"Skill already exists: {target.name}") from exc
    except OSError as exc:
        raise SkillSaveError("could not enable skill") from exc
    try:
        os.rename(source, target)
    except OSError as exc:
        _release_empty_claim(target)
        if exc.errno in {errno.EEXIST, errno.ENOTEMPTY}:
            raise SkillSaveError(f"Skill already exists: {target.name}") from exc
        raise


def _release_empty_claim(target: Path) -> None:
    """Remove the empty directory claimed for a rename that did not finish."""
    try:
        os.rmdir(target)
    except OSError:
        return


def _new_staging_path(target: Path) -> Path:
    """Return a staging directory no other enable will reuse.

    The pid and a random id are in the directory name. A marker file inside
    the tree would be copied into the published skill.
    """
    return target.with_name(f".{target.name}.enabling.{os.getpid()}.{uuid.uuid4().hex}")


def _skill_md_temp(source: Path) -> Path:
    """Return the ``SKILL.md`` temp written beside the drafts root."""
    return source.parent / f".{source.name}.SKILL.md.enabling"


def _is_enabling_name(name: str, skill: str) -> bool:
    prefix = f".{skill}.enabling"
    return name == prefix or name.startswith(f"{prefix}.")


def _enabling_pid(name: str, skill: str) -> int | None:
    """Return the pid encoded in ``.<skill>.enabling.<pid>.<id>``."""
    prefix = f".{skill}.enabling."
    if not name.startswith(prefix):
        return None
    token = name[len(prefix) :].split(".", 1)[0]
    if not token.isdigit():
        return None
    pid = int(token)
    if pid <= 0 or str(pid) != token:
        return None
    return pid


def _pid_alive(pid: int) -> bool:
    """Return whether ``pid`` is a running process.

    ``PermissionError`` means the process exists and belongs to someone else.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (OSError, OverflowError):
        return False
    return True


def _should_reap_enabling(path: Path, skill: str) -> bool:
    """Return whether ``path`` is a stale or dead-pid staging directory."""
    pid = _enabling_pid(path.name, skill)
    if pid is not None and not _pid_alive(pid):
        return True
    try:
        modified = path.lstat().st_mtime
    except OSError:
        return False
    return time.time() - modified > _STALE_ENABLING_SECONDS


def _unlink_file_or_symlink(path: Path) -> None:
    """Unlink a file or symlink without following it."""
    if not path.is_symlink() and not path.is_file():
        return
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise SkillSaveError("could not enable skill") from exc


def _reap_enabling(path: Path) -> None:
    """Remove one staging entry. A symlink is unlinked, not followed."""
    try:
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
        else:
            return
    except FileNotFoundError:
        return
    except OSError as exc:
        raise SkillSaveError("could not enable skill") from exc
    if path.exists() or path.is_symlink():
        raise SkillSaveError(f"enable was interrupted; {path.name} is still present")


def _recover_enabling(source: Path, target: Path) -> None:
    """Remove stale staging directories and a leftover ``SKILL.md`` temp.

    A staging directory is ``.<name>.enabling`` or
    ``.<name>.enabling.<pid>.<id>``. It is removed only when its mtime is
    older than ``_STALE_ENABLING_SECONDS`` or the pid in its name is not
    running. A young directory with a live pid, or with no pid, is left
    alone. The destination directory is never removed.
    """
    _unlink_file_or_symlink(_skill_md_temp(source))
    parent = target.parent
    if not parent.is_dir():
        return
    try:
        children = list(parent.iterdir())
    except OSError as exc:
        raise SkillSaveError("could not enable skill") from exc
    for child in children:
        if child.name == target.name or not _is_enabling_name(child.name, target.name):
            continue
        if not _should_reap_enabling(child, target.name):
            continue
        _reap_enabling(child)


def _rename_exclusive(source: Path, target: Path) -> None:
    """Move ``source`` to ``target`` without replacing an existing path.

    ``renameat2(RENAME_NOREPLACE)`` moves the draft in one call. When that
    call is unavailable or returns ``EXDEV``, the tree is copied to a unique
    ``.<name>.enabling.<pid>.<id>`` directory and that directory is renamed
    onto ``target`` once.
    """
    _recover_enabling(source, target)
    result = _renameat2(source, target, _RENAME_NOREPLACE)
    if result == 0:
        return
    if result == errno.EEXIST:
        raise SkillSaveError(f"Skill already exists: {target.name}")
    if result not in {None, errno.EXDEV}:
        raise SkillSaveError("could not enable skill")
    _stage_and_rename(source, target)


def _stage_and_rename(source: Path, target: Path) -> None:
    """Copy ``source`` to a unique hidden sibling, then rename that directory once."""
    temporary = _new_staging_path(target)
    try:
        shutil.copytree(source, temporary, symlinks=False)
        rename_without_replace(temporary, target)
    except FileExistsError as exc:
        raise SkillSaveError("could not enable skill") from exc
    except (OSError, shutil.Error, SkillSaveError):
        _discard_tree(temporary)
        raise
    _discard_tree(source)


def _discard_tree(path: Path) -> None:
    if path.is_symlink() or not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
        return
    path.unlink(missing_ok=True)


def _renameat2(source: Path, target: Path, flags: int) -> int | None:
    """Return 0, an errno, or None when ``renameat2`` is unavailable."""
    library = ctypes.util.find_library("c")
    if not library:
        return None
    libc = ctypes.CDLL(library, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        return None
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(target),
        flags,
    )
    if result == 0:
        return 0
    err = ctypes.get_errno()
    if err in {errno.EINVAL, errno.ENOSYS, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EPERM}:
        return None
    return err


def _arguments_json(arguments: dict[str, Any], secrets: Sequence[str]) -> str:
    """Redact string values, then serialize.

    json.dumps doubles backslashes, so a Windows path has to be scrubbed on
    the original string. Scrubbing the JSON text misses that doubled form.
    """
    redacted = _redact_data(arguments, secrets)
    try:
        return json.dumps(redacted, sort_keys=True, ensure_ascii=False)
    except TypeError:
        flattened = {
            str(key): _redact_text(str(value), secrets) for key, value in arguments.items()
        }
        return json.dumps(flattened, sort_keys=True, ensure_ascii=False)


def _redact_data(value: Any, secrets: Sequence[str]) -> Any:
    if isinstance(value, str):
        return _redact_text(value, secrets)
    if isinstance(value, dict):
        return {
            _redact_text(key, secrets) if isinstance(key, str) else key: _redact_data(item, secrets)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_data(item, secrets) for item in value]
    return value


def _one_line(text: str, limit: int) -> str:
    collapsed = " ".join(text.split())
    if not collapsed:
        return "(no output)"
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + "…"


def _redact_text(text: str, secrets: Sequence[str]) -> str:
    """Replace secrets, prefixed keys, and absolute paths with ``***``."""
    for secret in sorted({secret for secret in secrets if secret}, key=len, reverse=True):
        text = text.replace(secret, _REDACTED)
    held: list[str] = []

    def _hold(match: re.Match[str]) -> str:
        held.append(match.group(0))
        return f"\ue000url{len(held) - 1}\ue001"

    text = _scrub_secret_fragments(text)
    text = _HTTP_URL.sub(_hold, text)
    text = _KEY_SHAPED.sub(_REDACTED, text)
    text = _FILE_URL.sub(_REDACTED, text)
    text = _ABS_WINDOWS.sub(_REDACTED, text)
    text = _ABS_UNIX.sub(_REDACTED, text)
    for index, url in enumerate(held):
        text = text.replace(f"\ue000url{index}\ue001", _sanitize_url(url))
    return text


def _sanitize_url(url: str) -> str:
    """Drop userinfo and secret query params, then scrub credentials in the rest."""
    url = _USERINFO.sub(r"\1", url)
    url = _strip_secret_query(url)
    url = _scrub_secret_fragments(url)
    return _KEY_SHAPED.sub(_REDACTED, url)


def _scrub_secret_fragments(text: str) -> str:
    """Replace values in fragments such as ``#access_token=``."""
    names = "|".join(re.escape(name) for name in sorted(_SECRET_QUERY_NAMES, key=len, reverse=True))
    pattern = re.compile(rf"(?i)#({names})=([^\s\"'\\&#]*)")
    return pattern.sub(r"#\1=***", text)


def _strip_secret_query(url: str) -> str:
    if "?" not in url:
        return url
    base, rest = url.split("?", 1)
    fragment = ""
    if "#" in rest:
        rest, fragment = rest.split("#", 1)
        fragment = "#" + fragment
    kept = [
        part
        for part in rest.split("&")
        if part and part.split("=", 1)[0].lower() not in _SECRET_QUERY_NAMES
    ]
    if not kept:
        return base + fragment
    return base + "?" + "&".join(kept) + fragment
