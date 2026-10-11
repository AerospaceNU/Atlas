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
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from atlas.agent.layout import user_atlas_root
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
from atlas.agent.skills import _parse_frontmatter, _split_frontmatter

_MAX_DESCRIPTION = 200
_MAX_STEP_TEXT = 500
_MAX_RESPONSE = 2000
_DRAFTS_DIR = "skills-drafts"
_SKILLS_DIR = "skills"
_REDACTED = "***"
_RENAME_NOREPLACE = 1
_AT_FDCWD = -100
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
_PREVIEW_LIMIT = 400
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


class SkillPreview(BaseModel):
    """What ``/enable-skill`` shows before it moves a draft."""

    name: str
    description: str
    path: str
    content_sha256: str
    preview: str
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

    The description comes from the loader's frontmatter parser. ``content_sha256``
    is the SHA-256 of the ``SKILL.md`` bytes enable would publish, which is the
    same hash the loader records. ``preview`` is the start of the body.

    Args:
        workspace: Project directory that contains ``.atlas``.
        name: Skill slug previously written by :func:`save_skill`.
        scope: ``project`` or ``user``.
        home: User home for ``scope="user"``.

    Returns:
        The description, hash, and preview to show before promotion.

    Raises:
        SkillSaveError: If the draft is missing or the loader would skip it.
            The message never includes a host path.
    """
    try:
        loaded = _load_draft(workspace, name, scope=scope, home=home)
    except OSError as exc:
        raise SkillSaveError("could not enable skill") from exc
    published = _published_text(loaded.text)
    return SkillPreview(
        name=loaded.slug,
        description=loaded.description,
        path=logical_skill_path(scope, _DRAFTS_DIR, loaded.slug),
        content_sha256=hashlib.sha256(published.encode("utf-8")).hexdigest(),
        preview=_body_preview(loaded.body),
        confirm=f"{enable_command(loaded.slug)} confirm",
    )


def enable_skill(
    workspace: Path,
    name: str,
    *,
    scope: str = "project",
    home: Path | None = None,
) -> SavedSkill:
    """Move ``skills-drafts/<name>`` into ``.atlas/skills``.

    Call :func:`preview_skill` first so a person can see the description and
    body hash. The frontmatter is checked with the skills loader's parser
    before the directory moves. The ``name`` must match the folder. The draft
    notice is removed from the enabled file.

    Args:
        workspace: Project directory that contains ``.atlas``.
        name: Skill slug previously written by :func:`save_skill`.
        scope: ``project`` or ``user``.
        home: User home for ``scope="user"``.

    Returns:
        The enabled skill. ``path`` is the live ``.atlas/skills`` label.
        ``description`` is the parsed frontmatter description.

    Raises:
        SkillSaveError: If the draft is missing, the frontmatter would be
            skipped by the loader, or the destination is not a real directory
            inside the workspace or home. The message never includes a host path.
    """
    try:
        return _enable_skill(workspace, name, scope=scope, home=home)
    except OSError as exc:
        raise SkillSaveError("could not enable skill") from exc


def _enable_skill(
    workspace: Path,
    name: str,
    *,
    scope: str,
    home: Path | None,
) -> SavedSkill:
    loaded = _load_draft(workspace, name, scope=scope, home=home)
    skills = _skills_root(workspace, scope, home)
    if not skills.exists():
        skills.mkdir(parents=True)
    destination = skills / loaded.slug
    if destination.is_symlink() or not is_within(_scope_base(workspace, scope, home), destination):
        raise SkillSaveError("skill path escapes the atlas root")
    _rename_exclusive(loaded.source, destination)
    published = _published_text(loaded.text)
    _write_published(destination / "SKILL.md", published)
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
        drafts = user_atlas_root(base) / _DRAFTS_DIR
    else:
        drafts = project_skill_drafts_root(base)
    _refuse_escape(base, drafts)
    if drafts.exists() and not drafts.is_dir():
        raise SkillSaveError("skills-drafts directory is not a directory")
    return drafts


def _skills_root(workspace: Path, scope: str, home: Path | None) -> Path:
    base = _scope_base(workspace, scope, home)
    skills = user_skills_root(base) if scope == "user" else project_skills_root(base)
    _refuse_escape(base, skills)
    if skills.exists() and not skills.is_dir():
        raise SkillSaveError("skills directory is not a directory")
    return skills


def _refuse_escape(base: Path, path: Path) -> None:
    """Refuse a symlink, then require :func:`is_within` to keep ``path`` in ``base``.

    ``author_skill`` follows a directory symlink, so a drafts or skills
    directory that is itself a symlink is refused even when the target stays
    inside ``base``.
    """
    for candidate in (base / ".atlas", path):
        if candidate.is_symlink():
            raise SkillSaveError(f"{candidate.name} directory must not be a symlink")
    if not is_within(base, path):
        raise SkillSaveError("skill path escapes the atlas root")


class _LoadedDraft:
    def __init__(self, slug: str, description: str, body: str, text: str, source: Path) -> None:
        self.slug = slug
        self.description = description
        self.body = body
        self.text = text
        self.source = source


def _load_draft(workspace: Path, name: str, *, scope: str, home: Path | None) -> _LoadedDraft:
    slug = _skill_slug(name)
    base = _scope_base(workspace, scope, home)
    drafts = _drafts_root(workspace, scope, home)
    source = drafts / slug
    if source.is_symlink() or not source.is_dir() or not is_within(base, source):
        raise SkillSaveError("draft skill was not found")
    _refuse_symlinks(source)
    skill_file = source / "SKILL.md"
    if skill_file.is_symlink() or not skill_file.is_file() or not is_within(base, skill_file):
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


def _write_published(skill_file: Path, published: str) -> None:
    if skill_file.read_text(encoding="utf-8") == published:
        return
    temporary = skill_file.with_name(".SKILL.md.enabling")
    temporary.write_text(published, encoding="utf-8")
    os.replace(temporary, skill_file)


def _body_preview(body: str) -> str:
    notice_free = _published_text(body).strip()
    if len(notice_free) <= _PREVIEW_LIMIT:
        return notice_free
    return notice_free[: _PREVIEW_LIMIT - 1].rstrip() + "…"


def _refuse_symlinks(directory: Path) -> None:
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise SkillSaveError("draft skill must not contain a symlink")


def _validated_description(text: str, folder: str) -> tuple[str, str]:
    """Return the description and body when the loader would accept this skill."""
    try:
        frontmatter, body = _split_frontmatter(text)
        fields = _parse_frontmatter(frontmatter)
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


def _rename_exclusive(source: Path, target: Path) -> None:
    """Move ``source`` to ``target`` without replacing an existing path.

    ``renameat2(RENAME_NOREPLACE)`` claims the destination in one call. When
    the kernel cannot do that, ``mkdir`` claims the name and the children
    move in afterwards.
    """
    result = _renameat2(source, target, _RENAME_NOREPLACE)
    if result == 0:
        return
    if result == errno.EEXIST:
        raise SkillSaveError(f"Skill already exists: {target.name}")
    if result is not None:
        raise SkillSaveError("could not enable skill")
    try:
        os.mkdir(target)
    except FileExistsError as exc:
        raise SkillSaveError(f"Skill already exists: {target.name}") from exc
    for child in list(source.iterdir()):
        os.rename(child, target / child.name)
    os.rmdir(source)


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
    return _KEY_SHAPED.sub(_REDACTED, url)


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
