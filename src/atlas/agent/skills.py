"""Load Agent Skills and activate one by name.

Skills follow the Agent Skills layout: a directory whose ``SKILL.md`` starts
with YAML frontmatter (``name`` and ``description``) and continues with the
instruction body. Discovery reads two scopes. Project skills override user
skills. Inside one scope, ``.atlas/skills`` overrides ``.agents/skills``.

The system prompt receives only names and descriptions. ``use_skill`` reads
the instruction body when a task matches a description.
"""

from __future__ import annotations

import html
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal, cast

from pydantic import BaseModel, Field, create_model

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import Tool, ToolRegistry, ToolResult

_LOGGER = logging.getLogger(__name__)

_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_FRONTMATTER_KEY = re.compile(r"^([A-Za-z0-9_-]+):\s*(.*)$")
_MAX_NAME_LENGTH = 64
_MAX_DESCRIPTION_LENGTH = 1024
_MAX_LISTED_FILES = 50

_SKILL_INSTRUCTIONS = """The following skills provide specialized instructions for specific tasks.
When a task matches a skill's description, call the use_skill tool with the skill's name to load its full instructions."""


@dataclass(frozen=True)
class Skill:
    """One discovered skill.

    ``directory_label`` is a logical path (``~/.atlas/skills/name`` or
    ``.atlas/skills/name``). It is safe to show to the model. ``location`` and
    ``root`` stay on the host and are not included in prompts or tool results.
    """

    name: str
    description: str
    location: Path
    root: Path
    directory_label: str


def load_skills(*, workspace: Path | None = None, home: Path | None = None) -> list[Skill]:
    """Discover skills under ``workspace`` and ``home``.

    Missing directories contribute nothing. A skill file that cannot be parsed,
    or that has no name or description, is skipped. Project skills replace user
    skills with the same name. Within one scope, a skill in ``.atlas/skills``
    replaces one in ``.agents/skills``. The result is sorted by name.

    Args:
        workspace: Project root. ``.agents/skills`` and ``.atlas/skills`` are
            scanned when the directory exists.
        home: User home. ``~/.agents/skills`` and ``~/.atlas/skills`` are
            scanned when the directory exists.

    Returns:
        The skills that should be disclosed to the model.
    """
    found: dict[str, Skill] = {}
    for directory, label_prefix in _skill_roots(workspace=workspace, home=home):
        if not directory.is_dir():
            continue
        for skill in _scan_root(directory, label_prefix):
            previous = found.get(skill.name)
            if previous is not None:
                _LOGGER.warning(
                    "Skill %s from %s overrides %s",
                    skill.name,
                    skill.directory_label,
                    previous.directory_label,
                )
            found[skill.name] = skill
    return [found[name] for name in sorted(found)]


def render_system_prompt(base: str, skills: Sequence[Skill]) -> str:
    """Return ``base`` plus a skill catalog, or ``base`` when there are none.

    The catalog contains names and descriptions only. Instruction bodies stay
    out of the prompt until ``use_skill`` loads one.

    Args:
        base: The agent's system prompt without skills.
        skills: Skills from :func:`load_skills`.

    Returns:
        The system prompt to send on this session.
    """
    if not skills:
        return base
    blocks = ["<available_skills>"]
    for skill in sorted(skills, key=lambda item: item.name):
        blocks.append("  <skill>")
        blocks.append(f"    <name>{html.escape(skill.name)}</name>")
        blocks.append(f"    <description>{html.escape(skill.description)}</description>")
        blocks.append("  </skill>")
    blocks.append("</available_skills>")
    return base + "\n\n" + _SKILL_INSTRUCTIONS + "\n\n" + "\n".join(blocks)


def register_skill_tool(registry: ToolRegistry, skills: Sequence[Skill]) -> None:
    """Register ``use_skill`` when ``skills`` is non-empty.

    An empty catalog registers nothing, so the model is not offered a tool
    with no valid names.

    Args:
        registry: Session tool allow-list.
        skills: Skills the tool is allowed to activate.
    """
    if not skills:
        return
    registry.register(UseSkillTool(skills))


class UseSkillTool(Tool):
    """Load one skill's instructions and a listing of its other files.

    The name argument is limited to the discovered catalog. The result is the
    markdown body, without frontmatter, plus relative paths of bundled files.
    Those files are not read here.
    """

    name = "use_skill"
    description = (
        "Load one skill's full instructions by name and follow them. "
        "Call this when the task matches a skill listed in the system prompt."
    )
    trust: ClassVar[Literal["default", "opt_in"]] = "default"

    def __init__(self, skills: Sequence[Skill]) -> None:
        ordered = tuple(sorted(skills, key=lambda skill: skill.name))
        if not ordered:
            raise ValueError("use_skill requires at least one skill")
        self._by_name = {skill.name: skill for skill in ordered}
        names = tuple(self._by_name)
        name_type = cast(Any, Literal).__getitem__(names)
        self._input_model = create_model(
            "UseSkillInput",
            name=(name_type, Field(description="Name of the skill to load.")),
        )

    @property
    def input_model(self) -> type[BaseModel]:
        return self._input_model

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        del store  # The skill catalog is fixed at session start, not read from the sandbox.
        parsed = self._input_model.model_validate(arguments).model_dump()
        name = parsed["name"]
        if not isinstance(name, str):
            raise ValueError("skill name must be a string")
        return ToolResult(text=_render_skill_content(self._by_name[name]))


def _skill_roots(*, workspace: Path | None, home: Path | None) -> list[tuple[Path, str]]:
    """Return skill directories from lowest precedence to highest."""
    roots: list[tuple[Path, str]] = []
    if home is not None:
        roots.append((home / ".agents" / "skills", "~/.agents/skills"))
        roots.append((home / ".atlas" / "skills", "~/.atlas/skills"))
    if workspace is not None:
        roots.append((workspace / ".agents" / "skills", ".agents/skills"))
        roots.append((workspace / ".atlas" / "skills", ".atlas/skills"))
    return roots


def _scan_root(directory: Path, label_prefix: str) -> list[Skill]:
    """Load immediate child skills. The first name in sorted order wins."""
    found: dict[str, Skill] = {}
    children = sorted(
        (
            child
            for child in directory.iterdir()
            if child.is_dir() and not child.name.startswith(".")
        ),
        key=lambda child: child.name,
    )
    for child in children:
        skill = _load_skill_dir(child, label_prefix)
        if skill is None:
            continue
        if skill.name in found:
            _LOGGER.warning(
                "Skill %s from %s is shadowed by %s",
                skill.name,
                f"{label_prefix}/{child.name}",
                found[skill.name].directory_label,
            )
            continue
        found[skill.name] = skill
    return [found[name] for name in sorted(found)]


def _load_skill_dir(directory: Path, label_prefix: str) -> Skill | None:
    skill_md = directory / "SKILL.md"
    if not skill_md.is_file():
        return None
    text = _read_utf8(skill_md)
    if text is None:
        return None
    try:
        frontmatter, _body = _split_frontmatter(text)
        fields = _parse_frontmatter(frontmatter)
    except ValueError as exc:
        _LOGGER.warning("Skipping skill %s: %s", f"{label_prefix}/{directory.name}", exc)
        return None
    name = fields.get("name")
    description = fields.get("description")
    if not isinstance(name, str) or not name.strip():
        _LOGGER.warning("Skipping skill %s: missing name", f"{label_prefix}/{directory.name}")
        return None
    if not isinstance(description, str) or not description.strip():
        _LOGGER.warning(
            "Skipping skill %s: missing description",
            f"{label_prefix}/{directory.name}",
        )
        return None
    cleaned_name = name.strip()
    cleaned_description = description.strip()
    if "\n" in cleaned_name or "\r" in cleaned_name:
        _LOGGER.warning(
            "Skipping skill %s: name must be one line", f"{label_prefix}/{directory.name}"
        )
        return None
    _warn_spec(cleaned_name, cleaned_description, directory.name, label_prefix)
    try:
        root = directory.resolve()
    except OSError:
        _LOGGER.warning("Skipping skill %s: directory cannot be resolved", directory.name)
        return None
    return Skill(
        name=cleaned_name,
        description=cleaned_description,
        location=skill_md,
        root=root,
        directory_label=f"{label_prefix}/{directory.name}",
    )


def _warn_spec(name: str, description: str, directory_name: str, label_prefix: str) -> None:
    label = f"{label_prefix}/{directory_name}"
    if len(name) > _MAX_NAME_LENGTH or _NAME_RE.fullmatch(name) is None:
        _LOGGER.warning("Skill %s at %s does not match the Agent Skills name rules", name, label)
    if name != directory_name:
        _LOGGER.warning("Skill name %s does not match its directory %s", name, label)
    if len(description) > _MAX_DESCRIPTION_LENGTH:
        _LOGGER.warning("Skill %s description exceeds %s characters", name, _MAX_DESCRIPTION_LENGTH)


def _read_utf8(path: Path) -> str | None:
    try:
        data = path.read_bytes()
    except OSError:
        _LOGGER.warning("Could not read %s", path.name)
        return None
    if data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        _LOGGER.warning("Skipping skill file %s: not UTF-8", path.name)
        return None


def _split_frontmatter(text: str) -> tuple[str, str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("missing frontmatter")
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            frontmatter = "\n".join(lines[1:index])
            body = "\n".join(lines[index + 1 :]).strip()
            return frontmatter, body
    raise ValueError("frontmatter is not closed")


def _parse_frontmatter(text: str) -> dict[str, Any]:
    """Parse the small YAML subset used by skill frontmatter.

    Scalars may contain colons. ``>`` and ``|`` block scalars are accepted.
    A value that is empty and followed by indented ``key: value`` lines is a
    nested map. Anything else raises ``ValueError``.
    """
    fields: dict[str, Any] = {}
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.strip() or line.lstrip().startswith("#"):
            index += 1
            continue
        if line[0] in {" ", "\t"}:
            raise ValueError("frontmatter is not a mapping")
        match = _FRONTMATTER_KEY.match(line)
        if match is None:
            raise ValueError("frontmatter is not a mapping")
        key, raw = match.group(1), match.group(2).strip()
        if raw.startswith(">") or raw.startswith("|"):
            index += 1
            block: list[str] = []
            while index < len(lines) and (
                not lines[index].strip() or lines[index][0] in {" ", "\t"}
            ):
                block.append(lines[index])
                index += 1
            fields[key] = _block_scalar(raw, block)
            continue
        if raw == "":
            index += 1
            nested, index = _parse_nested(lines, index)
            fields[key] = nested
            continue
        fields[key] = _unquote(raw)
        index += 1
    return fields


def _parse_nested(lines: list[str], index: int) -> tuple[dict[str, str], int]:
    nested: dict[str, str] = {}
    while index < len(lines) and lines[index][:1] in {" ", "\t"}:
        stripped = lines[index].strip()
        index += 1
        if not stripped or stripped.startswith("#"):
            continue
        match = _FRONTMATTER_KEY.match(stripped)
        if match is None or not match.group(2).strip():
            raise ValueError("frontmatter is not a mapping")
        nested[match.group(1)] = _unquote(match.group(2).strip())
    return nested, index


def _block_scalar(style: str, lines: list[str]) -> str:
    content = [line.strip() for line in lines]
    while content and content[0] == "":
        content.pop(0)
    while content and content[-1] == "":
        content.pop()
    if style.startswith(">"):
        paragraphs: list[str] = []
        current: list[str] = []
        for line in content:
            if line == "":
                if current:
                    paragraphs.append(" ".join(current))
                    current = []
                continue
            current.append(line)
        if current:
            paragraphs.append(" ".join(current))
        return "\n".join(paragraphs)
    return "\n".join(content)


def _unquote(value: str) -> str:
    if len(value) < 2 or value[0] not in {'"', "'"} or value[-1] != value[0]:
        return value
    inner = value[1:-1]
    if value[0] == "'":
        return inner.replace("''", "'")
    return inner.replace(r"\n", "\n").replace(r"\"", '"').replace(r"\\", "\\")


def _render_skill_content(skill: Skill) -> str:
    resolved = _resolved_file(skill.location)
    if resolved is None or not _inside(skill.root, resolved):
        raise FileNotFoundError(f"Skill {skill.name} is missing")
    text = _read_utf8(skill.location)
    if text is None:
        raise FileNotFoundError(f"Skill {skill.name} is missing")
    try:
        _frontmatter, body = _split_frontmatter(text)
    except ValueError as exc:
        raise ValueError(f"Skill {skill.name} could not be read") from exc
    label = html.escape(skill.directory_label, quote=True)
    name = html.escape(skill.name, quote=True)
    lines = [
        f'<skill_content name="{name}">',
        body,
        "",
        f"Skill directory: {label}",
        "Relative paths in this skill are relative to the skill directory.",
    ]
    files, truncated = _list_resources(skill.root)
    if files or truncated:
        lines.append("")
        lines.append("<skill_resources>")
        lines.extend(f"  <file>{html.escape(path)}</file>" for path in files)
        if truncated:
            lines.append("  <truncated>true</truncated>")
        lines.append("</skill_resources>")
    lines.append("</skill_content>")
    return "\n".join(lines)


def _list_resources(root: Path) -> tuple[list[str], bool]:
    files: list[str] = []
    truncated = False
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.name == "SKILL.md" and path.parent.resolve() == root:
            continue
        resolved = _resolved_file(path)
        if resolved is None or not _inside(root, resolved):
            continue
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            continue
        if len(files) >= _MAX_LISTED_FILES:
            truncated = True
            break
        files.append(relative)
    return files, truncated


def _resolved_file(path: Path) -> Path | None:
    try:
        if not path.is_file():
            return None
        return path.resolve()
    except OSError:
        return None


def _inside(root: Path, candidate: Path) -> bool:
    return candidate == root or root in candidate.parents
