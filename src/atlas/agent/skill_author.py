"""Write an Agent Skill directory that follows the Agent Skills spec.

A skill is a directory named after its frontmatter ``name`` with a
``SKILL.md`` of YAML frontmatter plus markdown instructions. This module
checks those constraints and writes the directory. Discovering skills and
activating them in a session are separate.
"""

from __future__ import annotations

import os
import stat
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from atlas.agent.layout import project_atlas_root, user_atlas_root

MAX_SKILL_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024
MAX_COMPATIBILITY_LENGTH = 500

_FRONTMATTER_DELIMITER = "---"


class SkillAuthorError(ValueError):
    """The skill was not written.

    ``errors`` lists every problem found. Spec and path checks run before any
    file is created or replaced.
    """

    def __init__(self, errors: Sequence[str]) -> None:
        if not errors:
            raise ValueError("SkillAuthorError requires at least one error")
        self.errors = tuple(errors)
        super().__init__("; ".join(self.errors))


@dataclass(frozen=True)
class SkillDraft:
    """One skill to write as ``<name>/SKILL.md``.

    ``allowed_tools`` is the experimental ``allowed-tools`` frontmatter field.
    ``metadata`` values are strings. ``body`` is the markdown after the
    frontmatter.
    """

    name: str
    description: str
    body: str = ""
    license: str | None = None
    compatibility: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    allowed_tools: str | None = None


@dataclass(frozen=True)
class _PreparedSkill:
    name: str
    description: str
    body: str
    license: str | None
    compatibility: str | None
    metadata: dict[str, str]
    allowed_tools: str | None


def project_skills_root(workspace: Path) -> Path:
    """Return ``<workspace>/.atlas/skills`` without creating it.

    Args:
        workspace: Project directory that contains ``.atlas``.

    Returns:
        The project skills directory.
    """
    return project_atlas_root(workspace) / "skills"


def user_skills_root(home: Path | None = None) -> Path:
    """Return ``<home>/.atlas/skills`` without creating it.

    Args:
        home: Directory that contains ``.atlas``. ``None`` uses the real home.

    Returns:
        The user skills directory.
    """
    return user_atlas_root(home) / "skills"


def validate_skill(draft: SkillDraft) -> list[str]:
    """Return spec violations for ``draft``.

    An empty list means :func:`render_skill_md` can render the draft. The
    directory written for the skill uses the normalized name, so the
    frontmatter name matches that directory by construction.

    Args:
        draft: Skill fields to check.

    Returns:
        Human-readable problems. Empty when the draft is writable.
    """
    _prepared, errors = _prepare(draft)
    return errors


def render_skill_md(draft: SkillDraft) -> str:
    """Render ``SKILL.md`` text for ``draft``.

    Args:
        draft: Skill fields to render.

    Returns:
        The ``SKILL.md`` document, including the closing newline.

    Raises:
        SkillAuthorError: If ``draft`` violates the Agent Skills spec.
    """
    prepared, errors = _prepare(draft)
    if prepared is None:
        raise SkillAuthorError(errors)
    return _render(prepared)


def author_skill(
    skills_root: Path,
    draft: SkillDraft,
    *,
    files: Mapping[str, str] | None = None,
    overwrite: bool = False,
) -> Path:
    """Write ``<skills_root>/<name>/SKILL.md`` and any bundled ``files``.

    ``files`` maps paths relative to the skill directory (for example
    ``scripts/extract.py``) to UTF-8 text. Paths that leave the skill
    directory, name ``SKILL.md``, or address a symlink or hardlink are
    refused. An existing skill is left untouched unless ``overwrite`` is true,
    and a failed validation writes nothing.

    Args:
        skills_root: Directory that will contain one subdirectory per skill.
        draft: Skill fields to write.
        files: Extra text files bundled inside the skill directory.
        overwrite: Replace an existing ``SKILL.md`` and listed files.

    Returns:
        The skill directory that was written.

    Raises:
        SkillAuthorError: If the draft, a bundled path, or the destination is
            not safe to write.
    """
    prepared, errors = _prepare(draft)
    if prepared is None:
        raise SkillAuthorError(errors)
    bundled = {} if files is None else files
    root = Path(skills_root).expanduser().resolve()
    destination = root / prepared.name
    errors = _destination_errors(root, destination, bundled, overwrite=overwrite)
    if errors:
        raise SkillAuthorError(errors)

    rendered = _render(prepared)
    root.mkdir(parents=True, exist_ok=True)
    destination.mkdir(exist_ok=True)
    for relative, content in sorted(bundled.items()):
        path, locate_error = _locate_file(destination, relative)
        if path is None or locate_error is not None:
            raise SkillAuthorError([locate_error or "Skill file path is invalid"])
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_text(path, content, overwrite=overwrite and path.exists())
    _write_text(destination / "SKILL.md", rendered, overwrite=overwrite)
    return destination


def _prepare(draft: SkillDraft) -> tuple[_PreparedSkill | None, list[str]]:
    errors: list[str] = []
    name, name_errors = _normalize_name(draft.name)
    errors.extend(name_errors)
    description, description_errors = _required_text(
        draft.description, "description", MAX_DESCRIPTION_LENGTH
    )
    errors.extend(description_errors)
    license_text, license_errors = _optional_text(draft.license, "license", None)
    errors.extend(license_errors)
    compatibility, compatibility_errors = _optional_text(
        draft.compatibility, "compatibility", MAX_COMPATIBILITY_LENGTH
    )
    errors.extend(compatibility_errors)
    allowed_tools, allowed_errors = _optional_text(draft.allowed_tools, "allowed-tools", None)
    errors.extend(allowed_errors)
    metadata, metadata_errors = _normalize_metadata(draft.metadata)
    errors.extend(metadata_errors)
    body = draft.body.strip() if isinstance(draft.body, str) else ""
    if not isinstance(draft.body, str):
        errors.append("Field 'body' must be a string")
    if errors or name is None or description is None:
        return None, errors
    return (
        _PreparedSkill(
            name=name,
            description=description,
            body=body,
            license=license_text,
            compatibility=compatibility,
            metadata=metadata,
            allowed_tools=allowed_tools,
        ),
        [],
    )


def _normalize_name(name: object) -> tuple[str | None, list[str]]:
    if not isinstance(name, str) or not name.strip():
        return None, ["Field 'name' must be a non-empty string"]
    normalized = unicodedata.normalize("NFKC", name.strip())
    errors: list[str] = []
    if len(normalized) > MAX_SKILL_NAME_LENGTH:
        errors.append(
            f"Skill name '{normalized}' exceeds {MAX_SKILL_NAME_LENGTH} character limit "
            f"({len(normalized)} chars)"
        )
    if normalized != normalized.lower():
        errors.append(f"Skill name '{normalized}' must be lowercase")
    if normalized.startswith("-") or normalized.endswith("-"):
        errors.append("Skill name cannot start or end with a hyphen")
    if "--" in normalized:
        errors.append("Skill name cannot contain consecutive hyphens")
    if not all(character.isalnum() or character == "-" for character in normalized):
        errors.append(
            f"Skill name '{normalized}' contains invalid characters. "
            "Only letters, digits, and hyphens are allowed."
        )
    if _contains_delimiter(normalized):
        errors.append("Frontmatter field 'name' cannot contain '---'")
    return normalized, errors


def _required_text(value: object, field_name: str, limit: int) -> tuple[str | None, list[str]]:
    if not isinstance(value, str) or not value.strip():
        return None, [f"Field '{field_name}' must be a non-empty string"]
    text = value.strip()
    errors: list[str] = []
    if len(text) > limit:
        label = "Description" if field_name == "description" else field_name
        errors.append(f"{label} exceeds {limit} character limit ({len(text)} chars)")
    errors.extend(_delimiter_errors(text, field_name))
    return text, errors


def _optional_text(
    value: object, field_name: str, limit: int | None
) -> tuple[str | None, list[str]]:
    if value is None:
        return None, []
    if not isinstance(value, str):
        return None, [f"Field '{field_name}' must be a string"]
    text = value.strip()
    if not text:
        return None, []
    errors: list[str] = []
    if limit is not None and len(text) > limit:
        errors.append(
            f"Compatibility exceeds {limit} character limit ({len(text)} chars)"
            if field_name == "compatibility"
            else f"Field '{field_name}' exceeds {limit} character limit ({len(text)} chars)"
        )
    errors.extend(_delimiter_errors(text, field_name))
    return text, errors


def _normalize_metadata(metadata: object) -> tuple[dict[str, str], list[str]]:
    if metadata is None:
        return {}, []
    if isinstance(metadata, str) or not isinstance(metadata, Mapping):
        return {}, ["Field 'metadata' must be a mapping of strings"]
    errors: list[str] = []
    cleaned: dict[str, str] = {}
    for key, value in metadata.items():
        if not isinstance(key, str) or not key.strip() or "\n" in key or "\r" in key:
            errors.append("Field 'metadata' keys must be non-empty single-line strings")
            continue
        if not isinstance(value, str):
            errors.append(f"Field 'metadata' value for {key!r} must be a string")
            continue
        normalized_key = key.strip()
        if _contains_delimiter(normalized_key) or _contains_delimiter(value):
            errors.append("Frontmatter field 'metadata' cannot contain '---'")
            continue
        cleaned[normalized_key] = value
    return cleaned, errors


def _delimiter_errors(value: str, field_name: str) -> list[str]:
    if _contains_delimiter(value):
        return [f"Frontmatter field '{field_name}' cannot contain '---'"]
    return []


def _contains_delimiter(value: str) -> bool:
    return _FRONTMATTER_DELIMITER in value


def _render(prepared: _PreparedSkill) -> str:
    lines = [
        _FRONTMATTER_DELIMITER,
        f"name: {_yaml_string(prepared.name)}",
        f"description: {_yaml_string(prepared.description)}",
    ]
    if prepared.license is not None:
        lines.append(f"license: {_yaml_string(prepared.license)}")
    if prepared.compatibility is not None:
        lines.append(f"compatibility: {_yaml_string(prepared.compatibility)}")
    if prepared.metadata:
        lines.append("metadata:")
        for key in sorted(prepared.metadata):
            lines.append(f"  {_yaml_string(key)}: {_yaml_string(prepared.metadata[key])}")
    if prepared.allowed_tools is not None:
        lines.append(f"allowed-tools: {_yaml_string(prepared.allowed_tools)}")
    lines.append(_FRONTMATTER_DELIMITER)
    text = "\n".join(lines) + "\n"
    if prepared.body:
        text += "\n" + prepared.body + "\n"
    return text


def _yaml_string(value: str) -> str:
    pieces: list[str] = []
    for character in value:
        if character == "\\":
            pieces.append("\\\\")
        elif character == '"':
            pieces.append('\\"')
        elif character == "\n":
            pieces.append("\\n")
        elif character == "\r":
            pieces.append("\\r")
        elif character == "\t":
            pieces.append("\\t")
        elif ord(character) < 0x20:
            pieces.append(f"\\u{ord(character):04x}")
        else:
            pieces.append(character)
    return '"' + "".join(pieces) + '"'


def _destination_errors(
    root: Path,
    destination: Path,
    files: Mapping[str, str],
    *,
    overwrite: bool,
) -> list[str]:
    if root.exists() and not root.is_dir():
        return ["Skills root is not a directory"]
    if destination.is_symlink():
        return ["Skill directory escapes the skills root"]
    if destination.exists() and not destination.is_dir():
        return ["Skill path is not a directory"]

    errors: list[str] = []
    errors.extend(_target_errors(destination / "SKILL.md", overwrite=overwrite, label="SKILL.md"))
    for relative, content in files.items():
        if not isinstance(relative, str) or not isinstance(content, str):
            errors.append("Skill file paths and contents must be strings")
            continue
        path, locate_error = _locate_file(destination, relative)
        if path is None:
            errors.append(locate_error or "Skill file path is invalid")
            continue
        errors.extend(_target_errors(path, overwrite=overwrite, label=relative))
    return errors


def _locate_file(skill_dir: Path, relative: str) -> tuple[Path | None, str | None]:
    if (
        not relative
        or relative != relative.strip()
        or relative.startswith("/")
        or "\\" in relative
        or "\x00" in relative
    ):
        return None, "Skill file path must be a relative path inside the skill directory"
    path = Path(relative)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        return None, "Skill file path escapes the skill directory"
    if len(path.parts) == 1 and path.name.lower() == "skill.md":
        return None, "SKILL.md is written from the skill draft"
    root = skill_dir.resolve()
    candidate = (root / path).resolve()
    if root not in candidate.parents:
        return None, "Skill file path escapes the skill directory"
    return candidate, None


def _target_errors(path: Path, *, overwrite: bool, label: str) -> list[str]:
    if path.is_symlink():
        return [f"Skill path {label} escapes the skill directory"]
    if not path.exists():
        return []
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        return [f"Skill path {label} is not a file"]
    if info.st_nlink != 1:
        return ["Skill path is a hardlinked file that may lead outside the skill directory"]
    if not overwrite:
        return [f"Skill already exists: {label}"]
    return []


def _write_text(path: Path, text: str, *, overwrite: bool) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW
    flags |= os.O_TRUNC if overwrite else os.O_EXCL
    try:
        fd = os.open(path, flags, 0o644)
    except OSError as exc:
        raise SkillAuthorError([f"Could not write skill file {path.name}: {exc.strerror}"]) from exc
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise SkillAuthorError(
                ["Skill path is a hardlinked file that may lead outside the skill directory"]
            )
        handle.write(text)
