"""Validate and write Agent Skill directories (``<name>/SKILL.md``)."""

from __future__ import annotations

import os
import re
import shutil
import stat
import tempfile
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from atlas.agent.layout import (
    project_skill_drafts_root as project_skill_drafts_root,
)
from atlas.agent.layout import (
    project_skills_root as project_skills_root,
)
from atlas.agent.layout import (
    user_skills_root as user_skills_root,
)

MAX_SKILL_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024
MAX_COMPATIBILITY_LENGTH = 500

_FRONTMATTER_DELIMITER = "---"
_METADATA_KEY = re.compile(r"^[A-Za-z0-9_-]+$")
_FOLD_WIDTH = 74


class _YamlStr(str):
    style: str

    def __new__(cls, value: str, style: str = '"') -> _YamlStr:
        obj = super().__new__(cls, value)
        obj.style = style
        return obj


class _FrontmatterDumper(yaml.SafeDumper):
    pass


def _rep_yaml_str(dumper: yaml.SafeDumper, data: _YamlStr) -> yaml.ScalarNode:
    return dumper.represent_scalar("tag:yaml.org,2002:str", str(data), style=data.style)


_FrontmatterDumper.add_representer(_YamlStr, _rep_yaml_str)


class SkillAuthorError(ValueError):
    """Raised when a skill draft or destination fails validation."""

    def __init__(self, errors: Sequence[str]) -> None:
        if not errors:
            raise ValueError("SkillAuthorError requires at least one error")
        self.errors = tuple(errors)
        super().__init__("; ".join(self.errors))


@dataclass(frozen=True)
class SkillDraft:
    """Fields for one Agent Skill written as ``<name>/SKILL.md``."""

    name: str
    description: str
    body: str = ""
    license: str | None = None
    compatibility: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    allowed_tools: str | None = None


def normalize_skill_name(name: object) -> tuple[str | None, list[str]]:
    """NFKC-normalize a skill name and return any spec violations."""
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
    if not all(ch.isalnum() or ch == "-" for ch in normalized):
        errors.append(
            f"Skill name '{normalized}' contains invalid characters. "
            "Only letters, digits, and hyphens are allowed."
        )
    if _FRONTMATTER_DELIMITER in normalized:
        errors.append("Frontmatter field 'name' cannot contain '---'")
    return normalized, errors


def is_within(root: Path, path: Path) -> bool:
    """Return whether ``path`` resolves to ``root`` or a descendant inside it."""
    resolved_root = Path(root).expanduser().resolve()
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = resolved_root / candidate
    return candidate.resolve().is_relative_to(resolved_root)


def verify_contained_dir(
    base: Path,
    target: Path,
    *,
    atlas: Path | None = None,
    exact_root: Path | None = None,
) -> Path:
    """Verify ``target`` (and optional ``atlas`` parent) is not a symlink and stays contained."""
    for candidate in (atlas, target):
        if candidate is not None and candidate.is_symlink():
            raise SkillAuthorError([f"{candidate.name} directory must not be a symlink"])
    try:
        resolved = target.resolve()
    except OSError as exc:
        raise SkillAuthorError(["skill path escapes the atlas root"]) from exc
    if exact_root is not None and resolved != exact_root.expanduser().resolve():
        raise SkillAuthorError(["skill path escapes the atlas root"])
    allowed = (base,) if atlas is None else (base, atlas)
    if not any(is_within(root, resolved) for root in allowed):
        raise SkillAuthorError(["skill path escapes the atlas root"])
    return target


def validate_skill(draft: SkillDraft) -> list[str]:
    """Return spec violations for ``draft`` (empty when writable)."""
    return _prepare(draft)[1]


def render_skill_md(draft: SkillDraft) -> str:
    """Render ``SKILL.md`` text (including trailing newline) for ``draft``."""
    prepared, errors = _prepare(draft)
    if prepared is None:
        raise SkillAuthorError(errors)
    return _render(prepared)


def skill_directory_name(draft: SkillDraft) -> str:
    """Return the normalized directory name :func:`author_skill` writes for ``draft``."""
    prepared, errors = _prepare(draft)
    if prepared is None:
        raise SkillAuthorError(errors)
    return prepared.name


def author_skill(
    skills_root: Path,
    draft: SkillDraft,
    *,
    files: Mapping[str, str] | None = None,
    overwrite: bool = False,
) -> Path:
    """Write ``<skills_root>/<name>/SKILL.md`` and any bundled text ``files``."""
    prepared, errors = _prepare(draft)
    if prepared is None:
        raise SkillAuthorError(errors)
    bundled: Mapping[str, str] = {} if files is None else files
    root = Path(skills_root).expanduser().resolve()
    destination = root / prepared.name
    dest_errors = _destination_errors(root, destination, bundled, overwrite=overwrite)
    if dest_errors:
        raise SkillAuthorError(dest_errors)

    rendered = _render(prepared)
    root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=root, prefix=f".{prepared.name}.authoring."))
    try:
        if destination.exists():
            _copy_preserved(destination, staging, replaced={"SKILL.md", *bundled})
        for relative, content in sorted({**dict(bundled), "SKILL.md": rendered}.items()):
            if relative == "SKILL.md":
                path = staging / "SKILL.md"
            else:
                located, locate_error = _locate_file(staging, relative)
                if located is None:
                    raise SkillAuthorError([locate_error or "Skill file path is invalid"])
                path = located
            path.parent.mkdir(parents=True, exist_ok=True)
            _write_text(path, content, overwrite=False)
        _publish(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def _prepare(draft: SkillDraft) -> tuple[SkillDraft | None, list[str]]:
    name, errors = normalize_skill_name(draft.name)
    description, desc_errs = _check_text(
        draft.description, "description", MAX_DESCRIPTION_LENGTH, required=True
    )
    license_text, lic_errs = _check_text(draft.license, "license", None, required=False)
    compatibility, compat_errs = _check_text(
        draft.compatibility, "compatibility", MAX_COMPATIBILITY_LENGTH, required=False
    )
    allowed_tools, allow_errs = _check_text(
        draft.allowed_tools, "allowed-tools", None, required=False
    )
    metadata, meta_errs = _normalize_metadata(draft.metadata)
    errors.extend([*desc_errs, *lic_errs, *compat_errs, *allow_errs, *meta_errs])

    body = ""
    if not isinstance(draft.body, str):
        errors.append("Field 'body' must be a string")
    elif not (body := draft.body.strip()):
        errors.append("Field 'body' must be a non-empty string")

    if errors or name is None or description is None or not body:
        return None, errors
    return (
        SkillDraft(
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


def _check_text(
    value: object, field_name: str, limit: int | None, *, required: bool
) -> tuple[str | None, list[str]]:
    if value is None and not required:
        return None, []
    if not isinstance(value, str):
        msg = "a non-empty string" if required else "a string"
        return None, [f"Field '{field_name}' must be {msg}"]
    text = value.strip()
    if field_name == "description":
        text = text.replace("\r\n", "\n").replace("\r", "\n")
    if not text:
        return (
            (None, [f"Field '{field_name}' must be a non-empty string"]) if required else (None, [])
        )
    errors: list[str] = []
    if limit is not None and len(text) > limit:
        label = (
            field_name.capitalize()
            if field_name in {"description", "compatibility"}
            else f"Field '{field_name}'"
        )
        errors.append(f"{label} exceeds {limit} character limit ({len(text)} chars)")
    if _FRONTMATTER_DELIMITER in text:
        errors.append(f"Frontmatter field '{field_name}' cannot contain '---'")
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
        elif _METADATA_KEY.fullmatch(normalized_key := key.strip()) is None:
            errors.append("Field 'metadata' keys must be letters, digits, hyphens, and underscores")
        elif not isinstance(value, str) or not value.strip():
            errors.append(
                f"Field 'metadata' value for {normalized_key!r} must be a non-empty string"
            )
        elif _FRONTMATTER_DELIMITER in normalized_key or _FRONTMATTER_DELIMITER in value:
            errors.append("Frontmatter field 'metadata' cannot contain '---'")
        else:
            cleaned[normalized_key] = value
    return cleaned, errors


def _render(prepared: SkillDraft) -> str:
    def dump(data: dict[str, Any], width: int = 10_000) -> str:
        return yaml.dump(
            data, Dumper=_FrontmatterDumper, sort_keys=False, allow_unicode=True, width=width
        )

    parts = [dump({"name": _YamlStr(prepared.name)})]
    if "\n" in prepared.description:
        paragraphs = "\n".join(
            line.strip() for line in prepared.description.split("\n") if line.strip()
        )
        parts.append(dump({"description": _YamlStr(paragraphs + "\n", ">")}, width=_FOLD_WIDTH))
    else:
        parts.append(dump({"description": _YamlStr(prepared.description)}))
    extras: dict[str, Any] = {}
    if prepared.license is not None:
        extras["license"] = _YamlStr(prepared.license)
    if prepared.compatibility is not None:
        extras["compatibility"] = _YamlStr(prepared.compatibility)
    if prepared.metadata:
        extras["metadata"] = {k: _YamlStr(prepared.metadata[k]) for k in sorted(prepared.metadata)}
    if prepared.allowed_tools is not None:
        extras["allowed-tools"] = _YamlStr(prepared.allowed_tools)
    if extras:
        parts.append(dump(extras))
    return f"---\n{''.join(parts)}---\n\n{prepared.body}\n"


def _destination_errors(
    root: Path, destination: Path, files: Mapping[str, str], *, overwrite: bool
) -> list[str]:
    if root.exists() and not root.is_dir():
        return ["Skills root is not a directory"]
    if destination.is_symlink():
        return ["Skill directory escapes the skills root"]
    if destination.exists() and not destination.is_dir():
        return ["Skill path is not a directory"]
    errors = _target_errors(destination / "SKILL.md", overwrite=overwrite, label="SKILL.md")
    for relative, content in files.items():
        if not isinstance(relative, str) or not isinstance(content, str):
            errors.append("Skill file paths and contents must be strings")
            continue
        path, locate_error = _locate_file(destination, relative)
        if path is None:
            errors.append(locate_error or "Skill file path is invalid")
        else:
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
    if candidate == root or not is_within(root, candidate):
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
    return [] if overwrite else [f"Skill already exists: {label}"]


def _copy_preserved(destination: Path, staging: Path, *, replaced: set[str]) -> None:
    for path in sorted(destination.rglob("*")):
        if path.is_symlink():
            raise SkillAuthorError(["Skill directory contains a symlink"])
        if not path.is_file() or (relative := path.relative_to(destination).as_posix()) in replaced:
            continue
        info = path.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise SkillAuthorError(
                ["Skill path is a hardlinked file that may lead outside the skill directory"]
            )
        located, locate_error = _locate_file(staging, relative)
        if located is None:
            raise SkillAuthorError([locate_error or "Skill file path is invalid"])
        located.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, located)


def _publish(staging: Path, destination: Path) -> None:
    def rename(src: Path, dst: Path) -> None:
        try:
            os.rename(src, dst)
        except OSError as exc:
            raise SkillAuthorError([f"Could not write skill: {exc.strerror}"]) from exc

    if not destination.exists():
        rename(staging, destination)
        return
    backup = staging.with_name(f"{staging.name}.bak")
    rename(destination, backup)
    try:
        rename(staging, destination)
    except SkillAuthorError:
        if not destination.exists() and backup.exists():
            rename(backup, destination)
        raise
    shutil.rmtree(backup, ignore_errors=True)


def _write_text(path: Path, text: str, *, overwrite: bool) -> None:
    """Atomically write ``text`` to ``path``, refusing symlinks and hardlinks."""
    errors = _target_errors(path, overwrite=overwrite, label=path.name)
    if errors:
        raise SkillAuthorError(errors)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.chmod(tmp_path, 0o644)
        os.replace(tmp_path, path)
    except BaseException as exc:
        tmp_path.unlink(missing_ok=True)
        if isinstance(exc, OSError):
            raise SkillAuthorError(
                [f"Could not write skill file {path.name}: {exc.strerror}"]
            ) from exc
        raise
