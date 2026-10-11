"""Write an Agent Skill directory that follows the Agent Skills spec.

A skill is a directory named after its frontmatter ``name`` with a
``SKILL.md`` of YAML frontmatter plus markdown instructions. This module
checks those constraints and writes the directory. :func:`normalize_skill_name`
(also :func:`normalize_name`) and :func:`is_within` are the shared name and
path checks. Discovering skills and activating them in a session are separate.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import errno
import os
import re
import shutil
import stat
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from atlas.agent.layout import project_atlas_root, user_atlas_root

MAX_SKILL_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024
MAX_COMPATIBILITY_LENGTH = 500

_FRONTMATTER_DELIMITER = "---"
_METADATA_KEY = re.compile(r"^[A-Za-z0-9_-]+$")
_FOLD_WIDTH = 72
_RENAME_EXCHANGE = 2
_AT_FDCWD = -100
_libc: ctypes.CDLL | None = None
_renameat2: Any = None
_renameat2_loaded = False


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
    frontmatter and must be non-empty.
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


def project_skill_drafts_root(workspace: Path) -> Path:
    """Return ``<workspace>/.atlas/skills-drafts`` without creating it.

    Drafts are not loaded from ``.atlas/skills``. A person promotes one later.

    Args:
        workspace: Project directory that contains ``.atlas``.

    Returns:
        The project skill drafts directory.
    """
    return project_atlas_root(workspace) / "skills-drafts"


def user_skills_root(home: Path | None = None) -> Path:
    """Return ``<home>/.atlas/skills`` without creating it.

    Args:
        home: Directory that contains ``.atlas``. ``None`` uses the real home.

    Returns:
        The user skills directory.
    """
    return user_atlas_root(home) / "skills"


def normalize_skill_name(name: object) -> tuple[str | None, list[str]]:
    """NFKC-normalize a skill name and report spec violations.

    The skill loader and :func:`author_skill` share this check. ``None`` means
    there is no usable name. A non-empty invalid name is still returned so the
    caller can show it; callers reject the name when ``errors`` is not empty.
    :func:`normalize_name` is the same function.

    Args:
        name: Proposed skill name.

    Returns:
        The normalized name and any problems. An empty problem list means the
        name is writable.
    """
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


def is_within(root: Path, path: Path) -> bool:
    """Return whether ``path`` resolves to ``root`` or a file inside it.

    A relative ``path`` is joined to ``root`` first. Both paths are expanded
    and resolved, so a symlink that points outside ``root`` is not inside.

    Args:
        root: Directory that bounds ``path``.
        path: Path to test. It does not have to exist.

    Returns:
        True when the resolved path is ``root`` or a descendant of ``root``.
    """
    resolved_root = Path(root).expanduser().resolve()
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = resolved_root / candidate
    return candidate.resolve().is_relative_to(resolved_root)


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


def skill_directory_name(draft: SkillDraft) -> str:
    """Return the directory name :func:`author_skill` would write for ``draft``.

    Args:
        draft: Skill fields to check.

    Returns:
        The normalized skill name, which is also the directory name.

    Raises:
        SkillAuthorError: If ``draft`` violates the Agent Skills spec.
    """
    prepared, errors = _prepare(draft)
    if prepared is None:
        raise SkillAuthorError(errors)
    return prepared.name


normalize_name = normalize_skill_name


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
    bundled: Mapping[str, str] = {} if files is None else files
    root = Path(skills_root).expanduser().resolve()
    destination = root / prepared.name
    errors = _destination_errors(root, destination, bundled, overwrite=overwrite)
    errors.extend(_stranded_backup_errors(destination, overwrite=overwrite))
    if errors:
        raise SkillAuthorError(errors)
    if root.exists():
        _recover_replacing(destination)

    rendered = _render(prepared)
    root.mkdir(parents=True, exist_ok=True)
    staging = root / f".{prepared.name}.authoring"
    try:
        _reset_staging(staging)
        _copy_preserved(destination, staging, replaced={"SKILL.md", *bundled})
        _write_tree(staging, {**dict(bundled), "SKILL.md": rendered})
        _publish(staging, destination)
    except Exception:
        if staging.is_dir() and not staging.is_symlink():
            shutil.rmtree(staging, ignore_errors=True)
        raise
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
    body, body_errors = _normalize_body(draft.body)
    errors.extend(body_errors)
    if errors or name is None or description is None or not body:
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


_normalize_name = normalize_skill_name


def _required_text(value: object, field_name: str, limit: int) -> tuple[str | None, list[str]]:
    if not isinstance(value, str) or not value.strip():
        return None, [f"Field '{field_name}' must be a non-empty string"]
    text = value.strip()
    if field_name == "description":
        text = _normalize_newlines(text)
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
        normalized_key = key.strip()
        if _METADATA_KEY.fullmatch(normalized_key) is None:
            errors.append("Field 'metadata' keys must be letters, digits, hyphens, and underscores")
            continue
        if not isinstance(value, str) or not value.strip():
            errors.append(
                f"Field 'metadata' value for {normalized_key!r} must be a non-empty string"
            )
            continue
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


def _normalize_body(body: object) -> tuple[str, list[str]]:
    if not isinstance(body, str):
        return "", ["Field 'body' must be a string"]
    text = body.strip()
    if not text:
        return "", ["Field 'body' must be a non-empty string"]
    return text, []


def _render(prepared: _PreparedSkill) -> str:
    lines = [
        _FRONTMATTER_DELIMITER,
        f"name: {_yaml_string(prepared.name)}",
        *_description_lines(prepared.description),
    ]
    if prepared.license is not None:
        lines.append(f"license: {_yaml_string(prepared.license)}")
    if prepared.compatibility is not None:
        lines.append(f"compatibility: {_yaml_string(prepared.compatibility)}")
    if prepared.metadata:
        lines.append("metadata:")
        for key in sorted(prepared.metadata):
            lines.append(f"  {key}: {_yaml_string(prepared.metadata[key])}")
    if prepared.allowed_tools is not None:
        lines.append(f"allowed-tools: {_yaml_string(prepared.allowed_tools)}")
    lines.append(_FRONTMATTER_DELIMITER)
    return "\n".join(lines) + "\n\n" + prepared.body + "\n"


def _normalize_newlines(text: str) -> str:
    """Turn CR LF and bare CR into LF before a description is folded.

    The skills loader splits the file with ``str.splitlines``, which treats a
    raw carriage return as a line break. A CR left inside frontmatter splits a
    field in half and the loader skips the skill.
    """
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _description_lines(description: str) -> list[str]:
    """Render ``description`` so the skills loader can read it back.

    Carriage returns are normalized first. A single line stays a quoted
    scalar. A newline becomes a folded block (``>``): one paragraph per
    line, wrapped so the loader joins wrapped words with spaces and
    paragraphs with newlines.
    """
    description = _normalize_newlines(description)
    if "\n" not in description:
        return [f"description: {_yaml_string(description)}"]
    lines = ["description: >"]
    for index, paragraph in enumerate(description.split("\n")):
        if index:
            lines.append("")
        text = paragraph.strip()
        if not text:
            continue
        lines.extend(f"  {wrapped}" for wrapped in _wrap_plain(text, _FOLD_WIDTH))
    return lines


def _wrap_plain(paragraph: str, width: int) -> list[str]:
    words = paragraph.split(" ")
    lines: list[str] = []
    current = ""
    for word in words:
        if not word:
            continue
        piece = word if not current else f"{current} {word}"
        if current and len(piece) > width:
            lines.append(current)
            current = word
            continue
        current = piece
    if current:
        lines.append(current)
    return lines or [paragraph]


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
    if not overwrite:
        return [f"Skill already exists: {label}"]
    return []


def _copy_preserved(destination: Path, staging: Path, *, replaced: set[str]) -> None:
    """Copy files this write does not replace, keeping bytes and permissions.

    ``shutil.copy2`` keeps the mode bit, including execute, and does not
    decode the file as text. Symlinks and hardlinks are still refused.
    """
    if not destination.exists():
        return
    for path in sorted(destination.rglob("*")):
        if path.is_symlink():
            raise SkillAuthorError(["Skill directory contains a symlink"])
        if not path.is_file():
            continue
        relative = path.relative_to(destination).as_posix()
        if relative in replaced:
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


def _reset_staging(staging: Path) -> None:
    if staging.is_symlink():
        raise SkillAuthorError(["Skill staging path escapes the skills root"])
    if staging.exists():
        if not staging.is_dir():
            raise SkillAuthorError(["Skill staging path is not a directory"])
        shutil.rmtree(staging)
    staging.mkdir()


def _write_tree(staging: Path, payload: Mapping[str, str]) -> None:
    for relative, content in sorted(payload.items()):
        if not isinstance(relative, str) or not isinstance(content, str):
            raise SkillAuthorError(["Skill file paths and contents must be strings"])
        if relative == "SKILL.md":
            path = staging / "SKILL.md"
        else:
            located, locate_error = _locate_file(staging, relative)
            if located is None:
                raise SkillAuthorError([locate_error or "Skill file path is invalid"])
            path = located
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_text(path, content, overwrite=False)


def _replacing_path(destination: Path) -> Path:
    return destination.parent / f".{destination.name}.replacing"


def _stranded_backup_errors(destination: Path, *, overwrite: bool) -> list[str]:
    """Report a leftover ``.replacing`` directory without moving it.

    A refusing call must leave that backup where it is. Recovery runs only
    after these checks pass.
    """
    if overwrite or destination.exists() or destination.is_symlink():
        return []
    backup = _replacing_path(destination)
    if backup.exists() or backup.is_symlink():
        return ["Skill already exists: SKILL.md"]
    return []


def _recover_replacing(destination: Path) -> None:
    """Put a skill back if an earlier publish stopped after the backup rename.

    A crash between moving the live directory to ``.<name>.replacing`` and
    moving the staging directory into place leaves the skill missing. The next
    publish restores that backup. If both directories exist, the live one is
    the finished publish and the backup is removed.
    """
    backup = _replacing_path(destination)
    if backup.is_symlink():
        raise SkillAuthorError(["Skill staging path escapes the skills root"])
    if not backup.exists():
        return
    if not destination.exists():
        _rename(backup, destination)
        return
    if backup.is_dir():
        shutil.rmtree(backup)
        return
    backup.unlink()


def _publish(staging: Path, destination: Path) -> None:
    """Move ``staging`` to ``destination``.

    A new directory is one rename. Replacing an existing directory swaps the
    two with ``renameat2(RENAME_EXCHANGE)`` when the kernel allows it, so the
    skill name is never absent. Otherwise the live directory is renamed to
    ``.<name>.replacing``, the staging directory takes its place, and the
    backup is removed. ``_recover_replacing`` heals a backup left by a crash.
    """
    _recover_replacing(destination)
    if not destination.exists():
        _rename(staging, destination)
        return
    if _exchange_directories(staging, destination):
        shutil.rmtree(staging)
        return
    backup = _replacing_path(destination)
    if backup.exists() or backup.is_symlink():
        raise SkillAuthorError(["Skill staging path is not available"])
    _rename(destination, backup)
    try:
        _rename(staging, destination)
    except SkillAuthorError:
        if not destination.exists() and (backup.exists() or backup.is_symlink()):
            _rename(backup, destination)
        raise
    shutil.rmtree(backup)


def _exchange_directories(source: Path, target: Path) -> bool:
    """Atomically swap two directories.

    False for any errno other than ``ENOENT``, including ``EXDEV`` and
    ``EBUSY``, so the caller can fall back to a rename. A missing path is a
    real failure and is not treated as "the kernel cannot swap".
    """
    renameat2 = _cached_renameat2()
    if renameat2 is None:
        return False
    result = renameat2(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(target),
        _RENAME_EXCHANGE,
    )
    if result == 0:
        return True
    if ctypes.get_errno() == errno.ENOENT:
        raise SkillAuthorError([f"Could not write skill: {os.strerror(errno.ENOENT)}"])
    return False


def _cached_renameat2() -> Any:
    """Return ``renameat2`` from libc, loading it once."""
    global _libc, _renameat2, _renameat2_loaded
    if _renameat2_loaded:
        return _renameat2
    _renameat2_loaded = True
    library = ctypes.util.find_library("c")
    if not library:
        return None
    _libc = ctypes.CDLL(library, use_errno=True)
    renameat2 = getattr(_libc, "renameat2", None)
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
    _renameat2 = renameat2
    return renameat2


def _rename(source: Path, target: Path) -> None:
    try:
        os.rename(source, target)
    except OSError as exc:
        raise SkillAuthorError([f"Could not write skill: {exc.strerror}"]) from exc


def _write_text(path: Path, text: str, *, overwrite: bool) -> None:
    """Write ``text`` without truncating a hardlink.

    The file is opened without ``O_TRUNC``. ``st_nlink`` is checked on the
    open descriptor, and the file is truncated only after that check passes.
    """
    flags = os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW
    if not overwrite:
        flags |= os.O_EXCL
    try:
        fd = os.open(path, flags, 0o644)
    except OSError as exc:
        raise SkillAuthorError([f"Could not write skill file {path.name}: {exc.strerror}"]) from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise SkillAuthorError(
                ["Skill path is a hardlinked file that may lead outside the skill directory"]
            )
        os.ftruncate(fd, 0)
        data = text.encode("utf-8")
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise SkillAuthorError([f"Could not write skill file {path.name}"])
            view = view[written:]
    finally:
        os.close(fd)
