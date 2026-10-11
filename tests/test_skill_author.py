from __future__ import annotations

import os
import re
import shutil
import stat
from pathlib import Path
from typing import Any

import pytest

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import build_registry, default_registry
from atlas.agent.skill_author import (
    MAX_COMPATIBILITY_LENGTH,
    MAX_DESCRIPTION_LENGTH,
    MAX_SKILL_NAME_LENGTH,
    SkillAuthorError,
    SkillDraft,
    _write_text,
    author_skill,
    project_skills_root,
    render_skill_md,
    user_skills_root,
    validate_skill,
)
from atlas.agent.tools.author_skill import AuthorSkillTool


def _draft(**overrides: object) -> SkillDraft:
    fields: dict[str, object] = {
        "name": "ndvi-change",
        "description": "Compare two scenes. Use when the user asks about NDVI.",
        "body": "# NDVI change\n\nSubtract the later scene from the earlier one.",
    }
    fields.update(overrides)
    return SkillDraft(**fields)  # type: ignore[arg-type]


def test_author_skill_writes_frontmatter_and_body(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    draft = _draft()

    path = author_skill(root, draft)

    assert path == root.resolve() / "ndvi-change"
    assert (path / "SKILL.md").read_text(encoding="utf-8") == render_skill_md(draft)
    assert (path / "SKILL.md").read_text(encoding="utf-8") == (
        "---\n"
        'name: "ndvi-change"\n'
        'description: "Compare two scenes. Use when the user asks about NDVI."\n'
        "---\n"
        "\n"
        "# NDVI change\n"
        "\n"
        "Subtract the later scene from the earlier one.\n"
    )
    assert validate_skill(draft) == []


def test_author_skill_writes_optional_frontmatter_in_stable_order(tmp_path: Path) -> None:
    draft = _draft(
        body="Do the thing.",
        license="Apache-2.0",
        compatibility="Requires Python 3.12.",
        metadata={"version": "1.0", "author": "atlas"},
        allowed_tools="read_file bash_tool",
    )

    text = render_skill_md(draft)
    author_skill(tmp_path, draft)

    assert text == (
        "---\n"
        'name: "ndvi-change"\n'
        'description: "Compare two scenes. Use when the user asks about NDVI."\n'
        'license: "Apache-2.0"\n'
        'compatibility: "Requires Python 3.12."\n'
        "metadata:\n"
        '  author: "atlas"\n'
        '  version: "1.0"\n'
        'allowed-tools: "read_file bash_tool"\n'
        "---\n"
        "\n"
        "Do the thing.\n"
    )
    assert (tmp_path / "ndvi-change" / "SKILL.md").read_text(encoding="utf-8") == text


def test_render_folds_multiline_descriptions_for_the_loader() -> None:
    description = 'Use when: the user says "ndvi".\nThen continue.'
    draft = _draft(description=description, body="Keep the ---\ndelimiter in the body.")

    text = render_skill_md(draft)

    assert "description: >\n" in text
    assert '  Use when: the user says "ndvi".\n' in text
    assert "\n\n  Then continue.\n" in text
    assert text.endswith("Keep the ---\ndelimiter in the body.\n")
    fields, body = _parse_like_skills_loader(text)
    assert fields["name"] == "ndvi-change"
    assert fields["description"] == description
    assert body.startswith("Keep the ---")


def test_blank_optional_fields_are_omitted() -> None:
    text = render_skill_md(_draft(license="  ", compatibility="", allowed_tools="   ", metadata={}))

    assert "license:" not in text
    assert "compatibility:" not in text
    assert "allowed-tools:" not in text
    assert "metadata:" not in text


def test_name_is_nfkc_normalized_into_the_directory(tmp_path: Path) -> None:
    composed = "caf\u00e9"
    draft = _draft(name="cafe\u0301", description="A café skill.")

    path = author_skill(tmp_path, draft)

    assert path.name == composed
    assert f'name: "{composed}"\n' in (path / "SKILL.md").read_text(encoding="utf-8")


def test_fullwidth_letter_normalizes_to_ascii(tmp_path: Path) -> None:
    path = author_skill(tmp_path, _draft(name="\uff41"))

    assert path.name == "a"


def test_limits_are_inclusive(tmp_path: Path) -> None:
    draft = _draft(
        name="a" * MAX_SKILL_NAME_LENGTH,
        description="d" * MAX_DESCRIPTION_LENGTH + " ",
        compatibility="c" * MAX_COMPATIBILITY_LENGTH,
    )

    path = author_skill(tmp_path, draft)
    text = (path / "SKILL.md").read_text(encoding="utf-8")

    assert path.name == "a" * MAX_SKILL_NAME_LENGTH
    assert f'description: "{"d" * MAX_DESCRIPTION_LENGTH}"\n' in text
    assert f'compatibility: "{"c" * MAX_COMPATIBILITY_LENGTH}"\n' in text


@pytest.mark.parametrize(
    ("name", "match"),
    [
        ("", "non-empty"),
        ("   ", "non-empty"),
        (1, "non-empty"),
        ("NDVI", "lowercase"),
        ("-ndvi", "start or end"),
        ("ndvi-", "start or end"),
        ("ndvi--change", "consecutive"),
        ("ndvi_change", "invalid characters"),
        ("a" * (MAX_SKILL_NAME_LENGTH + 1), "character limit"),
    ],
)
def test_rejects_invalid_names(name: object, match: str) -> None:
    errors = validate_skill(_draft(name=name))

    assert errors
    assert any(match in error for error in errors)
    with pytest.raises(SkillAuthorError, match=match):
        render_skill_md(_draft(name=name))


def test_invalid_name_reports_every_problem() -> None:
    errors = validate_skill(_draft(name="A--B"))

    assert any("lowercase" in error for error in errors)
    assert any("consecutive" in error for error in errors)


@pytest.mark.parametrize(
    "description",
    ["", "   ", "d" * (MAX_DESCRIPTION_LENGTH + 1)],
)
def test_rejects_invalid_descriptions(description: str) -> None:
    errors = validate_skill(_draft(description=description))

    assert errors
    assert any("description" in error.lower() or "Description" in error for error in errors)


def test_rejects_frontmatter_delimiter_in_description() -> None:
    errors = validate_skill(_draft(description="Use --- carefully"))

    assert any("cannot contain '---'" in error for error in errors)


def test_rejects_oversized_compatibility() -> None:
    errors = validate_skill(_draft(compatibility="c" * (MAX_COMPATIBILITY_LENGTH + 1)))

    assert any("Compatibility exceeds" in error for error in errors)


def test_rejects_metadata_that_is_not_a_string_mapping() -> None:
    errors = validate_skill(_draft(metadata={"author": 1, "": "atlas"}))

    assert any("non-empty string" in error for error in errors)
    assert any("keys must be non-empty" in error for error in errors)


def test_rejects_non_mapping_metadata() -> None:
    errors = validate_skill(_draft(metadata="author=atlas"))

    assert errors == ["Field 'metadata' must be a mapping of strings"]


def test_invalid_draft_does_not_create_the_skills_root(tmp_path: Path) -> None:
    root = tmp_path / "skills"

    with pytest.raises(SkillAuthorError):
        author_skill(
            root,
            _draft(name="Bad Name"),
            files={"scripts/extract.py": "print(1)\n"},
        )

    assert not root.exists()


def test_author_skill_writes_bundled_files(tmp_path: Path) -> None:
    path = author_skill(
        tmp_path,
        _draft(),
        files={
            "scripts/extract.py": "print(1)\n",
            "references/guide.md": "# Guide\n",
        },
    )

    assert (path / "scripts" / "extract.py").read_text(encoding="utf-8") == "print(1)\n"
    assert (path / "references" / "guide.md").read_text(encoding="utf-8") == "# Guide\n"


def test_refuses_to_overwrite_unless_requested(tmp_path: Path) -> None:
    draft = _draft()
    author_skill(tmp_path, draft, files={"scripts/extract.py": "print(1)\n"})

    with pytest.raises(SkillAuthorError, match="already exists"):
        author_skill(tmp_path, _draft(body="changed"), files={"scripts/extract.py": "print(2)\n"})

    text = (tmp_path / "ndvi-change" / "SKILL.md").read_text(encoding="utf-8")
    script = (tmp_path / "ndvi-change" / "scripts" / "extract.py").read_text(encoding="utf-8")
    assert "changed" not in text
    assert script == "print(1)\n"


def test_overwrite_replaces_named_files_and_keeps_others(tmp_path: Path) -> None:
    path = author_skill(tmp_path, _draft(), files={"scripts/extract.py": "print(1)\n"})
    (path / "notes.txt").write_text("keep", encoding="utf-8")

    author_skill(
        tmp_path,
        _draft(body="changed"),
        files={"scripts/extract.py": "print(2)\n"},
        overwrite=True,
    )

    assert "changed" in (path / "SKILL.md").read_text(encoding="utf-8")
    assert (path / "scripts" / "extract.py").read_text(encoding="utf-8") == "print(2)\n"
    assert (path / "notes.txt").read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize(
    "relative",
    ["../outside.txt", "/etc/passwd", "scripts/../../outside.txt", "SKILL.md", "skill.md"],
)
def test_rejects_escaping_or_reserved_bundled_paths(tmp_path: Path, relative: str) -> None:
    root = tmp_path / "skills"

    with pytest.raises(SkillAuthorError, match="skill"):
        author_skill(root, _draft(), files={relative: "nope"})

    assert not root.exists()
    assert not (tmp_path / "outside.txt").exists()


def test_rejects_a_symlink_skill_directory(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "skills"
    root.mkdir()
    (root / "ndvi-change").symlink_to(outside, target_is_directory=True)

    with pytest.raises(SkillAuthorError, match="escapes"):
        author_skill(root, _draft())

    assert not (outside / "SKILL.md").exists()


def test_rejects_a_hardlinked_skill_file(tmp_path: Path) -> None:
    outside = tmp_path / "outside.md"
    outside.write_text("secret", encoding="utf-8")
    skill_dir = tmp_path / "skills" / "ndvi-change"
    skill_dir.mkdir(parents=True)
    try:
        os.link(outside, skill_dir / "SKILL.md")
    except OSError:
        pytest.skip("hardlinks are not supported on this filesystem")

    with pytest.raises(SkillAuthorError, match="hardlinked"):
        author_skill(tmp_path / "skills", _draft(body="changed"), overwrite=True)

    assert outside.read_text(encoding="utf-8") == "secret"


def test_rejects_a_bundled_symlink_that_leaves_the_skill(tmp_path: Path) -> None:
    outside = tmp_path / "outside.py"
    outside.write_text("secret", encoding="utf-8")
    path = author_skill(tmp_path, _draft())
    scripts = path / "scripts"
    scripts.mkdir()
    (scripts / "extract.py").symlink_to(outside)

    with pytest.raises(SkillAuthorError, match="escapes"):
        author_skill(
            tmp_path,
            _draft(),
            files={"scripts/extract.py": "print(1)\n"},
            overwrite=True,
        )

    assert outside.read_text(encoding="utf-8") == "secret"


def test_rejects_a_skills_root_that_is_a_file(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    root.write_text("not a directory", encoding="utf-8")

    with pytest.raises(SkillAuthorError, match="not a directory"):
        author_skill(root, _draft())

    assert root.read_text(encoding="utf-8") == "not a directory"


def test_rejects_an_empty_body(tmp_path: Path) -> None:
    root = tmp_path / "skills"

    errors = validate_skill(_draft(body="  \n\t"))

    assert "Field 'body' must be a non-empty string" in errors
    with pytest.raises(SkillAuthorError, match="body"):
        author_skill(root, _draft(body=""))
    assert not root.exists()


def test_write_text_rejects_a_hardlink_before_truncating(tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("secret-value", encoding="utf-8")
    link = tmp_path / "link.txt"
    try:
        os.link(outside, link)
    except OSError:
        pytest.skip("hardlinks are not supported on this filesystem")

    with pytest.raises(SkillAuthorError, match="hardlinked"):
        _write_text(link, "changed", overwrite=True)

    assert outside.read_text(encoding="utf-8") == "secret-value"
    assert link.read_text(encoding="utf-8") == "secret-value"


def test_write_text_replaces_a_regular_file_after_the_link_check(tmp_path: Path) -> None:
    path = tmp_path / "note.txt"
    path.write_text("old-content", encoding="utf-8")

    _write_text(path, "new", overwrite=True)

    assert path.read_text(encoding="utf-8") == "new"


def test_failed_publish_leaves_the_existing_skill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = author_skill(tmp_path, _draft(), files={"scripts/extract.py": "print(1)\n"})
    original = (path / "SKILL.md").read_text(encoding="utf-8")

    def fail_rename(*_args: object, **_kwargs: object) -> None:
        raise OSError(1, "nope")

    def refuse_exchange(*_args: object, **_kwargs: object) -> bool:
        return False

    monkeypatch.setattr("atlas.agent.skill_author._exchange_directories", refuse_exchange)
    monkeypatch.setattr(os, "rename", fail_rename)

    with pytest.raises(SkillAuthorError, match="Could not write skill"):
        author_skill(tmp_path, _draft(body="changed"), overwrite=True)

    assert (path / "SKILL.md").read_text(encoding="utf-8") == original
    assert (path / "scripts" / "extract.py").read_text(encoding="utf-8") == "print(1)\n"
    assert not (tmp_path / ".ndvi-change.authoring").exists()
    assert not (tmp_path / ".ndvi-change.replacing").exists()


def test_next_publish_recovers_a_leftover_replacing_directory(tmp_path: Path) -> None:
    path = author_skill(tmp_path, _draft())
    original = (path / "SKILL.md").read_text(encoding="utf-8")
    backup = tmp_path / ".ndvi-change.replacing"
    os.rename(path, backup)

    with pytest.raises(SkillAuthorError, match="already exists"):
        author_skill(tmp_path, _draft(body="changed"), overwrite=False)

    assert (path / "SKILL.md").read_text(encoding="utf-8") == original
    assert not backup.exists()

    author_skill(tmp_path, _draft(body="changed"), overwrite=True)

    assert "changed" in (path / "SKILL.md").read_text(encoding="utf-8")
    assert not backup.exists()
    assert not (tmp_path / ".ndvi-change.authoring").exists()


def test_live_skill_wins_when_a_replacing_backup_is_also_present(tmp_path: Path) -> None:
    path = author_skill(tmp_path, _draft())
    backup = tmp_path / ".ndvi-change.replacing"
    shutil.copytree(path, backup)
    (path / "SKILL.md").write_text("live-marker\n", encoding="utf-8")

    with pytest.raises(SkillAuthorError, match="already exists"):
        author_skill(tmp_path, _draft(), overwrite=False)

    assert (path / "SKILL.md").read_text(encoding="utf-8") == "live-marker\n"
    assert not backup.exists()


def test_overwrite_copies_preserved_bytes_and_permissions(tmp_path: Path) -> None:
    path = author_skill(tmp_path, _draft(), files={"scripts/extract.py": "print(1)\n"})
    binary = path / "scripts" / "run.bin"
    binary.write_bytes(b"\x00\xff")
    binary.chmod(0o755)

    author_skill(tmp_path, _draft(body="changed"), overwrite=True)

    assert binary.read_bytes() == b"\x00\xff"
    assert stat.S_IMODE(binary.stat().st_mode) == 0o755
    assert (path / "scripts" / "extract.py").read_text(encoding="utf-8") == "print(1)\n"
    assert "changed" in (path / "SKILL.md").read_text(encoding="utf-8")


def test_description_carriage_returns_are_normalized_before_folding(tmp_path: Path) -> None:
    description = "Compare scenes.\r\nUse when asked.\rThen stop."

    path = author_skill(tmp_path, _draft(description=description, body="Do the thing."))
    text = (path / "SKILL.md").read_bytes()

    assert b"\r" not in text
    fields, body = _parse_like_skills_loader(text.decode("utf-8"))
    assert fields["description"] == "Compare scenes.\nUse when asked.\nThen stop."
    assert body == "Do the thing."


def test_loader_parses_quoted_and_folded_frontmatter() -> None:
    description = "Compare two scenes.\nUse when the user asks about NDVI."
    draft = _draft(
        description=description,
        body="Do the thing.",
        license="Apache-2.0",
        compatibility="Requires Python 3.12.",
        metadata={"version": "1.0", "author": "atlas"},
        allowed_tools="read_file bash_tool",
    )

    fields, body = _parse_like_skills_loader(render_skill_md(draft))

    assert fields["name"] == "ndvi-change"
    assert fields["description"] == description
    assert fields["license"] == "Apache-2.0"
    assert fields["compatibility"] == "Requires Python 3.12."
    assert fields["metadata"] == {"author": "atlas", "version": "1.0"}
    assert fields["allowed-tools"] == "read_file bash_tool"
    assert body == "Do the thing."


def test_author_skill_tool_writes_the_project_skills_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, workspace = _project_store(tmp_path)
    calls: list[Path] = []

    def spy(root: Path, draft: SkillDraft, **kwargs: Any) -> Path:
        calls.append(root)
        return author_skill(root, draft, **kwargs)

    monkeypatch.setattr("atlas.agent.tools.author_skill.author_skill", spy)

    result = AuthorSkillTool().run(_tool_arguments(), store)

    skill = workspace / ".atlas" / "skills" / "ndvi-change" / "SKILL.md"
    assert calls == [project_skills_root(store.read_root or workspace)]
    assert skill.is_file()
    assert not (workspace / ".atlas" / "artifacts" / "session" / "skills").exists()
    assert result.artifacts == [".atlas/skills/ndvi-change/SKILL.md"]
    fields, body = _parse_like_skills_loader(skill.read_text(encoding="utf-8"))
    assert fields["name"] == "ndvi-change"
    assert body == "Subtract the later scene from the earlier one."


def test_author_skill_tool_writes_user_skills(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)
    store, workspace = _project_store(tmp_path)

    result = AuthorSkillTool().run(_tool_arguments(scope="user"), store)

    skill = home / ".atlas" / "skills" / "ndvi-change" / "SKILL.md"
    assert skill.is_file()
    assert not (workspace / ".atlas" / "skills").exists()
    assert result.artifacts == ["~/.atlas/skills/ndvi-change/SKILL.md"]
    fields, _body = _parse_like_skills_loader(skill.read_text(encoding="utf-8"))
    assert fields["name"] == "ndvi-change"


def test_author_skill_tool_refuses_a_skills_symlink_outside_the_workspace(tmp_path: Path) -> None:
    store, workspace = _project_store(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (workspace / ".atlas" / "skills").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="escapes"):
        AuthorSkillTool().run(_tool_arguments(), store)

    assert not any(outside.rglob("SKILL.md"))


def test_author_skill_tool_refuses_an_atlas_symlink_outside_the_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (workspace / ".atlas").symlink_to(outside, target_is_directory=True)
    store = LocalArtifactStore(tmp_path / "artifacts", read_root=workspace)

    with pytest.raises(ValueError, match="escapes"):
        AuthorSkillTool().run(_tool_arguments(), store)

    assert not any(outside.rglob("SKILL.md"))


def test_author_skill_tool_requires_a_workspace_read_root(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")

    with pytest.raises(ValueError, match="read root"):
        AuthorSkillTool().run(_tool_arguments(), store)

    assert not (tmp_path / "artifacts" / "skills").exists()


def test_author_skill_tool_overwrite_goes_through_author_skill(tmp_path: Path) -> None:
    store, workspace = _project_store(tmp_path)
    tool = AuthorSkillTool()
    tool.run(_tool_arguments(), store)
    skill = workspace / ".atlas" / "skills" / "ndvi-change" / "SKILL.md"
    original = skill.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="already exists"):
        tool.run(_tool_arguments(body="changed"), store)

    assert skill.read_text(encoding="utf-8") == original

    result = tool.run(_tool_arguments(body="changed", overwrite=True), store)

    assert "changed" in skill.read_text(encoding="utf-8")
    assert result.artifacts == [".atlas/skills/ndvi-change/SKILL.md"]


def _project_store(tmp_path: Path) -> tuple[LocalArtifactStore, Path]:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    artifact_root = workspace / ".atlas" / "artifacts" / "session"
    return LocalArtifactStore(artifact_root, read_root=workspace), workspace


def _tool_arguments(**overrides: object) -> dict[str, object]:
    arguments: dict[str, object] = {
        "name": "ndvi-change",
        "description": "Compare two scenes. Use when the user asks about NDVI.",
        "body": "Subtract the later scene from the earlier one.",
    }
    arguments.update(overrides)
    return arguments


def test_author_skill_tool_does_not_change_the_default_registry() -> None:
    names = [tool.name for tool in default_registry().definitions]

    assert names == [
        "bash_tool",
        "edit_file",
        "read_file",
        "write_file",
        "list_tools",
        "segment_landcover",
    ]
    assert "author_skill" not in names
    assert "stage_script_proposal" not in names
    opted_in = [
        tool.name
        for tool in build_registry(("atlas.agent.tools",), include_opt_in=True).definitions
    ]
    assert "author_skill" in opted_in
    assert AuthorSkillTool.trust == "opt_in"


def test_project_and_user_skills_roots(tmp_path: Path) -> None:
    assert project_skills_root(tmp_path) == tmp_path / ".atlas" / "skills"
    assert user_skills_root(tmp_path) == tmp_path / ".atlas" / "skills"
    assert not project_skills_root(tmp_path).exists()


# Mirrors ``_split_frontmatter`` / ``_parse_frontmatter`` on richardtang/skills-loading
# (atlas.agent.skills). Authoring has to stay readable by that loader.
_FRONTMATTER_KEY = re.compile(r"^([A-Za-z0-9_-]+):\s*(.*)$")


def _parse_like_skills_loader(text: str) -> tuple[dict[str, Any], str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise AssertionError("missing frontmatter")
    frontmatter = ""
    body = ""
    closed = False
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            frontmatter = "\n".join(lines[1:index])
            body = "\n".join(lines[index + 1 :]).strip()
            closed = True
            break
    if not closed:
        raise AssertionError("frontmatter is not closed")
    return _parse_frontmatter(frontmatter), body


def _parse_frontmatter(text: str) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.strip() or line.lstrip().startswith("#"):
            index += 1
            continue
        if line[0] in {" ", "\t"}:
            raise AssertionError("frontmatter is not a mapping")
        match = _FRONTMATTER_KEY.match(line)
        if match is None:
            raise AssertionError(f"frontmatter is not a mapping: {line!r}")
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
            raise AssertionError("frontmatter is not a mapping")
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
