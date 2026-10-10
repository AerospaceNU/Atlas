from __future__ import annotations

import os
from pathlib import Path

import pytest

from atlas.agent.skill_author import (
    MAX_COMPATIBILITY_LENGTH,
    MAX_DESCRIPTION_LENGTH,
    MAX_SKILL_NAME_LENGTH,
    SkillAuthorError,
    SkillDraft,
    author_skill,
    project_skills_root,
    render_skill_md,
    user_skills_root,
    validate_skill,
)


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
        '  "author": "atlas"\n'
        '  "version": "1.0"\n'
        'allowed-tools: "read_file bash_tool"\n'
        "---\n"
        "\n"
        "Do the thing.\n"
    )
    assert (tmp_path / "ndvi-change" / "SKILL.md").read_text(encoding="utf-8") == text


def test_render_quotes_colons_quotes_and_newlines() -> None:
    draft = _draft(
        description='Use when: the user says "ndvi".\nThen continue.',
        body="Keep the ---\ndelimiter in the body.",
    )

    text = render_skill_md(draft)

    assert 'description: "Use when: the user says \\"ndvi\\".\\nThen continue."\n' in text
    assert text.endswith("Keep the ---\ndelimiter in the body.\n")
    assert text.split("---", 2)[2].lstrip("\n").startswith("Keep the ---")


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

    assert any("must be a string" in error for error in errors)
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


def test_project_and_user_skills_roots(tmp_path: Path) -> None:
    assert project_skills_root(tmp_path) == tmp_path / ".atlas" / "skills"
    assert user_skills_root(tmp_path) == tmp_path / ".atlas" / "skills"
    assert not project_skills_root(tmp_path).exists()
