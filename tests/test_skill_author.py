from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

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
    is_within,
    normalize_skill_name,
    project_skill_drafts_root,
    project_skills_root,
    render_skill_md,
    user_skills_root,
    validate_skill,
)
from atlas.agent.skills import parse_skill_document
from atlas.agent.tools.author_skill import AuthorSkillInput, AuthorSkillTool


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
        allowed_tools="read_file bash",
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
        'allowed-tools: "read_file bash"\n'
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


def test_normalize_skill_name_is_the_public_nfkc_check() -> None:
    composed = "caf\u00e9-scan"
    normalized, errors = normalize_skill_name("  cafe\u0301-scan  ")

    assert normalized == composed
    assert errors == []
    upper, upper_errors = normalize_skill_name("NDVI")
    assert upper == "NDVI"
    assert any("lowercase" in error for error in upper_errors)
    missing, missing_errors = normalize_skill_name("  ")
    assert missing is None
    assert missing_errors


def test_is_within_follows_symlinks_and_keeps_paths_inside_the_root(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    root.mkdir()
    inside = root / "notes.txt"
    inside.write_text("ok", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    escaped = root / "escape"
    escaped.symlink_to(outside, target_is_directory=True)
    nested = root / "real"
    nested.mkdir()
    alias = root / "alias"
    alias.symlink_to(nested, target_is_directory=True)

    assert is_within(root, inside)
    assert is_within(root, root)
    assert is_within(root, Path("notes.txt"))
    assert is_within(root, alias)
    assert not is_within(root, outside)
    assert not is_within(root, escaped)
    assert not is_within(root, Path("../outside"))


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

    monkeypatch.setattr(os, "rename", fail_rename)

    with pytest.raises(SkillAuthorError, match="Could not write skill"):
        author_skill(tmp_path, _draft(body="changed"), overwrite=True)

    assert (path / "SKILL.md").read_text(encoding="utf-8") == original
    assert (path / "scripts" / "extract.py").read_text(encoding="utf-8") == "print(1)\n"
    assert [p.name for p in tmp_path.iterdir()] == ["ndvi-change"]


def test_mid_publish_failure_restores_existing_skill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = author_skill(tmp_path, _draft(), files={"scripts/extract.py": "print(1)\n"})
    original = (path / "SKILL.md").read_text(encoding="utf-8")
    real_rename = os.rename
    calls = 0

    def fail_second(src: object, dst: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError(1, "swap failed")
        real_rename(src, dst)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "rename", fail_second)

    with pytest.raises(SkillAuthorError, match="Could not write skill"):
        author_skill(tmp_path, _draft(body="changed"), overwrite=True)

    assert (path / "SKILL.md").read_text(encoding="utf-8") == original
    assert (path / "scripts" / "extract.py").read_text(encoding="utf-8") == "print(1)\n"
    assert [p.name for p in tmp_path.iterdir()] == ["ndvi-change"]


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
        allowed_tools="read_file bash",
    )

    fields, body = _parse_like_skills_loader(render_skill_md(draft))

    assert fields["name"] == "ndvi-change"
    assert fields["description"] == description
    assert fields["license"] == "Apache-2.0"
    assert fields["compatibility"] == "Requires Python 3.12."
    assert fields["metadata"] == {"author": "atlas", "version": "1.0"}
    assert fields["allowed-tools"] == "read_file bash"
    assert body == "Do the thing."


def test_author_skill_tool_writes_a_draft_not_a_live_skill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, workspace = _project_store(tmp_path)
    vetted = workspace / ".atlas" / "skills" / "ndvi-change"
    vetted.mkdir(parents=True)
    (vetted / "SKILL.md").write_text("vetted\n", encoding="utf-8")
    calls: list[tuple[Path, dict[str, Any]]] = []

    def spy(root: Path, draft: SkillDraft, **kwargs: Any) -> Path:
        calls.append((root, kwargs))
        return author_skill(root, draft, **kwargs)

    monkeypatch.setattr("atlas.agent.tools.author_skill.author_skill", spy)

    result = AuthorSkillTool().run(_tool_arguments(), store)

    draft = workspace / ".atlas" / "skills-drafts" / "ndvi-change" / "SKILL.md"
    assert calls == [(project_skill_drafts_root(store.read_root or workspace), {})]
    assert draft.is_file()
    assert (vetted / "SKILL.md").read_text(encoding="utf-8") == "vetted\n"
    assert not (workspace / ".atlas" / "artifacts" / "session" / "skills").exists()
    assert result.artifacts == [".atlas/skills-drafts/ndvi-change/SKILL.md"]
    assert "/enable-skill ndvi-change" in result.text
    fields, body = _parse_like_skills_loader(draft.read_text(encoding="utf-8"))
    assert fields["name"] == "ndvi-change"
    assert body == "Subtract the later scene from the earlier one."


def test_author_skill_tool_rejects_extra_fields(tmp_path: Path) -> None:
    assert set(AuthorSkillInput.model_fields) == {"name", "description", "body"}
    store, workspace = _project_store(tmp_path)

    with pytest.raises(ValidationError, match="overwrite"):
        AuthorSkillTool().run(_tool_arguments(overwrite=True, scope="user"), store)

    assert not (workspace / ".atlas" / "skills-drafts").exists()
    assert not (workspace / ".atlas" / "skills").exists()


def test_author_skill_tool_refuses_a_drafts_symlink_to_live_skills(tmp_path: Path) -> None:
    store, workspace = _project_store(tmp_path)
    skills = workspace / ".atlas" / "skills"
    vetted = skills / "ndvi-change"
    vetted.mkdir(parents=True)
    (vetted / "SKILL.md").write_text("vetted\n", encoding="utf-8")
    (workspace / ".atlas" / "skills-drafts").symlink_to(skills, target_is_directory=True)

    with pytest.raises(ValueError, match="drafts directory"):
        AuthorSkillTool().run(_tool_arguments(), store)

    assert (vetted / "SKILL.md").read_text(encoding="utf-8") == "vetted\n"
    assert (workspace / ".atlas" / "skills-drafts").is_symlink()


def test_author_skill_tool_refuses_a_drafts_symlink_outside_the_workspace(tmp_path: Path) -> None:
    store, workspace = _project_store(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (workspace / ".atlas" / "skills-drafts").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="drafts directory"):
        AuthorSkillTool().run(_tool_arguments(), store)

    assert not any(outside.rglob("SKILL.md"))


def test_author_skill_tool_refuses_an_atlas_symlink_outside_the_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (workspace / ".atlas").symlink_to(outside, target_is_directory=True)
    store = LocalArtifactStore(tmp_path / "artifacts", read_root=workspace)

    with pytest.raises(ValueError, match="drafts directory"):
        AuthorSkillTool().run(_tool_arguments(), store)

    assert not any(outside.rglob("SKILL.md"))


def test_author_skill_tool_uses_store_root_when_read_root_is_none(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    artifact_root = workspace / ".atlas" / "artifacts" / "session"
    store = LocalArtifactStore(workspace, artifact_root=artifact_root)
    assert store.read_root is None

    result = AuthorSkillTool().run(_tool_arguments(), store)

    draft = workspace / ".atlas" / "skills-drafts" / "ndvi-change" / "SKILL.md"
    assert draft.is_file()
    assert not (artifact_root / "skills-drafts").exists()
    assert not (workspace / ".atlas" / "skills").exists()
    assert result.artifacts == [".atlas/skills-drafts/ndvi-change/SKILL.md"]


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


def test_author_skill_tool_does_not_change_the_default_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ATLAS_HEURISTIC_MODELS", "1")
    names = [tool.name for tool in default_registry().definitions]

    core = {
        "bash",
        "edit_file",
        "glob",
        "grep",
        "ls",
        "patch_file",
        "read_file",
        "get_satellite_data",
        "spectral_indices",
        "web_fetch",
        "write_file",
    }
    assert core <= set(names)
    assert {
        "mask_clouds",
        "rgb_change",
        "flood_mask",
        "burn_scar",
        "segment_landcover",
    } <= set(names)
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
    assert project_skill_drafts_root(tmp_path) == tmp_path / ".atlas" / "skills-drafts"
    assert user_skills_root(tmp_path) == tmp_path / ".atlas" / "skills"
    assert not project_skills_root(tmp_path).exists()
    assert not project_skill_drafts_root(tmp_path).exists()


def _parse_like_skills_loader(text: str) -> tuple[dict[str, Any], str]:
    return parse_skill_document(text)
