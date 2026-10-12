"""Path helpers and the project versus user ``.atlas`` defaults."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from atlas.agent.layout import (
    ensure_project_layout,
    ensure_user_layout,
    project_agent_config_path,
    project_agents_md,
    project_agents_skills_dir,
    project_artifact_dir,
    project_artifacts_dir,
    project_atlas_root,
    project_data_dir,
    project_session_dir,
    project_sessions_dir,
    project_skill_drafts_root,
    project_skills_root,
    pull_manifest_path,
    user_agents_skills_dir,
    user_atlas_root,
    user_config_path,
    user_defaults_dir,
    user_keys_dir,
    user_skill_drafts_root,
    user_skills_root,
    user_weights_dir,
)
from atlas.cli.data import add_data_parser
from atlas.models.base import weights_root


def test_project_helpers_stay_under_the_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    assert project_atlas_root(workspace) == workspace / ".atlas"
    assert project_sessions_dir(workspace) == workspace / ".atlas" / "sessions"
    assert project_artifacts_dir(workspace) == workspace / ".atlas" / "artifacts"
    assert project_data_dir(workspace) == workspace / ".atlas" / "data"
    assert project_session_dir(workspace, "s1") == workspace / ".atlas" / "sessions" / "s1"
    assert project_artifact_dir(workspace, "s1") == workspace / ".atlas" / "artifacts" / "s1"
    assert project_skills_root(workspace) == workspace / ".atlas" / "skills"
    assert project_skill_drafts_root(workspace) == workspace / ".atlas" / "skills-drafts"
    assert project_agents_skills_dir(workspace) == workspace / ".agents" / "skills"
    assert project_agents_md(workspace) == workspace / "AGENTS.md"
    assert project_agent_config_path(workspace) == workspace / ".atlas" / "agent.toml"
    assert pull_manifest_path(project_data_dir(workspace)) == (
        workspace / ".atlas" / "data" / "manifest.json"
    )
    assert project_data_dir() == Path(".atlas/data")
    assert not project_data_dir().is_absolute()


def test_user_helpers_stay_under_the_home_directory(tmp_path: Path) -> None:
    home = tmp_path / "home"
    assert user_atlas_root(home) == home / ".atlas"
    assert user_keys_dir(home) == home / ".atlas" / "keys"
    assert user_weights_dir(home) == home / ".atlas" / "weights"
    assert user_defaults_dir(home) == home / ".atlas" / "defaults"
    assert user_skills_root(home) == home / ".atlas" / "skills"
    assert user_skill_drafts_root(home) == home / ".atlas" / "skills-drafts"
    assert user_agents_skills_dir(home) == home / ".agents" / "skills"
    assert user_config_path(home) == home / ".atlas" / "config.toml"


def test_ensure_project_layout_keeps_existing_data(tmp_path: Path) -> None:
    data = project_data_dir(tmp_path) / "keep.txt"
    data.parent.mkdir(parents=True)
    data.write_text("table", encoding="utf-8")

    root = ensure_project_layout(tmp_path)

    assert root == project_atlas_root(tmp_path)
    assert project_sessions_dir(tmp_path).is_dir()
    assert project_artifacts_dir(tmp_path).is_dir()
    assert data.read_text(encoding="utf-8") == "table"


def test_ensure_user_layout_keeps_existing_weights(tmp_path: Path) -> None:
    weight = user_weights_dir(tmp_path) / "model.json"
    weight.parent.mkdir(parents=True)
    weight.write_text("{}", encoding="utf-8")

    root = ensure_user_layout(tmp_path)

    assert root == user_atlas_root(tmp_path)
    assert user_keys_dir(tmp_path).is_dir()
    assert user_defaults_dir(tmp_path).is_dir()
    assert weight.read_text(encoding="utf-8") == "{}"


def test_data_cli_defaults_to_the_project_data_dir() -> None:
    parser = argparse.ArgumentParser()
    add_data_parser(parser.add_subparsers(dest="group"))

    args = parser.parse_args(["data"])

    assert args.out == project_data_dir()
    assert args.out == Path(".atlas/data")


def test_weights_root_prefers_env_then_checkout_then_user_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = tmp_path / "repo"
    checkout.mkdir()
    home = tmp_path / "home"
    monkeypatch.delenv("ATLAS_WEIGHTS_DIR", raising=False)
    monkeypatch.setattr("atlas.models.base.repo_root", lambda: checkout)
    monkeypatch.setattr("atlas.agent.layout.Path.home", lambda: home)

    assert weights_root() == (checkout / "weights").resolve()

    monkeypatch.setattr("atlas.models.base.repo_root", lambda: None)
    assert weights_root() == user_weights_dir().resolve()
    assert weights_root() == (home / ".atlas" / "weights").resolve()

    custom = tmp_path / "custom"
    monkeypatch.setenv("ATLAS_WEIGHTS_DIR", str(custom))
    assert weights_root() == custom.resolve()


def test_atlas_home_overrides_the_user_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    custom = tmp_path / "custom"
    other = tmp_path / "home"
    monkeypatch.setenv("ATLAS_HOME", str(custom))
    monkeypatch.setattr("atlas.agent.layout.Path.home", lambda: other)

    assert user_atlas_root() == custom
    assert user_atlas_root(other) == custom
    assert user_keys_dir() == custom / "keys"
    assert user_weights_dir(other) == custom / "weights"
    assert user_skills_root() == custom / "skills"
    assert user_skill_drafts_root(other) == custom / "skills-drafts"
    assert user_config_path() == custom / "config.toml"
    assert user_agents_skills_dir(other) == other / ".agents" / "skills"


def test_windows_style_home_stays_a_path_join() -> None:
    home = Path("C:/Users/atlas")
    workspace = Path("D:/work/atlas")

    assert user_atlas_root(home) == Path("C:/Users/atlas/.atlas")
    assert user_keys_dir(home) == Path("C:/Users/atlas/.atlas/keys")
    assert user_skills_root(home) == Path("C:/Users/atlas/.atlas/skills")
    assert user_config_path(home) == Path("C:/Users/atlas/.atlas/config.toml")
    assert project_atlas_root(workspace) == Path("D:/work/atlas/.atlas")
    assert project_data_dir(workspace) == Path("D:/work/atlas/.atlas/data")
    assert project_agents_md(workspace) == Path("D:/work/atlas/AGENTS.md")
    assert pull_manifest_path(project_data_dir(workspace)) == Path(
        "D:/work/atlas/.atlas/data/manifest.json"
    )
