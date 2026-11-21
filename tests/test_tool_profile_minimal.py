from __future__ import annotations

from atlas.agent.contracts import default_registry
from atlas.harness_eval.headless import MINIMAL_TOOLS, profile_registry


def test_minimal_profile_keeps_only_shell_and_file_tools() -> None:
    names = set(profile_registry("minimal").names)
    assert names == set(MINIMAL_TOOLS) & set(default_registry().names)
    assert "bash" in names and "edit_file" in names
    assert "grep" not in names and "web_fetch" not in names


def test_full_profile_is_unchanged() -> None:
    assert set(profile_registry("full").names) == set(default_registry().names)
