from __future__ import annotations

from pathlib import Path

import pytest

from atlas.agent.runtime import load_reasoning_effort


def _write(root: Path, text: str) -> None:
    (root / ".atlas").mkdir()
    (root / ".atlas" / "agent.toml").write_text(text, encoding="utf-8")


def test_unset_is_none(tmp_path: Path) -> None:
    assert load_reasoning_effort(tmp_path) is None


def test_valid_and_invalid(tmp_path: Path) -> None:
    _write(tmp_path, 'reasoning_effort = "medium"\n')
    assert load_reasoning_effort(tmp_path) == "medium"
    _write_bad = tmp_path / ".atlas" / "agent.toml"
    _write_bad.write_text('reasoning_effort = "extreme"\n', encoding="utf-8")
    with pytest.raises(ValueError):
        load_reasoning_effort(tmp_path)
    _write_bad.write_text("reasoning_effort = 3\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_reasoning_effort(tmp_path)
