from __future__ import annotations

from pathlib import Path

import pytest

import atlas.agent
from atlas.agent.system_prompt import _REQUIRED_RULES, load_system_prompt

FIRST_LINE = "You are Atlas, a local workspace assistant."
LAST_SENTENCE = "Finish with what you changed and how you checked it."


@pytest.fixture
def prompt() -> str:
    load_system_prompt.cache_clear()
    return load_system_prompt()


def test_prompt_has_the_coding_block(prompt: str) -> None:
    assert "Working on code" in prompt
    assert "Tool names are exact; the shell tool is bash." in prompt
    assert "read_file with offset and limit" in prompt
    assert "Do not end a turn with only a plan" in prompt


def test_required_rules_survive(prompt: str) -> None:
    for rule in _REQUIRED_RULES:
        assert rule in prompt


def test_first_line_is_unchanged(prompt: str) -> None:
    assert prompt.splitlines()[0] == FIRST_LINE


def test_prompt_forbids_hunting_for_upstream_fixes(prompt: str) -> None:
    assert "Do not search for an upstream or released fix" in prompt


def test_prompt_stays_compact(prompt: str) -> None:
    assert len(prompt) < 4600


def test_coding_block_is_at_the_end_of_the_file() -> None:
    path = Path(atlas.agent.__file__).parent / "system_prompt.md"
    text = path.read_text(encoding="utf-8")

    assert text.rstrip().endswith(LAST_SENTENCE)
    assert text.index("\nWorking on code\n") > text.index(
        "Only use tools on the supplied registry."
    )
    assert "\n\nWorking on code\n" in text
