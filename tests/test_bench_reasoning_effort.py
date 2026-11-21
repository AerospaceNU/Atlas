from __future__ import annotations

import pytest

from atlas.harness_eval import harness2_bench as bench


def test_effort_defaults_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(bench.ENV_REASONING_EFFORT, raising=False)
    assert bench._env_effort() is None


def test_effort_is_read_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(bench.ENV_REASONING_EFFORT, "low")
    assert bench._env_effort() == "low"


def test_invalid_effort_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(bench.ENV_REASONING_EFFORT, "turbo")
    with pytest.raises(ValueError):
        bench._env_effort()
