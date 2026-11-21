from __future__ import annotations

import pytest

from atlas.harness_eval import harness2_bench as hb


def test_profile_defaults_to_full(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(hb.ENV_TOOL_PROFILE, raising=False)
    assert hb._env_profile() == "full"


def test_profile_coding_and_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(hb.ENV_TOOL_PROFILE, "coding")
    assert hb._env_profile() == "coding"
    monkeypatch.setenv(hb.ENV_TOOL_PROFILE, "bogus")
    with pytest.raises(ValueError):
        hb._env_profile()


def test_env_float(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(hb.ENV_TEMPERATURE, raising=False)
    assert hb._env_float(hb.ENV_TEMPERATURE) is None
    monkeypatch.setenv(hb.ENV_TEMPERATURE, "0.3")
    assert hb._env_float(hb.ENV_TEMPERATURE) == 0.3
    monkeypatch.setenv(hb.ENV_TEMPERATURE, "x")
    with pytest.raises(ValueError):
        hb._env_float(hb.ENV_TEMPERATURE)
