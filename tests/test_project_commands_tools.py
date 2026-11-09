from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.tools.diagnostics import DiagnosticsTool
from atlas.agent.tools.run_tests import RunTestsTool


def _script(path: Path, body: str) -> None:
    """Write an executable shell script."""
    path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    path.chmod(0o755)


def test_explicit_command_reports_counts_and_nonzero_exit(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    command = [sys.executable, "-c", "print('3 passed, 1 failed'); raise SystemExit(1)"]

    result = RunTestsTool().run({"command": command}, store)

    assert "exit_code: 1" in result.text
    assert "passed=3 failed=1" in result.text
    assert "tail:" in result.text


def test_failing_test_names_are_listed(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    command = [
        sys.executable,
        "-c",
        "print('FAILED tests/test_x.py::test_y'); raise SystemExit(1)",
    ]

    result = RunTestsTool().run({"command": command}, store)

    assert "FAILED tests/test_x.py::test_y" in result.text


def test_explicit_command_times_out_without_raising(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    command = [sys.executable, "-c", "import time; time.sleep(10)"]

    result = RunTestsTool().run({"command": command, "timeout": 1}, store)

    assert "timed out after 1s" in result.text


def test_reports_no_detected_command(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)

    result = RunTestsTool().run({}, store)

    assert result.text.startswith("no test command detected")


def test_reports_a_missing_binary(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)

    result = RunTestsTool().run({"command": ["definitely-not-a-real-binary-xyz"]}, store)

    assert result.text == "command not found: definitely-not-a-real-binary-xyz"


def test_detects_pytest_from_pyproject(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")

    detected = RunTestsTool()._detect(tmp_path)

    assert detected is not None
    argv, source = detected
    assert "pytest" in argv
    assert source == "pyproject.toml/pytest"


def test_detects_cargo_from_manifest(tmp_path: Path) -> None:
    (tmp_path / "Cargo.toml").write_text("[package]\nname = 'demo'\n", encoding="utf-8")

    detected = RunTestsTool()._detect(tmp_path)

    assert detected is not None
    argv, source = detected
    assert argv[:2] == ["cargo", "test"]
    assert source == "Cargo.toml"


def test_diagnostics_parses_ruff_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _script(bin_dir / "ruff", "echo 'src/a.py:10:5: F401 unused'\nexit 1\n")
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}/usr/bin{os.pathsep}/bin")
    store = LocalArtifactStore(tmp_path)

    result = DiagnosticsTool().run({"tools": ["ruff"]}, store)

    assert "== ruff (exit 1, 1 diagnostics)" in result.text
    assert "src/a.py:10:5: F401 unused" in result.text


def test_diagnostics_reports_clean_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _script(bin_dir / "ruff", "exit 0\n")
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}/usr/bin{os.pathsep}/bin")
    store = LocalArtifactStore(tmp_path)

    result = DiagnosticsTool().run({"tools": ["ruff"]}, store)

    assert "== ruff (exit 0, 0 diagnostics)" in result.text
    assert "no diagnostics" in result.text


def test_diagnostics_reports_missing_tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    monkeypatch.setenv("PATH", str(bin_dir))
    store = LocalArtifactStore(tmp_path)

    result = DiagnosticsTool().run({"tools": ["mypy"]}, store)

    assert "missing: mypy" in result.text
    assert "no diagnostics tools available" in result.text


def test_diagnostics_times_out_without_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _script(bin_dir / "ruff", "sleep 10\n")
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}/usr/bin{os.pathsep}/bin")
    store = LocalArtifactStore(tmp_path)

    result = DiagnosticsTool().run({"tools": ["ruff"], "timeout": 1}, store)

    assert "[timed out after 1s]" in result.text
