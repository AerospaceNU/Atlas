from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
from pydantic import ValidationError

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.tools._process import run_captured
from atlas.agent.tools.BashTool import BashTool
from atlas.agent.tools.diagnostics import DiagnosticsTool
from atlas.agent.tools.git import GitDiffTool
from atlas.agent.tools.run_tests import RunTestsTool


def test_bash_timeout_returns_partial_output(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    command = "echo partial-out; echo partial-err >&2; sleep 20"

    result = BashTool().run({"command": command, "timeout": 1}, store)

    assert result.text.startswith("Command timed out after 1s:")
    assert "partial-out" in result.text
    assert "partial-err" in result.text
    assert "hint: rerun with a larger timeout (max 900)" in result.text


def test_bash_background_pattern_returns_quickly_and_kills_child(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    marker = tmp_path / "bg.marker"
    command = f"cd . && (sleep 2 && touch {marker}) & echo started"

    began = time.monotonic()
    result = BashTool().run({"command": command, "timeout": 30}, store)

    assert time.monotonic() - began < 5
    assert "exit_code: 0" in result.text
    assert "started" in result.text
    deadline = time.monotonic() + 2.5
    while time.monotonic() < deadline:
        assert not marker.exists()
        time.sleep(0.05)


def test_bash_timeout_above_old_limit_runs_normally(tmp_path: Path) -> None:
    result = BashTool().run({"command": "echo hi", "timeout": 301}, LocalArtifactStore(tmp_path))

    assert result.text.startswith("exit_code: 0")
    assert "note:" not in result.text


def test_bash_huge_timeout_is_clamped_with_a_note(tmp_path: Path) -> None:
    result = BashTool().run({"command": "echo hi", "timeout": 10**9}, LocalArtifactStore(tmp_path))

    assert result.text.startswith("note: timeout clamped to 900s\nexit_code: 0")


def test_bash_still_rejects_non_positive_timeouts(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        BashTool().run({"command": "echo hi", "timeout": 0}, LocalArtifactStore(tmp_path))


def test_long_output_keeps_the_last_line_and_a_marker(tmp_path: Path) -> None:
    command = "seq 1 200000; echo THE-LAST-LINE"

    result = BashTool().run({"command": command}, LocalArtifactStore(tmp_path))

    assert "THE-LAST-LINE" in result.text
    assert "... [truncated" in result.text
    assert result.text.startswith("exit_code: 0\nstdout:\n1\n")


def test_run_captured_handles_non_utf8_and_small_cap(tmp_path: Path) -> None:
    got = run_captured(
        ["bash", "-c", "printf 'a\\377b'; seq 1 100"], cwd=tmp_path, timeout=10, max_chars=50
    )

    assert got.exit_code == 0
    assert not got.timed_out
    assert "... [truncated" in got.stdout
    ok = run_captured(["bash", "-c", "printf 'a\\377b'"], cwd=tmp_path, timeout=10)
    assert ok.stdout == "a\N{REPLACEMENT CHARACTER}b"


@pytest.mark.parametrize(
    ("lock", "uv", "expected"),
    [
        (True, "/bin/uv", ["uv", "run", "pytest", "-q"]),
        (False, "/bin/uv", ["/x/python", "-m", "pytest", "-q"]),
        (True, None, ["/x/python", "-m", "pytest", "-q"]),
    ],
)
def test_run_tests_detection_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lock: bool,
    uv: str | None,
    expected: list[str],
) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='d'\n", encoding="utf-8")
    if lock:
        (tmp_path / "uv.lock").write_text("", encoding="utf-8")

    def fake_which(name: str) -> str | None:
        return {"uv": uv, "python": "/x/python"}.get(name)

    monkeypatch.setattr(shutil, "which", fake_which)

    detected = RunTestsTool()._detect(tmp_path)

    assert detected is not None
    assert detected[0] == expected


def test_run_tests_falls_back_to_python3_then_sys_executable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "pytest.ini").write_text("", encoding="utf-8")
    monkeypatch.setattr(shutil, "which", lambda name: "/y/python3" if name == "python3" else None)
    assert RunTestsTool()._detect(tmp_path) == (
        ["/y/python3", "-m", "pytest", "-q"],
        "pyproject.toml/pytest",
    )
    monkeypatch.setattr(shutil, "which", lambda name: None)
    detected = RunTestsTool()._detect(tmp_path)
    assert detected is not None
    assert detected[0][0] == sys.executable


def _run_echo(tmp_path: Path, text: str, code: int) -> str:
    script = f"import sys; sys.stdout.write({text!r}); raise SystemExit({code})"
    command = [sys.executable, "-c", script]
    return RunTestsTool().run({"command": command}, LocalArtifactStore(tmp_path)).text


def test_unittest_failure_summary(tmp_path: Path) -> None:
    text = (
        "FAIL: test_x (pkg.mod.Class)\nERROR: test_y (pkg.mod.Other)\n"
        "Ran 10 tests in 0.5s\n\nFAILED (failures=2, errors=1, skipped=3)\n"
    )

    out = _run_echo(tmp_path, text, 1)

    assert "passed=4 failed=2 error=1 skipped=3" in out
    assert "  FAIL: test_x (pkg.mod.Class)" in out
    assert "  ERROR: test_y (pkg.mod.Other)" in out


def test_unittest_ok_summary(tmp_path: Path) -> None:
    out = _run_echo(tmp_path, "Ran 7 tests in 0.1s\n\nOK\n", 0)

    assert "passed=7" in out


def test_sympy_summary(tmp_path: Path) -> None:
    out = _run_echo(tmp_path, "tests finished: 12 passed, 2 failed, 1 skipped\n", 1)

    assert "passed=12 failed=2 skipped=1" in out


def test_pytest_missing_hint(tmp_path: Path) -> None:
    out = _run_echo(tmp_path, "/usr/bin/python: No module named pytest\n", 1)

    assert "hint: pytest is not available to this interpreter" in out


def test_run_tests_timeout_keeps_partial_output(tmp_path: Path) -> None:
    script = (
        "import sys,time; print('2 passed'); print('FAILED t.py::x'); sys.stdout.flush(); "
        "time.sleep(20)"
    )

    result = RunTestsTool().run(
        {"command": [sys.executable, "-c", script], "timeout": 1}, LocalArtifactStore(tmp_path)
    )

    assert "timed out after 1s" in result.text
    assert "passed=2" in result.text
    assert "FAILED t.py::x" in result.text
    assert "tail:" in result.text


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_git_diff_of_a_latin1_file_does_not_raise(tmp_path: Path) -> None:
    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "a@b.c")
    git("config", "user.name", "t")
    (tmp_path / "f.txt").write_bytes(b"caf\xe9\n")
    git("add", ".")
    git("commit", "-qm", "x")
    (tmp_path / "f.txt").write_bytes(b"caf\xe9 au lait\n")

    result = GitDiffTool().run({}, LocalArtifactStore(tmp_path))

    assert "au lait" in result.text


def test_diagnostics_reports_a_crashing_checker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "ruff"
    script.write_text("#!/bin/sh\necho 'Traceback boom' >&2\nexit 2\n", encoding="utf-8")
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}/usr/bin{os.pathsep}/bin")

    result = DiagnosticsTool().run({"tools": ["ruff"]}, LocalArtifactStore(tmp_path))

    assert "[ruff failed: exit 2]" in result.text
    assert "Traceback boom" in result.text
    assert "no diagnostics" not in result.text


def test_sympy_summary_reads_only_the_counts_before_the_duration(tmp_path: Path) -> None:
    line = "tests finished: 5 passed, 1 failed, 0 skipped, 2 expected to fail, in 1.2 seconds\n"

    out = _run_echo(tmp_path, line, 1)

    assert "passed=5 failed=1 skipped=0 xfailed=2" in out


def test_sympy_summary_maps_expected_to_fail_but_passed(tmp_path: Path) -> None:
    out = _run_echo(
        tmp_path, "tests finished: 1 passed, 3 expected to fail but passed, in 1 seconds\n", 0
    )

    assert "passed=1 xpassed=3" in out


def test_run_captured_stops_a_command_that_floods_its_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("atlas.agent.tools._process.MAX_CAPTURE_BYTES", 200_000)
    began = time.monotonic()

    captured = run_captured(["bash", "-c", "yes"], cwd=tmp_path, timeout=30)

    assert time.monotonic() - began < 10
    assert captured.output_limited is True
    assert captured.timed_out is False
    assert captured.stdout.startswith("y\ny\n")


def test_run_captured_normal_output_is_not_limited(tmp_path: Path) -> None:
    captured = run_captured(["bash", "-c", "echo hi"], cwd=tmp_path, timeout=5)

    assert captured.output_limited is False
    assert captured.stdout == "hi\n"


def test_bash_and_run_tests_report_the_output_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("atlas.agent.tools._process.MAX_CAPTURE_BYTES", 200_000)
    store = LocalArtifactStore(tmp_path)
    note = "note: output exceeded 64 MB and the command was stopped"

    bash = BashTool().run({"command": "yes", "timeout": 30}, store)
    tests = RunTestsTool().run(
        {"command": [sys.executable, "-c", "while True: print('y' * 50)"], "timeout": 30}, store
    )

    assert note in bash.text
    assert "timed out" not in bash.text
    assert note in tests.text
