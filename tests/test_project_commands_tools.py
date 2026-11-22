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


def test_explicit_relative_executable_resolves_against_the_working_directory(
    tmp_path: Path,
) -> None:
    (tmp_path / "bin").mkdir()
    _script(tmp_path / "bin" / "test", 'echo "ran $1"; echo "2 passed"\n')
    _script(tmp_path / "runtests.sh", "echo dotslash\n")
    store = LocalArtifactStore(tmp_path)

    nested = RunTestsTool().run({"command": ["bin/test", "x"]}, store)
    dotted = RunTestsTool().run({"command": ["./runtests.sh"]}, store)

    assert "command not found" not in nested.text
    assert "ran x" in nested.text and "passed=2" in nested.text
    assert "dotslash" in dotted.text


def test_relative_file_that_is_not_executable_is_reported_not_raised(tmp_path: Path) -> None:
    script = tmp_path / "tests" / "runtests.py"
    script.parent.mkdir()
    script.write_text("print('x')\n")
    script.chmod(0o644)
    store = LocalArtifactStore(tmp_path)

    result = RunTestsTool().run({"command": ["tests/runtests.py"]}, store)

    assert result.text == "command not found: tests/runtests.py"


def test_relative_executable_that_does_not_exist_is_command_not_found(tmp_path: Path) -> None:
    result = RunTestsTool().run({"command": ["bin/nope"]}, LocalArtifactStore(tmp_path))

    assert result.text == "command not found: bin/nope"


def _counts(output: str) -> str:
    return RunTestsTool()._counts(output)


def test_unittest_expected_failures_are_not_failures() -> None:
    ok = "Ran 5 tests in 0.1s\n\nOK (expected failures=1)\n"
    bad = "Ran 5 tests in 0.1s\n\nFAILED (failures=2, expected failures=1)\n"

    assert _counts(ok) == "passed=4 xfailed=1"
    assert _counts(bad) == "passed=2 failed=2 xfailed=1"
    unexpected = "Ran 3 tests in 0.1s\n\nFAILED (unexpected successes=1)\n"
    assert _counts(unexpected) == "passed=2 xpassed=1"


def test_cargo_counts_are_summed_over_every_test_binary() -> None:
    output = (
        "test result: ok. 3 passed; 0 failed; 1 ignored; 0 measured\n"
        "test result: FAILED. 4 passed; 2 failed; 0 ignored; 0 measured\n"
    )

    assert _counts(output) == "passed=7 failed=2 ignored=1"


def test_failures_ignore_captured_log_lines_and_prefer_the_summary_section() -> None:
    output = "\n".join(
        [
            "ERROR some.logger:mod.py:3 something logged",
            "ERROR retrying connection",
            "=========== short test summary info ===========",
            "FAILED tests/test_a.py::test_one - assert 1 == 2",
            "ERROR tests/test_b.py::test_two",
            "FAILED tests/test_c.py",
            "=========== 2 failed in 1s ===========",
        ]
    )

    assert RunTestsTool()._failures(output) == [
        "FAILED tests/test_a.py::test_one - assert 1 == 2",
        "ERROR tests/test_b.py::test_two",
        "FAILED tests/test_c.py",
    ]


def test_failures_without_a_summary_section_skip_non_test_error_lines() -> None:
    output = "ERROR retrying connection\nFAILED tests/test_a.py::t\nERROR x - boom\n"

    assert RunTestsTool()._failures(output) == ["FAILED tests/test_a.py::t", "ERROR x - boom"]


def test_run_tests_huge_timeout_is_clamped_with_a_note(tmp_path: Path) -> None:
    from atlas.agent.artifacts import LocalArtifactStore
    from atlas.agent.tools.run_tests import RunTestsTool

    result = RunTestsTool().run(
        {"command": ["echo", "hi"], "timeout": 5000}, LocalArtifactStore(tmp_path)
    )
    lines = result.text.splitlines()
    assert lines[0].startswith("command: ")
    assert lines[1] == "note: timeout clamped to 900s"


def test_diagnostics_respects_request_path_for_cargo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tui = tmp_path / "tui"
    tui.mkdir()
    (tui / "Cargo.toml").write_text(
        "[package]\nname = 'tui'\nversion = '0.1.0'\n", encoding="utf-8"
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _script(bin_dir / "cargo", "pwd\necho 'src/main.rs:5:1: warning: unused'\nexit 0\n")
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}/usr/bin{os.pathsep}/bin")
    store = LocalArtifactStore(tmp_path)

    cargo_res = DiagnosticsTool().run({"path": "tui", "tools": ["cargo"]}, store)

    assert "== cargo (exit 0, 1 diagnostics)" in cargo_res.text
    assert "src/main.rs:5:1: warning: unused" in cargo_res.text
