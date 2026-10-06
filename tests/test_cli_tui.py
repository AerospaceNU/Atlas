from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from atlas.agent.contracts import CallTiming, ModelCall, TokenUsage
from atlas.agent.metrics import SessionTotals, apply_call, format_token_count, format_tui_status
from atlas.cli import main
from atlas.cli.tui import TuiLaunchError, ensure_tui_binary, launch_tui


def test_status_line_shows_context_tokens_spend_and_speed() -> None:
    text = format_tui_status(
        SessionTotals(
            model="example/model",
            prompt_tokens=12,
            completion_tokens=4,
            total_tokens=16,
            cost=0.02,
            context_used=12,
            context_limit=128000,
            last_http_status=200,
            last_elapsed_seconds=0.5,
            last_ttft_seconds=0.1,
            tokens_per_second=8.0,
            recent=["200 0.50s"],
        )
    )

    assert "context 12/128K" in text
    assert "tokens 16" in text
    assert "prompt" not in text
    assert "completion" not in text
    assert "spend $0.02" in text
    assert "8.0 tok/s" in text
    assert "http" not in text
    assert "latency" not in text
    assert "ttft" not in text
    assert "recent" not in text


def test_status_line_does_not_invent_a_context_window() -> None:
    text = format_tui_status(SessionTotals(context_used=9))

    assert "context 9/unknown" in text
    assert "128000" not in text
    assert "spend $0" in text
    assert "0.0 tok/s" in text
    assert "http" not in text
    assert "latency" not in text
    assert "ttft" not in text
    assert "recent" not in text


def test_output_speed_stays_when_a_later_call_writes_no_tokens() -> None:
    totals = SessionTotals(tokens_per_second=8.0)
    apply_call(
        totals,
        ModelCall(
            usage=TokenUsage(),
            timing=CallTiming(http_status=200, elapsed_seconds=4.0),
        ),
    )

    assert totals.tokens_per_second == 8.0
    assert "8.0 tok/s" in format_tui_status(totals)


def test_token_counts_use_three_sigfigs_in_k_m_and_b() -> None:
    assert format_token_count(355) == "355"
    assert format_token_count(380) == "380"
    assert format_token_count(128_000) == "128K"
    assert format_token_count(1_048_576) == "1.05M"
    assert format_token_count(1_500_000_000) == "1.50B"
    assert format_token_count(12_300_000_000) == "12.3B"
    text = format_tui_status(
        SessionTotals(
            model="google/gemini-3.8-flash",
            prompt_tokens=355,
            completion_tokens=25,
            total_tokens=380,
            context_used=355,
            context_limit=1_048_576,
            cost=0.00036,
            tokens_per_second=15.2,
        )
    )
    assert "context 355/1.05M" in text
    assert "tokens 380" in text
    assert "prompt" not in text
    assert "completion" not in text


def test_bare_atlas_opens_the_tui(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, list[str]] = {}

    def fake(argv: list[str]) -> int:
        seen["argv"] = argv
        return 7

    monkeypatch.setattr("atlas.cli.tui.launch_tui", fake)

    assert main([]) == 7
    assert seen["argv"] == []


def test_atlas_tui_forwards_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, list[str]] = {}

    def fake(argv: list[str]) -> int:
        seen["argv"] = argv
        return 0

    monkeypatch.setattr("atlas.cli.tui.launch_tui", fake)

    assert main(["tui", "--workspace", "/tmp/work"]) == 0
    assert seen["argv"] == ["--workspace", "/tmp/work"]


def test_data_command_does_not_open_the_tui(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fake(argv: list[str]) -> int:
        raise AssertionError(argv)

    monkeypatch.setattr("atlas.cli.tui.launch_tui", fake)

    assert main(["data", "--list"]) == 0
    assert "sentinel2" in capsys.readouterr().out


def test_root_help_mentions_tui_and_data(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])

    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "agent TUI" in out
    assert "atlas data" in out


def test_launch_replaces_the_process(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    binary = tmp_path / "atlas-tui"
    binary.write_text("", encoding="utf-8")
    recorded: dict[str, object] = {}

    monkeypatch.setattr("atlas.cli.tui.ensure_tui_binary", lambda: binary)

    def fake_exec(path: str, argv: list[str]) -> None:
        recorded["path"] = path
        recorded["argv"] = argv

    monkeypatch.setattr("atlas.cli.tui.os.execv", fake_exec)

    launch_tui(["--repo", "/tmp/atlas"])

    assert recorded["path"] == str(binary)
    assert recorded["argv"] == [str(binary), "--repo", "/tmp/atlas"]


def test_explicit_binary_is_used_without_building(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binary = tmp_path / "atlas-tui"
    binary.write_text("", encoding="utf-8")
    monkeypatch.setenv("ATLAS_TUI", str(binary))

    def fail_build(repo: Path, cargo: Path) -> Path:
        raise AssertionError((repo, cargo))

    monkeypatch.setattr("atlas.cli.tui._build", fail_build)

    assert ensure_tui_binary() == binary


def test_stale_binary_is_rebuilt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path)
    debug = repo / "tui" / "target" / "debug" / "atlas-tui"
    debug.parent.mkdir(parents=True)
    debug.write_text("old", encoding="utf-8")
    source = repo / "tui" / "src" / "main.rs"
    os.utime(debug, (1, 1))
    os.utime(source, (10, 10))
    monkeypatch.delenv("ATLAS_TUI", raising=False)
    monkeypatch.chdir(repo)
    monkeypatch.setattr("atlas.cli.tui._cargo_path", lambda: tmp_path / "cargo")

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        release = repo / "tui" / "target" / "release" / "atlas-tui"
        release.parent.mkdir(parents=True, exist_ok=True)
        release.write_text("new", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("atlas.cli.tui.subprocess.run", fake_run)

    assert ensure_tui_binary() == repo / "tui" / "target" / "release" / "atlas-tui"


def test_missing_cargo_reports_a_clear_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _fake_repo(tmp_path)
    monkeypatch.delenv("ATLAS_TUI", raising=False)
    monkeypatch.chdir(repo)
    monkeypatch.setattr("atlas.cli.tui._cargo_path", lambda: None)
    monkeypatch.setattr("atlas.cli.tui._which", lambda name: None)

    with pytest.raises(TuiLaunchError, match="cargo was not found"):
        ensure_tui_binary()

    assert launch_tui([]) == 1
    assert "cargo was not found" in capsys.readouterr().err


def _fake_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "atlas"
    (repo / "tui" / "src").mkdir(parents=True)
    (repo / "pyproject.toml").write_text('name = "atlas"\n', encoding="utf-8")
    (repo / "tui" / "Cargo.toml").write_text('[package]\nname = "atlas-tui"\n', encoding="utf-8")
    (repo / "tui" / "src" / "main.rs").write_text("fn main() {}\n", encoding="utf-8")
    return repo
