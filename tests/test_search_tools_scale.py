from __future__ import annotations

from pathlib import Path

import pytest

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.tools._walk import SKIP_DIR_NAMES
from atlas.agent.tools.glob import GlobTool
from atlas.agent.tools.grep import GrepTool
from atlas.agent.tools.ls import LsTool

GNU_NOTE = (
    "note: no match for the pattern as written; retried reading \\| \\( \\) as regex operators "
    "(this tool uses Python regex: write | ( ) unescaped)"
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_ignored_directory_does_not_eat_the_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = LocalArtifactStore(tmp_path)
    for index in range(20):
        _write(tmp_path / ".git" / f"obj{index}.txt", "x" * 1000)
    _write(tmp_path / "src.txt", "needle\n")
    monkeypatch.setattr(GrepTool, "max_scan_bytes", 5_000)

    result = GrepTool().run({"pattern": "needle"}, store)

    assert result.text == "src.txt:1:needle"


def test_exhausted_budget_is_reported_with_zero_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = LocalArtifactStore(tmp_path)
    for index in range(5):
        _write(tmp_path / f"a{index}.txt", "filler\n" * 10)
    _write(tmp_path / "z.txt", "needle\n")
    monkeypatch.setattr(GrepTool, "max_scan_bytes", 100)

    result = GrepTool().run({"pattern": "needle"}, store)

    assert result.text == ("no matches (scan limit reached after 2 files; narrow path or glob)")


def test_exhausted_budget_with_matches_keeps_trailing_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = LocalArtifactStore(tmp_path)
    _write(tmp_path / "a.txt", "needle\n" * 20)
    _write(tmp_path / "b.txt", "needle\n")
    monkeypatch.setattr(GrepTool, "max_scan_bytes", 10)

    result = GrepTool().run({"pattern": "needle"}, store)

    assert result.text.splitlines()[-1] == "... [scan limit reached]"


def test_binary_and_oversize_files_do_not_count_toward_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = LocalArtifactStore(tmp_path)
    (tmp_path / "a.bin").write_bytes(b"\x00" * 5000)
    _write(tmp_path / "b.txt", "x" * 5000)
    _write(tmp_path / "c.txt", "needle\n")
    monkeypatch.setattr(GrepTool, "max_scan_bytes", 5_001)
    monkeypatch.setattr(GrepTool, "max_file_bytes", 5_000)
    _write(tmp_path / "big.txt", "y" * 6000)

    result = GrepTool().run({"pattern": "needle"}, store)

    assert result.text == "c.txt:1:needle"


def test_ignored_directories_are_skipped_unless_asked(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    _write(tmp_path / ".git" / "a.txt", "needle git\n")
    _write(tmp_path / ".venv" / "lib" / "b.txt", "needle venv\n")
    _write(tmp_path / ".atlas" / "sessions" / "s.txt", "needle session\n")
    _write(tmp_path / "keep.txt", "needle keep\n")

    default = GrepTool().run({"pattern": "needle"}, store).text
    included = GrepTool().run({"pattern": "needle", "include_ignored": True}, store).text
    explicit = GrepTool().run({"pattern": "needle", "path": ".git"}, store).text
    explicit_session = GrepTool().run({"pattern": "needle", "path": ".atlas/sessions"}, store).text

    assert default == "keep.txt:1:needle keep"
    assert ".git/a.txt:1:needle git" in included
    assert ".venv/lib/b.txt:1:needle venv" in included
    assert ".atlas/sessions/s.txt:1:needle session" in included
    assert explicit == ".git/a.txt:1:needle git"
    assert explicit_session == ".atlas/sessions/s.txt:1:needle session"


def test_atlas_data_and_artifacts_are_searched_by_default(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    _write(tmp_path / ".atlas" / "data" / "scene.json", "needle\n")
    _write(tmp_path / ".atlas" / "artifacts" / "note.txt", "needle\n")
    _write(tmp_path / ".atlas" / "checkpoints" / "copy.txt", "needle\n")
    _write(tmp_path / ".atlas" / "tool-output" / "out.txt", "needle\n")

    result = GrepTool().run({"pattern": "needle"}, store).text

    assert result.splitlines() == [
        ".atlas/artifacts/note.txt:1:needle",
        ".atlas/data/scene.json:1:needle",
    ]


def test_context_layout_merges_windows_and_separates_groups(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    body = [f"line{n}" for n in range(1, 21)]
    body[4] = "hit five"
    body[6] = "hit seven"
    body[15] = "hit sixteen"
    _write(tmp_path / "f.txt", "\n".join(body) + "\n")

    result = GrepTool().run({"pattern": "hit", "context": 1}, store).text

    assert result.splitlines() == [
        "f.txt-4-line4",
        "f.txt:5:hit five",
        "f.txt-6-line6",
        "f.txt:7:hit seven",
        "f.txt-8-line8",
        "--",
        "f.txt-15-line15",
        "f.txt:16:hit sixteen",
        "f.txt-17-line17",
    ]


def test_context_separates_files_and_zero_context_has_no_separator(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    _write(tmp_path / "a.txt", "hit\n")
    _write(tmp_path / "b.txt", "hit\n")

    with_context = GrepTool().run({"pattern": "hit", "context": 2}, store).text
    plain = GrepTool().run({"pattern": "hit"}, store).text

    assert with_context.splitlines() == ["a.txt:1:hit", "--", "b.txt:1:hit"]
    assert plain.splitlines() == ["a.txt:1:hit", "b.txt:1:hit"]


def test_files_with_matches_and_count_modes(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    _write(tmp_path / "a.txt", "hit\nhit\n")
    _write(tmp_path / "b.txt", "hit\n")
    _write(tmp_path / "c.txt", "none\n")

    files = GrepTool().run({"pattern": "hit", "output_mode": "files_with_matches"}, store)
    counts = GrepTool().run({"pattern": "hit", "output_mode": "count"}, store)

    assert files.text.splitlines() == ["a.txt", "b.txt"]
    assert counts.text.splitlines() == ["a.txt:2", "b.txt:1", "total: 3"]


def test_gnu_alternation_is_retried_with_a_note(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    _write(tmp_path / "f.txt", "alpha\nbeta\n")

    result = GrepTool().run({"pattern": "alpha\\|beta"}, store).text

    assert result.splitlines() == [GNU_NOTE, "f.txt:1:alpha", "f.txt:2:beta"]


def test_literal_pipe_that_matches_is_not_rewritten(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    _write(tmp_path / "f.txt", "a|b\nalpha\n")

    result = GrepTool().run({"pattern": "a\\|b"}, store).text

    assert result == "f.txt:1:a|b"


def test_gnu_retry_that_finds_nothing_appends_the_hint(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    _write(tmp_path / "f.txt", "nothing\n")

    result = GrepTool().run({"pattern": "alpha\\|beta"}, store).text
    invalid = GrepTool().run({"pattern": "a\\(b"}, store).text

    assert result == f"no matches\n{GNU_NOTE}"
    assert invalid == f"no matches\n{GNU_NOTE}"


def test_result_is_capped_at_sixteen_thousand_characters(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    _write(tmp_path / "f.txt", "".join(f"hit {'x' * 100} {n}\n" for n in range(500)))

    result = GrepTool().run({"pattern": "hit", "max_matches": 1000}, store).text

    last = result.splitlines()[-1]
    assert last.startswith("... [") and 'output_mode="files_with_matches"]' in last
    assert len(result) <= 16_000
    shown = len(result.splitlines()) - 1
    assert f"... [{500 - shown} more matching lines;" in last


def test_multi_segment_glob_prunes_without_entering_ignored_directories(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    _write(tmp_path / "src" / "a" / "x.py", "x\n")
    _write(tmp_path / "src" / "x.py", "x\n")
    _write(tmp_path / ".venv" / "src" / "x.py", "x\n")
    _write(tmp_path / ".git" / "src" / "x.py", "x\n")

    default = GlobTool().run({"pattern": "src/**/x.py"}, store).text.splitlines()
    plain = GlobTool().run({"pattern": "src/x.py"}, store).text.splitlines()
    everything = GlobTool().run({"pattern": "src/**/x.py", "include_ignored": True}, store).text

    assert default == ["src/a/x.py", "src/x.py"]
    assert plain == ["src/x.py"]
    assert ".venv/src/x.py" in everything and ".git/src/x.py" in everything


def test_glob_prunes_ignored_directories(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    _write(tmp_path / "a.py", "a\n")
    _write(tmp_path / ".git" / "hook.py", "h\n")
    _write(tmp_path / ".venv" / "lib" / "v.py", "v\n")
    _write(tmp_path / "sub" / "node_modules" / "n.py", "n\n")
    _write(tmp_path / "sub" / "b.py", "b\n")

    default = GlobTool().run({"pattern": "**/*.py"}, store).text.splitlines()
    nested = GlobTool().run({"pattern": "sub/**/*.py"}, store).text.splitlines()
    everything = GlobTool().run({"pattern": "**/*.py", "include_ignored": True}, store)

    assert default == ["a.py", "sub/b.py"]
    assert nested == ["sub/b.py"]
    assert ".git/hook.py" in everything.text
    assert ".venv/lib/v.py" in everything.text
    assert "sub/node_modules/n.py" in everything.text


def test_glob_explicit_path_inside_ignored_directory_is_searched(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    _write(tmp_path / ".git" / "hook.py", "h\n")

    result = GlobTool().run({"pattern": "*.py", "path": ".git"}, store).text

    assert result == ".git/hook.py"


def test_ls_recursive_prunes_ignored_directories(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    _write(tmp_path / "a.txt", "a\n")
    _write(tmp_path / ".git" / "HEAD", "ref\n")
    _write(tmp_path / "__pycache__" / "x.pyc", "x\n")
    _write(tmp_path / ".atlas" / "data" / "d.json", "{}\n")
    _write(tmp_path / ".atlas" / "sessions" / "s.json", "{}\n")

    default = LsTool().run({"recursive": True}, store).text.splitlines()
    included = LsTool().run({"recursive": True, "include_ignored": True}, store).text

    assert ".git/" not in default and ".git/HEAD" not in default
    assert "__pycache__/" not in default
    assert ".atlas/sessions/" not in default
    assert ".atlas/data/d.json" in default
    assert "a.txt" in default
    assert ".git/HEAD" in included
    assert ".atlas/sessions/s.json" in included


def test_skip_names_cover_the_expected_directories() -> None:
    assert {".git", ".venv", "node_modules", "__pycache__", ".tox"} <= SKIP_DIR_NAMES


def test_glob_pattern_starting_in_a_skipped_directory_is_explicit(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    _write(tmp_path / ".git" / "config", "x")
    _write(tmp_path / "node_modules" / "pkg" / "index.js", "x")

    git = GlobTool().run({"pattern": ".git/*"}, store).text
    node = GlobTool().run({"pattern": "node_modules/pkg/*.js"}, store).text

    assert git.splitlines() == [".git/config"]
    assert node.splitlines() == ["node_modules/pkg/index.js"]


def test_grep_does_not_retry_after_the_scan_budget_stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = LocalArtifactStore(tmp_path)
    for index in range(5):
        _write(tmp_path / f"a{index}.txt", "filler\n" * 10)
    monkeypatch.setattr(GrepTool, "max_scan_bytes", 100)
    scans: list[str] = []
    real = GrepTool._scan

    def counting(self: GrepTool, request: object, *args: object) -> object:
        scans.append("scan")
        return real(self, request, *args)  # type: ignore[arg-type]

    monkeypatch.setattr(GrepTool, "_scan", counting)

    result = GrepTool().run({"pattern": "a\\|b"}, store).text

    assert len(scans) == 1
    assert result.startswith("no matches (scan limit reached")
    assert result.endswith(GNU_NOTE)


def test_grep_line_numbers_ignore_form_feeds_and_unicode_separators(tmp_path: Path) -> None:
    ff, ls = chr(12), chr(0x2028)
    body = f"one{ff}two\ntwo{ls}x\nneedle here\r\nlast\n"
    (tmp_path / "pg.py").write_bytes(body.encode("utf-8"))

    result = GrepTool().run({"pattern": "needle"}, LocalArtifactStore(tmp_path)).text

    assert result == "pg.py:3:needle here"
    count = GrepTool().run({"pattern": "^last$", "context": 1}, LocalArtifactStore(tmp_path)).text
    assert "pg.py:4:last" in count and "pg.py-3-needle here" in count


def test_grep_description_mentions_skipped_files() -> None:
    assert "Binary files and files over 1 MB are skipped" in GrepTool.description


def test_session_tool_output_is_skipped_but_other_artifacts_stay_visible(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    saved = tmp_path / ".atlas" / "artifacts" / "sess1" / "tool-output"
    _write(saved / "out.txt", "needle saved\n")
    _write(tmp_path / ".atlas" / "artifacts" / "sess1" / "report.txt", "needle report\n")
    _write(tmp_path / ".atlas" / "data" / "d.txt", "needle data\n")
    _write(tmp_path / "src" / "tool-output" / "mine.txt", "needle mine\n")

    default = GrepTool().run({"pattern": "needle"}, store).text
    included = GrepTool().run({"pattern": "needle", "include_ignored": True}, store).text
    explicit = (
        GrepTool()
        .run({"pattern": "needle", "path": ".atlas/artifacts/sess1/tool-output"}, store)
        .text
    )

    assert "saved" not in default
    assert "needle report" in default and "needle data" in default and "needle mine" in default
    assert "needle saved" in included
    assert explicit == ".atlas/artifacts/sess1/tool-output/out.txt:1:needle saved"
