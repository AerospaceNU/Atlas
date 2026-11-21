from __future__ import annotations

import re
from pathlib import Path

import pytest

from atlas.agent.artifacts import ArtifactError, LocalArtifactStore
from atlas.agent.tools.read_file import ReadFileTool
from atlas.agent.tools.read_files import ReadFilesTool


def _numbered_file(path: Path, count: int, width: int = 40) -> list[str]:
    lines = [f"line {i:06d} " + "x" * width for i in range(1, count + 1)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


def _read(store: LocalArtifactStore, **arguments: object) -> str:
    return ReadFileTool().run(arguments, store).text


def test_small_file_is_returned_byte_identical_to_the_old_output(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    (tmp_path / "a.txt").write_bytes(b"one\r\ntwo\rthree\n")

    assert _read(store, path="a.txt") == "one\ntwo\nthree\n"


def test_large_file_gets_the_default_window_and_a_continue_footer(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    lines = _numbered_file(tmp_path / "big.txt", 2000, width=5)

    text = _read(store, path="big.txt")

    body, footer = text.rsplit("\n", 1)
    assert body.splitlines() == lines[:500]
    assert footer == "... [lines 1-500 of 2000; continue with offset=501]"


def test_explicit_offset_and_limit(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    lines = _numbered_file(tmp_path / "big.txt", 100, width=5)

    text = _read(store, path="big.txt", offset=10, limit=3)

    assert text.splitlines() == [*lines[9:12], "... [lines 10-12 of 100; continue with offset=13]"]
    tail = _read(store, path="big.txt", offset=98, limit=50)
    assert tail.splitlines() == [*lines[97:], "... [lines 98-100 of 100]"]


def test_char_budget_cuts_at_a_line_boundary_and_following_the_footer_loses_nothing(
    tmp_path: Path,
) -> None:
    store = LocalArtifactStore(tmp_path)
    lines = _numbered_file(tmp_path / "big.txt", 3000, width=100)
    collected: list[str] = []
    offset = 1
    while True:
        text = _read(store, path="big.txt", offset=offset, limit=2000)
        *body, footer = text.split("\n")
        assert len("\n".join(body)) <= ReadFileTool.max_chars
        collected.extend(body)
        match = re.search(r"continue with offset=(\d+)\]$", footer)
        if match is None:
            assert footer.endswith("of 3000]")
            break
        assert int(match.group(1)) == offset + len(body)
        offset = int(match.group(1))
    assert collected == lines


def test_a_single_overlong_line_is_cut_with_a_marker(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    (tmp_path / "long.txt").write_text("y" * 50_000 + "\nnext\n", encoding="utf-8")

    text = _read(store, path="long.txt")

    first, footer = text.split("\n", 1)[0], text.split("\n")[-1]
    assert first.startswith("y" * ReadFileTool.max_chars)
    assert first.endswith(
        " ... [line 1 truncated: 34000 more characters; lines this long are better "
        "searched with grep or sliced through bash]"
    )
    assert footer == "... [lines 1-1 of 2; continue with offset=2]"


def test_offset_past_the_end(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    (tmp_path / "s.txt").write_text("a\nb\n", encoding="utf-8")

    assert (
        _read(store, path="s.txt", offset=9) == "[offset 9 is past the end of the file (2 lines)]"
    )


def test_a_300_kb_file_is_readable_in_windows(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    data = "".join(f"row {i:07d} {'z' * 40}\n" for i in range(6000))
    assert len(data) > 300_000
    (tmp_path / "wide.txt").write_text(data, encoding="utf-8")

    window = _read(store, path="wide.txt", offset=5000, limit=10)

    assert window.splitlines()[0].startswith("row 0004999")
    assert window.endswith("continue with offset=5010]")


def test_over_max_bytes_names_the_alternatives(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    (tmp_path / "huge.bin").write_bytes(b"x" * (ReadFileTool.max_bytes + 1))

    with pytest.raises(ArtifactError, match="byte read limit; search it with grep") as info:
        ReadFileTool().run({"path": "huge.bin"}, store)
    assert "huge.bin exceeds the 5000000 byte read limit" in str(info.value)


def test_read_files_stays_under_the_budget_with_truthful_headers(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    all_lines = {}
    for name in ("a.txt", "b.txt", "c.txt"):
        all_lines[name] = _numbered_file(tmp_path / name, 1000, width=60)

    result = ReadFilesTool().run({"files": [{"path": name} for name in all_lines]}, store)

    assert len(result.text) < ReadFilesTool.max_output_chars
    for name, lines in all_lines.items():
        match = re.search(rf"== {name} \(lines 1-(\d+) of 1000\)\n", result.text)
        assert match is not None
        shown = int(match.group(1))
        assert shown < 1000
        assert f"{shown:>6}| {lines[shown - 1]}" in result.text
        assert f"{shown + 1:>6}| " not in result.text.split(f"== {name}")[1].split("\n\n")[0]
        assert f"... [{1000 - shown} more lines; continue with offset={shown + 1}]" in result.text


def test_read_files_leaves_small_files_unchanged(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    (tmp_path / "a.txt").write_text("alpha\nbeta\n", encoding="utf-8")

    text = ReadFilesTool().run({"files": [{"path": "a.txt"}]}, store).text

    assert text == "== a.txt (lines 1-2 of 2)\n     1| alpha\n     2| beta"


def test_read_file_limit_above_the_maximum_is_clamped(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    (tmp_path / "huge.txt").write_text("".join(f"l{i}\n" for i in range(3000)), encoding="utf-8")
    text = _read(store, path="huge.txt", limit=10**6)
    assert "l0" in text
    assert "l1999" in text and "l2000" not in text.split("[")[0]
