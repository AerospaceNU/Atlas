from __future__ import annotations

import builtins
import io
import os
from pathlib import Path
from typing import Any

import pytest

from atlas.agent.artifacts import ArtifactError, LocalArtifactStore
from atlas.agent.tools.read_file import ReadFileTool


def _underreport_size_and_track_reads(
    monkeypatch: pytest.MonkeyPatch, target: Path, limit: int
) -> list[int]:
    original_stat, original_fstat, original_open = Path.stat, os.fstat, io.open
    identity = original_stat(target)
    sizes: list[int] = []

    def small_stat(info: os.stat_result) -> os.stat_result:
        if (info.st_dev, info.st_ino) == (identity.st_dev, identity.st_ino):
            values = list(info)
            values[6] = 0
            return os.stat_result(values)
        return info

    def path_stat(path: Path, *args: Any, **kwargs: Any) -> os.stat_result:
        return small_stat(original_stat(path, *args, **kwargs))

    class Reader:
        def __init__(self, stream: Any) -> None:
            self.stream = stream

        def __enter__(self) -> Reader:
            self.stream.__enter__()
            return self

        def __exit__(self, *args: Any) -> Any:
            return self.stream.__exit__(*args)

        def read(self, size: int = -1) -> Any:
            sizes.append(size)
            assert 0 <= size <= limit + 1, "file read must be bounded before allocation"
            return self.stream.read(size)

        def __getattr__(self, name: str) -> Any:
            return getattr(self.stream, name)

    def open_spy(file: Any, *args: Any, **kwargs: Any) -> Any:
        stream = original_open(file, *args, **kwargs)
        info = original_fstat(stream.fileno())
        return (
            Reader(stream)
            if (info.st_dev, info.st_ino) == (identity.st_dev, identity.st_ino)
            else stream
        )

    monkeypatch.setattr(Path, "stat", path_stat)
    monkeypatch.setattr(os, "fstat", lambda fd: small_stat(original_fstat(fd)))
    monkeypatch.setattr(io, "open", open_spy)
    monkeypatch.setattr(builtins, "open", open_spy)
    return sizes


@pytest.mark.parametrize("tool", [False, True], ids=["project_bytes", "read_file"])
def test_underreported_file_size_cannot_bypass_the_read_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tool: bool
) -> None:
    limit = ReadFileTool.max_bytes if tool else 32
    store = LocalArtifactStore(tmp_path / "workspace")
    target = tmp_path / "outside.txt"
    target.write_bytes(b"x" * (limit * 2))
    sizes = _underreport_size_and_track_reads(monkeypatch, target, limit)

    with pytest.raises(ArtifactError, match="read limit"):
        if tool:
            ReadFileTool().run({"path": str(target)}, store)
        else:
            store.read_project_bytes(str(target), max_bytes=limit)
    assert sizes == [limit + 1]


@pytest.mark.parametrize("tool", [False, True], ids=["project_bytes", "read_file"])
def test_a_file_that_grows_after_fstat_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tool: bool
) -> None:
    store = LocalArtifactStore(tmp_path)
    limit = ReadFileTool.max_bytes if tool else 32
    target = tmp_path / "growing.txt"
    target.write_bytes(b"x" * limit)
    original_fstat = os.fstat
    identity = target.stat()
    grew = False

    def grow_after_stat(fd: int) -> os.stat_result:
        nonlocal grew
        info = original_fstat(fd)
        if not grew and (info.st_dev, info.st_ino) == (identity.st_dev, identity.st_ino):
            grew = True
            writer = os.open(target, os.O_WRONLY | os.O_APPEND)
            try:
                os.write(writer, b"extra")
            finally:
                os.close(writer)
        return info

    monkeypatch.setattr(os, "fstat", grow_after_stat)
    with pytest.raises(ArtifactError, match="read limit"):
        if tool:
            ReadFileTool().run({"path": "growing.txt"}, store)
        else:
            store.read_project_bytes("growing.txt", max_bytes=limit)
    assert grew


def test_exact_byte_limits_and_universal_newlines_are_preserved(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    (tmp_path / "exact.txt").write_bytes("é".encode() * (ReadFileTool.max_bytes // 2))
    assert ReadFileTool().run({"path": "exact.txt"}, store).text == "é" * 50_000
    assert len(store.read_project_bytes("exact.txt", max_bytes=100_000)) == 100_000
    (tmp_path / "lines.txt").write_bytes(b"one\r\ntwo\rthree\n")
    assert ReadFileTool().run({"path": "lines.txt"}, store).text == "one\ntwo\nthree\n"


def test_zero_byte_limit_only_accepts_empty_files(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    (tmp_path / "empty.txt").write_bytes(b"")
    (tmp_path / "one.txt").write_bytes(b"x")
    assert store.read_project_bytes("empty.txt", max_bytes=0) == b""
    with pytest.raises(ArtifactError, match="read limit"):
        store.read_project_bytes("one.txt", max_bytes=0)
