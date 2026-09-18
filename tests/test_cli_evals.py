from __future__ import annotations

import json
from pathlib import Path

import pytest

from atlas.cli import main

_DATE = "2024-07-01:2024-07-07"
_COORDS = "-71.12,42.32,-71.02,42.40"


@pytest.mark.evals
def test_cli_live_search_only(tmp_path: Path) -> None:
    code = main(
        [
            "data",
            "sentinel2",
            "optical",
            "--date",
            _DATE,
            "--coords",
            _COORDS,
            "--limit",
            "1",
            "--search-only",
            "--out",
            str(tmp_path),
        ]
    )
    assert code == 0
    payload = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    src = next(item for item in payload["sources"] if item["source"] == "sentinel2")
    assert src["error"] is None
    assert src["n_scenes"] >= 1


@pytest.mark.evals
def test_cli_live_download_skips_existing(tmp_path: Path) -> None:
    argv = [
        "data",
        "sentinel2",
        "optical",
        "--date",
        _DATE,
        "--coords",
        _COORDS,
        "--limit",
        "1",
        "--out",
        str(tmp_path),
    ]
    assert main(argv) == 0
    rasters = [path for path in (tmp_path / "sentinel2").glob("*") if path.suffix != ".json"]
    assert rasters
    first = rasters[0]
    assert first.stat().st_size > 0
    stamp = first.stat().st_mtime_ns
    assert main(argv) == 0
    assert first.stat().st_mtime_ns == stamp
    payload = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    src = next(item for item in payload["sources"] if item["source"] == "sentinel2")
    assert src["skipped"]
