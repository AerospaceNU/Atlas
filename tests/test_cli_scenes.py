from __future__ import annotations

import argparse
import contextlib
import io
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

import httpx
import pytest

from atlas.agent.layout import pull_manifest_path
from atlas.cli import main
from atlas.cli.data import run_data
from atlas.data.base import Asset, BBox, DataPullClient, PullRequest, PullResult, Scene, SceneKind

TEST_COORDS = "-71.12,42.32,-71.02,42.40"


class OfflineClient(DataPullClient):
    satellite: ClassVar[str] = "sentinel2"
    scene_kind: ClassVar[SceneKind] = SceneKind.optical
    search_url: ClassVar[str] = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
    collection = "sentinel-2-l2a"

    def __init__(self, scenes: list[Scene]) -> None:
        self._scenes = scenes

    async def search(self, request: PullRequest) -> PullResult:
        return PullResult(request=request, scenes=self._scenes)


def _scene(scene_id: str, day: int) -> Scene:
    return Scene(
        id=scene_id,
        datetime=datetime(2024, 7, day, 15, 30, tzinfo=UTC),
        bbox=BBox(west=-71.5, south=42.0, east=-70.5, north=42.8),
        platform="sentinel-2a",
        cloud_cover=10.0,
        # Not a GeoTIFF, so the pull fetches it whole instead of as a COG window.
        assets={
            "visual": Asset(
                href="https://example.com/visual.nc?sig=SECRET",
                media_type="application/x-netcdf",
                roles=["visual"],
            )
        },
    )


def _pull(out: Path, scenes: list[Scene], *, search_only: bool) -> int:
    args = argparse.Namespace(
        list_catalog=False,
        satellite="sentinel2",
        kind="optical",
        date="2024-07-01:2024-07-07",
        coords=TEST_COORDS,
        out=out,
        limit=10,
        min_cloud_cover=None,
        max_cloud_cover=None,
        concurrency=2,
        search_only=search_only,
        asset=None,
        yes=True,
    )
    http = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b"II*\x00"))
    )
    return run_data(args, clients={"sentinel2": OfflineClient(scenes)}, http=http)


def _scenes_json(out: Path, *extra: str) -> dict[str, Any]:
    result: dict[str, Any] = json.loads(_run_scenes(out, "--json", *extra))
    return result


def _run_scenes(out: Path, *extra: str) -> str:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = main(["scenes", "--out", str(out), "--coords", TEST_COORDS, *extra])
    assert code == 0
    return buffer.getvalue()


def test_pull_then_query_offline_keeps_index_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "data"
    assert _pull(out, [_scene("S2A_0702", 2), _scene("S2A_0705", 5)], search_only=True) == 0

    def no_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("atlas scenes must not open an HTTP client")

    monkeypatch.setattr(httpx.AsyncClient, "__init__", no_network)
    monkeypatch.setattr(httpx.Client, "__init__", no_network)
    found = _scenes_json(out, "--date", "2024-07-01:2024-07-07")
    monkeypatch.undo()
    monkeypatch.chdir(tmp_path)

    ids = [card["index_id"] for card in found["groups"][0]["cards"]]
    assert len(ids) == 2
    assert {card["availability"] for card in found["groups"][0]["cards"]} == {"found"}

    assert _pull(out, [_scene("S2A_0702", 2)], search_only=False) == 0
    after = _scenes_json(out, "--date", "2024-07-01:2024-07-07")
    cards = after["groups"][0]["cards"]
    assert [card["index_id"] for card in cards] == ids
    assert cards[0]["availability"] == "downloaded"
    assert cards[0]["local_path"] == "data/sentinel2/S2A_0702_visual.nc"
    assert cards[1]["availability"] == "found"
    assert cards[0]["found_via"] == ["sentinel2"]
    assert cards[0]["catalog"] == "planetarycomputer.microsoft.com"
    assert "http" not in json.dumps(after)
    snapshots = sqlite3.connect(out / "index.sqlite").execute("SELECT scene_json FROM scenes")
    stored = " ".join(raw for (raw,) in snapshots.fetchall())
    assert "example.com/visual.nc" in stored
    assert "SECRET" not in stored


def test_before_after_reports_missing_date(tmp_path: Path) -> None:
    assert _pull(tmp_path, [_scene("S2A_0702", 2)], search_only=True) == 0
    result = _scenes_json(tmp_path, "--on", "2024-07-02", "--on", "2024-07-20")
    assert [group["gap"] for group in result["groups"]] == [False, True]
    text = _run_scenes(tmp_path, "--on", "2024-07-02", "--on", "2024-07-20")
    assert "2024-07-20: no scenes found" in text
    assert "sc_" in text


def test_no_index_and_usage_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        main(
            [
                "scenes",
                "--out",
                str(tmp_path),
                "--coords",
                TEST_COORDS,
                "--date",
                "2024-07-01:2024-07-02",
            ]
        )
        == 1
    )
    assert "no scene index" in capsys.readouterr().err
    assert _pull(tmp_path, [_scene("S2A_0702", 2)], search_only=True) == 0
    assert main(["scenes", "--out", str(tmp_path), "--coords", TEST_COORDS]) == 2
    assert main(["scenes", "--out", str(tmp_path), "--date", "2024-07-01:2024-07-02"]) == 2
    assert main(["scenes", "--out", str(tmp_path), "--rebuild"]) == 0
    assert "rebuilt 1 scene cards" in capsys.readouterr().err


def test_index_failure_is_reported_and_keeps_the_pull_exit_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # An index written by a newer Atlas cannot be updated; the pull itself still succeeds.
    out = tmp_path / "data"
    out.mkdir()
    db = sqlite3.connect(out / "index.sqlite")
    with db:
        db.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        db.execute("INSERT INTO meta VALUES ('schema_version', '99')")
    db.close()
    assert _pull(out, [_scene("S2A_0702", 2)], search_only=True) == 0
    err = capsys.readouterr().err
    assert "scene index not updated" in err
    assert "IndexVersionError" in err
    assert pull_manifest_path(out).exists()
    db = sqlite3.connect(out / "index.sqlite")
    tables = {name for (name,) in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    db.close()
    assert tables == {"meta"}


def test_cloud_order_flag(tmp_path: Path) -> None:
    scenes = [
        _scene("S2A_0702", 2).model_copy(update={"cloud_cover": 40.0}),
        _scene("S2A_0705", 5).model_copy(update={"cloud_cover": 3.0}),
    ]
    assert _pull(tmp_path, scenes, search_only=True) == 0
    result = _scenes_json(tmp_path, "--date", "2024-07-01:2024-07-07", "--order", "cloud")
    assert [card["item_id"] for card in result["groups"][0]["cards"]] == ["S2A_0705", "S2A_0702"]
