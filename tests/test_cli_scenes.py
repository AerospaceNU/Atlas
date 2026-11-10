from __future__ import annotations

import argparse
import contextlib
import io
import json
import sqlite3
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, ClassVar

import httpx
import pytest

from atlas.agent.layout import pull_manifest_path
from atlas.archive import Availability, SceneIndex
from atlas.cli import main
from atlas.cli.data import run_data
from atlas.cli.download import DownloadOutcome, DownloadReport, write_aoi
from atlas.cli.indexing import index_pull
from atlas.compile.product import SceneCatalog, SourceScenes
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
    layers = [card["layers"]["visual"] for card in found["groups"][0]["cards"]]
    assert {layer["availability"] for layer in layers} == {"absent"}

    assert _pull(out, [_scene("S2A_0702", 2)], search_only=False) == 0
    after = _scenes_json(out, "--date", "2024-07-01:2024-07-07")
    cards = after["groups"][0]["cards"]
    assert [card["index_id"] for card in cards] == ids
    assert cards[0]["layers"]["visual"]["availability"] == "full"
    assert cards[0]["layers"]["visual"]["locator"] == {
        "scheme": "workspace",
        "key": "data/sentinel2/S2A_0702_visual.nc",
    }
    assert cards[1]["layers"]["visual"]["availability"] == "absent"
    assert (cards[0]["source"], cards[0]["found_in"]) == ("sentinel2", ["sentinel2"])
    assert "http" not in json.dumps(after)
    table = _run_scenes(out, "--date", "2024-07-01:2024-07-07")
    assert "full" in table  # LOCAL column
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
    cards = result["groups"][0]["cards"]
    assert [card["provider_id"] for card in cards] == ["S2A_0705", "S2A_0702"]


def test_index_pull_records_a_chip_with_its_window(tmp_path: Path) -> None:
    chip = BBox(west=-71.12, south=42.32, east=-71.02, north=42.40)
    scene = _scene("S2A_0702", 2).model_copy(
        update={"source": "sentinel2", "collection": "sentinel-2-l2a"}
    )
    path = tmp_path / "sentinel2" / "S2A_0702_visual.tif"
    path.parent.mkdir()
    path.write_bytes(b"II*\x00")
    write_aoi(path, chip)
    catalog = SceneCatalog(
        request=PullRequest(start_date=date(2024, 7, 1), end_date=date(2024, 7, 7), bbox=chip),
        sources=[SourceScenes(source="sentinel2", collection="sentinel-2-l2a", scenes=[scene])],
    )
    report = DownloadReport(
        outcomes=[
            DownloadOutcome(
                source="sentinel2", scene_id="S2A_0702", status="saved", path=path, asset="visual"
            )
        ]
    )
    summary = index_pull(tmp_path, catalog, report, {"sentinel2": OfflineClient([scene])})
    assert summary is not None and summary.stats is not None
    index_id = summary.index_id("sentinel2", "S2A_0702")
    assert index_id is not None
    assert summary.describe(path="index.sqlite") == "scene index index.sqlite: 1 new, 0 refreshed"
    with SceneIndex.at(tmp_path) as index:
        card = index.lookup(index_id).card
    assert card is not None
    assert card.layers["visual"].availability is Availability.windowed
    assert card.layers["visual"].window == chip
