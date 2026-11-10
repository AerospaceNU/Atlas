"""The find_scenes tool: offline lookups in the scene index for the agent."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

import httpx
import pytest
from pydantic import ValidationError

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import default_registry
from atlas.agent.layout import project_data_dir
from atlas.agent.system_prompt import load_system_prompt
from atlas.agent.tools.find_scenes import FindScenesTool
from atlas.agent.tools.retrieve_data_tool import GetDataTool
from atlas.archive import IndexEntry, SavedLayer, SceneIndex, make_index_id
from atlas.data.base import Asset, BBox, DataPullClient, PullRequest, PullResult, Scene, SceneKind

AOI = BBox(west=-71.12, south=42.32, east=-71.02, north=42.40)
WEEK = {"start_date": "2024-07-01", "end_date": "2024-07-07"}
MODULE = "atlas.agent.tools.retrieve_data_tool"


def _scene(scene_id: str, day: int, cloud: float | None) -> Scene:
    return Scene(
        id=scene_id,
        source="sentinel2",
        collection="sentinel-2-l2a",
        datetime=datetime(2024, 7, day, 15, 30, tzinfo=UTC),
        bbox=BBox(west=-71.5, south=42.0, east=-70.5, north=42.8),
        platform="sentinel-2a",
        cloud_cover=cloud,
        assets={
            "visual": Asset(href="https://example.com/visual.tif?sig=SECRET", roles=["visual"])
        },
    )


class FakePullClient(DataPullClient):
    satellite: ClassVar[str] = "sentinel2"
    scene_kind: ClassVar[SceneKind] = SceneKind.optical

    def __init__(self, scenes: list[Scene]) -> None:
        self._scenes = scenes

    async def search(self, request: PullRequest) -> PullResult:
        return PullResult(request=request, scenes=self._scenes)


def _find(store: LocalArtifactStore, **arguments: Any) -> dict[str, Any]:
    body: dict[str, Any] = {"bbox": AOI.model_dump(), **WEEK}
    body.update(arguments)
    text = FindScenesTool().run(body, store).text
    payload: dict[str, Any] = json.loads(text.splitlines()[0])
    return payload


def _ids(group: dict[str, Any]) -> list[str]:
    return [card["provider_id"] for card in group["cards"]]


def test_tool_is_in_the_default_registry() -> None:
    assert "find_scenes" in {tool.name for tool in default_registry().definitions}


def test_search_once_then_find_the_clearest_without_searching_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenes = [_scene("cloudy", 2, 70.0), _scene("clear", 4, 3.5), _scene("hazy", 6, 20.0)]
    monkeypatch.setattr(f"{MODULE}.sources", lambda *_: {"sentinel2": FakePullClient(scenes)})
    store = LocalArtifactStore(tmp_path)
    searched = GetDataTool().run(
        {"data_type": "optical", **WEEK, "bbox": AOI.model_dump(), "sources": ["sentinel2"]},
        store,
    )
    clear_id = make_index_id("sentinel2", "sentinel-2-l2a", "clear")
    assert clear_id in searched.text

    def no_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("find_scenes must not open an HTTP client")

    monkeypatch.setattr(httpx.AsyncClient, "__init__", no_network)
    monkeypatch.setattr(httpx.Client, "__init__", no_network)
    payload = _find(store, satellites=["sentinel2"], order="cloud", limit=1)

    [group] = payload["groups"]
    assert [card["index_id"] for card in group["cards"]] == [clear_id]
    assert group["cards"][0]["cloud_cover"] == 3.5
    assert group["matched"] == 3
    assert "http" not in json.dumps(payload)


def test_on_dates_report_a_gap(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    with SceneIndex.at(store.resolve("data")) as index:
        index.upsert([IndexEntry(_scene("before", 2, 5.0))])
    payload = _find(store, start_date=None, end_date=None, on_dates=["2024-07-02", "2024-07-20"])
    before, after = payload["groups"]
    assert (_ids(before), before["gap"]) == (["before"], False)
    assert (after["cards"], after["gap"]) == ([], True)


def test_reads_both_data_folders_as_one_index(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    agent_dir = store.resolve("data")
    cli_dir = project_data_dir(tmp_path)
    saved = cli_dir / "sentinel2" / "shared_visual.tif"
    saved.parent.mkdir(parents=True)
    saved.write_bytes(b"II*\x00")
    with SceneIndex.at(agent_dir) as index:
        index.upsert([IndexEntry(_scene("agent", 2, 40.0)), IndexEntry(_scene("shared", 3, 9.0))])
    with SceneIndex.at(cli_dir) as index:
        index.upsert(
            [
                IndexEntry(_scene("cli", 5, 1.0)),
                IndexEntry(_scene("shared", 3, 9.0), saved=SavedLayer("visual", saved)),
            ]
        )

    payload = _find(store, order="cloud", limit=2)

    [group] = payload["groups"]
    assert _ids(group) == ["cli", "shared"]  # the clearest overall, across both folders
    assert group["matched"] == 3  # "shared" is in both and counts once
    shared = group["cards"][1]["layers"]["visual"]
    assert shared["availability"] == "full"  # the copy with local data wins
    assert shared["locator"] == {
        "scheme": "workspace",
        "key": ".atlas/data/sentinel2/shared_visual.tif",
    }
    assert payload["index_dirs"] == ["data", ".atlas/data"]


def test_no_index_says_so_and_creates_nothing(tmp_path: Path) -> None:
    result = FindScenesTool().run({"bbox": AOI.model_dump(), **WEEK}, LocalArtifactStore(tmp_path))
    assert "no scene index" in result.text
    assert not list(tmp_path.rglob("index.sqlite"))


def test_an_unreadable_index_is_reported_and_the_other_still_answers(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    with SceneIndex.at(store.resolve("data")) as index:
        index.upsert([IndexEntry(_scene("ok", 2, 5.0))])
    newer = project_data_dir(tmp_path) / "index.sqlite"
    newer.parent.mkdir(parents=True)
    db = sqlite3.connect(newer)
    with db:
        db.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        db.execute("INSERT INTO meta VALUES ('schema_version', '99')")
    db.close()

    text = FindScenesTool().run({"bbox": AOI.model_dump(), **WEEK}, store).text

    assert _ids(json.loads(text.splitlines()[0])["groups"][0]) == ["ok"]
    assert "cannot read .atlas/data/index.sqlite" in text and "newer Atlas" in text


def test_a_point_and_radius_become_the_area(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    with SceneIndex.at(store.resolve("data")) as index:
        index.upsert([IndexEntry(_scene("here", 2, 5.0))])
    payload = _find(store, bbox=None, latitude=42.36, longitude=-71.06)
    assert _ids(payload["groups"][0]) == ["here"]
    assert payload["query"]["bbox"]["west"] == pytest.approx(
        -71.06 - 2.0 / 111.32 / 0.7396, abs=1e-3
    )


@pytest.mark.parametrize(
    "arguments",
    [
        {**WEEK},
        {"bbox": AOI.model_dump(), "latitude": 42.36, "longitude": -71.06, **WEEK},
        {"bbox": AOI.model_dump()},
        {"bbox": AOI.model_dump(), **WEEK, "limit": 21},
    ],
)
def test_bad_input_is_rejected(tmp_path: Path, arguments: dict[str, Any]) -> None:
    with pytest.raises((ValidationError, ValueError)):
        FindScenesTool().run(arguments, LocalArtifactStore(tmp_path))


def test_the_system_prompt_requires_scene_citations() -> None:
    prompt = load_system_prompt()
    assert "Scene answers must cite the index_id (sc_...) of every scene they rely on." in prompt
    assert "find_scenes before searching a catalog again" in prompt
