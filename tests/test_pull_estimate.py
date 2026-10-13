"""Size estimate and confirmation before a pull downloads anything."""

from __future__ import annotations

import argparse
import asyncio
import io
import sys
from datetime import UTC, date, datetime
from pathlib import Path

import httpx

from atlas.cli import _attach_negative_option_values
from atlas.cli.data import add_data_parser, run_data
from atlas.cli.estimate import (
    CONFIRM_PROMPT,
    CONFIRMATION_REQUIRED,
    DOWNLOAD_CANCELLED,
    EXIT_CANCELLED,
    EXIT_REFUSED,
    NO_COVERAGE,
    confirm_download,
    estimate_pull,
    render_pull_estimate,
)
from atlas.compile.product import SceneCatalog, SourceScenes
from atlas.data.base import Asset, BBox, DataPullClient, PullRequest, PullResult, Scene
from atlas.data.credentials import MissingCredentialError
from atlas.data.stac import item_to_scene

AOI = BBox(west=-71.12, south=42.32, east=-71.02, north=42.40)
_WHEN = datetime(2024, 7, 1, tzinfo=UTC)


class _Stdin(io.StringIO):
    """stdin stand-in that records reads and can pretend to be a terminal."""

    def __init__(self, text: str = "", *, tty: bool) -> None:
        super().__init__(text)
        self._tty = tty
        self.reads = 0

    def isatty(self) -> bool:
        return self._tty

    def readline(self, size: int | None = -1) -> str:
        self.reads += 1
        if not self._tty:
            raise AssertionError("stdin was read")
        return super().readline(-1 if size is None else size)


class _Catalog(DataPullClient):
    def __init__(
        self,
        scenes: list[Scene] | None = None,
        error: BaseException | None = None,
    ) -> None:
        self._scenes = list(scenes or [])
        self._error = error
        self.searches = 0

    async def search(self, request: PullRequest) -> PullResult:
        self.searches += 1
        if self._error is not None:
            raise self._error
        return PullResult(request=request, scenes=self._scenes)


def _scene(
    scene_id: str,
    *,
    size: int | None = None,
    key: str = "data",
    roles: list[str] | None = None,
    href: str | None = None,
) -> Scene:
    body: dict[str, object] = {
        "href": href or f"https://example.com/{scene_id}.bin",
        "media_type": "application/octet-stream",
        "roles": roles if roles is not None else ["data"],
    }
    if size is not None:
        body["size_bytes"] = size
    return Scene(
        id=scene_id,
        source="sentinel2",
        datetime=_WHEN,
        bbox=AOI,
        platform="sentinel-2",
        assets={key: Asset.model_validate(body)},
    )


def _request() -> PullRequest:
    return PullRequest(
        start_date=date(2024, 7, 1),
        end_date=date(2024, 7, 7),
        bbox=AOI,
    )


def _catalog(*sources: SourceScenes) -> SceneCatalog:
    return SceneCatalog(request=_request(), sources=list(sources))


def _source(
    name: str,
    scenes: list[Scene],
    *,
    error: str | None = None,
) -> SourceScenes:
    return SourceScenes(source=name, collection=name, scenes=scenes, error=error)


def _feature(scene_id: str, file_size: object) -> dict[str, object]:
    asset: dict[str, object] = {
        "href": f"https://example.com/{scene_id}.tif",
        "type": "image/tiff; application=geotiff",
        "roles": ["data"],
    }
    if file_size is not None:
        asset["file:size"] = file_size
    return {
        "id": scene_id,
        "bbox": [-71.12, 42.32, -71.02, 42.40],
        "properties": {"datetime": "2024-07-01T00:00:00Z", "platform": "sentinel-2"},
        "assets": {"data": asset},
    }


def _args(tmp_path: Path, *extra: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="atlas")
    sub = parser.add_subparsers(dest="group")
    add_data_parser(sub)
    return parser.parse_args(
        _attach_negative_option_values(
            [
                "data",
                "sentinel2",
                "optical",
                "--date",
                "2024-07-01:2024-07-07",
                "--coords",
                "-71.12,42.32,-71.02,42.40",
                "--out",
                str(tmp_path),
                *extra,
            ]
        )
    )


def _clients() -> dict[str, _Catalog]:
    return {
        "sentinel2": _Catalog([_scene("sized", size=1000), _scene("unsized")]),
        "firms": _Catalog(error=MissingCredentialError("EARTHDATA_TOKEN")),
        "goes": _Catalog([]),
    }


def _run(
    tmp_path: Path,
    stdin: _Stdin,
    *extra: str,
    clients: dict[str, _Catalog] | None = None,
) -> tuple[int, str, list[str]]:
    err = io.StringIO()
    gets: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        text = err.getvalue()
        assert "Pull estimate" in text
        assert text.index("Pull estimate") < text.index(": pulling ")
        gets.append(str(request.url))
        return httpx.Response(200, content=b"tif-bytes")

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    previous_in, previous_err = sys.stdin, sys.stderr
    sys.stdin, sys.stderr = stdin, err
    try:
        code = run_data(_args(tmp_path, *extra), clients=clients or _clients(), http=http)
    finally:
        sys.stdin, sys.stderr = previous_in, previous_err
        asyncio.run(http.aclose())
    return code, err.getvalue(), gets


def test_estimate_uses_catalog_sizes_and_leaves_the_rest_unknown() -> None:
    sized = item_to_scene(_feature("sized", 1000))
    unsized = item_to_scene(_feature("unsized", "1000"))
    odd = item_to_scene(_feature("odd", 1.5))
    negative = item_to_scene(_feature("negative", -5))
    flag = item_to_scene(_feature("flag", True))
    missing = item_to_scene(_feature("missing", None))
    scenes = [sized, unsized, odd, negative, flag, missing]
    assert all(scene is not None for scene in scenes)
    assert sized is not None and sized.assets["data"].size_bytes == 1000
    for scene in scenes[1:]:
        assert scene is not None
        assert scene.assets["data"].size_bytes is None

    kept = [scene for scene in scenes if scene is not None]
    estimate = estimate_pull(_catalog(_source("sentinel2", kept)))
    assert estimate.scene_count == 6
    assert estimate.known_bytes == 1000
    assert estimate.unknown_scenes == 5
    assert estimate.bytes_text == "1000 (5 unknown)"
    assert estimate.skipped == ()

    visual = Scene(
        id="both",
        source="sentinel2",
        datetime=_WHEN,
        bbox=AOI,
        platform="sentinel-2",
        assets={
            "visual": Asset(
                href="https://example.com/visual.tif",
                media_type="image/tiff; application=geotiff",
                roles=["visual"],
                size_bytes=10,
            ),
            "B04": Asset(
                href="https://example.com/red.tif",
                media_type="image/tiff; application=geotiff",
                roles=["data"],
                size_bytes=20,
            ),
            "thumbnail": Asset(
                href="https://example.com/thumb.jpg",
                media_type="image/jpeg",
                roles=["thumbnail"],
                size_bytes=1,
            ),
        },
        band_aliases={"red": "B04"},
    )
    catalog = _catalog(_source("sentinel2", [visual]))
    assert estimate_pull(catalog).known_bytes == 10
    assert estimate_pull(catalog, asset_name="red").known_bytes == 20

    browse = _scene("preview", size=99, key="thumbnail", roles=["thumbnail"])
    browse_estimate = estimate_pull(_catalog(_source("sentinel2", [browse])))
    assert browse_estimate.bytes_text == "0"
    assert browse_estimate.unknown_scenes == 0


def test_estimate_does_not_treat_an_extract_asset_size_as_the_download() -> None:
    scene = _scene("month", size=50_000_000)
    estimate = estimate_pull(
        _catalog(_source("fldas", [scene])),
        unknown_byte_sources={"fldas"},
    )
    assert estimate.scene_count == 1
    assert estimate.known_bytes == 0
    assert estimate.bytes_text == "unknown"


def test_estimate_reports_skipped_sources_and_does_not_invent_bytes() -> None:
    sized = _scene("sized", size=1000)
    credential = "EARTHDATA_TOKEN is not set; set it with /key earthdata or the env var"
    estimate = estimate_pull(
        _catalog(
            _source("sentinel2", [sized, _scene("unsized")]),
            _source("firms", [sized], error=credential),
            _source("goes", []),
        )
    )
    assert estimate.scene_count == 2
    assert estimate.bytes_text == "1000 (1 unknown)"
    assert [(item.source, item.reason) for item in estimate.skipped] == [
        ("firms", credential),
        ("goes", NO_COVERAGE),
    ]
    text = render_pull_estimate(estimate)
    assert "scenes: 2" in text
    assert "bytes: 1000 (1 unknown)" in text
    assert f"firms: {credential}" in text
    assert f"goes: {NO_COVERAGE}" in text


def test_confirm_accepts_yes_and_refuses_a_pipe_without_reading() -> None:
    pipe = _Stdin(tty=False)
    err = io.StringIO()
    assert confirm_download(assume_yes=False, stdin=pipe, stderr=err) == EXIT_REFUSED
    assert pipe.reads == 0
    assert CONFIRMATION_REQUIRED in err.getvalue()
    assert CONFIRM_PROMPT not in err.getvalue()

    skipped = _Stdin("n\n", tty=True)
    assert confirm_download(assume_yes=True, stdin=skipped, stderr=io.StringIO()) is None
    assert skipped.reads == 0

    yes = io.StringIO()
    assert confirm_download(assume_yes=False, stdin=_Stdin(" YES \n", tty=True), stderr=yes) is None
    no = io.StringIO()
    assert (
        confirm_download(assume_yes=False, stdin=_Stdin("\n", tty=True), stderr=no)
        == EXIT_CANCELLED
    )
    assert DOWNLOAD_CANCELLED in no.getvalue()


def test_pull_prints_the_estimate_before_downloading(tmp_path: Path) -> None:
    code, err, gets = _run(tmp_path, _Stdin(tty=False), "--yes")
    assert code == 0
    assert "scenes: 2" in err
    assert "bytes: 1000 (1 unknown)" in err
    assert "EARTHDATA_TOKEN is not set" in err
    assert f"goes: {NO_COVERAGE}" in err
    assert CONFIRM_PROMPT not in err
    assert len(gets) == 2
    assert (tmp_path / "manifest.json").is_file()


def test_pull_confirm_yes_downloads_and_no_does_not(tmp_path: Path) -> None:
    agreed, yes_err, gets = _run(tmp_path / "yes", _Stdin("y\n", tty=True))
    assert agreed == 0
    assert CONFIRM_PROMPT in yes_err
    assert "scenes: 2" in yes_err
    assert "bytes: 1000 (1 unknown)" in yes_err
    assert len(gets) == 2
    assert list((tmp_path / "yes").rglob("*.bin"))

    declined, no_err, no_gets = _run(tmp_path / "no", _Stdin("n\n", tty=True))
    assert declined == EXIT_CANCELLED
    assert DOWNLOAD_CANCELLED in no_err
    assert "scenes: 2" in no_err
    assert ": pulling " not in no_err
    assert no_gets == []
    assert not (tmp_path / "no" / "manifest.json").exists()
    assert list((tmp_path / "no").rglob("*.bin")) == []


def test_yes_flag_skips_the_prompt(tmp_path: Path) -> None:
    stdin = _Stdin("n\n", tty=False)
    code, err, gets = _run(tmp_path, stdin, "--yes")
    assert code == 0
    assert stdin.reads == 0
    assert CONFIRM_PROMPT not in err
    assert "Pull estimate" in err
    assert len(gets) == 2


def test_noninteractive_pull_without_yes_refuses(tmp_path: Path) -> None:
    stdin = _Stdin(tty=False)
    code, err, gets = _run(tmp_path, stdin)
    assert code == EXIT_REFUSED
    assert stdin.reads == 0
    assert CONFIRMATION_REQUIRED in err
    assert CONFIRM_PROMPT not in err
    assert "scenes: 2" in err
    assert "bytes: 1000 (1 unknown)" in err
    assert gets == []
    assert not (tmp_path / "manifest.json").exists()
    assert list(tmp_path.rglob("*.bin")) == []


def test_search_only_does_not_prompt_or_download(tmp_path: Path) -> None:
    stdin = _Stdin(tty=False)
    code, err, gets = _run(tmp_path, stdin, "--search-only")
    assert code == 0
    assert stdin.reads == 0
    assert "Pull estimate" not in err
    assert CONFIRM_PROMPT not in err
    assert gets == []
    assert (tmp_path / "manifest.json").is_file()
