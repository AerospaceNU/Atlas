from __future__ import annotations

import asyncio
import csv
import json
import re
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import unquote

import httpx
import pytest

from atlas.cli import main
from atlas.cli.data import run_data
from atlas.data import FldasClient
from atlas.data.base import Asset, BBox, DataPullClient, PullRequest, PullResult, Scene, SceneKind
from atlas.data.registry import SOURCES, sources
from atlas.data.sources import fldas as fldas_module
from atlas.data.sources.fldas import (
    FILL_VALUE,
    MISSING_TOKEN_MESSAGE,
    FldasAuthError,
    FldasError,
    FldasRequestError,
    _axis_constraint,
    _expand_sampled_axis,
    parse_dap2,
    parse_points_csv,
)
from atlas.data.sources.fldas import _Axis as Axis

X_AXIS = (-71.15, -71.05, -70.95, -70.85)
Y_AXIS = (42.15, 42.25, 42.35, 42.45)
FL_DAS_OPENDAP = (
    "https://opendap.earthdata.nasa.gov/collections/C1563089663-GES_DISC/granules/"
    "FLDAS_NOAH01_C_GL_M.001%3AFLDAS_NOAH01_C_GL_M.A202001.001.nc"
)
FULL_FILE = (
    "https://data.gesdisc.earthdata.nasa.gov/data/FLDAS/FLDAS_NOAH01_C_GL_M.001/"
    "FLDAS_NOAH01_C_GL_M.A202001.001.nc"
)
BOSTON = BBox(west=-71.2, south=42.2, east=-70.9, north=42.5)
JANUARY = (date(2020, 1, 1), date(2020, 1, 31))


def _request(bbox: BBox = BOSTON, *, limit: int = 1) -> PullRequest:
    return PullRequest(start_date=JANUARY[0], end_date=JANUARY[1], bbox=bbox, limit=limit)


@pytest.fixture(autouse=True)
def _isolate_earthdata_env(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep offline tests off a developer's real .env so credentials never leak in."""
    if request.node.get_closest_marker("evals") is not None:
        return
    monkeypatch.delenv("EARTHDATA_TOKEN", raising=False)
    monkeypatch.delenv("EARTHDATA_USERNAME", raising=False)
    monkeypatch.delenv("EARTHDATA_PASSWORD", raising=False)
    monkeypatch.setattr(fldas_module, "load_dotenv", lambda *a, **k: False)


def _granule_item(*, granule_ur: str | None = None) -> dict[str, Any]:
    ur = granule_ur or "FLDAS_NOAH01_C_GL_M.001:FLDAS_NOAH01_C_GL_M.A202001.001.nc"
    return {
        "meta": {"concept-id": "G-FLDAS-1"},
        "umm": {
            "GranuleUR": ur,
            "TemporalExtent": {
                "RangeDateTime": {
                    "BeginningDateTime": "2020-01-02T00:00:00.000Z",
                    "EndingDateTime": "2020-01-31T23:59:59.000Z",
                }
            },
            "SpatialExtent": {
                "HorizontalSpatialDomain": {
                    "Geometry": {
                        "BoundingRectangles": [
                            {
                                "WestBoundingCoordinate": -180.0,
                                "SouthBoundingCoordinate": -60.0,
                                "EastBoundingCoordinate": 180.0,
                                "NorthBoundingCoordinate": 90.0,
                            }
                        ]
                    }
                }
            },
            "RelatedUrls": [
                {"Type": "GET DATA", "URL": FULL_FILE, "Description": "HTTPS NetCDF"},
                {
                    "Type": "GET DATA VIA DIRECT ACCESS",
                    "URL": "s3://gesdisc-cumulus-prod-protected/FLDAS/A202001.001.nc",
                },
                {
                    "Type": "USE SERVICE API",
                    "Subtype": "OPENDAP DATA",
                    "URL": FL_DAS_OPENDAP,
                    "Description": "OPeNDAP request URL",
                },
            ],
        },
    }


def _value(name: str, y: int, x: int) -> float:
    """Deterministic sample value; Evap_tavg carries one product fill cell."""
    if name == "Evap_tavg" and y == 1 and x == 0:
        return FILL_VALUE
    return float(_variable_index(name) * 100 + y * 10 + x)


def _array_text(name: str, y0: int, y1: int, x0: int, x1: int) -> str:
    rows = []
    for index in range(y0, y1 + 1):
        cells = ", ".join(repr(_value(name, index, x)) for x in range(x0, x1 + 1))
        rows.append(f"[0][{index - y0}], {cells}")
    header = f"{name}.{name}[1][{y1 - y0 + 1}][{x1 - x0 + 1}]"
    return "\n".join([header, *rows])


def _axis_text(name: str, axis: tuple[float, ...], low: int, high: int) -> str:
    values = ", ".join(repr(axis[i]) for i in range(low, high + 1))
    return f"{name}.{name}[{high - low + 1}]\n{values}"


def _slice_requests(query: str) -> list[tuple[str, list[tuple[int, int]]]]:
    parsed: list[tuple[str, list[tuple[int, int]]]] = []
    for part in query.split(","):
        name = part.split("[", 1)[0]
        pairs = [(int(lo), int(hi)) for lo, hi in re.findall(r"\[(\d+):(\d+)\]", part)]
        parsed.append((name, pairs))
    return parsed


def _dap2_text(query: str) -> str:
    blocks: list[str] = ["Dataset {\n    (no header needed for the test parser)\n} fldas;\n"]
    for name, pairs in _slice_requests(query):
        if name == "X":
            (low, high) = pairs[0]
            blocks.append(_axis_text("X", X_AXIS, low, high))
        elif name == "Y":
            (low, high) = pairs[0]
            blocks.append(_axis_text("Y", Y_AXIS, low, high))
        else:
            (y0, y1), (x0, x1) = pairs[1], pairs[2]
            blocks.append(_array_text(name, y0, y1, x0, x1))
    return "\n\n".join(blocks) + "\n"


_DEFAULT_ORDER = (
    "SoilMoi00_10cm_tavg",
    "SoilMoi10_40cm_tavg",
    "SoilMoi40_100cm_tavg",
    "SoilMoi100_200cm_tavg",
    "Evap_tavg",
    "Qs_tavg",
    "Qsb_tavg",
    "SWE_inst",
    "SnowDepth_inst",
    "RadT_tavg",
)


def _variable_index(name: str) -> int:
    return _DEFAULT_ORDER.index(name) if name in _DEFAULT_ORDER else 0


class FakeFldas:
    """MockTransport-backed transport recording every OPeNDAP query."""

    def __init__(
        self,
        *,
        status: int = 200,
        html: bool = False,
        items: list[dict[str, Any]] | None = None,
    ) -> None:
        self.status = status
        self.html = html
        self.items = [_granule_item()] if items is None else items
        self.queries: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.startswith("https://cmr.earthdata.nasa.gov/"):
            return httpx.Response(200, json={"items": self.items})
        query = unquote(url.split("?", 1)[1]) if "?" in url else ""
        self.queries.append(query)
        if self.html:
            return httpx.Response(
                200,
                text="<!DOCTYPE html><html><body>login</body></html>",
                headers={"content-type": "text/html"},
            )
        if self.status != 200:
            return httpx.Response(self.status, text="denied")
        if query in {"X,Y", _axis_constraint()}:
            return httpx.Response(
                200, text="\n".join([_axis_full("X", X_AXIS), _axis_full("Y", Y_AXIS)]) + "\n"
            )
        return httpx.Response(200, text=_dap2_text(query))

    def client(self, **kwargs: Any) -> FldasClient:
        transport = httpx.MockTransport(self.handler)
        http = httpx.AsyncClient(transport=transport)
        return FldasClient(client=http, **kwargs)


def _axis_full(name: str, axis: tuple[float, ...]) -> str:
    values = ", ".join(repr(value) for value in axis)
    return f"{name}.{name}[{len(axis)}]\n{values}"


def _scene(scene_id: str = "g1", *, day: int = 2) -> Scene:
    return Scene(
        id=scene_id,
        source="fldas",
        collection="FLDAS_NOAH01_C_GL_M",
        kind=SceneKind.landsurface,
        datetime=datetime(2020, 1, day, tzinfo=UTC),
        bbox=BBox(west=-180, south=-60, east=180, north=90),
        platform="FLDAS",
        assets={"opendap": Asset(href=FL_DAS_OPENDAP)},
    )


def _rows(result: Any) -> list[dict[str, str]]:
    header = list(result.columns)
    return [dict(zip(header, row, strict=True)) for row in result.rows]


def test_fldas_catalog_hint_and_kind_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["data", "fldas"]) == 0
    out = capsys.readouterr().out
    assert "landsurface" in out
    assert "atlas data fldas landsurface --date START:END" in out

    fake = FakeFldas()
    client = _FakeFldasSource(fake, token="token")
    _injected(monkeypatch, client)
    code = run_data(_pull_args(tmp_path, kind=None))
    assert code == 0
    assert (tmp_path / "fldas.csv").exists()
    asyncio.run(client.aclose())


def test_fldas_is_registered_under_landsurface() -> None:
    from atlas.cli.catalog import resolve_sources

    assert SOURCES["fldas"] is FldasClient
    assert resolve_sources("fldas", "landsurface") == ["fldas"]
    assert FldasClient.scene_kind is SceneKind.landsurface
    assert FldasClient.satellite == "fldas"


@pytest.mark.asyncio
async def test_search_reads_cmr_and_exposes_only_the_opendap_asset() -> None:
    fake = FakeFldas()
    client = fake.client()
    try:
        result = await client.search(_request())
    finally:
        await client.aclose()
    assert [scene.id for scene in result.scenes] == [
        "FLDAS_NOAH01_C_GL_M.001:FLDAS_NOAH01_C_GL_M.A202001.001.nc"
    ]
    scene = result.scenes[0]
    assert list(scene.assets) == ["opendap"]
    assert scene.assets["opendap"].href == FL_DAS_OPENDAP
    assert scene.datetime == datetime(2020, 1, 2, tzinfo=UTC)
    assert scene.bbox.as_list() == [-180.0, -60.0, 180.0, 90.0]
    assert scene.platform == "FLDAS"
    assert scene.instrument == "Noah"
    assert scene.collection == "FLDAS_NOAH01_C_GL_M"
    assert scene.gsd_m == pytest.approx(11132.0)
    assert scene.properties["full_file_url"] == FULL_FILE
    hrefs = [asset.href for asset in scene.assets.values()]
    assert hrefs == [FL_DAS_OPENDAP]
    assert "s3://" not in " ".join(hrefs)
    assert fake.queries == []


@pytest.mark.asyncio
async def test_search_sends_short_name_version_bbox_and_temporal() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"items": [_granule_item()]})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = FldasClient(client=http)
    try:
        await client.search(_request())
    finally:
        await client.aclose()
    url = seen[0].url
    assert str(url).startswith("https://cmr.earthdata.nasa.gov/search/granules.umm_json")
    params = dict(url.params)
    assert params["short_name"] == "FLDAS_NOAH01_C_GL_M"
    assert params["version"] == "001"
    assert params["bounding_box"] == "-71.2,42.2,-70.9,42.5"
    assert params["temporal"] == "2020-01-01T00:00:00Z,2020-01-31T23:59:59Z"
    assert seen[0].headers.get("Authorization") is None


@pytest.mark.asyncio
async def test_wrapping_bbox_raises_before_http() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"no request expected: {request.url}")

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = FldasClient(client=http)
    wrapping = BBox(west=170, south=-10, east=-170, north=10)
    with pytest.raises(FldasRequestError, match="west<=east"):
        await client.search(_request(wrapping))
    with pytest.raises(FldasRequestError, match="west<=east"):
        await client.extract_training_table(_request(wrapping), [_scene()])
    await client.aclose()


@pytest.mark.asyncio
async def test_constraint_window_is_narrow() -> None:
    fake = FakeFldas()
    client = fake.client(token="token")
    try:
        await client.extract_training_table(_request(), [_scene()], variables=["Evap_tavg"])
    finally:
        await client.aclose()
    axis_query, data_query = fake.queries
    assert axis_query == _axis_constraint()
    assert "Evap_tavg[0:0][1:3][0:2]" in data_query
    assert "X[0:2]" in data_query
    assert "Y[1:3]" in data_query
    assert "3599" not in data_query
    assert "1499" not in data_query


@pytest.mark.asyncio
async def test_extract_writes_cells_fill_values_and_empty_cells() -> None:
    fake = FakeFldas()
    client = fake.client(token="token")
    try:
        table = await client.extract_training_table(
            _request(), [_scene()], variables=["Evap_tavg", "SoilMoi00_10cm_tavg"]
        )
    finally:
        await client.aclose()
    assert table.columns == (
        "longitude",
        "latitude",
        "time",
        "Evap_tavg",
        "SoilMoi00_10cm_tavg",
    )
    rows = _rows(table)
    assert len(rows) == 9
    first = rows[0]
    assert first["longitude"] == "-71.15"
    assert first["latitude"] == "42.25"
    assert first["time"] == "2020-01-02"
    assert first["Evap_tavg"] == ""  # the fixture writes the fill value at (y=1, x=0)
    assert first["SoilMoi00_10cm_tavg"] == "10.0"
    assert rows[1]["Evap_tavg"] == "411.0"
    assert rows[1]["SoilMoi00_10cm_tavg"] == "11.0"
    assert all("cell_longitude" not in row for row in rows)


@pytest.mark.asyncio
async def test_extract_uses_one_data_request_per_scene_and_one_axis_read() -> None:
    fake = FakeFldas()
    client = fake.client(token="token")
    try:
        table = await client.extract_training_table(
            _request(), [_scene("g1", day=2), _scene("g2", day=3)], variables=["Evap_tavg"]
        )
    finally:
        await client.aclose()
    assert len(fake.queries) == 3
    assert fake.queries[0] == _axis_constraint()
    assert all(query.startswith("Evap_tavg[") for query in fake.queries[1:])
    assert table.n_rows == 18


@pytest.mark.asyncio
async def test_a_failed_month_is_skipped_and_the_rest_are_kept(tmp_path: Path) -> None:
    fake = FakeFldas()
    client = fake.client(token="token", table_path=tmp_path / "fldas.csv")

    async def flaky(scene: Scene, names: object, x_lo: int, x_hi: int, y_lo: int, y_hi: int):
        if scene.id == "g2":
            raise FldasError("Earthdata OPeNDAP returned 502")
        return await FldasClient._read_granule(client, scene, names, x_lo, x_hi, y_lo, y_hi)

    client._read_granule = flaky  # type: ignore[method-assign]
    try:
        table = await client.extract_training_table(
            _request(), [_scene("g1", day=2), _scene("g2", day=3)], variables=["Evap_tavg"]
        )
    finally:
        await client.aclose()
    assert table.n_rows == 9
    assert table.skipped and "2020-01-03" in table.skipped[0]
    lines = (tmp_path / "fldas.csv").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 10
    assert all(line.split(",")[2] == "2020-01-02" for line in lines[1:])


@pytest.mark.asyncio
async def test_points_produce_one_data_request_and_cell_columns(tmp_path: Path) -> None:
    points_file = tmp_path / "points.csv"
    points_file.write_text(
        "Lon,Lat\n-71.18,42.22\n-70.92,42.48\n-71.12,42.32\n-70.98,42.42\n-71.02,42.28\n",
        encoding="utf-8",
    )
    points = parse_points_csv(points_file)
    fake = FakeFldas()
    client = fake.client(token="token")
    try:
        table = await client.extract_training_table(
            _request(), [_scene()], points=points, variables=["Evap_tavg"]
        )
    finally:
        await client.aclose()
    # Five points still collapse into a single data request per granule.
    assert len(fake.queries) == 2
    assert fake.queries[1] == "Evap_tavg[0:0][1:3][0:2],X[0:2],Y[1:3]"
    assert table.columns == (
        "longitude",
        "latitude",
        "cell_longitude",
        "cell_latitude",
        "time",
        "Evap_tavg",
    )
    rows = _rows(table)
    assert len(rows) == 5
    assert [row["longitude"] for row in rows] == [
        "-71.18",
        "-70.92",
        "-71.12",
        "-70.98",
        "-71.02",
    ]
    assert [row["latitude"] for row in rows] == [
        "42.22",
        "42.48",
        "42.32",
        "42.42",
        "42.28",
    ]
    assert [row["cell_longitude"] for row in rows] == [
        "-71.15",
        "-70.95",
        "-71.15",
        "-70.95",
        "-71.05",
    ]
    assert [row["cell_latitude"] for row in rows] == [
        "42.25",
        "42.45",
        "42.35",
        "42.45",
        "42.25",
    ]
    assert [row["time"] for row in rows] == ["2020-01-02"] * 5


@pytest.mark.asyncio
async def test_unknown_variable_raises_before_http() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"no request expected: {request.url}")

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = FldasClient(client=http)
    with pytest.raises(FldasRequestError, match="unknown FLDAS variable"):
        await client.extract_training_table(_request(), [_scene()], variables=["NotAField"])
    await client.aclose()


@pytest.mark.asyncio
async def test_off_grid_and_outside_bbox_points_raise() -> None:
    fake = FakeFldas()
    client = fake.client(token="token")
    try:
        with pytest.raises(FldasRequestError, match="outside the request bbox"):
            await client.extract_training_table(
                _request(), [_scene()], points=[(-75.0, 42.3)], variables=["Evap_tavg"]
            )
        with pytest.raises(FldasRequestError, match="FLDAS grid"):
            await client.extract_training_table(
                _request(BBox(west=-71.2, south=-61.0, east=-70.9, north=-59.0)),
                [_scene()],
                points=[(-71.15, -60.5)],
                variables=["Evap_tavg"],
            )
        with pytest.raises(FldasRequestError, match="more than half a cell"):
            await client.extract_training_table(
                _request(BBox(west=-71.2, south=44.0, east=-70.9, north=46.0)),
                [_scene()],
                points=[(-71.15, 45.0)],
                variables=["Evap_tavg"],
            )
    finally:
        await client.aclose()
    # Only the rejected wrapping/off-grid cases short-circuit; the axis read of the
    # preceding valid request is expected.
    assert fake.queries
    assert all(query == _axis_constraint() for query in fake.queries)


@pytest.mark.asyncio
async def test_bbox_without_cell_centers_raises() -> None:
    fake = FakeFldas()
    client = fake.client(token="token")
    try:
        with pytest.raises(FldasRequestError, match="no X grid-cell center"):
            await client.extract_training_table(
                _request(BBox(west=-72.0, south=42.2, east=-71.5, north=42.5)),
                [_scene()],
                variables=["Evap_tavg"],
            )
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_auth_errors_name_the_token_variable_without_leaking() -> None:
    secret = "s3cr3t-password"
    unauthorized = FakeFldas(status=401)
    client = unauthorized.client(username="user", password=secret)
    try:
        with pytest.raises(FldasAuthError) as excinfo:
            await client.extract_training_table(_request(), [_scene()], variables=["Evap_tavg"])
    finally:
        await client.aclose()
    message = str(excinfo.value)
    assert message == MISSING_TOKEN_MESSAGE
    assert secret not in message

    html_login = FakeFldas(html=True)
    client = html_login.client(token="tok3n-secret")
    try:
        with pytest.raises(FldasAuthError) as excinfo:
            await client.extract_training_table(_request(), [_scene()], variables=["Evap_tavg"])
    finally:
        await client.aclose()
    message = str(excinfo.value)
    assert "EARTHDATA_TOKEN" in message
    assert "tok3n-secret" not in message


@pytest.mark.asyncio
async def test_extract_empty_scene_list_is_an_error() -> None:
    fake = FakeFldas()
    client = fake.client()
    try:
        with pytest.raises(Exception, match="no FLDAS granules"):
            await client.extract_training_table(_request(), [], variables=["Evap_tavg"])
    finally:
        await client.aclose()


class _FakeFldasSource(FldasClient):
    """Search returns a fixed scene; OPeNDAP calls go to the fake transport."""

    def __init__(self, fake: FakeFldas, **kwargs: Any) -> None:
        self._http = httpx.AsyncClient(transport=httpx.MockTransport(fake.handler))
        super().__init__(client=self._http, **kwargs)
        self.fake = fake

    async def aclose(self) -> None:
        await self._http.aclose()

    async def search(self, request: PullRequest) -> PullResult:
        return PullResult(request=request, scenes=[_scene()])


def _pull_args(tmp_path: Path, **overrides: Any) -> Any:
    import argparse

    values: dict[str, Any] = {
        "list_catalog": False,
        "satellite": "fldas",
        "kind": "landsurface",
        "date": "2020-01-01:2020-01-31",
        "coords": "-71.2,42.2,-70.9,42.5",
        "out": tmp_path,
        "limit": 1,
        "min_cloud_cover": None,
        "max_cloud_cover": None,
        "concurrency": 2,
        "search_only": False,
        "asset": None,
        "table": None,
        "variables": None,
        "points": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _injected(monkeypatch: pytest.MonkeyPatch, client: DataPullClient) -> None:
    monkeypatch.setattr("atlas.cli.data.sources", lambda *names: {name: client for name in names})
    token = getattr(client, "_token", None)
    if isinstance(token, str) and token.strip():
        monkeypatch.setenv("EARTHDATA_TOKEN", token)


def test_run_data_writes_the_training_table_and_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeFldas()
    client = _FakeFldasSource(fake, token="token")
    _injected(monkeypatch, client)
    assert run_data(_pull_args(tmp_path)) == 0
    table = tmp_path / "fldas.csv"
    assert table.is_file()
    lines = table.read_text(encoding="utf-8").splitlines()
    assert lines[0].split(",")[:4] == ["longitude", "latitude", "time", "SoilMoi00_10cm_tavg"]
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["sources"][0]["saved"] == ["fldas.csv"]
    assert manifest["sources"][0]["error"] is None
    assert len(fake.queries) == 2
    asyncio.run(client.aclose())


def test_run_data_table_flag_and_variables_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeFldas()
    client = _FakeFldasSource(fake, token="token")
    _injected(monkeypatch, client)
    target = tmp_path / "nested" / "custom.csv"
    args = _pull_args(tmp_path, table=target, variables="Evap_tavg")
    assert run_data(args) == 0
    with target.open(newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    assert header == ["longitude", "latitude", "time", "Evap_tavg"]
    assert (tmp_path / "fldas.csv").exists() is False
    asyncio.run(client.aclose())


def test_search_only_writes_manifest_without_opendap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeFldas()
    client = _FakeFldasSource(fake)  # no credentials: CMR search is public
    _injected(monkeypatch, client)
    assert run_data(_pull_args(tmp_path, search_only=True)) == 0
    assert fake.queries == []
    assert not (tmp_path / "fldas.csv").exists()
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["sources"][0]["n_scenes"] == 1
    assert manifest["sources"][0]["saved"] == []
    asyncio.run(client.aclose())


def test_unknown_variable_exits_2_without_http(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = FakeFldas()
    client = _FakeFldasSource(fake, token="token")
    _injected(monkeypatch, client)
    assert run_data(_pull_args(tmp_path, variables="Nope_tavg")) == 2
    assert "unknown FLDAS variable" in capsys.readouterr().err
    assert fake.queries == []
    asyncio.run(client.aclose())


def test_asset_flag_with_extract_exits_2(tmp_path: Path) -> None:
    import argparse

    args = argparse.Namespace(
        list_catalog=False,
        satellite="fldas",
        kind="landsurface",
        date="2020-01-01:2020-01-31",
        coords="-71.2,42.2,-70.9,42.5",
        out=tmp_path,
        limit=1,
        min_cloud_cover=None,
        max_cloud_cover=None,
        concurrency=2,
        search_only=False,
        asset="opendap",
        table=None,
        variables=None,
        points=None,
    )
    assert run_data(args) == 2


def test_wrapping_bbox_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "fldas",
            "landsurface",
            "--date",
            "2020-01-01:2020-01-31",
            "--coords",
            "170,-10,-170,10",
            "--search-only",
        ]
    )
    assert code == 2
    assert "west<=east" in capsys.readouterr().err


def test_multi_source_extract_flags_exit_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "data",
            "all",
            "all",
            "--date",
            "2020-01-01:2020-01-31",
            "--coords",
            "-71.2,42.2,-70.9,42.5",
            "--out",
            str(tmp_path),
            "--variables",
            "Evap_tavg",
        ]
    )
    assert code == 2
    assert "runs alone" in capsys.readouterr().err


def test_mixed_catalog_pull_still_writes_the_fldas_table(tmp_path: Path) -> None:
    fake = FakeFldas()
    fldas = _FakeFldasSource(fake, token="token")
    scene = Scene(
        id="s2-1",
        source="sentinel2",
        datetime=datetime(2020, 1, 15, tzinfo=UTC),
        bbox=BBox(west=-71.2, south=42.2, east=-70.9, north=42.5),
        platform="sentinel-2",
        assets={
            "data": Asset(
                href="https://example.com/scene.tif",
                media_type="image/tiff; application=geotiff",
                roles=["data"],
            )
        },
    )

    class _Sentinel(DataPullClient):
        satellite: ClassVar[str] = "sentinel2"
        scene_kind: ClassVar[SceneKind] = SceneKind.optical

        async def search(self, request: PullRequest) -> PullResult:
            return PullResult(request=request, scenes=[scene])

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "example.com"
        return httpx.Response(200, content=b"tif-bytes")

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    args = _pull_args(tmp_path, satellite="all", kind="all")
    try:
        code = run_data(
            args,
            clients={"fldas": fldas, "sentinel2": _Sentinel()},
            http=http,
        )
    finally:
        asyncio.run(fldas.aclose())
        asyncio.run(http.aclose())
    assert code == 0
    assert (tmp_path / "fldas.csv").exists()
    saved = list(tmp_path.glob("**/*.tif"))
    assert saved and saved[0].stat().st_size > 0


def test_cloud_ascii_rows_become_a_yx_array() -> None:
    text = """Dataset: FLDAS_NOAH01_C_GL_M.A202001.001.nc
Y, 42.05, 42.15, 42.25
X, -71.35, -71.25, -71.15
Evap_tavg.X, -71.35, -71.25, -71.15
Evap_tavg.Evap_tavg[Evap_tavg.time=13879][Evap_tavg.Y=42.05], 1.0, 2.0, 3.0
Evap_tavg.Evap_tavg[Evap_tavg.time=13879][Evap_tavg.Y=42.15], 4.0, -9999.0, 6.0
Evap_tavg.Evap_tavg[Evap_tavg.time=13879][Evap_tavg.Y=42.25], 7.0, 8.0, 9.0
"""
    arrays = parse_dap2(text, "cloud")
    assert arrays["Y"].values == (42.05, 42.15, 42.25)
    assert arrays["X"].values == (-71.35, -71.25, -71.15)
    assert arrays["Evap_tavg"].dims == (1, 3, 3)
    assert arrays["Evap_tavg"].values[4] == FILL_VALUE


def test_two_axis_centers_rebuild_the_regular_grid() -> None:
    axis = _expand_sampled_axis(Axis("X", (-179.95, 179.95)), 3599)
    assert len(axis.values) == 3600
    assert axis.values[0] == -179.95
    assert axis.values[1] == -179.85
    assert axis.values[-1] == 179.95


def test_dap2_grid_maps_do_not_replace_the_variable() -> None:
    text = """
SoilMoi00_10cm_tavg.SoilMoi00_10cm_tavg[1][2][2]
[0][0], 0.11, 0.12
[0][1], 0.21, 0.22

SoilMoi00_10cm_tavg.X[2]
-71.05, -70.95

SoilMoi00_10cm_tavg.Y[2]
42.25, 42.35

X.X[4]
-71.15, -71.05, -70.95, -70.85
"""
    arrays = parse_dap2(text, "grid")
    assert arrays["SoilMoi00_10cm_tavg"].dims == (1, 2, 2)
    assert arrays["SoilMoi00_10cm_tavg"].values[0] == 0.11
    assert arrays["SoilMoi00_10cm_tavg.X"].values == (-71.05, -70.95)
    assert arrays["X"].values[0] == -71.15


def test_all_landsurface_resolves_to_just_fldas() -> None:
    from atlas.cli.catalog import resolve_sources

    assert resolve_sources("all", "landsurface") == ["fldas"]


def test_empty_cell_window_is_usage_exit_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = FakeFldas()
    client = _FakeFldasSource(fake, token="token")
    _injected(monkeypatch, client)
    args = _pull_args(tmp_path, coords="-72.0,42.2,-71.5,42.5")
    assert run_data(args) == 2
    assert "no X grid-cell center" in capsys.readouterr().err
    assert not (tmp_path / "fldas.csv").exists()
    asyncio.run(client.aclose())


def test_missing_token_names_the_dotenv_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "data",
            "fldas",
            "landsurface",
            "--date",
            "2020-01-01:2020-01-31",
            "--coords",
            "-71.2,42.2,-70.9,42.5",
            "--out",
            str(tmp_path),
        ]
    )
    assert code == 1
    assert capsys.readouterr().err.strip() == MISSING_TOKEN_MESSAGE
    assert not (tmp_path / "fldas.csv").exists()


def test_auth_failure_returns_exit_1_not_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = FakeFldas(status=401)
    client = _FakeFldasSource(fake)
    _injected(monkeypatch, client)
    assert run_data(_pull_args(tmp_path)) == 1
    err = capsys.readouterr().err
    assert MISSING_TOKEN_MESSAGE in err
    assert not (tmp_path / "fldas.csv").exists()
    asyncio.run(client.aclose())


def test_main_searches_only_through_the_fldas_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeFldas()
    client = _FakeFldasSource(fake, token="token")
    _injected(monkeypatch, client)
    code = main(
        [
            "data",
            "fldas",
            "landsurface",
            "--search-only",
            "--date",
            "2020-01-01:2020-01-31",
            "--coords",
            "-71.2,42.2,-70.9,42.5",
            "--limit",
            "1",
            "--out",
            str(tmp_path),
        ]
    )
    assert code == 0
    assert fake.queries == []
    asyncio.run(client.aclose())


def test_dap2_parser_accepts_prefix_and_scientific_notation() -> None:
    text = """Dataset {
    Float32 SoilMoi00_10cm_tavg[time = 1][Y = 2][X = 2];
    Float64 X[X = 4];
    Float64 Y[Y = 4];
} example;
---------------------------------------------
SoilMoi00_10cm_tavg.SoilMoi00_10cm_tavg[1][2][2]
[0][0], 0.11, 0.12
[0][1], 0.21, -9999.0

SoilMoi10_40cm_tavg[1][1][2]
[0][0], 1.5e-2, -2.5E+3

X.X[4]
-71.15, -71.05, -70.95, -70.85

Y.Y[4]
42.15, 42.25, 42.35, 42.45
"""
    arrays = parse_dap2(text, "test")
    assert set(arrays) == {"SoilMoi00_10cm_tavg", "SoilMoi10_40cm_tavg", "X", "Y"}
    assert arrays["SoilMoi00_10cm_tavg"].dims == (1, 2, 2)
    assert arrays["SoilMoi00_10cm_tavg"].values == (0.11, 0.12, 0.21, FILL_VALUE)
    assert arrays["SoilMoi10_40cm_tavg"].dims == (1, 1, 2)
    assert arrays["SoilMoi10_40cm_tavg"].values == (0.015, -2500.0)
    assert arrays["X"].dims == (4,)
    assert arrays["Y"].values == Y_AXIS


def test_parse_points_csv_accepts_xy_header_case_insensitively(tmp_path: Path) -> None:
    path = tmp_path / "xy.csv"
    path.write_text("x,y\n-71.18,42.22\n", encoding="utf-8")
    assert parse_points_csv(path) == [(-71.18, 42.22)]
    with pytest.raises(FldasRequestError, match="needs a column named"):
        bad = tmp_path / "bad.csv"
        bad.write_text("a,b\n1,2\n", encoding="utf-8")
        parse_points_csv(bad)


def test_main_writes_a_points_table(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    points_file = tmp_path / "pts.csv"
    points_file.write_text("longitude,latitude\n-71.18,42.22\n-70.98,42.42\n", encoding="utf-8")
    fake = FakeFldas()
    client = _FakeFldasSource(fake, token="token")
    _injected(monkeypatch, client)
    table = tmp_path / "out" / "pts-table.csv"
    code = main(
        [
            "data",
            "fldas",
            "landsurface",
            "--date",
            "2020-01-01:2020-01-31",
            "--coords",
            "-71.2,42.2,-70.9,42.5",
            "--limit",
            "1",
            "--out",
            str(tmp_path / "out"),
            "--table",
            str(table),
            "--points",
            str(points_file),
            "--variables",
            "Evap_tavg",
        ]
    )
    assert code == 0
    lines = table.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "longitude,latitude,cell_longitude,cell_latitude,time,Evap_tavg"
    assert len(lines) == 3
    assert lines[1].startswith("-71.18,42.22,-71.15,42.25,2020-01-02,")
    assert not (tmp_path / "out" / "fldas.csv").exists()
    asyncio.run(client.aclose())


@pytest.mark.asyncio
async def test_constructor_table_path_writes_and_overwrites(tmp_path: Path) -> None:
    target = tmp_path / "deep" / "table.csv"
    target.parent.mkdir(parents=True)
    target.write_text("stale\n", encoding="utf-8")
    fake = FakeFldas()
    client = fake.client(token="token", table_path=target)
    try:
        await client.extract_training_table(_request(), [_scene()], variables=["Evap_tavg"])
        await client.extract_training_table(_request(), [_scene()], variables=["Qs_tavg"])
    finally:
        await client.aclose()
    lines = target.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "longitude,latitude,time,Qs_tavg"
    assert len(lines) == 10
    assert "stale" not in lines


@pytest.mark.asyncio
async def test_empty_points_sequence_is_a_request_error() -> None:
    fake = FakeFldas()
    client = fake.client()
    try:
        with pytest.raises(FldasRequestError, match="no points"):
            await client.extract_training_table(
                _request(), [_scene()], points=[], variables=["Evap_tavg"]
            )
    finally:
        await client.aclose()
    assert fake.queries == []


@pytest.mark.asyncio
async def test_descending_axis_direction_is_discovered() -> None:
    """A north-to-south Y axis must still map onto the same lat/lon cells."""
    x_desc = tuple(reversed(X_AXIS))
    y_desc = tuple(reversed(Y_AXIS))
    queries: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        query = unquote(str(request.url).split("?", 1)[1])
        queries.append(query)
        if query in {"X,Y", _axis_constraint()}:
            axes = "\n".join([_axis_full("X", x_desc), _axis_full("Y", y_desc)])
            return httpx.Response(200, text=f"{axes}\n")
        return httpx.Response(200, text=_dap2_text(query))

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = FldasClient(client=http, token="token")
    try:
        table = await client.extract_training_table(_request(), [_scene()], variables=["Evap_tavg"])
    finally:
        await client.aclose()
    rows = _rows(table)
    assert len(rows) == 9
    assert table.columns == ("longitude", "latitude", "time", "Evap_tavg")
    assert [row["longitude"] for row in rows[:3]] == ["-71.15", "-71.05", "-70.95"]
    assert [row["latitude"] for row in rows[::3]] == ["42.25", "42.35", "42.45"]
    # The window is still narrow: three of the four cells on each axis.
    assert queries == [
        _axis_constraint(),
        "Evap_tavg[0:0][0:2][1:3],X[1:3],Y[0:2]",
    ]


def test_registry_exposes_fldas_client() -> None:
    assert isinstance(sources("fldas")["fldas"], FldasClient)


def test_module_does_not_link_the_global_netcdf() -> None:
    scene = _scene()
    hrefs = [asset.href for asset in scene.assets.values()]
    assert hrefs == [FL_DAS_OPENDAP]
    assert FULL_FILE not in hrefs


@pytest.mark.evals
@pytest.mark.asyncio
async def test_live_fldas_cmr_search_january_2020() -> None:
    client = FldasClient()
    try:
        result = await client.search(_request())
    finally:
        await client.aclose()
    assert result.scenes, "FLDAS CMR search returned no granules for January 2020"
    scene = result.scenes[0]
    assert scene.assets["opendap"].href.startswith("https://")
    assert scene.bbox.south <= 42.2 <= scene.bbox.north
