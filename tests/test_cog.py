"""Windowed COG reads must not transfer the rest of the granule."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import struct
import zlib
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx
import numpy as np
import pytest
from numpy.typing import NDArray

import atlas.data.cog as cog
from atlas.cli.data import run_data
from atlas.cli.download import download_assets
from atlas.data.base import Asset, BBox, DataPullClient, PullRequest, PullResult, Scene, SceneKind
from atlas.data.cog import (
    CogError,
    CogLimitError,
    CogMissError,
    CogRangeError,
    CogRead,
    CogUnsupportedError,
    encode_cog,
    model_bounds,
    open_cog,
    read_cog_window,
)

# pyproj (EPSG:32619) envelope of lon/lat [-71.10, 42.30, -71.00, 42.40], 3x3 samples.
_UTM_WEST = 326898.562214
_UTM_SOUTH = 4685022.640626
_UTM_EAST = 335403.039470
_UTM_NORTH = 4696325.489341


@dataclass
class _Field:
    tag: int
    type_code: int
    count: int
    data: bytes
    inline: bool
    offset: int = 0


@dataclass(frozen=True)
class BuiltCog:
    blob: bytes
    tile_spans: list[tuple[int, int]]
    width: int
    height: int
    origin_x: float
    origin_y: float
    pixel_size_x: float
    pixel_size_y: float


def _pack_values(endian: str, type_code: int, values: list[int] | list[float]) -> bytes:
    if type_code == 3:
        return struct.pack(endian + f"{len(values)}H", *values)
    if type_code == 4:
        return struct.pack(endian + f"{len(values)}I", *[int(value) for value in values])
    if type_code == 12:
        return struct.pack(endian + f"{len(values)}d", *[float(value) for value in values])
    raise AssertionError(type_code)


def _field(tag: int, type_code: int, values: list[int] | list[float], endian: str) -> _Field:
    data = _pack_values(endian, type_code, values)
    return _Field(tag, type_code, len(values), data, inline=len(data) <= 4)


def _entry(field: _Field, endian: str) -> bytes:
    if field.inline:
        value = field.data + bytes(4 - len(field.data))
    else:
        value = struct.pack(endian + "I", field.offset)
    return struct.pack(endian + "HHI", field.tag, field.type_code, field.count) + value


def build_cog(
    pixels: NDArray[np.uint8] | NDArray[np.uint16],
    *,
    tile: int,
    origin_x: float,
    origin_y: float,
    pixel_size_x: float,
    pixel_size_y: float,
    epsg: int | None = 4326,
    compression: str = "none",
    predictor: bool = False,
    endian: str = "<",
    trailer: int = 0,
    pad: int = 0,
    declared_tile_bytes: int | None = None,
    dimensions: tuple[int, int] | None = None,
    tile_blobs: list[bytes] | None = None,
    place_tiles_at: int | None = None,
    nodata: str | None = None,
    tile_gap: int = 0,
) -> BuiltCog:
    """Build a classic tiled GeoTIFF the production reader does not share code with."""
    if dimensions is None:
        bands, height, width = (int(pixels.shape[0]), int(pixels.shape[1]), int(pixels.shape[2]))
    else:
        height, width = dimensions
        bands = int(pixels.shape[0])
    dtype = np.dtype(f"{endian}u{pixels.dtype.itemsize}")
    tiles_x = math.ceil(width / tile)
    tiles_y = math.ceil(height / tile)
    if compression == "deflate":
        compression_code = 8
    elif compression == "lzw":
        compression_code = 5
    else:
        compression_code = 1
    if tile_blobs is not None:
        stored = list(tile_blobs)
    elif dimensions is not None:
        stored = [b"\x00"] * (tiles_x * tiles_y)
    else:
        source = np.asarray(pixels, dtype=dtype)
        stored = []
        for ty in range(tiles_y):
            for tx in range(tiles_x):
                block = np.zeros((tile, tile, bands), dtype=dtype)
                row0, col0 = ty * tile, tx * tile
                row1, col1 = min(row0 + tile, height), min(col0 + tile, width)
                block[: row1 - row0, : col1 - col0, :] = np.moveaxis(
                    source[:, row0:row1, col0:col1], 0, -1
                )
                if predictor:
                    for col in range(block.shape[1] - 1, 0, -1):
                        block[:, col, :] -= block[:, col - 1, :]
                raw = np.ascontiguousarray(block).tobytes()
                stored.append(zlib.compress(raw) if compression == "deflate" else raw)

    ntiles = len(stored)
    key = 2048 if epsg in {4326, 4269, 4258, 4267} else 3072
    fields = [
        _field(256, 4, [width], endian),
        _field(257, 4, [height], endian),
        _field(258, 3, [pixels.dtype.itemsize * 8] * bands, endian),
        _field(259, 3, [compression_code], endian),
        _field(262, 3, [2 if bands == 3 else 1], endian),
        _field(277, 3, [bands], endian),
        _field(284, 3, [1], endian),
        _field(317, 3, [2 if predictor else 1], endian),
        _field(322, 4, [tile], endian),
        _field(323, 4, [tile], endian),
        _field(324, 4, [0] * ntiles, endian),
        _field(
            325,
            4,
            [
                declared_tile_bytes if declared_tile_bytes is not None else len(blob)
                for blob in stored
            ],
            endian,
        ),
        _field(339, 3, [1], endian),
        _field(33550, 12, [pixel_size_x, pixel_size_y, 0.0], endian),
        _field(33922, 12, [0.0, 0.0, 0.0, origin_x, origin_y, 0.0], endian),
    ]
    if epsg is not None:
        fields.append(_field(34735, 3, [1, 1, 0, 1, key, 0, 1, epsg], endian))
    if nodata is not None:
        raw_nodata = nodata.encode("ascii") + b"\x00"
        fields.append(_Field(42113, 2, len(raw_nodata), raw_nodata, inline=len(raw_nodata) <= 4))
    fields.sort(key=lambda item: item.tag)
    ifd_at = 8 + pad
    ifd_size = 2 + len(fields) * 12 + 4
    cursor = ifd_at + ifd_size
    for field in fields:
        if field.inline:
            continue
        field.offset = cursor
        cursor += len(field.data)
    spans: list[tuple[int, int]] = []
    offsets: list[int] = []
    if place_tiles_at is None:
        for index, blob in enumerate(stored):
            offsets.append(cursor)
            spans.append((cursor, cursor + len(blob) - 1))
            cursor += len(blob)
            if tile_gap and index + 1 < len(stored):
                cursor += tile_gap
    else:
        tile_cursor = place_tiles_at
        for blob in stored:
            length = declared_tile_bytes if declared_tile_bytes is not None else max(len(blob), 1)
            offsets.append(tile_cursor)
            spans.append((tile_cursor, tile_cursor + length - 1))
            tile_cursor += length
    offset_field = next(field for field in fields if field.tag == 324)
    offset_field.data = _pack_values(endian, 4, offsets)
    count_field = next(field for field in fields if field.tag == 325)
    count_field.data = _pack_values(
        endian,
        4,
        [declared_tile_bytes if declared_tile_bytes is not None else len(blob) for blob in stored],
    )

    order = 0x4949 if endian == "<" else 0x4D4D
    out = bytearray(struct.pack(endian + "HHI", order, 42, ifd_at))
    out += bytes(pad)
    out += struct.pack(endian + "H", len(fields))
    for field in fields:
        out += _entry(field, endian)
    out += struct.pack(endian + "I", 0)
    for field in fields:
        if not field.inline:
            out += field.data
    if place_tiles_at is None:
        for index, blob in enumerate(stored):
            out += blob
            if tile_gap and index + 1 < len(stored):
                out += bytes(tile_gap)
    out += bytes(trailer)
    return BuiltCog(
        blob=bytes(out),
        tile_spans=spans,
        width=width,
        height=height,
        origin_x=origin_x,
        origin_y=origin_y,
        pixel_size_x=pixel_size_x,
        pixel_size_y=pixel_size_y,
    )


def _grid(
    height: int, width: int, *, bands: int = 1, dtype: type[np.generic] = np.uint8
) -> NDArray[np.generic]:
    pixels = np.zeros((bands, height, width), dtype=dtype)
    for band in range(bands):
        for row in range(height):
            for col in range(width):
                pixels[band, row, col] = band * 40 + row * 10 + col
    return pixels


def _geographic() -> BuiltCog:
    return build_cog(
        _grid(8, 8),
        tile=2,
        origin_x=-10.0,
        origin_y=10.0,
        pixel_size_x=0.25,
        pixel_size_y=0.25,
        trailer=100_000,
        pad=32,
    )


class _CountingStream(httpx.AsyncByteStream):
    """Yields nothing until iterated, so a closed 200 is observable."""

    def __init__(self, log: _RangeLog, payload: bytes) -> None:
        self._log = log
        self._payload = payload

    async def __aiter__(self) -> AsyncIterator[bytes]:
        self._log.body_reads += 1
        yield self._payload

    async def aclose(self) -> None:
        return None


class _RangeLog:
    def __init__(
        self,
        blob: bytes,
        *,
        honor_range: bool = True,
        allow_full: bool = False,
        fail_first: int = 0,
        retry_after: str | None = None,
        short_reads: int = 0,
    ) -> None:
        self.blob = blob
        self.honor_range = honor_range
        self.allow_full = allow_full
        self.fail_first = fail_first
        self.retry_after = retry_after
        self.short_reads = short_reads
        self.failures = 0
        self.calls: list[tuple[int, int] | None] = []
        self.body_reads = 0

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        header = request.headers.get("range")
        if header is None:
            self.calls.append(None)
            if not self.allow_full:
                return httpx.Response(500, content=b"full granule GET")
            return httpx.Response(200, content=self.blob)
        start_s, end_s = header.removeprefix("bytes=").split("-", 1)
        start, end = int(start_s), int(end_s)
        self.calls.append((start, end))
        if self.fail_first > 0:
            self.fail_first -= 1
            self.failures += 1
            headers = {} if self.retry_after is None else {"Retry-After": self.retry_after}
            return httpx.Response(503, headers=headers, content=b"busy")
        if not self.honor_range:
            return httpx.Response(200, stream=_CountingStream(self, self.blob))
        if start < 0 or end < start or end >= len(self.blob):
            return httpx.Response(416)
        body = self.blob[start : end + 1]
        if self.short_reads > 0:
            self.short_reads -= 1
            body = body[:-1]
        return httpx.Response(
            206,
            content=body,
            headers={"Content-Range": f"bytes {start}-{end}/{len(self.blob)}"},
        )

    def range_calls(self) -> list[tuple[int, int]]:
        return [call for call in self.calls if call is not None]


def _tiles_touched(calls: list[tuple[int, int]], spans: list[tuple[int, int]]) -> set[int]:
    hit: set[int] = set()
    for start, end in calls:
        for index, (span_start, span_end) in enumerate(spans):
            if start <= span_end and end >= span_start:
                hit.add(index)
    return hit


class _Pull(DataPullClient):
    satellite = "sentinel2"
    scene_kind = SceneKind.optical

    def __init__(self, scenes: list[Scene]) -> None:
        self._scenes = scenes

    async def search(self, request: PullRequest) -> PullResult:
        return PullResult(request=request, scenes=self._scenes)

    async def aclose(self) -> None:
        return None


def _scene(href: str) -> Scene:
    return Scene(
        id="chip",
        source="sentinel2",
        datetime=datetime(2024, 7, 1, tzinfo=UTC),
        bbox=BBox(west=-10.0, south=8.0, east=-8.0, north=10.0),
        platform="sentinel-2",
        assets={
            "visual": Asset(
                href=href,
                media_type="image/tiff; application=geotiff",
                roles=["visual"],
            )
        },
    )


@pytest.mark.asyncio
async def test_read_window_fetches_only_the_overlapping_tile() -> None:
    built = _geographic()
    log = _RangeLog(built.blob)
    async with httpx.AsyncClient(transport=log.transport()) as http:
        read = await read_cog_window(
            http,
            "https://example.com/granule.tif",
            west=-9.5,
            south=8.5,
            east=-9.0,
            north=9.0,
        )
    # cols 2:4, rows 4:6 is tile (1, 2) of a 4 by 4 tile grid: index 9.
    assert _tiles_touched(log.range_calls(), built.tile_spans) == {9}
    assert read.bytes_fetched == sum(end - start + 1 for start, end in log.range_calls())
    assert read.bytes_fetched < len(built.blob) // 2
    assert read.col_off == 2
    assert read.row_off == 4
    expected = np.array([[[42, 43], [52, 53]]], dtype=np.uint8)
    assert np.array_equal(read.array, expected)
    assert read.origin_x == pytest.approx(-9.5)
    assert read.origin_y == pytest.approx(9.0)


@pytest.mark.asyncio
async def test_window_is_clipped_to_the_raster_edge() -> None:
    built = _geographic()
    log = _RangeLog(built.blob)
    async with httpx.AsyncClient(transport=log.transport()) as http:
        read = await read_cog_window(
            http,
            "https://example.com/granule.tif",
            west=-10.5,
            south=9.5,
            east=-9.5,
            north=10.5,
        )
    assert _tiles_touched(log.range_calls(), built.tile_spans) == {0}
    assert np.array_equal(read.array, np.array([[[0, 1], [10, 11]]], dtype=np.uint8))
    assert read.col_off == 0
    assert read.row_off == 0


@pytest.mark.asyncio
async def test_window_outside_the_raster_does_not_fetch_tiles() -> None:
    built = _geographic()
    log = _RangeLog(built.blob)
    async with httpx.AsyncClient(transport=log.transport()) as http:
        with pytest.raises(CogMissError, match="does not overlap"):
            await read_cog_window(
                http,
                "https://example.com/granule.tif",
                west=0.0,
                south=0.0,
                east=1.0,
                north=1.0,
            )
    assert _tiles_touched(log.range_calls(), built.tile_spans) == set()


@pytest.mark.asyncio
async def test_deflate_and_horizontal_predictor_round_trip() -> None:
    pixels = _grid(8, 8)
    built = build_cog(
        pixels,
        tile=2,
        origin_x=-10.0,
        origin_y=10.0,
        pixel_size_x=0.25,
        pixel_size_y=0.25,
        compression="deflate",
        predictor=True,
        trailer=50_000,
    )
    log = _RangeLog(built.blob)
    async with httpx.AsyncClient(transport=log.transport()) as http:
        read = await read_cog_window(
            http,
            "https://example.com/granule.tif",
            west=-9.5,
            south=8.5,
            east=-9.0,
            north=9.0,
        )
    assert _tiles_touched(log.range_calls(), built.tile_spans) == {9}
    assert np.array_equal(read.array, pixels[:, 4:6, 2:4])
    assert read.bytes_fetched < len(built.blob) // 2


@pytest.mark.asyncio
async def test_big_endian_uint16_window() -> None:
    pixels = _grid(4, 4, dtype=np.uint16)
    built = build_cog(
        pixels,
        tile=2,
        origin_x=0.0,
        origin_y=4.0,
        pixel_size_x=1.0,
        pixel_size_y=1.0,
        endian=">",
        trailer=10_000,
    )
    log = _RangeLog(built.blob)
    async with httpx.AsyncClient(transport=log.transport()) as http:
        read = await read_cog_window(
            http,
            "https://example.com/granule.tif",
            west=2.0,
            south=0.0,
            east=4.0,
            north=2.0,
        )
    assert np.array_equal(read.array, pixels[:, 2:4, 2:4])
    assert read.bytes_fetched < len(built.blob)


@pytest.mark.asyncio
async def test_multiband_window() -> None:
    pixels = _grid(4, 4, bands=3)
    built = build_cog(
        pixels,
        tile=2,
        origin_x=0.0,
        origin_y=4.0,
        pixel_size_x=1.0,
        pixel_size_y=1.0,
    )
    log = _RangeLog(built.blob)
    async with httpx.AsyncClient(transport=log.transport()) as http:
        read = await read_cog_window(
            http,
            "https://example.com/granule.tif",
            west=0.0,
            south=2.0,
            east=2.0,
            north=4.0,
        )
    assert np.array_equal(read.array, pixels[:, 0:2, 0:2])
    assert _tiles_touched(log.range_calls(), built.tile_spans) == {0}


@pytest.mark.asyncio
async def test_server_that_ignores_range_is_rejected() -> None:
    built = _geographic()
    log = _RangeLog(built.blob, honor_range=False)
    async with httpx.AsyncClient(transport=log.transport()) as http:
        with pytest.raises(CogRangeError, match="whole object"):
            await read_cog_window(
                http,
                "https://example.com/granule.tif",
                west=-9.5,
                south=8.5,
                east=-9.0,
                north=9.0,
            )
    assert log.body_reads == 0
    assert len(log.calls) == 1


@pytest.mark.asyncio
async def test_untiled_tiff_is_not_windowed() -> None:
    # Classic header plus one ImageWidth tag and no tile index.
    body = struct.pack("<HHI", 0x4949, 42, 8)
    body += struct.pack("<H", 1)
    body += struct.pack("<HHI", 256, 4, 1) + struct.pack("<I", 4)
    body += struct.pack("<I", 0)
    log = _RangeLog(body)
    async with httpx.AsyncClient(transport=log.transport()) as http:
        with pytest.raises(CogUnsupportedError, match="not tiled"):
            await read_cog_window(
                http,
                "https://example.com/plain.tif",
                west=0.0,
                south=0.0,
                east=1.0,
                north=1.0,
            )


@pytest.mark.asyncio
async def test_encoded_window_round_trips() -> None:
    built = _geographic()
    async with httpx.AsyncClient(transport=_RangeLog(built.blob).transport()) as http:
        read = await read_cog_window(
            http,
            "https://example.com/granule.tif",
            west=-9.5,
            south=8.5,
            east=-9.0,
            north=9.0,
        )
    encoded = encode_cog(read)
    assert len(encoded) < len(built.blob)
    async with httpx.AsyncClient(transport=_RangeLog(encoded).transport()) as http:
        again = await read_cog_window(
            http,
            "https://example.com/window.tif",
            west=read.origin_x,
            south=read.origin_y - read.pixel_size_y * read.array.shape[1],
            east=read.origin_x + read.pixel_size_x * read.array.shape[2],
            north=read.origin_y,
        )
    assert np.array_equal(again.array, read.array)
    assert again.epsg == 4326


def test_model_bounds_matches_published_utm_and_mercator() -> None:
    utm = model_bounds(32619, west=-71.10, south=42.30, east=-71.00, north=42.40)
    assert utm is not None
    for got, want in zip(utm, (_UTM_WEST, _UTM_SOUTH, _UTM_EAST, _UTM_NORTH), strict=True):
        assert abs(got - want) < 0.01
    mercator = model_bounds(3857, west=-71.0, south=42.0, east=-71.0, north=42.0)
    assert mercator is not None
    assert abs(mercator[0] - -7903683.846322423) < 0.01
    assert abs(mercator[1] - 5160979.444049781) < 0.01
    southern = model_bounds(32734, west=18.5, south=-33.9, east=18.5, north=-33.9)
    assert southern is not None
    assert abs(southern[0] - 268832.299584) < 0.01
    assert abs(southern[1] - 6246117.788051) < 0.01


def test_model_bounds_refuses_antimeridian_unknown_and_polar() -> None:
    assert model_bounds(4326, west=170.0, south=-10.0, east=-170.0, north=10.0) is None
    assert model_bounds(26919, west=-71.0, south=42.0, east=-70.0, north=43.0) is None
    assert model_bounds(32619, west=-71.0, south=85.0, east=-70.0, north=86.0) is None
    with pytest.raises(CogMissError):
        model_bounds(4326, west=0.0, south=2.0, east=1.0, north=1.0)


@pytest.mark.asyncio
async def test_download_reads_a_cog_window_instead_of_the_whole_granule(tmp_path: Path) -> None:
    built = _geographic()
    log = _RangeLog(built.blob)
    scene = _scene("https://example.com/granule.tif")
    async with httpx.AsyncClient(transport=log.transport()) as http:
        report = await download_assets(
            [scene],
            tmp_path,
            http=http,
            clients={},
            bbox=BBox(west=-9.5, south=8.5, east=-9.0, north=9.0),
        )
    assert report.outcomes[0].status == "saved"
    assert None not in log.calls
    saved = report.saved_paths()["sentinel2"][0].read_bytes()
    assert len(saved) < len(built.blob)
    assert _tiles_touched(log.range_calls(), built.tile_spans) == {9}
    check = _RangeLog(saved)
    async with httpx.AsyncClient(transport=check.transport()) as http:
        read = await read_cog_window(
            http,
            "https://example.com/saved.tif",
            west=-9.5,
            south=8.5,
            east=-9.0,
            north=9.0,
        )
    assert np.array_equal(read.array, np.array([[[42, 43], [52, 53]]], dtype=np.uint8))


@pytest.mark.asyncio
async def test_download_windows_a_utm_cog(tmp_path: Path) -> None:
    pixels = _grid(80, 80, dtype=np.uint16)
    built = build_cog(
        pixels,
        tile=16,
        origin_x=300_000.0,
        origin_y=4_720_000.0,
        pixel_size_x=500.0,
        pixel_size_y=500.0,
        epsg=32619,
        trailer=200_000,
    )
    log = _RangeLog(built.blob)
    scene = _scene("https://example.com/utm.tif")
    async with httpx.AsyncClient(transport=log.transport()) as http:
        report = await download_assets(
            [scene],
            tmp_path,
            http=http,
            clients={},
            bbox=BBox(west=-71.10, south=42.30, east=-71.00, north=42.40),
        )
    assert report.outcomes[0].status == "saved"
    assert None not in log.calls
    assert len(report.saved_paths()["sentinel2"][0].read_bytes()) < len(built.blob)
    touched = _tiles_touched(log.range_calls(), built.tile_spans)
    assert touched
    assert len(touched) < len(built.tile_spans)
    saved = report.saved_paths()["sentinel2"][0].read_bytes()
    check = _RangeLog(saved)
    async with httpx.AsyncClient(transport=check.transport()) as http:
        read = await read_cog_window(
            http,
            "https://example.com/saved.tif",
            west=_UTM_WEST,
            south=_UTM_SOUTH,
            east=_UTM_EAST,
            north=_UTM_NORTH,
        )
    assert np.array_equal(read.array, pixels[:, 47:70, 53:71])


@pytest.mark.asyncio
async def test_download_of_unknown_crs_fetches_the_whole_asset(tmp_path: Path) -> None:
    built = build_cog(
        _grid(4, 4),
        tile=2,
        origin_x=300_000.0,
        origin_y=4_700_000.0,
        pixel_size_x=10.0,
        pixel_size_y=10.0,
        epsg=26919,
    )
    log = _RangeLog(built.blob, allow_full=True)
    async with httpx.AsyncClient(transport=log.transport()) as http:
        report = await download_assets(
            [_scene("https://example.com/nad83.tif")],
            tmp_path,
            http=http,
            clients={},
            bbox=BBox(west=-71.1, south=42.3, east=-71.0, north=42.4),
        )
    assert report.outcomes[0].status == "saved"
    assert None in log.calls
    assert report.saved_paths()["sentinel2"][0].read_bytes() == built.blob


@pytest.mark.asyncio
async def test_empty_window_does_not_reuse_an_older_chip(tmp_path: Path) -> None:
    built = _geographic()
    log = _RangeLog(built.blob)
    dest = tmp_path / "sentinel2" / "chip_visual.tif"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"yesterday")
    async with httpx.AsyncClient(transport=log.transport()) as http:
        report = await download_assets(
            [_scene("https://example.com/granule.tif")],
            tmp_path,
            http=http,
            clients={},
            bbox=BBox(west=1.0, south=1.0, east=2.0, north=2.0),
        )
    assert report.outcomes[0].status == "failed"
    assert report.outcomes[0].path is None
    assert "does not overlap" in (report.outcomes[0].error or "")
    assert report.saved_paths() == {}
    assert not dest.exists()


@pytest.mark.asyncio
async def test_window_with_pixels_replaces_an_older_chip(tmp_path: Path) -> None:
    built = _geographic()
    log = _RangeLog(built.blob)
    dest = tmp_path / "sentinel2" / "chip_visual.tif"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"yesterday")
    async with httpx.AsyncClient(transport=log.transport()) as http:
        report = await download_assets(
            [_scene("https://example.com/granule.tif")],
            tmp_path,
            http=http,
            clients={},
            bbox=BBox(west=-9.5, south=8.5, east=-9.0, north=9.0),
        )
    assert report.outcomes[0].status == "saved"
    saved = dest.read_bytes()
    assert saved != b"yesterday"
    assert len(saved) < len(built.blob)
    assert report.saved_paths()["sentinel2"] == [dest]


@pytest.mark.asyncio
async def test_ignored_range_does_not_keep_an_older_chip(tmp_path: Path) -> None:
    built = _geographic()
    log = _RangeLog(built.blob, honor_range=False, allow_full=True)
    dest = tmp_path / "sentinel2" / "chip_visual.tif"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"yesterday")
    async with httpx.AsyncClient(transport=log.transport()) as http:
        report = await download_assets(
            [_scene("https://example.com/granule.tif")],
            tmp_path,
            http=http,
            clients={},
            bbox=BBox(west=-9.5, south=8.5, east=-9.0, north=9.0),
        )
    assert report.outcomes[0].status == "failed"
    assert "whole object" in (report.outcomes[0].error or "")
    assert not dest.exists()
    assert log.body_reads == 0


@pytest.mark.asyncio
async def test_download_miss_does_not_fetch_the_granule(tmp_path: Path) -> None:
    built = _geographic()
    log = _RangeLog(built.blob)
    async with httpx.AsyncClient(transport=log.transport()) as http:
        report = await download_assets(
            [_scene("https://example.com/granule.tif")],
            tmp_path,
            http=http,
            clients={},
            bbox=BBox(west=1.0, south=1.0, east=2.0, north=2.0),
        )
    assert report.outcomes[0].status == "failed"
    assert "does not overlap" in (report.outcomes[0].error or "")
    assert None not in log.calls
    assert _tiles_touched(log.range_calls(), built.tile_spans) == set()
    assert list(tmp_path.rglob("*.tif")) == []


@pytest.mark.asyncio
async def test_range_ignored_is_not_buffered_or_downloaded_again(tmp_path: Path) -> None:
    built = _geographic()
    log = _RangeLog(built.blob, honor_range=False, allow_full=True)
    async with httpx.AsyncClient(transport=log.transport()) as http:
        report = await download_assets(
            [_scene("https://example.com/granule.tif")],
            tmp_path,
            http=http,
            clients={},
            bbox=BBox(west=-9.5, south=8.5, east=-9.0, north=9.0),
        )
    assert report.outcomes[0].status == "failed"
    assert "whole object" in (report.outcomes[0].error or "")
    assert None not in log.calls
    assert log.body_reads == 0
    assert list(tmp_path.rglob("*.tif")) == []


@pytest.mark.asyncio
async def test_range_above_the_byte_limit_is_not_requested() -> None:
    built = build_cog(
        _grid(4, 4),
        tile=2,
        origin_x=0.0,
        origin_y=4.0,
        pixel_size_x=1.0,
        pixel_size_y=1.0,
        declared_tile_bytes=8 * 1024 * 1024 + 1,
    )
    log = _RangeLog(built.blob)
    async with httpx.AsyncClient(transport=log.transport()) as http:
        with pytest.raises(CogRangeError, match="byte limit"):
            await read_cog_window(
                http,
                "https://example.com/granule.tif",
                west=0.0,
                south=2.0,
                east=2.0,
                north=4.0,
            )
    assert all(end - start + 1 <= 8 * 1024 * 1024 for start, end in log.range_calls())


def _short_ifd(entries: list[tuple[int, int, int]]) -> bytes:
    """Classic little-endian IFD of inline integer fields. ``entries`` are tag, type, value."""
    body = struct.pack("<H", len(entries))
    for tag, type_code, value in sorted(entries):
        body += struct.pack("<HHI", tag, type_code, 1)
        if type_code == 3:
            body += struct.pack("<H", value) + b"\x00\x00"
        else:
            body += struct.pack("<I", value)
    body += struct.pack("<I", 0)
    return struct.pack("<HHI", 0x4949, 42, 8) + body


@pytest.mark.asyncio
async def test_ifd_entry_cap_stops_before_the_directory() -> None:
    blob = struct.pack("<HHIH", 0x4949, 42, 8, 200)
    log = _RangeLog(blob)
    async with httpx.AsyncClient(transport=log.transport()) as http:
        with pytest.raises(CogLimitError, match="200 entries"):
            await read_cog_window(
                http,
                "https://example.com/granule.tif",
                west=0.0,
                south=0.0,
                east=1.0,
                north=1.0,
            )
    assert log.range_calls() == [(0, 7), (8, 9)]


@pytest.mark.asyncio
async def test_tile_count_cap_stops_before_the_offset_table() -> None:
    blob = _short_ifd(
        [
            (256, 4, 100_000),
            (257, 4, 100_000),
            (258, 3, 8),
            (322, 4, 1),
            (323, 4, 1),
        ]
    )
    log = _RangeLog(blob)
    async with httpx.AsyncClient(transport=log.transport()) as http:
        with pytest.raises(CogLimitError, match="too many tiles"):
            await read_cog_window(
                http,
                "https://example.com/granule.tif",
                west=0.0,
                south=0.0,
                east=1.0,
                north=1.0,
            )
    assert all(end < len(blob) for _, end in log.range_calls())


@pytest.mark.asyncio
async def test_window_pixel_cap_stops_before_tile_payloads() -> None:
    built = build_cog(
        np.zeros((1, 1, 1), dtype=np.uint8),
        tile=1000,
        origin_x=0.0,
        origin_y=5000.0,
        pixel_size_x=1.0,
        pixel_size_y=1.0,
        dimensions=(5000, 5000),
        place_tiles_at=1_000_000,
    )
    log = _RangeLog(built.blob)
    async with httpx.AsyncClient(transport=log.transport()) as http:
        opened = await open_cog(http, "https://example.com/granule.tif")
        assert opened.epsg == 4326
        with pytest.raises(CogLimitError, match="too many pixels"):
            await opened.read_window(west=0.0, south=0.0, east=5000.0, north=5000.0)
    assert all(start < 1_000_000 for start, _ in log.range_calls())


@pytest.mark.asyncio
async def test_deflate_bomb_is_rejected(tmp_path: Path) -> None:
    bomb = zlib.compress(b"\x00" * 200_000)
    built = build_cog(
        np.zeros((1, 2, 2), dtype=np.uint8),
        tile=2,
        origin_x=0.0,
        origin_y=2.0,
        pixel_size_x=1.0,
        pixel_size_y=1.0,
        compression="deflate",
        tile_blobs=[bomb],
    )
    log = _RangeLog(built.blob, allow_full=True)
    async with httpx.AsyncClient(transport=log.transport()) as http:
        with pytest.raises(CogLimitError, match="exceeded its dimensions"):
            await read_cog_window(
                http,
                "https://example.com/granule.tif",
                west=0.0,
                south=0.0,
                east=2.0,
                north=2.0,
            )
        report = await download_assets(
            [_scene("https://example.com/granule.tif")],
            tmp_path,
            http=http,
            clients={},
            bbox=BBox(west=0.0, south=0.0, east=2.0, north=2.0),
        )
    assert report.outcomes[0].status == "failed"
    assert "exceeded its dimensions" in (report.outcomes[0].error or "")
    assert None not in log.calls
    assert list(tmp_path.rglob("*.tif")) == []


def test_run_data_passes_the_aoi_into_the_cog_read(tmp_path: Path) -> None:
    built = _geographic()
    log = _RangeLog(built.blob)
    scene = _scene("https://example.com/granule.tif")
    args = argparse.Namespace(
        list_catalog=False,
        satellite="sentinel2",
        kind="optical",
        date="2024-07-01:2024-07-07",
        coords="-9.5,8.5,-9.0,9.0",
        out=tmp_path,
        limit=10,
        min_cloud_cover=None,
        max_cloud_cover=None,
        concurrency=2,
        search_only=False,
        asset=None,
        table=None,
        variables=None,
        points=None,
    )
    http = httpx.AsyncClient(transport=log.transport())
    try:
        assert run_data(args, clients={"sentinel2": _Pull([scene])}, http=http) == 0
    finally:
        asyncio.run(http.aclose())
    assert None not in log.calls
    saved = next(tmp_path.rglob("*.tif")).read_bytes()
    assert len(saved) < len(built.blob)
    assert _tiles_touched(log.range_calls(), built.tile_spans) == {9}


def test_manifest_records_bytes_read_and_granule_size(tmp_path: Path) -> None:
    built = _geographic()
    log = _RangeLog(built.blob)
    scene = _scene("https://example.com/granule.tif")
    args = argparse.Namespace(
        list_catalog=False,
        satellite="sentinel2",
        kind="optical",
        date="2024-07-01:2024-07-07",
        coords="-9.5,8.5,-9.0,9.0",
        out=tmp_path,
        limit=10,
        min_cloud_cover=None,
        max_cloud_cover=None,
        concurrency=2,
        search_only=False,
        asset=None,
        table=None,
        variables=None,
        points=None,
    )
    http = httpx.AsyncClient(transport=log.transport())
    try:
        assert run_data(args, clients={"sentinel2": _Pull([scene])}, http=http) == 0
    finally:
        asyncio.run(http.aclose())
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    source = manifest["sources"][0]
    assert source["saved"] == ["sentinel2/chip_visual.tif"]
    assert source["windows"] == [
        {
            "id": "chip",
            "bytes_read": sum(end - start + 1 for start, end in log.range_calls()),
            "granule_size": len(built.blob),
        }
    ]
    assert source["windows"][0]["bytes_read"] < source["windows"][0]["granule_size"]


@pytest.mark.asyncio
async def test_adjacent_tile_ranges_are_fetched_together() -> None:
    pixels = _grid(8, 8)
    built = build_cog(
        pixels,
        tile=2,
        origin_x=-10.0,
        origin_y=10.0,
        pixel_size_x=0.25,
        pixel_size_y=0.25,
    )
    log = _RangeLog(built.blob)
    async with httpx.AsyncClient(transport=log.transport()) as http:
        read = await read_cog_window(
            http,
            "https://example.com/granule.tif",
            west=-10.0,
            south=9.5,
            east=-9.0,
            north=10.0,
        )
    tile_calls = [call for call in log.range_calls() if _tiles_touched([call], built.tile_spans)]
    assert tile_calls
    assert len(tile_calls) == 1
    assert _tiles_touched(tile_calls, built.tile_spans) == {0, 1}
    assert np.array_equal(read.array, pixels[:, 0:2, 0:4])


@pytest.mark.asyncio
@pytest.mark.parametrize("tile_gap", [100, 9000])
async def test_small_gaps_between_tiles_are_coalesced(tile_gap: int) -> None:
    pixels = _grid(8, 8)
    built = build_cog(
        pixels,
        tile=2,
        origin_x=-10.0,
        origin_y=10.0,
        pixel_size_x=0.25,
        pixel_size_y=0.25,
        tile_gap=tile_gap,
    )
    log = _RangeLog(built.blob)
    async with httpx.AsyncClient(transport=log.transport()) as http:
        read = await read_cog_window(
            http,
            "https://example.com/granule.tif",
            west=-10.0,
            south=9.5,
            east=-9.0,
            north=10.0,
        )
    tile_calls = [call for call in log.range_calls() if _tiles_touched([call], built.tile_spans)]
    assert len(tile_calls) == (1 if tile_gap <= cog._COALESCE_GAP else 2)
    assert _tiles_touched(tile_calls, built.tile_spans) == {0, 1}
    assert np.array_equal(read.array, pixels[:, 0:2, 0:4])


@pytest.mark.asyncio
async def test_a_short_deflate_tile_does_not_say_exceeded() -> None:
    built = build_cog(
        np.zeros((1, 2, 2), dtype=np.uint8),
        tile=2,
        origin_x=0.0,
        origin_y=2.0,
        pixel_size_x=1.0,
        pixel_size_y=1.0,
        compression="deflate",
        tile_blobs=[zlib.compress(b"\x00")],
    )
    log = _RangeLog(built.blob)
    async with httpx.AsyncClient(transport=log.transport()) as http:
        with pytest.raises(CogError, match="shorter than its dimensions") as raised:
            await read_cog_window(
                http,
                "https://example.com/granule.tif",
                west=0.0,
                south=0.0,
                east=2.0,
                north=2.0,
            )
    assert "exceeded" not in str(raised.value)


def _patch_backoff(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record range waits and pin jitter at zero. ``Retry-After`` is not jittered."""
    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(cog.random, "uniform", lambda _low, _high: 0.0)
    monkeypatch.setattr(cog.asyncio, "sleep", fake_sleep)
    return delays


@pytest.mark.asyncio
async def test_transient_range_status_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    delays = _patch_backoff(monkeypatch)
    built = _geographic()
    log = _RangeLog(built.blob, fail_first=1)
    async with httpx.AsyncClient(transport=log.transport()) as http:
        read = await read_cog_window(
            http,
            "https://example.com/granule.tif",
            west=-9.5,
            south=8.5,
            east=-9.0,
            north=9.0,
        )
    assert log.failures == 1
    assert delays == [0.5]
    assert np.array_equal(read.array, np.array([[[42, 43], [52, 53]]], dtype=np.uint8))
    assert _tiles_touched(log.range_calls(), built.tile_spans) == {9}


@pytest.mark.asyncio
async def test_range_retries_wait_half_a_second_then_one_then_two(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delays = _patch_backoff(monkeypatch)
    built = _geographic()
    log = _RangeLog(built.blob, fail_first=3)
    async with httpx.AsyncClient(transport=log.transport()) as http:
        read = await read_cog_window(
            http,
            "https://example.com/granule.tif",
            west=-9.5,
            south=8.5,
            east=-9.0,
            north=9.0,
        )
    assert log.failures == 3
    assert delays == [0.5, 1.0, 2.0]
    assert np.array_equal(read.array, np.array([[[42, 43], [52, 53]]], dtype=np.uint8))


@pytest.mark.asyncio
async def test_range_backoff_includes_jitter(monkeypatch: pytest.MonkeyPatch) -> None:
    delays = _patch_backoff(monkeypatch)

    def uniform(low: float, high: float) -> float:
        assert low < 0.0 < high
        return high

    monkeypatch.setattr(cog.random, "uniform", uniform)
    built = _geographic()
    log = _RangeLog(built.blob, fail_first=3)
    async with httpx.AsyncClient(transport=log.transport()) as http:
        await read_cog_window(
            http,
            "https://example.com/granule.tif",
            west=-9.5,
            south=8.5,
            east=-9.0,
            north=9.0,
        )
    fraction = cog._JITTER_FRACTION
    assert delays == [
        0.5 + 0.5 * fraction,
        1.0 + 1.0 * fraction,
        2.0 + 2.0 * fraction,
    ]


@pytest.mark.asyncio
async def test_retry_after_header_sets_the_range_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    delays = _patch_backoff(monkeypatch)
    built = _geographic()
    log = _RangeLog(built.blob, fail_first=1, retry_after="2")
    async with httpx.AsyncClient(transport=log.transport()) as http:
        await read_cog_window(
            http,
            "https://example.com/granule.tif",
            west=-9.5,
            south=8.5,
            east=-9.0,
            north=9.0,
        )
    assert delays == [2.0]


@pytest.mark.asyncio
async def test_retry_after_is_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    delays = _patch_backoff(monkeypatch)
    built = _geographic()
    log = _RangeLog(built.blob, fail_first=1, retry_after="120")
    async with httpx.AsyncClient(transport=log.transport()) as http:
        await read_cog_window(
            http,
            "https://example.com/granule.tif",
            west=-9.5,
            south=8.5,
            east=-9.0,
            north=9.0,
        )
    assert delays == [60.0]


@pytest.mark.asyncio
async def test_past_retry_after_date_waits_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    delays = _patch_backoff(monkeypatch)
    built = _geographic()
    log = _RangeLog(built.blob, fail_first=1, retry_after="Wed, 21 Oct 2015 07:28:00 GMT")
    async with httpx.AsyncClient(transport=log.transport()) as http:
        await read_cog_window(
            http,
            "https://example.com/granule.tif",
            west=-9.5,
            south=8.5,
            east=-9.0,
            north=9.0,
        )
    assert delays == [0.0]


@pytest.mark.asyncio
@pytest.mark.parametrize("header", ["inf", "+inf", "-inf", "nan", "NaN"])
async def test_non_finite_retry_after_falls_back_to_backoff(
    monkeypatch: pytest.MonkeyPatch, header: str
) -> None:
    delays = _patch_backoff(monkeypatch)
    built = _geographic()
    log = _RangeLog(built.blob, fail_first=1, retry_after=header)
    async with httpx.AsyncClient(transport=log.transport()) as http:
        await read_cog_window(
            http,
            "https://example.com/granule.tif",
            west=-9.5,
            south=8.5,
            east=-9.0,
            north=9.0,
        )
    assert delays == [0.5]
    assert all(math.isfinite(delay) for delay in delays)


@pytest.mark.asyncio
async def test_a_short_range_is_retried_once(monkeypatch: pytest.MonkeyPatch) -> None:
    delays = _patch_backoff(monkeypatch)
    built = _geographic()
    log = _RangeLog(built.blob, short_reads=1)
    async with httpx.AsyncClient(transport=log.transport()) as http:
        read = await read_cog_window(
            http,
            "https://example.com/granule.tif",
            west=-9.5,
            south=8.5,
            east=-9.0,
            north=9.0,
        )
    assert log.calls[0] == log.calls[1] == (0, 7)
    assert delays == []
    assert np.array_equal(read.array, np.array([[[42, 43], [52, 53]]], dtype=np.uint8))


@pytest.mark.asyncio
async def test_a_second_short_range_fails_the_scene(tmp_path: Path) -> None:
    built = _geographic()
    log = _RangeLog(built.blob, short_reads=10, allow_full=True)
    async with httpx.AsyncClient(transport=log.transport()) as http:
        with pytest.raises(CogRangeError, match="not the requested length"):
            await read_cog_window(
                http,
                "https://example.com/granule.tif",
                west=-9.5,
                south=8.5,
                east=-9.0,
                north=9.0,
            )
        report = await download_assets(
            [_scene("https://example.com/granule.tif")],
            tmp_path,
            http=http,
            clients={},
            bbox=BBox(west=-9.5, south=8.5, east=-9.0, north=9.0),
        )
    assert log.calls[:2] == [(0, 7), (0, 7)]
    assert log.calls[2:4] == [(0, 7), (0, 7)]
    assert None not in log.calls
    assert report.outcomes[0].status == "failed"
    assert "not the requested length" in (report.outcomes[0].error or "")
    assert list(tmp_path.rglob("*.tif")) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("nodata", [None, "0", "-9999"])
async def test_gdal_nodata_is_carried_through_the_window(nodata: str | None) -> None:
    built = build_cog(
        _grid(4, 4),
        tile=2,
        origin_x=0.0,
        origin_y=4.0,
        pixel_size_x=1.0,
        pixel_size_y=1.0,
        nodata=nodata,
    )
    async with httpx.AsyncClient(transport=_RangeLog(built.blob).transport()) as http:
        read = await read_cog_window(
            http,
            "https://example.com/granule.tif",
            west=0.0,
            south=2.0,
            east=2.0,
            north=4.0,
        )
    assert read.nodata == nodata
    encoded = encode_cog(read)
    async with httpx.AsyncClient(transport=_RangeLog(encoded).transport()) as http:
        again = await read_cog_window(
            http,
            "https://example.com/window.tif",
            west=read.origin_x,
            south=read.origin_y - read.pixel_size_y * read.array.shape[1],
            east=read.origin_x + read.pixel_size_x * read.array.shape[2],
            north=read.origin_y,
        )
    assert again.nodata == nodata
    assert np.array_equal(again.array, read.array)


def test_bigtiff_and_lzw_are_documented_as_later_work() -> None:
    text = cog.__doc__ or ""
    assert "BigTIFF" in text
    assert "LZW" in text
    assert "later work" in text


@pytest.mark.asyncio
async def test_bigtiff_is_rejected_as_later_work() -> None:
    blob = struct.pack("<HHI", 0x4949, 43, 8)
    log = _RangeLog(blob)
    async with httpx.AsyncClient(transport=log.transport()) as http:
        with pytest.raises(CogUnsupportedError, match="BigTIFF is later work"):
            await read_cog_window(
                http,
                "https://example.com/granule.tif",
                west=0.0,
                south=0.0,
                east=1.0,
                north=1.0,
            )
    assert log.range_calls() == [(0, 7)]


@pytest.mark.asyncio
async def test_lzw_is_rejected_as_later_work() -> None:
    built = build_cog(
        _grid(4, 4),
        tile=2,
        origin_x=0.0,
        origin_y=4.0,
        pixel_size_x=1.0,
        pixel_size_y=1.0,
        compression="lzw",
    )
    log = _RangeLog(built.blob)
    async with httpx.AsyncClient(transport=log.transport()) as http:
        with pytest.raises(CogUnsupportedError, match="LZW compression is later work"):
            await read_cog_window(
                http,
                "https://example.com/granule.tif",
                west=0.0,
                south=2.0,
                east=2.0,
                north=4.0,
            )
    assert _tiles_touched(log.range_calls(), built.tile_spans) == set()


def test_encode_rejects_an_empty_window() -> None:
    with pytest.raises(CogError, match="empty"):
        encode_cog(
            CogRead(
                array=np.zeros((1, 0, 1), dtype=np.uint8),
                origin_x=0.0,
                origin_y=0.0,
                pixel_size_x=1.0,
                pixel_size_y=1.0,
                epsg=4326,
                col_off=0,
                row_off=0,
                bytes_fetched=0,
            )
        )
