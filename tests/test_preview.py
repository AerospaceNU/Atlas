"""Display previews beside downloaded GeoTIFFs. No network: MockTransport only."""

from __future__ import annotations

import struct
from datetime import UTC, datetime
from pathlib import Path

import httpx
import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image

from atlas.cli.data import _merge_source
from atlas.cli.download import download_assets, retire_product
from atlas.cli.preview import (
    MAX_SIDE,
    needs_preview,
    preview_path,
    render_preview,
    write_file_preview,
    write_preview,
)
from atlas.data.base import Asset, BBox, Scene
from atlas.data.cog import CogLimitError, CogUnsupportedError, encode_cog, read_overview
from atlas.data.cog import CogRead as _CogRead

_TILE = 16


class _Entries:
    def __init__(self, entries: list[tuple[int, int, int, bytes]]) -> None:
        self.entries = entries

    def add(self, tag: int, type_code: int, values: list[int] | list[float]) -> None:
        if type_code == 2:
            data = bytes(int(value) for value in values)
            count = len(data)
        else:
            code = {3: "H", 4: "I", 12: "d"}[type_code]
            data = struct.pack(f"<{len(values)}{code}", *values)
            count = len(values)
        self.entries.append((tag, type_code, count, data))


def _tiled_tiff(
    levels: list[NDArray[np.generic]],
    *,
    mask_after: int | None = None,
    nodata: str | None = None,
) -> bytes:
    """Classic little-endian tiled TIFF: IFD0 georeferenced, then one IFD per overview.

    Each level is ``(bands, height, width)``, uncompressed, in 16 px tiles.
    ``mask_after`` inserts a 1-band transparency-mask IFD after that level.
    """
    ifds: list[tuple[NDArray[np.generic], int]] = []
    for index, level in enumerate(levels):
        ifds.append((level, 0 if index == 0 else 1))
        if mask_after == index:
            ifds.append((np.full((1, *level.shape[1:]), 255, dtype=np.uint8), 4 | 1))
    out = bytearray(b"II*\x00\x00\x00\x00\x00")
    ifd_offsets: list[int] = []
    next_slots: list[int] = []
    for number, (pixels, subfile) in enumerate(ifds):
        bands, height, width = pixels.shape
        tiles_x = -(-width // _TILE)
        tiles_y = -(-height // _TILE)
        offsets: list[int] = []
        counts: list[int] = []
        for ty in range(tiles_y):
            for tx in range(tiles_x):
                block = np.zeros((_TILE, _TILE, bands), dtype=pixels.dtype.newbyteorder("<"))
                part = pixels[:, ty * _TILE : (ty + 1) * _TILE, tx * _TILE : (tx + 1) * _TILE]
                block[: part.shape[1], : part.shape[2], :] = np.transpose(part, (1, 2, 0))
                raw = block.tobytes()
                offsets.append(len(out))
                counts.append(len(raw))
                out += raw
        kind = pixels.dtype.kind
        fmt = {"u": 1, "i": 2, "f": 3}[kind]
        bits = pixels.dtype.itemsize * 8
        extra = bytearray()
        entries: list[tuple[int, int, int, bytes]] = []
        add = _Entries(entries).add
        add(254, 4, [subfile])
        add(256, 4, [width])
        add(257, 4, [height])
        add(258, 3, [bits] * bands)
        add(259, 3, [1])
        add(262, 3, [2 if bands >= 3 else 1])
        add(277, 3, [bands])
        add(284, 3, [1])
        add(322, 3, [_TILE])
        add(323, 3, [_TILE])
        add(324, 4, offsets)
        add(325, 4, counts)
        add(339, 3, [fmt] * bands)
        if number == 0:
            add(33550, 12, [1.0, 1.0, 0.0])
            add(33922, 12, [0.0, 0.0, 0.0, -71.0, 42.0, 0.0])
            add(34735, 3, [1, 1, 0, 1, 2048, 0, 1, 4326])
            if nodata is not None:
                add(42113, 2, list((nodata + "\x00").encode()))
        entries.sort()
        ifd_at = len(out) + (len(out) % 2)
        out += b"\x00" * (ifd_at - len(out))
        ifd_offsets.append(ifd_at)
        data_at = ifd_at + 2 + 12 * len(entries) + 4
        body = bytearray(struct.pack("<H", len(entries)))
        for tag, type_code, count, data in entries:
            if len(data) <= 4:
                body += struct.pack("<HHI", tag, type_code, count) + data.ljust(4, b"\x00")
            else:
                body += struct.pack("<HHII", tag, type_code, count, data_at + len(extra))
                extra += data
                if len(extra) % 2:
                    extra += b"\x00"
        next_slots.append(ifd_at + len(body))
        body += b"\x00\x00\x00\x00"
        out += body + extra
    struct.pack_into("<I", out, 4, ifd_offsets[0])
    for slot, target in zip(next_slots, ifd_offsets[1:], strict=False):
        struct.pack_into("<I", out, slot, target)
    return bytes(out)


def _ramp(
    height: int, width: int, *, bands: int = 1, dtype: type[np.generic] = np.uint16
) -> NDArray[np.generic]:
    base = np.arange(height * width, dtype=np.float64).reshape(height, width)
    return np.stack([(base + band * 7).astype(dtype) for band in range(bands)])


def _read(array: NDArray[np.generic], *, nodata: str | None = None) -> _CogRead:
    return _CogRead(
        array=array,  # type: ignore[arg-type]
        origin_x=-71.0,
        origin_y=42.0,
        pixel_size_x=0.001,
        pixel_size_y=0.001,
        epsg=4326,
        col_off=0,
        row_off=0,
        bytes_fetched=0,
        nodata=nodata,
    )


# --- stretch rules ---------------------------------------------------------


def test_uint16_band_is_stretched_to_the_full_byte_range() -> None:
    dark = (np.arange(100, dtype=np.uint16).reshape(1, 10, 10) + 7000).astype(np.uint16)
    image = render_preview(dark)
    assert image.shape == (10, 10)
    assert image.min() == 0
    assert image.max() == 255


def test_gdal_nodata_is_excluded_and_black() -> None:
    band = np.full((1, 10, 10), -9999, dtype=np.int16)
    band[0, :5, :] = np.arange(50, dtype=np.int16).reshape(5, 10) + 100
    image = render_preview(band, "-9999")
    assert (image[5:, :] == 0).all()
    # The valid half spans 0-255, so -9999 did not set the low percentile.
    assert image[:5].max() == 255
    assert np.count_nonzero(image[:5]) > 40


def test_nan_is_excluded_from_float_stretch() -> None:
    band = np.linspace(-20.0, 5.0, 100, dtype=np.float32).reshape(1, 10, 10)
    band[0, 0, :] = np.nan
    image = render_preview(band)
    assert (image[0] == 0).all()
    assert image[1:].max() == 255


def test_mostly_zero_band_without_nodata_treats_zero_as_fill() -> None:
    band = np.zeros((1, 10, 10), dtype=np.uint16)
    band[0, 0, :] = np.arange(10, dtype=np.uint16) + 5000
    image = render_preview(band)
    assert image[0].min() == 0 and image[0].max() == 255
    assert np.count_nonzero(image[0]) == 9
    assert (image[1:] == 0).all()


def test_zero_counts_as_data_when_nodata_is_set() -> None:
    band = np.zeros((1, 10, 10), dtype=np.uint16)
    band[0, 0, :] = 1000
    image = render_preview(band, "65535")
    # Zero is valid and is the low end, so the 1000s are bright, not mid-grey fill.
    assert (image[0] == 255).all()


def test_four_bands_preview_rgb_from_the_first_three() -> None:
    pixels = np.zeros((4, 4, 4), dtype=np.uint8)
    pixels[0] = 200  # red
    pixels[1, :, :2] = 10
    pixels[1, :, 2:] = 20
    pixels[3] = 255  # NIR must not appear
    pixels[0, 0, 0] = 100
    image = render_preview(pixels)
    assert image.shape == (4, 4, 3)
    assert image[:, 2:, 1].min() == 255
    assert image[:, :2, 1].max() == 0


def test_constant_band_is_mid_grey_not_an_error() -> None:
    image = render_preview(np.full((1, 3, 3), 42, dtype=np.float32))
    assert (image == 128).all()


def test_preview_is_downsampled_to_the_side_cap() -> None:
    big = np.zeros((1, 10, MAX_SIDE * 2 + 1), dtype=np.uint16)
    image = render_preview(big)
    assert max(image.shape) <= MAX_SIDE


def test_palette_and_rgb_bytes_need_no_preview() -> None:
    assert not needs_preview(np.zeros((1, 2, 2), dtype=np.uint8), (0,) * 768)
    assert not needs_preview(np.zeros((3, 2, 2), dtype=np.uint8))
    assert needs_preview(np.zeros((4, 2, 2), dtype=np.uint8))
    assert needs_preview(np.zeros((1, 2, 2), dtype=np.uint8))
    assert needs_preview(np.zeros((3, 2, 2), dtype=np.uint16))
    assert needs_preview(np.zeros((1, 2, 2), dtype=np.float32))


def test_write_preview_names_the_png_and_skips_rgb(tmp_path: Path) -> None:
    data = tmp_path / "scene_red.tif"
    data.write_bytes(b"data")
    written = write_preview(data, _ramp(4, 4))
    assert written == tmp_path / "scene_red.preview.png"
    assert Image.open(written).mode == "L"
    assert data.read_bytes() == b"data"
    # The same name later holding an image that displays drops the old preview.
    assert write_preview(data, np.zeros((3, 4, 4), dtype=np.uint8)) is None
    assert not written.exists()


# --- overview reader -------------------------------------------------------


@pytest.mark.asyncio
async def test_overview_reader_picks_the_smallest_level_over_the_floor(tmp_path: Path) -> None:
    full = _ramp(64, 48, bands=4, dtype=np.uint8)
    half = full[:, ::2, ::2].copy()
    quarter = full[:, ::4, ::4].copy()
    path = tmp_path / "naip.tif"
    path.write_bytes(_tiled_tiff([full, half, quarter], mask_after=0))
    before = path.read_bytes()
    read = await read_overview(path, min_long_side=30)
    assert np.array_equal(read.array, half)
    assert read.pixel_size_x == pytest.approx(2.0)
    assert read.epsg == 4326
    assert path.read_bytes() == before
    smallest = await read_overview(path, min_long_side=1)
    assert np.array_equal(smallest.array, quarter)
    whole = await read_overview(path, min_long_side=10_000)
    assert np.array_equal(whole.array, full)


@pytest.mark.asyncio
async def test_overview_reader_refuses_a_full_level_over_the_pixel_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("atlas.data.cog._MAX_WINDOW_PIXELS", 100)
    path = tmp_path / "big.tif"
    path.write_bytes(_tiled_tiff([_ramp(32, 32)]))
    with pytest.raises(CogLimitError):
        await read_overview(path)


@pytest.mark.asyncio
async def test_untiled_file_is_not_previewed(tmp_path: Path) -> None:
    path = tmp_path / "strip.tif"
    Image.fromarray(np.zeros((8, 8), dtype=np.uint16)).save(path)
    with pytest.raises(CogUnsupportedError):
        await write_file_preview(path)
    assert not preview_path(path).exists()


# --- download hook ---------------------------------------------------------


def _scene(href: str, key: str = "image") -> Scene:
    return Scene(
        id="tile",
        source="naip",
        datetime=datetime(2023, 9, 1, tzinfo=UTC),
        bbox=BBox(west=-71.0, south=41.9, east=-70.9, north=42.0),
        platform="naip",
        assets={
            key: Asset(href=href, media_type="image/tiff; application=geotiff", roles=["data"])
        },
    )


def _whole_file(blob: bytes) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if "range" in request.headers:
            return httpx.Response(500)
        return httpx.Response(200, content=blob, headers={"content-type": "image/tiff"})

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_whole_file_download_gets_a_preview_from_an_overview(tmp_path: Path) -> None:
    full = _ramp(64, 64, bands=4, dtype=np.uint8)
    blob = _tiled_tiff([full, full[:, ::2, ::2].copy()])
    async with httpx.AsyncClient(transport=_whole_file(blob)) as http:
        report = await download_assets(
            [_scene("https://example.com/naip.tif")], tmp_path, http=http, clients={}
        )
    outcome = report.outcomes[0]
    assert outcome.status == "saved"
    assert outcome.path is not None and outcome.path.read_bytes() == blob
    assert outcome.preview == tmp_path / "naip" / "tile_image.preview.png"
    with Image.open(outcome.preview) as image:
        assert image.mode == "RGB"
        assert image.size == (64, 64)  # the overview floor exceeds both levels
    assert report.saved_paths() == {"naip": [outcome.path]}


@pytest.mark.asyncio
async def test_preview_failure_does_not_fail_the_download(tmp_path: Path) -> None:
    async with httpx.AsyncClient(transport=_whole_file(b"II*\x00garbage")) as http:
        report = await download_assets(
            [_scene("https://example.com/naip.tif")], tmp_path, http=http, clients={}
        )
    outcome = report.outcomes[0]
    assert outcome.status == "saved"
    assert outcome.error is None
    assert outcome.preview is None
    assert outcome.preview_error is not None
    assert outcome.preview_error.startswith("preview not written:")


@pytest.mark.asyncio
async def test_skipped_whole_file_without_a_preview_gets_one(tmp_path: Path) -> None:
    dest = tmp_path / "naip" / "tile_image.tif"
    dest.parent.mkdir()
    dest.write_bytes(_tiled_tiff([_ramp(32, 32)]))

    def refuse(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("a skipped file is not fetched")

    async with httpx.AsyncClient(transport=httpx.MockTransport(refuse)) as http:
        report = await download_assets(
            [_scene("https://example.com/naip.tif")], tmp_path, http=http, clients={}
        )
    assert report.outcomes[0].status == "skipped"
    assert report.outcomes[0].preview == preview_path(dest)
    assert preview_path(dest).is_file()


@pytest.mark.asyncio
async def test_windowed_chip_preview_uses_the_window_pixels(tmp_path: Path) -> None:
    pixels = _ramp(32, 32)
    blob = encode_cog(
        _CogRead(
            array=pixels,  # type: ignore[arg-type]
            origin_x=-71.0,
            origin_y=42.0,
            pixel_size_x=0.003125,
            pixel_size_y=0.003125,
            epsg=4326,
            col_off=0,
            row_off=0,
            bytes_fetched=0,
        )
    )

    def ranged(request: httpx.Request) -> httpx.Response:
        start_s, end_s = request.headers["range"].removeprefix("bytes=").split("-")
        start, end = int(start_s), min(int(end_s), len(blob) - 1)
        return httpx.Response(
            206,
            content=blob[start : end + 1],
            headers={"Content-Range": f"bytes {start}-{end}/{len(blob)}"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(ranged)) as http:
        report = await download_assets(
            [_scene("https://example.com/red.tif", key="red")],
            tmp_path,
            http=http,
            clients={},
            bbox=BBox(west=-70.99, south=41.95, east=-70.95, north=41.99),
        )
    outcome = report.outcomes[0]
    assert outcome.status == "saved", outcome.error
    assert outcome.preview is not None
    with Image.open(outcome.preview) as image:
        assert image.mode == "L"
        assert image.getextrema() == (0, 255)


@pytest.mark.asyncio
async def test_rgb_chip_gets_no_preview(tmp_path: Path) -> None:
    rgb = np.stack([np.full((32, 32), value, dtype=np.uint8) for value in (10, 20, 30)])
    blob = encode_cog(_read(rgb))

    def ranged(request: httpx.Request) -> httpx.Response:
        start_s, end_s = request.headers["range"].removeprefix("bytes=").split("-")
        start, end = int(start_s), min(int(end_s), len(blob) - 1)
        return httpx.Response(
            206,
            content=blob[start : end + 1],
            headers={"Content-Range": f"bytes {start}-{end}/{len(blob)}"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(ranged)) as http:
        report = await download_assets(
            [_scene("https://example.com/visual.tif", key="visual")],
            tmp_path,
            http=http,
            clients={},
            bbox=BBox(west=-70.99, south=41.99, east=-70.98, north=41.999),
        )
    outcome = report.outcomes[0]
    assert outcome.status == "saved", outcome.error
    assert outcome.preview is None and outcome.preview_error is None
    assert not list((tmp_path / "naip").glob("*.png"))


def test_retiring_a_product_retires_its_preview(tmp_path: Path) -> None:
    data = tmp_path / "tile_image.tif"
    data.write_bytes(b"x")
    preview = preview_path(data)
    preview.write_bytes(b"png")
    retire_product(data)
    assert not data.exists() and not preview.exists()
    assert (tmp_path / "tile_image.preview.png.stale").read_bytes() == b"png"


def test_manifest_previews_accumulate_across_pages() -> None:
    prior = {"previews": ["naip/a.preview.png"], "scene_ids": ["a"]}
    current = {"previews": ["naip/b.preview.png"], "scene_ids": ["b"], "error": None}
    merged = _merge_source(current, prior)
    assert merged["previews"] == ["naip/a.preview.png", "naip/b.preview.png"]
    failed = _merge_source({"previews": [], "error": "boom"}, prior)
    assert failed["previews"] == ["naip/a.preview.png"]
