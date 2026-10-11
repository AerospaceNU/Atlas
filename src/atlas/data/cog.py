"""Read a window of a Cloud Optimized GeoTIFF without the rest of the granule.

The header and the internal tiles that overlap the requested box are fetched
with HTTP Range requests. Pixel tiles outside that box stay on the server.
The full-resolution IFD is the one that is read; overview levels are not
substituted for it.
"""

from __future__ import annotations

import math
import struct
import zlib
from dataclasses import dataclass
from typing import Final
from urllib.parse import urlparse

import httpx
import numpy as np
from numpy.typing import NDArray

# Degree CRS values we can pass through without a projection.
GEOGRAPHIC_EPSG: Final[frozenset[int]] = frozenset({4326, 4269, 4258, 4267})

_WEB_MERCATOR: Final[int] = 3857
_MERCATOR_RADIUS: Final[float] = 6378137.0
_UTM_NORTH_MIN: Final[int] = 32601
_UTM_NORTH_MAX: Final[int] = 32660
_UTM_SOUTH_MIN: Final[int] = 32701
_UTM_SOUTH_MAX: Final[int] = 32760
_USER_DEFINED_EPSG: Final[int] = 32767

_TYPE_SIZE: Final[dict[int, int]] = {
    1: 1,  # BYTE
    3: 2,  # SHORT
    4: 4,  # LONG
    12: 8,  # DOUBLE
}
_TAG_WIDTH: Final[int] = 256
_TAG_HEIGHT: Final[int] = 257
_TAG_BITS: Final[int] = 258
_TAG_COMPRESSION: Final[int] = 259
_TAG_PHOTOMETRIC: Final[int] = 262
_TAG_SAMPLES: Final[int] = 277
_TAG_PLANAR: Final[int] = 284
_TAG_PREDICTOR: Final[int] = 317
_TAG_TILE_WIDTH: Final[int] = 322
_TAG_TILE_HEIGHT: Final[int] = 323
_TAG_TILE_OFFSETS: Final[int] = 324
_TAG_TILE_COUNTS: Final[int] = 325
_TAG_SAMPLE_FORMAT: Final[int] = 339
_TAG_PIXEL_SCALE: Final[int] = 33550
_TAG_TIEPOINT: Final[int] = 33922
_TAG_GEOKEYS: Final[int] = 34735
_GEO_GEOGRAPHIC: Final[int] = 2048
_GEO_PROJECTED: Final[int] = 3072
_COMPRESS_NONE: Final[int] = 1
_COMPRESS_DEFLATE: Final[frozenset[int]] = frozenset({8, 32946})
_MISS: Final[str] = "AOI does not overlap the COG"


class CogError(Exception):
    """A COG window could not be read."""


class CogMissError(CogError):
    """The requested box does not overlap the raster."""


class CogUnsupportedError(CogError):
    """The object is not a tiled GeoTIFF this reader can window."""


class CogRangeError(CogError):
    """The server did not return the requested byte range."""


@dataclass(frozen=True)
class CogRead:
    """Pixels and georeferencing for one window of a COG.

    ``array`` has shape ``(bands, height, width)``. ``origin_x`` and
    ``origin_y`` are the upper-left corner of this window in the raster CRS.
    ``pixel_size_y`` is positive; row 0 is the northern edge.
    """

    array: NDArray[np.uint8] | NDArray[np.uint16]
    origin_x: float
    origin_y: float
    pixel_size_x: float
    pixel_size_y: float
    epsg: int | None
    col_off: int
    row_off: int
    bytes_fetched: int


@dataclass
class _Image:
    endian: str
    width: int
    height: int
    samples: int
    bits: int
    compression: int
    predictor: int
    tile_width: int
    tile_height: int
    tile_offsets: list[int]
    tile_counts: list[int]
    origin_x: float
    origin_y: float
    pixel_size_x: float
    pixel_size_y: float
    epsg: int | None


class _RangeReader:
    """HTTP Range reader that remembers spans it has already fetched."""

    def __init__(self, http: httpx.AsyncClient, href: str) -> None:
        self._http = http
        self._href = href
        self._spans: list[tuple[int, int, bytes]] = []
        self.bytes_fetched = 0

    async def read(self, start: int, end: int) -> bytes:
        if end < start or start < 0:
            raise CogError("invalid byte range")
        cached = self._cached(start, end)
        if cached is not None:
            return cached
        response = await self._http.get(
            self._href,
            headers={"Range": f"bytes={start}-{end}"},
            follow_redirects=True,
        )
        if response.status_code == 200:
            raise CogRangeError("server returned the whole object instead of the requested range")
        if response.status_code == 416:
            raise CogRangeError("server rejected the byte range")
        if response.status_code != 206:
            response.raise_for_status()
            raise CogRangeError(f"unexpected status {response.status_code} for a range request")
        data = response.content
        expected = end - start + 1
        if len(data) != expected:
            raise CogRangeError("range response was not the requested length")
        self._spans.append((start, end, data))
        self.bytes_fetched += len(data)
        return data

    def _cached(self, start: int, end: int) -> bytes | None:
        for span_start, span_end, data in self._spans:
            if span_start <= start and end <= span_end:
                offset = start - span_start
                return data[offset : offset + (end - start + 1)]
        return None


class CogFile:
    """A tiled GeoTIFF whose header has been read and whose pixels have not."""

    def __init__(self, reader: _RangeReader, image: _Image) -> None:
        self._reader = reader
        self._image = image

    @property
    def epsg(self) -> int | None:
        """EPSG code from the GeoTIFF keys, if one was declared."""
        return self._image.epsg

    async def read_window(
        self,
        *,
        west: float,
        south: float,
        east: float,
        north: float,
    ) -> CogRead:
        """Read the pixels that overlap a box in the raster CRS.

        Args:
            west: Left edge in the raster CRS.
            south: Bottom edge in the raster CRS.
            east: Right edge in the raster CRS.
            north: Top edge in the raster CRS.

        Returns:
            The overlapping pixels and the georeferencing of that window.

        Raises:
            CogMissError: The box does not cover any pixel.
            CogError: A tile could not be decoded.
            CogRangeError: The server did not honor Range.
        """
        image = self._image
        col0, row0, col1, row1 = _pixel_window(
            image,
            west=west,
            south=south,
            east=east,
            north=north,
        )
        tiles_x = math.ceil(image.width / image.tile_width)
        tx0 = col0 // image.tile_width
        tx1 = (col1 - 1) // image.tile_width
        ty0 = row0 // image.tile_height
        ty1 = (row1 - 1) // image.tile_height
        out_dtype = np.dtype(np.uint16 if image.bits == 16 else np.uint8)
        out: NDArray[np.uint8] | NDArray[np.uint16] = np.zeros(
            (image.samples, row1 - row0, col1 - col0),
            dtype=out_dtype,
        )
        for ty in range(ty0, ty1 + 1):
            for tx in range(tx0, tx1 + 1):
                index = ty * tiles_x + tx
                await _copy_tile(self._reader, image, out, index, tx, ty, col0, row0, col1, row1)
        return CogRead(
            array=out,
            origin_x=image.origin_x + col0 * image.pixel_size_x,
            origin_y=image.origin_y - row0 * image.pixel_size_y,
            pixel_size_x=image.pixel_size_x,
            pixel_size_y=image.pixel_size_y,
            epsg=image.epsg,
            col_off=col0,
            row_off=row0,
            bytes_fetched=self._reader.bytes_fetched,
        )


async def open_cog(http: httpx.AsyncClient, href: str) -> CogFile:
    """Open a tiled GeoTIFF and read its header, not its pixels.

    Args:
        http: Client used for range requests.
        href: Absolute http(s) URL of the COG.

    Returns:
        A handle that can read one or more windows.

    Raises:
        CogUnsupportedError: The object is not a tiled GeoTIFF this reader understands.
        CogRangeError: The server did not honor Range.
        httpx.HTTPError: The request failed.
    """
    parsed = urlparse(href)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise CogUnsupportedError("COG href must be an absolute http(s) URL")
    reader = _RangeReader(http, href)
    image = await _parse_image(reader)
    return CogFile(reader, image)


async def read_cog_window(
    http: httpx.AsyncClient,
    href: str,
    *,
    west: float,
    south: float,
    east: float,
    north: float,
) -> CogRead:
    """Read the pixels of ``href`` that overlap a box in the raster CRS.

    Issues HTTP Range requests for the TIFF header and for the internal tiles
    that intersect the box. The rest of the granule stays on the server.

    Args:
        http: Client used for range requests. The server must honor Range.
        href: Absolute http(s) URL of a tiled GeoTIFF.
        west: Left edge in the raster CRS.
        south: Bottom edge in the raster CRS.
        east: Right edge in the raster CRS.
        north: Top edge in the raster CRS.

    Returns:
        The windowed pixels and the georeferencing of that window.

    Raises:
        CogMissError: The box does not cover any pixel.
        CogUnsupportedError: The object is not a tiled GeoTIFF this reader understands.
        CogRangeError: The server did not honor Range.
        httpx.HTTPError: The request failed.
    """
    cog = await open_cog(http, href)
    return await cog.read_window(west=west, south=south, east=east, north=north)


def model_bounds(
    epsg: int,
    *,
    west: float,
    south: float,
    east: float,
    north: float,
) -> tuple[float, float, float, float] | None:
    """Map a longitude/latitude box into the raster CRS.

    Geographic CRS values are returned unchanged. WGS84 UTM and Web Mercator
    are projected from a 3 by 3 sample of the box, which is enough for an AOI
    that fits in one scene. ``None`` means this module cannot place the box
    in that CRS, and the caller should fetch the whole asset instead.

    Args:
        epsg: Raster CRS.
        west: Western edge in degrees.
        south: Southern edge in degrees.
        east: Eastern edge in degrees.
        north: Northern edge in degrees.

    Returns:
        ``(west, south, east, north)`` in the raster CRS, or ``None`` when the
        CRS is unsupported or the box crosses the antimeridian.

    Raises:
        CogMissError: ``north`` is south of ``south``.
    """
    coords = (west, south, east, north)
    if not all(math.isfinite(value) for value in coords):
        return None
    if north < south:
        raise CogMissError(_MISS)
    if epsg in GEOGRAPHIC_EPSG:
        if west > east:
            return None
        return west, south, east, north
    if west > east:
        return None
    if _project_point(epsg, west, south) is None:
        return None
    lons = (west, (west + east) / 2.0, east)
    lats = (south, (south + north) / 2.0, north)
    xs: list[float] = []
    ys: list[float] = []
    for lon in lons:
        for lat in lats:
            point = _project_point(epsg, lon, lat)
            if point is None:
                return None
            xs.append(point[0])
            ys.append(point[1])
    return min(xs), min(ys), max(xs), max(ys)


def encode_cog(read: CogRead) -> bytes:
    """Encode ``read`` as a single-tile little-endian GeoTIFF.

    Args:
        read: Window returned by :func:`read_cog_window`.

    Returns:
        A classic TIFF whose first tile is the window.

    Raises:
        CogError: ``read.array`` is not uint8 or uint16.
    """
    array = read.array
    if array.dtype == np.uint8:
        bits = 8
    elif array.dtype == np.uint16:
        bits = 16
    else:
        raise CogError(f"cannot encode dtype {array.dtype}")
    if array.ndim != 3:
        raise CogError("window array must have shape (bands, height, width)")
    bands, height, width = (int(array.shape[0]), int(array.shape[1]), int(array.shape[2]))
    if bands < 1 or height < 1 or width < 1:
        raise CogError("window array is empty")
    endian = "<"
    cube = np.transpose(array, (1, 2, 0)).astype(np.dtype(f"{endian}u{bits // 8}"), copy=False)
    tile = cube.tobytes()
    scale = struct.pack(endian + "3d", read.pixel_size_x, read.pixel_size_y, 0.0)
    tie = struct.pack(endian + "6d", 0.0, 0.0, 0.0, read.origin_x, read.origin_y, 0.0)
    geokeys = _geokey_bytes(read.epsg, endian)
    payloads: list[bytes] = [scale, tie]
    if geokeys is not None:
        payloads.append(geokeys)
    # Width through tiepoint is 13 tags; the GeoKey directory is the 14th.
    n_entries = 14 if geokeys is not None else 13
    ifd_size = 2 + n_entries * 12 + 4
    cursor = 8 + ifd_size
    payload_offsets = []
    for payload in payloads:
        payload_offsets.append(cursor)
        cursor += len(payload)
    tile_offset = cursor
    photometric = 2 if bands == 3 else 1
    fields: list[tuple[int, int, int, bytes]] = [
        (_TAG_WIDTH, 4, 1, struct.pack(endian + "I", width)),
        (_TAG_HEIGHT, 4, 1, struct.pack(endian + "I", height)),
        (_TAG_BITS, 3, 1, struct.pack(endian + "H", bits) + b"\x00\x00"),
        (_TAG_COMPRESSION, 3, 1, struct.pack(endian + "H", _COMPRESS_NONE) + b"\x00\x00"),
        (_TAG_PHOTOMETRIC, 3, 1, struct.pack(endian + "H", photometric) + b"\x00\x00"),
        (_TAG_SAMPLES, 3, 1, struct.pack(endian + "H", bands) + b"\x00\x00"),
        (_TAG_TILE_WIDTH, 4, 1, struct.pack(endian + "I", width)),
        (_TAG_TILE_HEIGHT, 4, 1, struct.pack(endian + "I", height)),
        (_TAG_TILE_OFFSETS, 4, 1, struct.pack(endian + "I", tile_offset)),
        (_TAG_TILE_COUNTS, 4, 1, struct.pack(endian + "I", len(tile))),
        (_TAG_SAMPLE_FORMAT, 3, 1, struct.pack(endian + "H", 1) + b"\x00\x00"),
        (_TAG_PIXEL_SCALE, 12, 3, struct.pack(endian + "I", payload_offsets[0])),
        (_TAG_TIEPOINT, 12, 6, struct.pack(endian + "I", payload_offsets[1])),
    ]
    if geokeys is not None:
        fields.append(
            (_TAG_GEOKEYS, 3, len(geokeys) // 2, struct.pack(endian + "I", payload_offsets[2]))
        )
    fields.sort(key=lambda item: item[0])
    out = bytearray(struct.pack(endian + "HHI", 0x4949, 42, 8))
    out += struct.pack(endian + "H", len(fields))
    for tag, type_code, count, value in fields:
        out += struct.pack(endian + "HHI", tag, type_code, count)
        out += value
    out += struct.pack(endian + "I", 0)
    for payload in payloads:
        out += payload
    out += tile
    return bytes(out)


def _geokey_bytes(epsg: int | None, endian: str) -> bytes | None:
    if epsg is None or epsg == _USER_DEFINED_EPSG:
        return None
    key = _GEO_GEOGRAPHIC if epsg in GEOGRAPHIC_EPSG else _GEO_PROJECTED
    values = (1, 1, 0, 1, key, 0, 1, epsg)
    return struct.pack(endian + f"{len(values)}H", *values)


def _project_point(epsg: int, lon: float, lat: float) -> tuple[float, float] | None:
    if epsg == _WEB_MERCATOR:
        return _web_mercator(lon, lat)
    zone = _utm_zone(epsg)
    if zone is None or abs(lat) > 84.0:
        return None
    number, northern = zone
    return _wgs84_utm(lon, lat, number, northern=northern)


def _utm_zone(epsg: int) -> tuple[int, bool] | None:
    if _UTM_NORTH_MIN <= epsg <= _UTM_NORTH_MAX:
        return epsg - 32600, True
    if _UTM_SOUTH_MIN <= epsg <= _UTM_SOUTH_MAX:
        return epsg - 32700, False
    return None


def _web_mercator(lon_deg: float, lat_deg: float) -> tuple[float, float]:
    lat = min(max(lat_deg, -85.05112878), 85.05112878)
    x_coord = math.radians(lon_deg) * _MERCATOR_RADIUS
    y_coord = math.log(math.tan(math.pi / 4.0 + math.radians(lat) / 2.0)) * _MERCATOR_RADIUS
    return x_coord, y_coord


def _wgs84_utm(lon_deg: float, lat_deg: float, zone: int, *, northern: bool) -> tuple[float, float]:
    semi_major = 6378137.0
    flattening = 1.0 / 298.257223563
    scale = 0.9996
    eccentricity_sq = flattening * (2.0 - flattening)
    second_eccentricity_sq = eccentricity_sq / (1.0 - eccentricity_sq)
    lon0 = math.radians(zone * 6.0 - 183.0)
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    tan_lat = math.tan(lat)
    radius = semi_major / math.sqrt(1.0 - eccentricity_sq * sin_lat * sin_lat)
    tan_sq = tan_lat * tan_lat
    c_term = second_eccentricity_sq * cos_lat * cos_lat
    a_term = cos_lat * (lon - lon0)
    e4 = eccentricity_sq * eccentricity_sq
    e6 = e4 * eccentricity_sq
    meridian = semi_major * (
        (1.0 - eccentricity_sq / 4.0 - 3.0 * e4 / 64.0 - 5.0 * e6 / 256.0) * lat
        - (3.0 * eccentricity_sq / 8.0 + 3.0 * e4 / 32.0 + 45.0 * e6 / 1024.0) * math.sin(2.0 * lat)
        + (15.0 * e4 / 256.0 + 45.0 * e6 / 1024.0) * math.sin(4.0 * lat)
        - (35.0 * e6 / 3072.0) * math.sin(6.0 * lat)
    )
    easting = (
        scale
        * radius
        * (
            a_term
            + (1.0 - tan_sq + c_term) * a_term**3 / 6.0
            + (
                5.0
                - 18.0 * tan_sq
                + tan_sq * tan_sq
                + 72.0 * c_term
                - 58.0 * second_eccentricity_sq
            )
            * a_term**5
            / 120.0
        )
        + 500_000.0
    )
    northing = scale * (
        meridian
        + radius
        * tan_lat
        * (
            a_term**2 / 2.0
            + (5.0 - tan_sq + 9.0 * c_term + 4.0 * c_term * c_term) * a_term**4 / 24.0
            + (
                61.0
                - 58.0 * tan_sq
                + tan_sq * tan_sq
                + 600.0 * c_term
                - 330.0 * second_eccentricity_sq
            )
            * a_term**6
            / 720.0
        )
    )
    if not northern:
        northing += 10_000_000.0
    return easting, northing


async def _parse_image(reader: _RangeReader) -> _Image:
    header = await reader.read(0, 7)
    if header[:2] == b"II":
        endian = "<"
    elif header[:2] == b"MM":
        endian = ">"
    else:
        raise CogUnsupportedError("not a TIFF")
    magic = struct.unpack_from(endian + "H", header, 2)[0]
    if magic != 42:
        raise CogUnsupportedError("only classic TIFF COGs can be windowed")
    ifd_at = struct.unpack_from(endian + "I", header, 4)[0]
    count_raw = await reader.read(ifd_at, ifd_at + 1)
    count = struct.unpack(endian + "H", count_raw)[0]
    if count < 1 or count > 4096:
        raise CogUnsupportedError("TIFF IFD is not a COG header")
    ifd_end = ifd_at + 2 + count * 12 + 4 - 1
    ifd = await reader.read(ifd_at, ifd_end)
    entries: dict[int, tuple[int, int, bytes]] = {}
    for index in range(count):
        offset = 2 + index * 12
        tag, type_code, number = struct.unpack_from(endian + "HHI", ifd, offset)
        entries[tag] = (type_code, number, ifd[offset + 8 : offset + 12])
    return await _image_from_entries(reader, endian, entries)


async def _image_from_entries(
    reader: _RangeReader,
    endian: str,
    entries: dict[int, tuple[int, int, bytes]],
) -> _Image:
    if _TAG_TILE_WIDTH not in entries or _TAG_TILE_HEIGHT not in entries:
        raise CogUnsupportedError("COG is not tiled")
    if (
        _TAG_PLANAR in entries
        and _int_value(await _entry_ints(reader, endian, entries[_TAG_PLANAR])) != 1
    ):
        raise CogUnsupportedError("separate-plane TIFFs are not windowed")
    width = _int_value(await _entry_ints(reader, endian, entries[_TAG_WIDTH]))
    height = _int_value(await _entry_ints(reader, endian, entries[_TAG_HEIGHT]))
    samples = 1
    if _TAG_SAMPLES in entries:
        samples = _int_value(await _entry_ints(reader, endian, entries[_TAG_SAMPLES]))
    bits_values = await _entry_ints(reader, endian, entries[_TAG_BITS])
    if len(bits_values) not in {1, samples} or any(bits != bits_values[0] for bits in bits_values):
        raise CogUnsupportedError("mixed bit depths are not windowed")
    bits = bits_values[0]
    if bits not in {8, 16}:
        raise CogUnsupportedError(f"{bits}-bit samples are not windowed")
    if _TAG_SAMPLE_FORMAT in entries:
        formats = await _entry_ints(reader, endian, entries[_TAG_SAMPLE_FORMAT])
        if any(fmt != 1 for fmt in formats):
            raise CogUnsupportedError("only unsigned integer samples are windowed")
    compression = _COMPRESS_NONE
    if _TAG_COMPRESSION in entries:
        compression = _int_value(await _entry_ints(reader, endian, entries[_TAG_COMPRESSION]))
    if compression != _COMPRESS_NONE and compression not in _COMPRESS_DEFLATE:
        raise CogUnsupportedError(f"TIFF compression {compression} is not windowed")
    predictor = 1
    if _TAG_PREDICTOR in entries:
        predictor = _int_value(await _entry_ints(reader, endian, entries[_TAG_PREDICTOR]))
    if predictor not in {1, 2}:
        raise CogUnsupportedError(f"TIFF predictor {predictor} is not windowed")
    tile_width = _int_value(await _entry_ints(reader, endian, entries[_TAG_TILE_WIDTH]))
    tile_height = _int_value(await _entry_ints(reader, endian, entries[_TAG_TILE_HEIGHT]))
    if min(width, height, samples, tile_width, tile_height) < 1:
        raise CogUnsupportedError("TIFF dimensions are empty")
    tiles_x = math.ceil(width / tile_width)
    tiles_y = math.ceil(height / tile_height)
    offsets = await _entry_ints(reader, endian, entries[_TAG_TILE_OFFSETS])
    counts = await _entry_ints(reader, endian, entries[_TAG_TILE_COUNTS])
    if len(offsets) != tiles_x * tiles_y or len(counts) != len(offsets):
        raise CogUnsupportedError("tile index does not match the raster dimensions")
    if _TAG_PIXEL_SCALE not in entries or _TAG_TIEPOINT not in entries:
        raise CogUnsupportedError("COG has no ModelPixelScale and ModelTiepoint")
    scale = await _entry_doubles(reader, endian, entries[_TAG_PIXEL_SCALE])
    tie = await _entry_doubles(reader, endian, entries[_TAG_TIEPOINT])
    if len(scale) < 2 or len(tie) < 6:
        raise CogUnsupportedError("GeoTIFF georeferencing is incomplete")
    pixel_size_x = scale[0]
    pixel_size_y = scale[1]
    if pixel_size_x <= 0 or pixel_size_y <= 0 or not math.isfinite(pixel_size_x + pixel_size_y):
        raise CogUnsupportedError("pixel scale must be finite and positive")
    origin_x = tie[3] - tie[0] * pixel_size_x
    origin_y = tie[4] + tie[1] * pixel_size_y
    epsg = None
    if _TAG_GEOKEYS in entries:
        epsg = _epsg_from_geokeys(await _entry_ints(reader, endian, entries[_TAG_GEOKEYS]))
    return _Image(
        endian=endian,
        width=width,
        height=height,
        samples=samples,
        bits=bits,
        compression=compression,
        predictor=predictor,
        tile_width=tile_width,
        tile_height=tile_height,
        tile_offsets=offsets,
        tile_counts=counts,
        origin_x=origin_x,
        origin_y=origin_y,
        pixel_size_x=pixel_size_x,
        pixel_size_y=pixel_size_y,
        epsg=epsg,
    )


def _epsg_from_geokeys(values: list[int]) -> int | None:
    if len(values) < 4:
        return None
    projected: int | None = None
    geographic: int | None = None
    for index in range(values[3]):
        base = 4 + index * 4
        if base + 3 >= len(values):
            break
        key_id, location, count, value = values[base : base + 4]
        if location != 0 or count != 1 or value in {0, _USER_DEFINED_EPSG}:
            continue
        if key_id == _GEO_PROJECTED:
            projected = value
        elif key_id == _GEO_GEOGRAPHIC:
            geographic = value
    return projected if projected is not None else geographic


async def _entry_ints(
    reader: _RangeReader,
    endian: str,
    entry: tuple[int, int, bytes],
) -> list[int]:
    type_code, count, inline = entry
    raw = await _entry_bytes(reader, endian, type_code, count, inline)
    if type_code == 1:
        fmt = "B"
    elif type_code == 3:
        fmt = "H"
    elif type_code == 4:
        fmt = "I"
    else:
        raise CogUnsupportedError(f"TIFF type {type_code} is not an integer field")
    try:
        return list(struct.unpack(endian + f"{count}{fmt}", raw))
    except struct.error as exc:
        raise CogUnsupportedError("TIFF field is truncated") from exc


async def _entry_doubles(
    reader: _RangeReader,
    endian: str,
    entry: tuple[int, int, bytes],
) -> list[float]:
    type_code, count, inline = entry
    if type_code != 12:
        raise CogUnsupportedError("GeoTIFF scale and tiepoint must be doubles")
    raw = await _entry_bytes(reader, endian, type_code, count, inline)
    try:
        return list(struct.unpack(endian + f"{count}d", raw))
    except struct.error as exc:
        raise CogUnsupportedError("TIFF field is truncated") from exc


async def _entry_bytes(
    reader: _RangeReader,
    endian: str,
    type_code: int,
    count: int,
    inline: bytes,
) -> bytes:
    size_of = _TYPE_SIZE.get(type_code)
    if size_of is None or count < 1:
        raise CogUnsupportedError(f"TIFF type {type_code} is not supported")
    size = size_of * count
    if size <= 4:
        return inline[:size]
    (offset,) = struct.unpack(endian + "I", inline)
    return await reader.read(offset, offset + size - 1)


def _int_value(values: list[int]) -> int:
    if len(values) != 1:
        raise CogUnsupportedError("expected a single TIFF value")
    return values[0]


def _pixel_window(
    image: _Image,
    *,
    west: float,
    south: float,
    east: float,
    north: float,
) -> tuple[int, int, int, int]:
    coords = (west, south, east, north)
    if not all(math.isfinite(value) for value in coords) or east < west or north < south:
        raise CogError("window bounds are invalid")
    if east == west or north == south:
        raise CogMissError(_MISS)
    col0 = math.floor(_snap((west - image.origin_x) / image.pixel_size_x))
    col1 = math.ceil(_snap((east - image.origin_x) / image.pixel_size_x))
    row0 = math.floor(_snap((image.origin_y - north) / image.pixel_size_y))
    row1 = math.ceil(_snap((image.origin_y - south) / image.pixel_size_y))
    col0 = max(0, col0)
    row0 = max(0, row0)
    col1 = min(image.width, col1)
    row1 = min(image.height, row1)
    if col1 <= col0 or row1 <= row0:
        raise CogMissError(_MISS)
    return col0, row0, col1, row1


def _snap(value: float) -> float:
    nearest = round(value)
    if abs(value - nearest) <= 1e-8:
        return float(nearest)
    return value


async def _copy_tile(
    reader: _RangeReader,
    image: _Image,
    out: NDArray[np.uint8] | NDArray[np.uint16],
    index: int,
    tx: int,
    ty: int,
    col0: int,
    row0: int,
    col1: int,
    row1: int,
) -> None:
    offset = image.tile_offsets[index]
    count = image.tile_counts[index]
    if count < 1 or offset < 0:
        raise CogError("tile offset is empty")
    payload = await reader.read(offset, offset + count - 1)
    cube = _decode_tile(payload, image)
    tile_col0 = tx * image.tile_width
    tile_row0 = ty * image.tile_height
    overlap_c0 = max(col0, tile_col0)
    overlap_c1 = min(col1, tile_col0 + image.tile_width, image.width)
    overlap_r0 = max(row0, tile_row0)
    overlap_r1 = min(row1, tile_row0 + image.tile_height, image.height)
    src_c0 = overlap_c0 - tile_col0
    src_r0 = overlap_r0 - tile_row0
    window = cube[
        src_r0 : src_r0 + (overlap_r1 - overlap_r0), src_c0 : src_c0 + (overlap_c1 - overlap_c0), :
    ]
    out[:, overlap_r0 - row0 : overlap_r1 - row0, overlap_c0 - col0 : overlap_c1 - col0] = (
        np.transpose(window, (2, 0, 1))
    )


def _decode_tile(payload: bytes, image: _Image) -> NDArray[np.generic]:
    if image.compression == _COMPRESS_NONE:
        raw = payload
    else:
        try:
            raw = zlib.decompress(payload)
        except zlib.error as exc:
            raise CogError("deflate tile could not be decoded") from exc
    itemsize = image.bits // 8
    expected = image.tile_width * image.tile_height * image.samples * itemsize
    if len(raw) < expected:
        raise CogError("tile expanded to fewer bytes than its dimensions")
    dtype = np.dtype(f"{image.endian}u{itemsize}")
    cube = np.frombuffer(raw[:expected], dtype=dtype).reshape(
        image.tile_height, image.tile_width, image.samples
    )
    cube = np.array(cube, copy=True)
    if image.predictor == 2:
        for col in range(1, image.tile_width):
            cube[:, col, :] += cube[:, col - 1, :]
    return cube
