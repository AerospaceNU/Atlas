"""Server-side mosaic rendering over an AOI (the "A+" path).

Registers a STAC search with Planetary Computer's mosaic API, which composites
ACROSS all matching scenes server-side (clearest pixel first for optical,
newest first for SAR), then fetches the web-mercator tiles covering the AOI,
stitches them, and crops to the exact bbox.

The render recipe (band combo / color formula) is reused verbatim from a
representative scene's `rendered_preview` asset, so each collection renders the
way Planetary Computer intends without hardcoding per-collection recipes here.

NOTE: this currently targets the Planetary Computer mosaic API and assumes the
client exposes `.collection` and `._build_query_filter` (i.e. a
PlanetaryComputerClient subclass). Local-COG compositing is a future path.
"""

from __future__ import annotations

import io
import math
import urllib.parse as up
from pathlib import Path
from typing import Any

import httpx
from PIL import Image

from atlas.compile.product import MosaicResult
from atlas.data.base import PullRequest, Scene
from atlas.data.planetary_computer import PlanetaryComputerClient

DATA_API = "https://planetarycomputer.microsoft.com/api/data/v1"
DEFAULT_ZOOM = 13
DEFAULT_SCALE = 2  # @2x tiles


def _deg2tile(lon: float, lat: float, z: int) -> tuple[int, int]:
    n = 2**z
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)
    return x, y


def _global_px(lon: float, lat: float, z: int, tile_px: int) -> tuple[float, float]:
    world = tile_px * (2**z)
    px = (lon + 180.0) / 360.0 * world
    py = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * world
    return px, py


def _register_search(
    client: PlanetaryComputerClient, request: PullRequest, http: httpx.Client
) -> str:
    query = client._build_query_filter(request)
    body: dict[str, Any] = {
        "collections": [client.collection],
        "bbox": request.bbox.as_list(),
        "datetime": (
            f"{request.start_date.isoformat()}T00:00:00Z/{request.end_date.isoformat()}T23:59:59Z"
        ),
    }
    if query:
        body["query"] = query
    # Clearest pixel wins for optical; newest wins for SAR (no cloud field).
    if query and "eo:cloud_cover" in query:
        body["sortby"] = [{"field": "eo:cloud_cover", "direction": "asc"}]
    else:
        body["sortby"] = [{"field": "datetime", "direction": "desc"}]

    resp = http.post(f"{DATA_API}/mosaic/register", json=body, timeout=60.0)
    resp.raise_for_status()
    search_id = resp.json()["searchid"]
    if not isinstance(search_id, str):
        raise TypeError(f"Expected string searchid, got {type(search_id)}")
    return search_id


def _render_params(sample: Scene, collection: str) -> str:
    """Reuse the collection's rendered_preview recipe, minus item/collection."""
    preview = sample.assets.get("rendered_preview")
    if preview is None:
        raise ValueError(f"Scene {sample.id} has no rendered_preview asset to derive a recipe")
    pairs = up.parse_qsl(up.urlsplit(preview.href).query, keep_blank_values=True)
    kept = [(k, v) for k, v in pairs if k not in ("item", "collection")]
    kept.insert(0, ("collection", collection))  # mosaic tiles require a collection param
    return up.urlencode(kept, doseq=True)


def build_mosaic(
    client: PlanetaryComputerClient,
    request: PullRequest,
    sample: Scene,
    *,
    source: str,
    out_path: Path,
    zoom: int = DEFAULT_ZOOM,
    scale: int = DEFAULT_SCALE,
    http: httpx.Client | None = None,
) -> MosaicResult:
    """Render a gap-filled mosaic for one source over `request.bbox`.

    `sample` is any scene from this source's search (used only for the render
    recipe). Writes a PNG to `out_path` and returns a `MosaicResult`.
    """
    owns_http = http is None
    http = http or httpx.Client()
    tile_px = 256 * scale
    try:
        search_id = _register_search(client, request, http)
        params = _render_params(sample, client.collection)

        bbox = request.bbox
        x0, y0 = _deg2tile(bbox.west, bbox.north, zoom)  # top-left tile
        x1, y1 = _deg2tile(bbox.east, bbox.south, zoom)  # bottom-right tile
        cols, rows = x1 - x0 + 1, y1 - y0 + 1

        canvas = Image.new("RGBA", (cols * tile_px, rows * tile_px))
        n_tiles = 0
        for ty in range(y0, y1 + 1):
            for tx in range(x0, x1 + 1):
                url = f"{DATA_API}/mosaic/tiles/{search_id}/{zoom}/{tx}/{ty}@{scale}x.png?{params}"
                resp = http.get(url, timeout=120.0)
                resp.raise_for_status()
                tile = Image.open(io.BytesIO(resp.content)).convert("RGBA")
                canvas.paste(tile, ((tx - x0) * tile_px, (ty - y0) * tile_px))
                n_tiles += 1

        ox, oy = x0 * tile_px, y0 * tile_px
        left, top = _global_px(bbox.west, bbox.north, zoom, tile_px)
        right, bottom = _global_px(bbox.east, bbox.south, zoom, tile_px)
        crop = canvas.crop(
            (round(left - ox), round(top - oy), round(right - ox), round(bottom - oy))
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(out_path)
        return MosaicResult(
            source=source,
            collection=client.collection,
            search_id=search_id,
            n_tiles=n_tiles,
            width=crop.size[0],
            height=crop.size[1],
            path=str(out_path),
        )
    finally:
        if owns_http:
            http.close()
