from __future__ import annotations

import argparse
import asyncio
import io
import json
import random
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

import httpx
from PIL import Image

from atlas.data.planetary_computer import PlanetaryComputerClient
from atlas.models.base import repo_root

EUROSAT_URLS = (
    "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1",
    "http://madm.dfki.de/files/sentinel/EuroSAT.zip",
)
SYNTHETIC_SIZE = 64
_STAC_SEARCH = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
_STAC_SOURCE = "planetary_computer"
_STAC_COLLECTION = "sentinel-2-l2a"
_PREVIEW_ASSET_KEYS = ("rendered_preview", "visual", "thumbnail")
_SAS_HOST_SUFFIX = ".blob.core.windows.net"

# Weak labels by place: real Sentinel-2 L2A true-color previews.
STAC_TARGETS: tuple[dict[str, Any], ...] = (
    {
        "name": "bare",
        "bbox": [5.5, 18.5, 7.5, 20.5],
        "datetime": "2024-06-01T00:00:00Z/2024-08-31T23:59:59Z",
        "query": {"eo:cloud_cover": {"lte": 5}},
    },
    {
        "name": "vegetation",
        "bbox": [-62.2, -3.4, -61.6, -2.8],
        "datetime": "2024-06-01T00:00:00Z/2024-08-31T23:59:59Z",
        "query": {"eo:cloud_cover": {"lte": 10}},
    },
    {
        "name": "water",
        "bbox": [5.0, 42.2, 6.2, 43.2],
        "datetime": "2024-06-01T00:00:00Z/2024-08-31T23:59:59Z",
        "query": {"eo:cloud_cover": {"lte": 20}},
    },
    {
        "name": "built-up",
        "bbox": [2.25, 48.82, 2.42, 48.92],
        "datetime": "2024-06-01T00:00:00Z/2024-08-31T23:59:59Z",
        "query": {"eo:cloud_cover": {"lte": 20}},
    },
    {
        "name": "cloud",
        "bbox": [-10.5, 51.0, -5.5, 55.5],
        "datetime": "2024-01-01T00:00:00Z/2024-03-31T23:59:59Z",
        "query": {"eo:cloud_cover": {"gte": 80}},
    },
)

# EuroSAT RGB is labeled Sentinel-2. Cloud and bare are not in EuroSAT, so those
# come from Planetary Computer true-color previews over known AOIs.
EUROSAT_TO_CLASS: dict[str, str] = {
    "Forest": "vegetation",
    "HerbaceousVegetation": "vegetation",
    "Pasture": "vegetation",
    "PermanentCrop": "vegetation",
    "AnnualCrop": "vegetation",
    "SeaLake": "water",
    "River": "water",
    "Residential": "built-up",
    "Industrial": "built-up",
    "Highway": "built-up",
}

SYNTHETIC_COLORS: dict[str, tuple[int, int, int]] = {
    "bare": (180, 140, 90),
    "built-up": (128, 128, 128),
    "water": (30, 90, 180),
    "vegetation": (40, 160, 60),
    "cloud": (245, 245, 245),
}


def _root() -> Path:
    found = repo_root()
    if found is None:
        raise SystemExit("No checkout found")
    return found


def _save_chip(image: Image.Image, destination: Path) -> None:
    """Write RGB PNG at native size. Never resizes the scene."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(destination)


def write_synthetic(data_root: Path, per_split: int = 3) -> None:
    """Solid RGB chips for offline tests; not satellite."""
    for split in ("train", "val"):
        for name, color in SYNTHETIC_COLORS.items():
            for index in range(per_split):
                chip = Image.new("RGB", (SYNTHETIC_SIZE, SYNTHETIC_SIZE), color)
                _save_chip(chip, data_root / split / name / f"{index}.png")


def _download_zip(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 1_000_000:
        return
    last_error: Exception | None = None
    for candidate in (url, *EUROSAT_URLS):
        try:
            with (
                httpx.Client(timeout=120.0, follow_redirects=True) as client,
                client.stream("GET", candidate) as response,
            ):
                response.raise_for_status()
                tmp = dest.with_suffix(".partial")
                with tmp.open("wb") as handle:
                    for chunk in response.iter_bytes():
                        handle.write(chunk)
                tmp.replace(dest)
                return
        except httpx.HTTPError as exc:
            last_error = exc
            continue
    raise RuntimeError(f"Could not download EuroSAT RGB: {last_error}")


def _extract_eurosat(zip_path: Path, extract_dir: Path) -> Path:
    marker = extract_dir / ".extracted"
    if not marker.is_file():
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(extract_dir)
        marker.write_text("ok\n", encoding="utf-8")
    return extract_dir


def _class_source_dir(extracted: Path, eurosat_name: str) -> Path:
    matches = [path for path in extracted.rglob(eurosat_name) if path.is_dir()]
    if not matches:
        raise FileNotFoundError(f"EuroSAT class folder missing: {eurosat_name}")
    return matches[0]


def layout_eurosat(
    extracted: Path,
    data_root: Path,
    *,
    train_n: int,
    val_n: int,
    rng: random.Random,
) -> None:
    """Copy a seeded subset of EuroSAT JPEGs into train/val class folders as PNG."""
    buckets: dict[str, list[Path]] = {
        "vegetation": [],
        "water": [],
        "built-up": [],
    }
    for eurosat_name, atlas_name in EUROSAT_TO_CLASS.items():
        source = _class_source_dir(extracted, eurosat_name)
        buckets[atlas_name].extend(sorted(source.glob("*.jpg")) + sorted(source.glob("*.jpeg")))
    for atlas_name, files in buckets.items():
        rng.shuffle(files)
        chosen = files[: train_n + val_n]
        train_files = chosen[:train_n]
        val_files = chosen[train_n : train_n + val_n]
        for index, src in enumerate(train_files):
            with Image.open(src) as image:
                _save_chip(image, data_root / "train" / atlas_name / f"eurosat_{index:04d}.png")
        for index, src in enumerate(val_files):
            with Image.open(src) as image:
                _save_chip(image, data_root / "val" / atlas_name / f"eurosat_{index:04d}.png")


@dataclass(frozen=True, slots=True)
class AssetPick:
    """One downloadable asset plus the scene provenance it came from."""

    scene_id: str
    collection: str
    datetime: str | None
    cloud_cover: float | None
    bbox: list[float] | None
    asset: str
    href: str


def _asset_href(spec: object) -> str | None:
    href = spec.get("href") if isinstance(spec, dict) else None
    return href if isinstance(href, str) and href else None


def _is_data_asset(spec: object) -> bool:
    roles = spec.get("roles") if isinstance(spec, dict) else None
    return isinstance(roles, list) and "data" in roles


def _choose_asset(feature: dict[str, Any]) -> tuple[str, str] | None:
    """Prefer a browsable preview, then fall back to any data asset."""
    assets = feature.get("assets")
    if not isinstance(assets, dict):
        return None
    for key in _PREVIEW_ASSET_KEYS:
        href = _asset_href(assets.get(key))
        if href is not None:
            return key, href
    for key, spec in assets.items():
        href = _asset_href(spec)
        if href is not None and _is_data_asset(spec):
            return str(key), href
    return None


def _float_or_none(raw: object) -> float | None:
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        return None
    return float(raw)


def _asset_pick(feature: dict[str, Any], *, collection: str) -> AssetPick | None:
    scene_id = feature.get("id")
    chosen = _choose_asset(feature)
    if chosen is None or not isinstance(scene_id, str) or not scene_id:
        return None
    props = feature.get("properties")
    if not isinstance(props, dict):
        props = {}
    scene_dt = props.get("datetime")
    bbox = feature.get("bbox")
    key, href = chosen
    return AssetPick(
        scene_id=scene_id,
        collection=str(feature.get("collection") or collection),
        datetime=scene_dt if isinstance(scene_dt, str) else None,
        cloud_cover=_float_or_none(props.get("eo:cloud_cover")),
        bbox=[float(value) for value in bbox] if isinstance(bbox, list) else None,
        asset=key,
        href=href,
    )


def _asset_suffix(pick: AssetPick) -> str:
    """Previews are re-encoded as PNG; data assets keep their own extension."""
    if pick.asset in _PREVIEW_ASSET_KEYS:
        return ".png"
    return PurePosixPath(urlparse(pick.href).path).suffix.lower() or ".bin"


def _needs_signing(href: str) -> bool:
    host = urlparse(href).hostname
    return host is not None and host.endswith(_SAS_HOST_SUFFIX)


def _write_asset(content: bytes, destination: Path) -> None:
    if destination.suffix == ".png":
        with Image.open(io.BytesIO(content)) as image:
            _save_chip(image, destination)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)


def _write_sidecar(pick: AssetPick, destination: Path) -> None:
    payload = {
        "id": pick.scene_id,
        "source": _STAC_SOURCE,
        "collection": pick.collection,
        "datetime": pick.datetime,
        "cloud_cover": pick.cloud_cover,
        "bbox": pick.bbox,
        "asset": pick.asset,
        "href": pick.href,
    }
    sidecar = destination.with_suffix(".json")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


async def _stac_features(
    client: httpx.AsyncClient,
    *,
    bbox: list[float],
    datetime_range: str,
    query: dict[str, Any],
    limit: int,
) -> list[dict[str, Any]]:
    body = {
        "collections": [_STAC_COLLECTION],
        "bbox": bbox,
        "datetime": datetime_range,
        "limit": limit,
        "query": query,
    }
    response = await client.post(_STAC_SEARCH, json=body)
    response.raise_for_status()
    payload = response.json()
    features = payload.get("features") if isinstance(payload, dict) else None
    if not isinstance(features, list):
        return []
    return [item for item in features if isinstance(item, dict)]


async def _download_assets(
    client: httpx.AsyncClient,
    signer: PlanetaryComputerClient,
    picks: list[AssetPick],
    destinations: list[Path],
) -> int:
    saved = 0
    for pick, dest in zip(picks, destinations, strict=True):
        try:
            href = await signer.sign_href(pick.href) if _needs_signing(pick.href) else pick.href
            response = await client.get(href, follow_redirects=True)
            response.raise_for_status()
            _write_asset(response.content, dest)
            _write_sidecar(pick, dest)
            saved += 1
        except (httpx.HTTPError, OSError):
            continue
    return saved


def _split_destinations(
    picks: list[AssetPick], *, data_root: Path, atlas_name: str, val_n: int
) -> list[Path]:
    val_count = min(val_n, max(1, len(picks) // 5))
    train_count = len(picks) - val_count
    groups = (("train", picks[:train_count]), ("val", picks[train_count:]))
    return [
        data_root / split / atlas_name / f"stac_{index:04d}{_asset_suffix(pick)}"
        for split, group in groups
        for index, pick in enumerate(group)
    ]


async def fetch_stac_class(
    client: httpx.AsyncClient,
    signer: PlanetaryComputerClient,
    *,
    atlas_name: str,
    bbox: list[float],
    datetime_range: str,
    query: dict[str, Any],
    data_root: Path,
    train_n: int,
    val_n: int,
) -> None:
    features = await _stac_features(
        client,
        bbox=bbox,
        datetime_range=datetime_range,
        query=query,
        limit=max(train_n + val_n, 10) * 2,
    )
    picks = [
        pick
        for pick in (_asset_pick(item, collection=_STAC_COLLECTION) for item in features)
        if pick is not None
    ][: train_n + val_n]
    if len(picks) < 4:
        keys = []
        if features:
            raw_assets = features[0].get("assets")
            if isinstance(raw_assets, dict):
                keys = list(raw_assets)
        raise RuntimeError(
            f"STAC returned {len(features)} {atlas_name} items, "
            f"{len(picks)} downloadable assets (asset keys={keys})"
        )
    dests = _split_destinations(picks, data_root=data_root, atlas_name=atlas_name, val_n=val_n)
    saved = await _download_assets(client, signer, picks, dests)
    if saved < 4:
        raise RuntimeError(f"Only downloaded {saved} {atlas_name} STAC assets")


async def fetch_stac_extras(data_root: Path, *, train_n: int, val_n: int) -> None:
    timeout = httpx.Timeout(60.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        signer = PlanetaryComputerClient(collection=_STAC_COLLECTION, client=client)
        for target in STAC_TARGETS:
            name = target["name"]
            assert isinstance(name, str)
            bbox = target["bbox"]
            datetime_range = target["datetime"]
            query = target["query"]
            assert isinstance(bbox, list)
            assert isinstance(datetime_range, str)
            assert isinstance(query, dict)
            print(f"STAC {name}…", file=sys.stderr)
            await fetch_stac_class(
                client,
                signer,
                atlas_name=name,
                bbox=[float(v) for v in bbox],
                datetime_range=datetime_range,
                query=query,
                data_root=data_root,
                train_n=train_n,
                val_n=val_n,
            )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Download real Sentinel-2 chips into train/ and val/ class folders."
    )
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument(
        "--eurosat",
        action="store_true",
        help="Also download labeled EuroSAT RGB chips (vegetation/water/built-up).",
    )
    parser.add_argument("--per-class", type=int, default=20)
    parser.add_argument("--val-per-class", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    data_root = _root() / "local" / "models" / "segment_landcover"
    data_root.mkdir(parents=True, exist_ok=True)
    if args.synthetic:
        write_synthetic(data_root)
        print(data_root)
        return
    print("Fetching Sentinel-2 L2A true-color previews from Planetary Computer", file=sys.stderr)
    asyncio.run(fetch_stac_extras(data_root, train_n=args.per_class, val_n=args.val_per_class))
    if args.eurosat:
        cache = data_root / "cache"
        zip_path = cache / "EuroSAT_RGB.zip"
        print(f"Downloading EuroSAT RGB → {zip_path}", file=sys.stderr)
        _download_zip(EUROSAT_URLS[0], zip_path)
        extracted = _extract_eurosat(zip_path, cache / "eurosat")
        rng = random.Random(args.seed)
        layout_eurosat(
            extracted,
            data_root,
            train_n=args.per_class,
            val_n=args.val_per_class,
            rng=rng,
        )
    print(data_root)


if __name__ == "__main__":
    main(sys.argv[1:])
