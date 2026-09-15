# Planned model-as-tool plugins

Local CPU inference on session RGB PNGs (mosaics, GIBS snapshots, chips).
No GPU. Tile large mosaics at 256–512; do not run detectors on a full mosaic
in one pass. Spectral / SAR / foundation models are out of scope here.

## classify_chip

Cheap land-cover triage of one chip (bare soil, built-up, water, vegetation,
cloud) before change or a VLM.

- **Input:** RGB PNG, `H×W×3` uint8, resized to `224×224×3`
- **Output:** `C` class scores, `C=5` (bare, built-up, water, vegetation, cloud)

## detect_objects

Locate ships, aircraft, and vehicles on high-res chips (NAIP / Maxar-looking
PNG). Nano YOLO only.

- **Input:** RGB PNG tile, `640×640×3` uint8
- **Output:** `N×6` boxes `(x1, y1, x2, y2, score, class_id)` in pixel coords

## rgb_change

Visual before/after hotspots on a PNG pair (absdiff / grayscale delta + blob
polygons). Not NDVI/NBR.

- **Input:** two RGB PNGs, each `H×W×3` uint8, same `H×W`
- **Output:** heatmap `H×W` float32 in `[0, 1]` and optional polygons

## mask_clouds

Per-pixel cloud / valid mask on true-color PNG so later tools skip junk.

- **Input:** RGB PNG tile, `256×256×3` or `512×512×3` uint8
- **Output:** mask `H×W` uint8 (`0` clear, `1` cloud), same spatial size as input

## burn_scar

Burned-area mask from dark scars in true color. Not NBR severity.

- **Input:** RGB PNG tile, `256×256×3` or `512×512×3` uint8
- **Output:** mask `H×W` uint8 (`0` unburned, `1` burn), same spatial size as input

## flood_mask

Open-water mask on optical PNG. Not under-cloud SAR flood.

- **Input:** RGB PNG tile, `256×256×3` or `512×512×3` uint8
- **Output:** mask `H×W` uint8 (`0` dry, `1` water), same spatial size as input
