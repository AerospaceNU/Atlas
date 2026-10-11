# rgb_change

Visual before/after hotspots on a PNG pair (absdiff / grayscale delta + blob
polygons). Not NDVI/NBR. No GPU.

The heatmap is the mean absolute channel difference divided by 255, so each
pixel is float32 in `[0, 1]`. Blobs are 4-connected components
(`scipy.ndimage.label`) at or above 0.2 with at least 16 pixels. Each blob is
the convex hull of those pixels. A hull with fewer than 3 points is dropped.
Rasters above `_MAX_PIXELS` are rejected.

## Spec

Visual before/after hotspots on a PNG pair (absdiff / grayscale delta + blob
polygons). Not NDVI/NBR.

- **Input:** two RGB PNGs, each `H×W×3` uint8, same `H×W`
- **Output:** heatmap `H×W` float32 in `[0, 1]` and optional polygons
