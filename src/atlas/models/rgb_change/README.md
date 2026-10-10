# rgb_change

Visual before/after hotspots on a PNG pair. Not NDVI or NBR.

The heatmap is the mean absolute channel difference divided by 255, so each
pixel is float32 in `[0, 1]`. A blob is a 4-connected region at or above 0.2
with at least 16 pixels. Each blob is the convex hull of those pixels.

## Tool I/O

- **Input:** two workspace-relative RGB PNGs with the same `H×W`.
- **Output:** `artifacts/rgb_change_heatmap.npy` (`H×W` float32 in `[0, 1]`) and
  JSON with `changed_fraction` plus polygons (`area`, `points` as `[x, y]`).
