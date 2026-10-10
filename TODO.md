# Model-as-tool plugins

Local CPU inference on session RGB PNGs (mosaics, GIBS snapshots, chips).
No GPU. Spectral / SAR / foundation models are out of scope.

These plugins are registered as agent tools. Contracts live in each package
README under `src/atlas/models/`.

- `segment_landcover` — per-pixel land cover at native resolution
- `rgb_change` — before/after heatmap and blob polygons
- `mask_clouds` — cloud mask on a 256 or 512 tile
- `burn_scar` — burn mask on a 256 or 512 tile
- `flood_mask` — open-water mask on a 256 or 512 tile
