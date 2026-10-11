# Model-as-tool plugins

Local CPU inference on session RGB PNGs (mosaics, GIBS snapshots, chips).
No GPU. Tile large mosaics at 256–512; do not run detectors on a full mosaic
in one pass. Spectral / SAR / foundation models are out of scope.

The input and output specs live in each package README:

- `src/atlas/models/segment_landcover/README.md`
- `src/atlas/models/rgb_change/README.md`
- `src/atlas/models/mask_clouds/README.md`
- `src/atlas/models/burn_scar/README.md`
- `src/atlas/models/flood_mask/README.md`
- `src/atlas/models/unet_water/README.md`
- `src/atlas/models/lgbm_clouds/README.md`
