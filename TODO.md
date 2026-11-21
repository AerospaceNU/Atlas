# Model-as-tool plugins

Scope is no longer RGB-only: plugins run on prepared analysis stacks
(reflectance, dB) as well as session RGB PNGs. CPU only, no GPU. Large rasters
are tiled by the runtime; never run a detector on a full mosaic in one pass.

Plugins follow the `model.toml` schema 2: tiers `baseline`, `learned`,
`heuristic`; runtimes `rules` and `onnx`; pinned weights installed by a human with
`atlas models pull`. The contract is in
[docs/satellite-analysis.md](docs/satellite-analysis.md).

Packs today:

- `cloud_mask` (OmniCloudMask, ONNX; hidden until its weight is published)
- `burn_severity` (rules)
- `water_extent` (rules)
- `sar_flood` (rules)

Next:

- Fields of The World U-Net and HOT fAIr buildings, both ONNX. Each needs a
  maintainer export and a licence check before it ships.
- Prithvi is deferred.
- `remote` and `torch` runtimes parse in the manifest but do not run.

The RGB-preview heuristics (`segment_landcover`, `rgb_change`, `mask_clouds`,
`burn_scar`, `flood_mask`, `unet_water`, `lgbm_clouds`) keep their package
READMEs under `src/atlas/models/` and need `ATLAS_HEURISTIC_MODELS=1`.
