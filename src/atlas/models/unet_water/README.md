# unet_water

Open-water mask from a small CPU U-Net (encoder, 2x pool, bottleneck, skip,
decoder). No GPU and no torch. Spectral / SAR / foundation models are out of
scope.

Large mosaics are tiled at the weight file's `tile` edge. Production tiles are
256 or 512. Each window keeps an 8 pixel margin of context, the margin is
cropped, and the cores are stitched so the mask stays the original HxW.

## Weights

Weights are not committed. `weight` in `plugin.toml` is a key relative to
`ATLAS_WEIGHTS_DIR` or `<repo>/weights`. `sha256` is empty until you pin a
file:

```bash
uv run python src/atlas/models/unet_water/train.py
uv run python src/atlas/models/unet_water/export.py
```

`export.py` replaces the weight file atomically and prints `restart to load`
plus the sha256. Paste that digest into `sha256` in `plugin.toml` and restart.
While `sha256` is empty the tool is not registered. `ATLAS_ALLOW_UNPINNED=1`
registers it anyway, logs a warning, and marks the description unpinned.
A non-empty pin must match, or inference fails closed. Errors name the
relative key, not a host path.

The JSON object must contain:

- `runtime`: `unet_cpu`
- `classes`: the same list as `plugin.toml`
- `tile`: an even edge listed in `input.sizes` (256 or 512)
- `scale`, `mean`, `std`: per-channel normalization,
  `(rgb * scale - mean) / std`, applied before the convolutions.
  `scale` is finite and non-zero. Every weight array is finite.
- `enc_w`, `enc_b`, `bn_w`, `bn_b`, `dec_w`, `dec_b`, `head_w`, `head_b`

Conv kernels are 3x3, shaped `(Cout, Cin, 3, 3)`. The hidden width is capped
at 32 channels. `train.py` writes `scale = 1/255`, `mean = 0`, `std = 1`, and
a nearest class-mean head. Chips live in `data-dir/<class>/*.png`.

The tool is omitted from the session registry until that weight file exists
and `sha256` is set, unless `ATLAS_ALLOW_UNPINNED=1`.

## Spec

- **Input:** RGB PNG, `HxWx3` uint8, any size up to the pixel cap
- **Output:** mask `HxW` uint8 (class index), same spatial size as the input,
  plus JSON with `width`, `height`, per-class `fractions`, and
  `positive_fraction` for class 1
