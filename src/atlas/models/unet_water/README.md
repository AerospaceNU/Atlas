# unet_water

Open-water mask from a small CPU U-Net (encoder, 2x pool, bottleneck, skip,
decoder). No GPU. Spectral / SAR / foundation models are out of scope.

Large mosaics are tiled at the weight file's `tile` edge (256 or 512 in
production, 32 allowed for fixtures). Tiles are stitched, so the mask stays
the original HxW. The input is not rejected for being larger than one tile.

## Weights

`weight` in `plugin.toml` is a key relative to `ATLAS_WEIGHTS_DIR` or
`<repo>/weights`. `sha256` pins the bytes when it is set. The JSON object must
contain:

- `runtime`: `unet_cpu`
- `classes`: the same list as `plugin.toml`
- `tile`: an even edge listed in `input.sizes`
- `enc_w`, `enc_b`, `bn_w`, `bn_b`, `dec_w`, `dec_b`, `head_w`, `head_b`

Conv kernels are 3x3, shaped `(Cout, Cin, 3, 3)`. A missing file or a hash
mismatch fails closed. The error names the relative key, not a host path.

## Spec

- **Input:** RGB PNG, `HxWx3` uint8, any size up to the pixel cap
- **Output:** mask `HxW` uint8 (`0` land, `1` water), same spatial size as the
  input, plus JSON with `width`, `height`, and `positive_fraction`
