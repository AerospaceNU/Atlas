# lgbm_clouds

Per-pixel cloud mask from a LightGBM model on RGB features. No GPU. Spectral /
SAR / foundation models are out of scope.

Large mosaics are tiled at the weight file's `tile` edge and stitched. The
output mask has the same HxW as the input.

## Weights

`weight` in `plugin.toml` is a key relative to `ATLAS_WEIGHTS_DIR` or
`<repo>/weights`. `sha256` pins the bytes when it is set. The JSON object must
contain:

- `runtime`: `lightgbm`
- `classes`: the same list as `plugin.toml` (`clear`, then `cloud`)
- `tile`: an edge listed in `input.sizes`
- `features`: `["r", "g", "b"]`
- `model`: a LightGBM text model (`Booster.model_to_string()`)

Class `cloud` is positive when the predicted probability is at least 0.5.
A missing file or a hash mismatch fails closed. The error names the relative
key, not a host path.

## Spec

- **Input:** RGB PNG, `HxWx3` uint8, any size up to the pixel cap
- **Output:** mask `HxW` uint8 (`0` clear, `1` cloud), same spatial size as the
  input, plus JSON with `width`, `height`, and `positive_fraction`
