# lgbm_clouds

Per-pixel cloud mask from a LightGBM model. LightGBM is an optional extra, not
a core dependency:

```bash
uv sync --extra lightgbm
```

If the extra or `libgomp` cannot be imported, this tool is skipped and the
other Atlas tools still register. No GPU. Spectral / SAR / foundation models
are out of scope.

Large mosaics are tiled at the weight file's `tile` edge (256 or 512) and
stitched. The output mask has the same HxW as the input.

Features are raw `r`, `g`, and `b` on the **0-255** scale stored in the PNG.
They are not divided by 255. Train and score on that same scale.

## Weights

Weights are not committed. `weight` in `plugin.toml` is a key relative to
`ATLAS_WEIGHTS_DIR` or `<repo>/weights`. `sha256` is empty until you pin a
file:

```bash
uv run python src/atlas/models/lgbm_clouds/train.py
uv run python src/atlas/models/lgbm_clouds/export.py
```

`export.py` replaces the weight file atomically and prints `restart to load`
plus the sha256. Paste that digest into `sha256` in `plugin.toml` and restart.
While `sha256` is empty the tool is not registered. `ATLAS_ALLOW_UNPINNED=1`
registers it anyway, logs a warning, and marks the description unpinned.
A non-empty pin must match, or inference fails closed. Errors name the
relative key, not a host path.

The JSON object must contain:

- `runtime`: `lightgbm`
- `classes`: the same list as `plugin.toml` (`clear`, then `cloud`)
- `tile`: an edge listed in `input.sizes` (256 or 512)
- `features`: `["r", "g", "b"]`
- `model`: a LightGBM text model (`Booster.model_to_string()`)

The booster must be binary: `num_model_per_iteration() == 1`. Class `cloud`
is positive when the predicted probability is at least 0.5.

The tool is omitted from the session registry until that weight file exists
and `sha256` is set, unless `ATLAS_ALLOW_UNPINNED=1`.

## Spec

- **Input:** RGB PNG, `HxWx3` uint8, any size up to the pixel cap
- **Output:** mask `HxW` uint8 (`0` clear, `1` cloud), same spatial size as the
  input, plus JSON with `width`, `height`, and `positive_fraction`
