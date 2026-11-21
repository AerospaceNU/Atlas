# Satellite analysis

Reference for the typed satellite tools, the stack format, model plugins and the
script path. The agent's own short version is `atlas geo guide TOPIC`.

Needs `uv sync --extra geo --extra onnx` for reprojection, geographic inputs
and learned models. Rule-based paths run without them.

## 1. Stack format

A stack is a directory `.atlas/analysis/stacks/<id>/` under the workspace. It is
persistent: a new session opens it by id, and removing a session does not remove
it. Stacks from older sessions under `<artifact_root>/analysis/stacks` are still
found by id (read-only fallback; nothing is migrated). Run ids are unique per
workspace:

- `stack.json` (schema `atlas.stack/1`, written last as the commit marker)
- `data.npy`: float32, layers x H x W
- `qa.npy`: optional uint8 QA planes, 0/1
- `preview.png`

`stack.json` keys: `id`, `params_hash`, `created`, `warnings`, `grid` (crs,
transform, size, bounds, `pixel_area_m2`), `aoi`, `layers`, `qa`, `provenance`,
`parents`, `data_sha256`, `run_id`.

- **Identity.** The id is the given `name`, else a prepared
  `<sensor>_<YYYYMMDD>_<hash>` or derived `<parent>-<op>-<hash>`. The same id and
  `params_hash` is reused. The same id with different parameters is refused
  unless `overwrite=true`, which needs approval.
- **Units.** Reflectance 0-1, backscatter dB, metres, K, class codes. NaN is no
  data. Layers are lowercase common names; several dates use `red@YYYYMMDD`.
- **Valid pixels.** `valid("clear")` is finite in the used layers and none of
  nodata, saturated, cloud, cloud_shadow, cirrus, outside_aoi in this stack's
  planes or in `mask_with` stacks. Snow, water and adjacent stay valid.
- **QA presets** by name: `s2-scl`, `landsat-qa-pixel`, `hls-fmask`.
  `dilate_px` grows cloud, shadow and cirrus. Planes are never burned into data.
- Derived stacks hold only their own layers; combining stacks needs the same grid.

## 2. Tools

| Tool | Does | Writes |
|---|---|---|
| `get_satellite_data` | search; `download=true` pulls chips to `.atlas/data/<source>/` (workspace, persistent) | chips |
| `prepare_stack` | chips or rasters to one stack | stack |
| `inspect_stack` | layers, valid %, QA fractions | none |
| `band_math` | expression or `index=` | stack |
| `classify` | rules, thresholds, preset or otsu | stack |
| `change_detect` | difference, ratio, log_ratio_db, transitions | stack |
| `area_table` | class or zonal hectares with valid denominators | table, csv |
| `composite` | per-pixel temporal composite | stack |
| `time_series` | index series, trend, month_anomaly | table, chart |
| `render_map` | PNG with legend, scale bar | figure |
| `export_geotiff` | GeoTIFF, GeoJSON, CSV | export |
| `satellite_guide` | guide topics | none |

Results are at most 1,500 characters and end with `run: rNNNN`. Errors end in a
`Fix:` line. Mutating cases are approval-gated: `overwrite=true`, and
`render_map` or `export_geotiff` with an explicit `path`.
`get_satellite_data` is not gated and has no size limit; windows over 36 MP are
refused as a memory guard, and a call stops at 600 s. Every tool definition the
model sees is at most 1,600 characters.
`spectral_indices` is the legacy `.npy` tool; use `band_math` on stacks.

## 3. Models

Model tools are built from packaged `model.toml` manifests (schema 2), never
from names the model supplies. Fields: `name`, `version`, `tier`
(`baseline`, `learned`, `heuristic`), `status`, `task`, `license`, `citation`,
`weights` (file, url, sha256, size, upstream), `input` (mode single|pair|series,
sensors, bands, unit, resolution, pair rules), `runtime` (`rules` or `onnx`;
`remote` and `torch` parse only), `output`, `params`.

Weights live in the user weights directory (`~/.atlas/weights`) or
`ATLAS_WEIGHTS_DIR`. `atlas models pull NAME` installs and `atlas models verify`
re-hashes them. Weight status is `ready`, `missing`, `corrupt`, `local` (pinned,
no public URL, install with `--from PATH`) or `unpublished` (no pin). A model
whose weight is not installed is hidden from the agent. Heuristic RGB tools join
only with `ATLAS_HEURISTIC_MODELS=1`.

| Model | Tier | Notes and limits |
|---|---|---|
| `cloud_mask` | learned, onnx | OmniCloudMask V4, MIT. Local weight (no public URL); install with `atlas models pull cloud_mask --from PATH --accept-license`. |
| `burn_severity` | baseline, rules | dNBR classes from fixed thresholds; not calibrated per fire. |
| `water_extent` | baseline, rules | Index thresholds; shadow and wet soil confuse it. |
| `sar_flood` | baseline, rules | VV pair in dB, same relative orbit; urban and vegetated areas give false results. |

## 4. Script path

If no typed tool fits, the agent writes `.atlas/analysis/scripts/<name>.py` and runs
it with bash as `"$ATLAS_PYTHON"`. There is no python-exec tool. The bash
environment sets `ATLAS_PYTHON`, `ATLAS_WORKSPACE`, `ATLAS_ANALYSIS_DIR` (the
persistent analysis directory) and `ATLAS_ARTIFACT_ROOT` (the session artifact
root; only a fallback for scripts run outside the agent).
Scripts import from `atlas.geo` (`open_stack`, `workspace`, `eval_expr`,
`class_table`, `render_map`, `write_geotiff`, `finish`). `finish({...})` records
the script's sha256 in `.atlas/analysis/runs.jsonl` and prints `run: rNNNN`.

## 5. Limits

- Stacks and pulls persist across sessions; `atlas geo stacks` lists them.
- No download limit. A pulled window is at most 36 MP (a memory guard); larger
  AOIs are refused.
- A stack is at most 200e6 cells; one UTM zone per AOI.
- Long tools stop at 600 s and say so.
- Per-pixel cloud masking exists only through QA planes or `cloud_mask`.

## 6. Citations and provenance

Every typed result carries a run id. The agent cites the tool name, run id and
artifact path for each number, and the model tool name, version and licence for
model output. `.atlas/analysis/runs.jsonl` has one line per run with parameters and
input hashes; each stack's `stack.json` carries layer sources and `sha256`.
