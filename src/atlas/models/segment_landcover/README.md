# segment_landcover

Per-pixel land-cover map on an RGB satellite scene. The model does **not**
resize the input. A 4000×3000 mosaic stays 4000×3000; each pixel gets one of
`bare`, `built-up`, `water`, `vegetation`, `cloud`.

This demo uses nearest RGB centroid (CPU). Swap `infer.py` for a U-Net later;
the tool contract stays the same.

## Layout

Copy this directory to `src/atlas/models/<name>/`. Keep `plugin.toml`.
The agent loads every `plugin.toml` at boot and registers `name` as a tool.

Weights are **not** in `src/`. `plugin.toml` stores a key relative to the
weights root:

```toml
weight = "segment_landcover/model.json"
```

Resolved as `ATLAS_WEIGHTS_DIR` or `<repo>/weights`. Never put a machine
absolute path in the manifest. The agent never runs `acquire.sh`, `train.py`,
or `export.py`.

## Train

Default acquire pulls Sentinel-2 L2A true-color previews from Planetary
Computer (weak labels by place). `--synthetic` writes solid-color PNGs.

```bash
bash src/atlas/models/segment_landcover/acquire.sh
uv run python src/atlas/models/segment_landcover/train.py
uv run python src/atlas/models/segment_landcover/export.py
uv run python src/atlas/models/segment_landcover/preview.py
```

`preview.py` runs the tool at **native** size and writes a sheet of
RGB | overlay thumbnails (thumbnails are display-only).

## Tool I/O

- **Input:** workspace-relative RGB PNG, any H×W
- **Output:** same-size palette `*_labels.png`, RGB `*_overlay.png`, and JSON
  with `width`, `height`, and per-class pixel fractions
