# cloud_mask (OmniCloudMask V4)

Cloud and cloud-shadow segmentation. Learned model (two U-Nets, RegNetY-004 and
EdgeNeXt-small, logits averaged) converted to one ONNX file. Runs on CPU through
onnxruntime (`uv sync --extra onnx`).

- Input: a prepared stack with `red`, `green`, `nir` reflectance at 5-50 m, at least 32 px
  a side. Sentinel-2 `nir` must be B8A, Landsat 8/9 `nir` B5. A red-green-blue stack, a 1 m
  stack and a 16x16 stack are refused with the fix.
- Output: class layer `cloud_mask` (0 clear, 1 thick_cloud, 2 thin_cloud, 3 cloud_shadow)
  and QA planes `cloud` (1), `cirrus` (2), `cloud_shadow` (3). NaN where every input band is
  NaN. Upstream labels no data as class 0; Atlas counts NaN as not valid.
- Per-input z-score and the ensemble are inside the ONNX graph. Pixels that are exactly zero
  in a band are excluded from that band's statistics.
- Citation: Wright et al. (2025), Remote Sensing of Environment 322, 114694,
  doi:10.1016/j.rse.2025.114694. Upstream repo `NickWright/OmniCloudMask`, revision
  `66b606a7199bd80b499f8a871930a7d5c7ef94eb`.

## Licence

V4 weights are MIT (Government of Western Australia, DPIRD). The V1 weights in the same
upstream repository are CC BY-NC 4.0 (non-commercial) and are not used here; do not swap
them in. See `NOTICE` for the MIT, RegNetY and EdgeNeXt notices.

## Weights

Local weight, hosted on the machine that uses it: `url` in `model.toml` is empty (there is
no public URL) and the status is `local` until installed. The sha256 pin is the hash of the
maintainer conversion and is always checked. The file lands in the user weight store
(`~/.atlas/weights`, or `ATLAS_WEIGHTS_DIR`). The tool registers whenever that verified file is
present and onnxruntime imports. To install:

    uv run python tools/models/convert_omnicloudmask.py WEIGHTS_DIR omnicloudmask_v4.onnx
    atlas models pull cloud_mask --from omnicloudmask_v4.onnx

The converter needs torch in a throwaway environment (command in its docstring). A
different torch, onnx or onnxruntime version can produce different bytes, so compare
the printed sha256 with the pin before pulling.

## Known limits

- Thin cloud and cirrus: the least reliable class. The test scene had no thin_cloud pixels,
  so the cirrus plane is untested on real data.
- Shadow is inferred from appearance, not sun geometry; terrain shadow and dark burn scars
  are confused with it.
- Snow, ice and bright bare ground can be labelled cloud; dark water can be labelled shadow.
- Tile-extent dependence: z-score uses the statistics of whatever the network sees.
  Inputs up to 1,048,576 px run in one pass; larger ones run in 1024 px tiles with a
  128 px margin (core 768 px), each tile normalised on its own.
- Pixels that are zero in every band are treated as no data.

## Measured

One real Sentinel-2 L2A chip, 1024 x 1024 at 10 m (B04, B03, B8A; 11,447 px are zero in every
band and enter as NaN), macOS CPU, 8 threads, onnxruntime 1.32.0, upstream omnicloudmask 1.8.1,
converter output sha256 `d31d7a9a...8e43` (re-converted from the safetensors with torch: same
hash, self-parity ok):

| Check | Result |
|---|---|
| 512 px chips vs upstream omnicloudmask 1.8.1 | 0 of 1,048,576 px differ (class counts identical) |
| Atlas tool, one 1024 px pass | 1.3-1.5 s of tool time (8 threads); peak RSS about 745 MB including Python |
| Same pass vs upstream's own tiling | 99.75% agree (2,619 px differ by design: different tiles, different z-score) |
| 2048 x 2048 mosaic (flipped copies), Atlas tiled 1024/128 vs one full-extent pass | 99.24% agree (31,873 px differ) |
| 2048 x 2048, tiled 1024/128 (9 tiles) | 9.6 s; peak RSS about 1.1 GB |

Earlier runs on the same scene gave 98.8% agreement at 256 px non-overlapping tiles. There
is no fixed pass bar; the figures show how much tile extent can move the answer. The scene
held no thin_cloud pixels.
