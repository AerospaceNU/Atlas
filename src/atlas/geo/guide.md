# Satellite analysis guide

Served by satellite_guide(topic) and `atlas geo guide TOPIC`. Topics: workflow, stack, masks, areas, models, render, recipes, scripts, limits.

## topic: workflow
Call order. Each step reads the previous result; do not skip prepare_stack.
1. get_satellite_data(assets=[...], select=..., polygon=...) searches; add download=true to pull chips into data/<source>/.
2. prepare_stack(scenes="last_pull", name=...) builds ONE co-registered physical-unit stack.
3. inspect_stack(stack=...) shows layers, valid %, QA fractions.
4. band_math, classify, change_detect, area_table, time_series, composite, or a model tool. Each writes a NEW stack or table.
5. render_map(stack=..., layers=[...]) to look at the result.
6. Report.
Errors end in a "Fix:" line. Run the named call verbatim. Never retry the same call unchanged.
Report: units, valid fraction, hectares with aoi_px and valid_px denominators, model limits, citations, and for each number the tool name, run id (rNNNN) and artifact path. Never read areas off an image; quote the tool's numbers.
Topics: stack, masks, areas, models, render, recipes, scripts, limits.

## topic: stack
A stack is a directory analysis/stacks/<id>/ with stack.json, data.npy (float32, layers x H x W), optional qa.npy and preview.png. Other analysis dirs: analysis/{figures,tables,exports,scripts} and analysis/runs.jsonl.
Layers: lowercase common names (red, nir, swir22, vv, scl, landcover). Several dates: red@YYYYMMDD. Units: reflectance 0-1, backscatter dB, metres, K, class codes. NaN means no data.
Stacks are immutable. A tool that changes anything writes a new stack named <parent>-<op>-<hash>, or name=... if given. Same name with different parameters is refused unless overwrite=true (approval needed).
Derived stacks hold only their own layers. To keep inputs, use keep_layers on a model tool. Combining stacks needs the same grid.
Raw pulls stay in data/<source>/ with a .meta.json sidecar; prepare_stack finds chips by sidecar.
Stacks do NOT survive a new session. List them: atlas geo stacks.
inspect_stack(stack="latest") opens the newest stack.

## topic: masks
Clouds and bad pixels are QA planes in the stack, never burned into the data. Valid pixels for areas: finite in the used layers AND none of nodata, saturated, cloud, cloud_shadow, cirrus, outside_aoi. Snow, water and adjacent stay valid.
mask="clear" (default) applies that rule. mask="none" ignores QA. A list of semantics picks exactly those.
mask_with=[stack ids] adds the QA planes of other stacks on the same grid, e.g. the product of cloud_mask.
prepare_stack: qa="auto" builds planes from SCL, QA_PIXEL or Fmask; dilate_px=2 grows cloud, shadow and cirrus on Sentinel-2.
Per-pixel cloud masking exists only through QA planes or the cloud_mask model. Heuristic RGB tools (mask_clouds, burn_scar, flood_mask) are previews, not trained models.
Always report valid_fraction. Below about 50 percent valid, say the estimate is weak. Dates below 20 percent valid are flagged in time_series.
Polygon AOI adds outside_aoi, excluded from valid_px but counted in aoi_px.

## topic: areas
area_table(stack, layer, zones, mask) returns per class: px, ha, pct of valid. Hectares = px x pixel_area_m2 / 1e4. The result states aoi_px and valid_px; quote both.
zones takes either an inline or file GeoJSON, or {"stack": id, "layer": name}: a categorical layer on the same grid, giving a class-by-zone cross-tab. Example: flood extent by land cover is area_table(stack="flood", layer="sar_flood", zones={"stack":"wc", "layer":"landcover"}).
Large tables show 12 class or 20 zone rows, then "+N more: <csv path>". Read the csv with read_file; never guess hidden rows.
For a continuous layer, area_table returns a zonal statistic (stat=mean|median|min|max|sum).
Percent of valid is the share of usable pixels; percent of AOI uses aoi_px. Say which.
A class absent from the table has zero valid pixels, not unknown area.
classify(stack, layer, rules|thresholds+labels|preset|method="otsu") makes the categorical layer first. change_detect(stack, layer, method=difference|ratio|log_ratio_db|transitions) compares dates.

## topic: models
Model tools run only on prepared stacks, never on raw chips. Tool input: stack, name, keep_layers, params, mask_with. A model tool refuses a stack that does not fit its input contract and names the fix; follow it.
Cite the model's tool name, version, run id and licence. Report its known limits. Heuristic RGB tools are previews, not models.
{{MODELS}}
Hidden models are not callable. You cannot pull weights; tell the user to run the pull command shown, then start a new session.

## topic: render
render_map(stack, layers=[one layer or three for RGB], composite, stretch, colormap, overlay, compare, date, attach) writes a PNG under analysis/figures/ plus a .json sidecar.
Stretch is the 2-98 percentile of valid pixels; the result states the range. overlay=<class layer> outlines classes over a base. compare=<stack> draws two panels on the same grid.
attach=true shows you the PNG. If you cannot see images, or attach=false, the result carries a 12x24 text grid and legend with hectares instead; read those.
Legend areas, scale bar and north arrow are drawn by the tool. Images are 800 px on the longest side by default and under 250 KB.
Use render_map to check that a product looks plausible; do not measure from it. Take numbers from area_table.
export_geotiff(stack, layers, format=geotiff|geojson|csv) writes under analysis/exports/. An explicit path outside it needs approval.
time_series and area_table can add a chart PNG with chart=true.

## topic: recipes
(a) Burn severity, Sentinel-2:
get_satellite_data(data_type="optical", sources=["sentinel2"], assets=["nir","swir22"], select="clearest", download=true) for a pre-fire and a post-fire window (two calls). Then prepare_stack(scenes="last_pull", name="fire"); burn_severity(stack="fire"); area_table(stack=<product id>, layer="burn_severity"). Report moderate_low and above as burned; state the class set is not calibrated.
(b) SAR flood with land cover:
get_satellite_data(data_type="sar", assets=["vv"], orbit="8:ascending", select="oldest" then "newest") for a dry and a flood date on the SAME orbit; then sources=["worldcover"], assets=["landcover"]. prepare_stack(scenes="last_pull", name="flood"); sar_flood(stack="flood"); area_table(stack=<product id>, layer="sar_flood", zones={"stack":"flood", "layer":"landcover"}) if land cover shares the grid, else prepare both into one stack.
(c) NDVI trend:
get_satellite_data(assets=["red","nir","scl"], select="per_month", download=true); prepare_stack(scenes="last_pull", name="ts", qa="auto"); time_series(stack="ts", index="ndvi", trend="month_anomaly"). month_anomaly removes the seasonal cycle; plain ols does not. Report slope per year and n_dates.

## topic: scripts
Typed tools first. If none fits, write a script with write_file to analysis/scripts/<name>.py, then run it with bash: "$ATLAS_PYTHON" analysis/scripts/<name>.py. Write only under analysis/. Print at most 40 lines. End with finish({...}), which records the script sha256 and prints run: rNNNN.
Imports: from atlas.geo import open_stack, workspace, eval_expr, class_table, render_map, write_geotiff, finish
1. Hectares of dNBR above 0.27 on stack fire:
```python
import numpy as np
from atlas.geo import open_stack, eval_expr, finish
s = open_stack("fire"); a, b = s.dates()[0], s.dates()[-1]
nbr = lambda d: eval_expr("(n - s) / (n + s)", {"n": s.band("nir", date=d), "s": s.band("swir22", date=d)})
dnbr = nbr(a) - nbr(b); v = s.valid("clear") & np.isfinite(dnbr)
px = int(((dnbr > 0.27) & v).sum()); ha = px * s.pixel_area_m2 / 1e4
print(f"burned_ha={ha:.4g} px={px} valid_px={int(v.sum())} aoi_px={v.size}")
finish({"burned_ha": ha, "burned_px": px, "valid_px": int(v.sum())})
```
2. Ratio layer saved as GeoTIFF:
```python
from atlas.geo import open_stack, eval_expr, write_geotiff, workspace, finish
s = open_stack("fire"); d = s.dates()[-1]
r = eval_expr("nir / swir22", {"nir": s.band("nir", date=d), "swir22": s.band("swir22", date=d)})
out = s.derive(op="ratio", name="fire-ratio", overwrite=True, params={}, layers=[{"name": "ratio", "kind": "index", "unit": "1"}], data=r[None])
p = workspace() / "analysis" / "exports" / "fire_ratio.tif"; p.parent.mkdir(parents=True, exist_ok=True)
info = write_geotiff(out, ["ratio"], p)
finish({"bytes": info["bytes"]}, outputs=[p])
```

## topic: limits
Window cap: 36 MP per pulled window; larger AOIs are refused with advice to shrink the box. Stacks are capped at 200e6 cells. Long tools stop at 600 s and say so.
Per-pixel cloud masking only through QA planes or cloud_mask. Heuristic RGB tools (mask_clouds, burn_scar, flood_mask, segment_landcover, rgb_change) work on 8-bit PNG previews and are not trained models.
nir is B8A (20 m) on Sentinel-2; nir_broad is B08 (10 m). Landsat nir is B5. The result names the real band.
SAR: urban and vegetated areas give double-bounce returns that fool fixed-threshold flood rules. Use the same relative orbit and direction for a pair. Sentinel-1 is shown in dB.
Landsat GeoTIFFs use PixelIsPoint; the reader shifts half a pixel and logs it in the layer processing.
Fixed thresholds (dNBR classes, water, flood) are fire-, scene- and sensor-specific, not calibrated. Say so.
One UTM zone per AOI. FIRMS, GIBS and browse sources are search-only here; use atlas data for them.
Stacks do not persist across sessions. Downloads are bounded by max_download_mb (default 2048).
