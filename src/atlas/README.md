# atlas

Satellite scene search, consolidation, and a bounded tool-using agent.

## `agent`

The tool loop over one local artifact directory.

| Module | What it holds |
|---|---|
| `artifacts` | The workspace sandbox (`LocalArtifactStore`) and image metadata. Paths stay relative to the store root. |
| `contracts` | Tool and model types, the tool registry, and discovery of tool classes. |
| `runtime` | The bounded agent loop: call a tool, record the step, stop at the call budget. |
| `session` | A multi-turn conversation on top of that loop, plus the `python -m atlas.agent` entry. |
| `model` | The OpenRouter chat backend. |
| `tools/` | Concrete tools. `read_file` reads text in the workspace and does not run it. |
| `__main__` | Module entry that starts a session. |

## `cli`

The `atlas` command.

| Module | What it holds |
|---|---|
| `__init__` | Argument routing. No args opens the TUI; `atlas data` and a bare satellite name stay on search. |
| `catalog` | Source names, kind aliases, and the catalog listing. |
| `data` | Search: pick sources, run a pull, print scenes. |
| `download` | Fetch scene assets (previews and thumbnails are refused). |
| `progress` | Per-source search progress for the data command. |
| `tui` | Find or build the Rust `atlas-tui` binary and replace this process with it. |

## `compile`

Agent-agnostic consolidation. Same path for interactive use and batch jobs.

| Module | What it holds |
|---|---|
| `pipeline` | `consolidate`: search, select, optionally render mosaics. |
| `aggregate` | Fan a request out across sources. One source's error does not abort the rest. |
| `select` | Metadata filters: cloud, id dedup, clearest-then-newest, a cheap coverage estimate. |
| `mosaic` | Planetary Computer mosaic render, written as `<source>.png`. |
| `product` | Result types: per-source scenes, the catalog, and a rendered mosaic. |

## `data`

STAC and related search clients. `search` is the only required method, and it is async.

| Module | What it holds |
|---|---|
| `base` | Shared models: bbox, scene, pull request/result, and the client interface. |
| `registry` | Stable source names and `sources()` / `all_sources()`. |
| `stac` | Generic STAC API client. |
| `planetary_computer` | Microsoft Planetary Computer search and asset signing. |
| `sentinel1`, `sentinel2`, `landsat8`, `landsat9` | Those collections on Planetary Computer. Landsat 8 and 9 share a collection and are split by platform. |
| `pc_more` | Extra Planetary Computer collections (ASTER, ALOS, DEM, WorldCover, GOES CMI, MODIS, precipitation). |
| `modis_base` | Shared NASA CMR client for MODIS granules. |
| `modis_active_fire`, `modis_land_cover`, `modis_surface`, `modis_vegetation` | Those MODIS products. |
| `hls_landsat`, `hls_sentinel` | Harmonized Landsat–Sentinel. |
| `naip` | NAIP aerial imagery. |
| `cop_dem` | Copernicus DEM (30 m and 90 m). |
| `earth_search` | Element84 Earth Search (Sentinel-1 GRD). |
| `usgs` | USGS Landsat Collection 2. |
| `cdse` | Copernicus Data Space: Sentinel-3, Sentinel-5P, burnt area. |
| `cmr_stac` | NASA CMR STAC: GEDI, ICESat-2, SMAP, VIIRS. |
| `digitalearth` | Digital Earth Australia and Africa Sentinel-2. |
| `inpe` | CBERS-4 and Amazonia-1. |
| `maxar_opendata` | Maxar Open Data. |
| `firms` | NASA FIRMS active-fire detections. |
| `gibs` | NASA GIBS browse imagery (true color, night lights, flood, snow, thermal). |
| `goes`, `himawari` | Geostationary imagery. |

## `models`

Plugins the agent can call. Each plugin is a subpackage with a `plugin.toml`.

| Module | What it holds |
|---|---|
| `base` | Plugin manifest, weights loading, and shared image helpers. |
| `registry` | Find every `plugin.toml` and register its tool. |
| `segment_landcover/` | Per-pixel land-cover labels on an RGB scene. See that folder's README for train and export. |
