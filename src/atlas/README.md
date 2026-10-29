# atlas

Satellite scene search, consolidation, and a bounded tool-using agent.

## `agent`

The tool loop over one local artifact directory.

| Module | What it holds |
|---|---|
| `artifacts` | The workspace sandbox (`LocalArtifactStore`) and image metadata. Paths stay relative to the store root. |
| `contracts` | Tool and model types, the tool registry, and discovery of tool classes. |
| `runtime` | The bounded agent loop: call a tool, record the step, stop at the call budget. A run starts with the system prompt, then project instructions when the workspace root has `AGENTS.md`. |
| `instructions` | Load `AGENTS.md` from the workspace root. Nested directories are not searched, and a closer file does not win over the root file. The text is delimited project guidance after the system prompt, not part of it. A missing file is silent. An unreadable or malformed file, including a FIFO, is skipped. Loading it cannot add tools or change the sandbox. |
| `system_prompt` | The package system prompt (`system_prompt.md`). A workspace file cannot replace it. The text is cached for the life of the process, so an edit applies on the next start. |
| `layout` | Project `.atlas/` and user `~/.atlas/` paths. `ATLAS_HOME` overrides the user root. The directory list is `docs/atlas-layout.md`. |
| `session` | A multi-turn conversation on top of that loop, plus the `python -m atlas.agent` entry. New sessions and artifacts are created under the project layout. |
| `skills` | Load Agent Skills from the workspace read root and activate one with `use_skill`. `.atlas/skills-drafts` is not loaded. Home skills stay off unless `ATLAS_HOME_SKILLS=1` or `home_skills = true` in `~/.atlas/config.toml`. |
| `analysis_skill` | Save a finished analysis as a draft under `.atlas/skills-drafts/`. `/enable-skill` shows the description and body hash before moving it into `.atlas/skills`. |
| `model` | The OpenRouter chat backend. |
| `tools/` | Concrete tools. `read_file` reads text in the workspace and does not run it. |
| `skill_author` | Write an Agent Skill directory (`SKILL.md`) that matches the Agent Skills spec. `normalize_skill_name` (`normalize_name`) and `is_within` are the shared name and path checks. |
| `tools/author_skill` | Opt-in tool that writes one draft under `.atlas/skills-drafts`. It does not enable the skill or change the default registry. |
| `__main__` | Module entry that starts a session. |

## `cli`

The `atlas` command.

| Module | What it holds |
|---|---|
| `__init__` | Argument routing. No args opens the TUI; `atlas data` and a bare satellite name stay on search. |
| `catalog` | Source names, kind aliases, and the catalog listing. |
| `data` | Search: pick sources, run a pull, print scenes. |
| `download` | Fetch scene assets (previews and thumbnails are refused). Geographic, Web Mercator, and WGS84 UTM COGs are windowed to the request box. |
| `progress` | Per-source search progress for the data command. |
| `tui` | Find or build the Rust `atlas-tui` binary and replace this process with it. |

## `compile`

Agent-agnostic consolidation. Same path for interactive use and batch jobs.

| Module | What it holds |
|---|---|
| `pipeline` | `consolidate`: search, select, optionally render mosaics. |
| `aggregate` | Fan a request out across sources. One source's error does not abort the rest. A rate-limited source (HTTP 429) is retried while the others continue. Each source has a time budget. Recorded HTTP errors are the status and reason only. |
| `select` | Metadata filters: cloud, id dedup, clearest-then-newest, a cheap coverage estimate. |
| `mosaic` | Planetary Computer mosaic render, written as `<source>.png`. |
| `product` | Result types: per-source scenes, the catalog, and a rendered mosaic. |

## `data`

STAC and related search clients. `search` is the only required method, and it is async.

| Module | What it holds |
|---|---|
| `base` | Shared models: bbox, polygon AOI, scene, pull request/result, and the client interface. |
| `cog` | Read a COG window with HTTP range requests instead of the whole granule. BigTIFF and LZW are later work. |
| `registry` | Stable source names and `sources()` / `all_sources()`. |
| `stac` | Generic STAC API client. |
| `planetary_computer` | Microsoft Planetary Computer search and asset signing. |
| `sentinel1`, `sentinel2`, `landsat8`, `landsat9` | Those collections on Planetary Computer. Landsat 8 and 9 share a collection and are split by platform. |
| `pc_more` | Extra Planetary Computer collections (ASTER, ALOS, DEM, WorldCover, GOES CMI, MODIS, precipitation). |
| `modis_base` | Shared NASA CMR client for MODIS granules. A 401 or 403 names `EARTHDATA_TOKEN` as missing or rejected and does not print the value. Downloads send the token as a bearer header to `https` hosts under `nasa.gov` only. |
| `modis_active_fire`, `modis_land_cover`, `modis_surface`, `modis_vegetation` | Those MODIS products. |
| `hls_landsat`, `hls_sentinel` | Harmonized Landsat–Sentinel. |
| `naip` | NAIP aerial imagery. |
| `cop_dem` | Copernicus DEM (30 m and 90 m). |
| `earth_search` | Element84 Earth Search (Sentinel-1 GRD, Sentinel-2 L2A). |
| `cdse` | Copernicus Data Space: Sentinel-3, Sentinel-5P, burnt area. A 401 or 403 names `CDSE_USERNAME` and `CDSE_PASSWORD` as missing or rejected. CDSE is set with the env var only. The value is not printed. Downloads use each asset's OData `alternate.https` link with a bearer token from the CDSE identity service, sent only to `download.dataspace.copernicus.eu` and `zipper.dataspace.copernicus.eu`. |
| `cmr_stac` | NASA CMR STAC: GEDI, ICESat-2, SMAP, VIIRS. A 401 or 403 names `EARTHDATA_TOKEN` as missing or rejected and does not print the value. Downloads send the token as a bearer header to `https` hosts under `nasa.gov` only. |
| `digitalearth` | Digital Earth Australia and Africa Sentinel-2. |
| `inpe` | CBERS-4 and Amazonia-1. |
| `maxar_opendata` | Maxar Open Data. |
| `firms` | NASA FIRMS active-fire detections. `FIRMS_MAP_KEY` comes from the session, the environment, or `~/.atlas/keys`. A missing key names `FIRMS_MAP_KEY` and says to set it with `/key firms` or the env var. A rejected key names the variable and the status. The value is not printed. A dateline box is two area requests. |
| `gibs` | NASA GIBS browse imagery (true color, night lights, flood, snow, thermal). A dateline box is two WMS GetMap requests. |
| `goes`, `himawari` | Geostationary imagery. GOES picks a satellite per side of a dateline box. Himawari lists the full-disk slot once when either side meets the disk. |

## `models`

Plugins the agent can call. Each plugin is a subpackage with a `plugin.toml`.

| Module | What it holds |
|---|---|
| `base` | Plugin manifest, weights loading, and shared image helpers. |
| `registry` | Find every `plugin.toml` and register its tool. |
| `binary_mask` | Heuristic 0/1 masks. Large mosaics are tiled, then stitched. |
| `unet_cpu` | CPU U-Net runtime: JSON weight contract and tiled mosaic inference. |
| `lgbm_runtime` | LightGBM runtime (optional `lightgbm` extra). JSON weight contract, raw 0-255 RGB features. |
| `segment_landcover/` | Per-pixel land-cover labels on an RGB scene. See that folder's README for train and export. |
| `rgb_change/` | Before/after heatmap and blob polygons on an RGB pair. |
| `mask_clouds/` | Heuristic cloud mask. Mosaics are tiled at 512 and stitched. |
| `burn_scar/` | Heuristic burn-scar mask. Mosaics are tiled at 512 and stitched. |
| `flood_mask/` | Heuristic open-water mask. Mosaics are tiled at 512 and stitched. |
| `unet_water/` | Open-water CPU U-Net plugin. |
| `lgbm_clouds/` | Cloud LightGBM plugin. |
