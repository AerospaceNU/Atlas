# water_extent

Open-water mask from MNDWI on one prepared date.

Baseline rule pack: verify against imagery. It is a fixed-threshold rule, not a trained model.

## Inputs
Single-date stack from `prepare_stack` with layers `green` and `swir16`, reflectance. Sources: sentinel2 (B11), landsat8/9 (SR_B6), HLS.

## Formulas
`mndwi = (green - swir16) / (green + swir16)`; a zero denominator gives NaN. The continuous index is not an output layer; use `band_math` on the input stack to get it. An NDWI (green, nir) variant is not shipped: the manifest cannot declare optional bands. Use `band_math` plus `classify` for NDWI.

## Classes
| code | class | rule |
|---|---|---|
| 1 | land | mndwi <= threshold |
| 2 | water | mndwi > threshold |

## Parameters
`threshold` 0.0 (-0.5 to 0.5).

## Known limits
- Threshold 0.0 is a textbook default, not tuned to any scene; adjust it and check against imagery.
- Cloud shadow, dark asphalt, burn scars and snow can read as water.
- Turbid or vegetated water can be missed.
- Mixed shoreline pixels at 20 m resolution are classified by their blend, not their majority cover.
- No split between permanent and seasonal water; one date gives one extent.

## References
- Xu, H. (2006). Int. J. Remote Sensing 27(14), 3025-3033.

## Example
`water_extent(stack="s2_20240601")` or `water_extent(stack="s2_20240601", threshold=0.1)`.
