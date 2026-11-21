# burn_severity

Burn severity from dNBR on a pre/post stack. The result states that the threshold set is fire- and sensor-specific and not calibrated.

Baseline rule pack: verify against imagery.

## Inputs
Pair stack from `prepare_stack` with layers `nir` and `swir22` on two dates (pre = earliest, post = latest), reflectance, same grid. Sources: sentinel2 (B8A, B12), landsat8/9 (SR_B5, SR_B7), HLS.

## Formulas
`NBR = (nir - swir22) / (nir + swir22)`; `dnbr = NBR_pre - NBR_post`. A zero denominator gives NaN. The continuous dNBR is not an output layer; compute it with `band_math` on the input stack if needed.

## Classes
| code | class | rule |
|---|---|---|
| 1 | regrowth | dnbr < t_regrowth |
| 2 | unburned | dnbr < t_unburned |
| 3 | low | dnbr < t_low |
| 4 | moderate_low | dnbr < t_mod_low |
| 5 | moderate_high | dnbr < t_mod_high |
| 6 | high | dnbr >= t_mod_high |

First match wins. Cloudy or NaN pixels are NaN, never a class.

## Parameters
`t_regrowth` -0.10, `t_unburned` 0.10, `t_low` 0.27, `t_mod_low` 0.44, `t_mod_high` 0.66 (each -1 to 2). They must stay strictly increasing; an unordered override silently collapses classes.

## Known limits
- Threshold set is fire- and sensor-specific, not calibrated; the defaults are the USGS/FIREMON dNBR table, originally derived from Landsat.
- Pre and post scenes must be the same season and phenology; vegetation change unrelated to fire shows up as dNBR.
- Cloud and shadow are masked only through the QA planes of the stack; unmasked haze or shadow biases dNBR.
- Sentinel-2 NIR is B8A (20 m) rather than B08 (10 m); this shifts dNBR relative to Landsat-derived thresholds.
- Dark soil, water and burned-then-flooded ground give spurious dNBR.
- No field-validated Composite Burn Index (CBI) calibration; thresholds t_regrowth < t_unburned < t_low < t_mod_low < t_mod_high must stay strictly increasing.

## References
- Key, C.H. and Benson, N.C. (2006). FIREMON, RMRS-GTR-164-CD, USDA Forest Service.

## Example
`burn_severity(stack="pairstack")` or `burn_severity(stack="pairstack", t_mod_high=0.5)`.
