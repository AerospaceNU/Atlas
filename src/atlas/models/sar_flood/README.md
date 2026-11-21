# sar_flood

Flood extent from a Sentinel-1 VV pair in dB.

Baseline rule pack: verify against imagery. It is a fixed-threshold rule, not a trained model.

## Inputs
Pair stack with layer `vv` in dB on two dates (pre = earliest, post = latest), same grid, same relative orbit and orbit direction (both are checked).

## Formulas
`diff = vv_post - vv_pre`. The continuous diff is not an output layer; use `band_math` on the input stack to get it.

## Classes
| code | class | rule |
|---|---|---|
| 1 | permanent_water | pre.vv < t_water and post.vv < t_water |
| 2 | flooded | post.vv < t_water and diff < t_change |
| 3 | dry | post.vv >= t_water or diff >= t_change |

First match wins, so permanent water is taken before flooded.

## Parameters
`t_water` -15 dB (-30 to -5), `t_change` -3 dB (-10 to 0).

## Known limits
- Urban and vegetated areas hide flooding through double-bounce backscatter that stays bright.
- Smooth surfaces (tarmac, sand, ice) and radar shadow read as water.
- Wind-roughened water is missed because it stays bright.
- Wet snow lowers backscatter and can read as water.
- Fixed thresholds on RTC gamma0 dB, with no per-scene tuning.
- The pair must share relative orbit and orbit direction; mixed geometry is refused.

## References
- No single source; fixed backscatter and change thresholds follow common SAR flood-mapping practice.

## Example
`sar_flood(stack="s1_pair")` or `sar_flood(stack="s1_pair", t_water=-16)`.
