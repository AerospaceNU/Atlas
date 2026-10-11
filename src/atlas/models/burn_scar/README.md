# burn_scar

Heuristic burned-area mask from dark scars in true color. Not NBR severity
and not a trained model. No GPU.

A pixel is burn only when red is strictly greater than green and blue, green
is below 90, and luma is at least 25 and below 80. Black and dark grey stay
unburned.

## Spec

Burned-area mask from dark scars in true color. Not NBR severity.

- **Input:** RGB PNG tile, `256×256×3` or `512×512×3` uint8
- **Output:** mask `H×W` uint8 (`0` unburned, `1` burn), same spatial size as input

Tile large mosaics at 256–512. Windows use the larger declared edge (512) and
are stitched, so the mask stays the original HxW instead of rejecting a
non-tile input.
