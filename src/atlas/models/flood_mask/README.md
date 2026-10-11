# flood_mask

Heuristic open-water mask on optical PNG. Not under-cloud SAR flood and not a
trained model. No GPU.

A pixel is water when blue is at least 80 and exceeds red by more than 30 and
green by more than 20.

## Spec

Open-water mask on optical PNG. Not under-cloud SAR flood.

- **Input:** RGB PNG tile, `256×256×3` or `512×512×3` uint8
- **Output:** mask `H×W` uint8 (`0` dry, `1` water), same spatial size as input

Tile large mosaics at 256–512. Windows use the larger declared edge (512) and
are stitched, so the mask stays the original HxW instead of rejecting a
non-tile input.
