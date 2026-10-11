# mask_clouds

Heuristic per-pixel cloud / valid mask on true-color PNG so later tools skip
junk. Not a trained model. No GPU. Spectral / SAR / foundation models are out
of scope.

A pixel is cloud when the channel mean is at least 200 and
`max(R, G, B) - min(R, G, B)` is at most 25.

## Spec

Per-pixel cloud / valid mask on true-color PNG so later tools skip junk.

- **Input:** RGB PNG tile, `256×256×3` or `512×512×3` uint8
- **Output:** mask `H×W` uint8 (`0` clear, `1` cloud), same spatial size as input

Tile large mosaics at 256–512; do not run the mask on a full mosaic in one
pass. Windows use the larger declared edge (512). Each window is masked and
stitched, so a mosaic that is not itself 256 or 512 still returns a mask at
the original HxW. Smaller images run as one window.
