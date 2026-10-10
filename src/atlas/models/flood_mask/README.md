# flood_mask

Open-water mask on an optical PNG. Not under-cloud SAR flood.

No weights. A pixel is water when blue is at least 80 and exceeds red by more
than 30 and green by more than 20. Cloud and vegetation stay dry.

## Tool I/O

- **Input:** workspace-relative RGB PNG, `256×256` or `512×512`. Other sizes are
  rejected. The tile is not resized.
- **Output:** same-size mask PNG (`0` dry, `1` water) and JSON with `width`,
  `height`, and `positive_fraction`.
