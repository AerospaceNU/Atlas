# burn_scar

Burned-area mask from dark red-brown scars in true color. Not NBR severity.

No weights. A pixel is burn when its luma is below 80, red is at least green
and blue, and green is below 90. Dark blue water and green vegetation stay
unburned.

## Tool I/O

- **Input:** workspace-relative RGB PNG, `256×256` or `512×512`. Other sizes are
  rejected. The tile is not resized.
- **Output:** same-size mask PNG (`0` unburned, `1` burn) and JSON with
  `width`, `height`, and `positive_fraction`.
