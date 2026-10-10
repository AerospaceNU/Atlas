# mask_clouds

Per-pixel cloud mask on a true-color PNG tile so later tools can skip junk.

No weights. A pixel is cloud when it is bright and nearly neutral: the channel
mean is at least 200 and `max(R, G, B) - min(R, G, B)` is at most 25.

## Tool I/O

- **Input:** workspace-relative RGB PNG, `256×256` or `512×512`. Other sizes are
  rejected. The tile is not resized.
- **Output:** same-size mask PNG (`0` clear, `1` cloud) and JSON with `width`,
  `height`, and `positive_fraction`.
