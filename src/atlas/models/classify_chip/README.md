# classify_chip

Template model-as-tool plugin. Cheap nearest-centroid on mean RGB so the
layout is real; replace `train.py` / `infer.py` when you ship a CNN.

## Layout

Copy this directory to `src/atlas/models/<name>/`. Keep `plugin.toml`.
The agent loads every `plugin.toml` at boot and registers `name` as a tool.

Weights are **not** in `src/`. `plugin.toml` stores a key relative to the
weights root:

```toml
weight = "classify_chip/model.json"
```

Resolved as `ATLAS_WEIGHTS_DIR` or `<repo>/weights` (see `atlas.models.base.weights_root`).
Never put a machine absolute path in the manifest.

The agent never runs `acquire.sh`, `train.py`, or `export.py`.

## Train a demo weight

From the repo root:

```bash
bash src/atlas/models/classify_chip/acquire.sh
uv run python src/atlas/models/classify_chip/train.py
uv run python src/atlas/models/classify_chip/export.py
```

Paste the printed sha256 into `plugin.toml` when you pin an official export.
Empty `sha256` means “file must exist”; a non-empty value fails closed on mismatch.

## Tool I/O

- **Input:** workspace-relative RGB PNG, resized to `224×224×3`
- **Output:** `artifacts/classify_chip.json` with `path`, `label`, and `scores`
  (`bare`, `built-up`, `water`, `vegetation`, `cloud`)
