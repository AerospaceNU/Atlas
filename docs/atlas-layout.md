# Atlas file layout

Project files live in `.atlas/` under the working directory. User files live in
`~/.atlas/`. Path helpers are in `src/atlas/agent/layout.py`.

## Project (`.atlas/` in the working directory)

| Path | What it holds |
|---|---|
| `sessions/<id>/` | One agent session. `meta.json` is the id and model. `transcript.jsonl` is the redacted TUI transcript. No keys. |
| `artifacts/<id>/` | Files for that session. Removing the session removes this directory. |
| `data/` | Data CLI pulls. This is the existing `.atlas/data` default. |

`agent.toml`, `skills/`, and `skills-drafts/` also live in this `.atlas/` directory.

## User (`~/.atlas/`)

| Path | What it holds |
|---|---|
| `keys/` | API keys. The directory is mode `0700` and each file is mode `0600`. Keys are not written into the project. |
| `weights/` | User-level weights. This is the existing `~/.atlas/weights` fallback. |
| `defaults/` | User defaults. |

`ATLAS_WEIGHTS_DIR` overrides the weights location. When a checkout is visible,
`<repo>/weights` is used before `~/.atlas/weights`.
