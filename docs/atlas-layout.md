# Atlas file layout

Project files live in `.atlas/` under the working directory. User files live in
`~/.atlas/`. Path helpers are in `src/atlas/agent/layout.py`.

These are the directories Atlas already used. Sessions, artifacts, data pulls,
skills, and skill drafts were already under the project `.atlas/` directory.
Keys, weights, and defaults were already under the user directory. Nothing is
read from an older location outside `.atlas/`.

`AGENTS.md` stays at the workspace root. `.agents/skills` stays beside
`.atlas`, not inside it. A data pull writes `manifest.json` in its output
directory, which defaults to `.atlas/data`.

## Project (`.atlas/` in the working directory)

| Path | What it holds |
|---|---|
| `sessions/<id>/` | One agent session. `meta.json` is the id and model. No keys. |
| `artifacts/<id>/` | Files for that session. Removing the session removes this directory. |
| `data/` | Data CLI pulls. This is the existing `.atlas/data` default. `manifest.json` is the pull manifest. |
| `skills/` | Vetted skills. |
| `skills-drafts/` | Draft skills. They are not loaded until a person enables one. |
| `agent.toml` | Workspace model and tool-call budget. |

## User (`~/.atlas/`)

| Path | What it holds |
|---|---|
| `keys/` | API keys. The directory is mode `0700` and each file is mode `0600`. Keys are not written into the project. |
| `weights/` | User-level weights. This is the existing `~/.atlas/weights` fallback. |
| `defaults/` | User defaults. |
| `skills/` | User skills, loaded only when home skills are on. |
| `skills-drafts/` | User skill drafts. |
| `config.toml` | `home_skills = true` turns user skills on when `ATLAS_HOME_SKILLS` is unset. |

`ATLAS_HOME` overrides this user root. Set it to the directory that would
otherwise be `~/.atlas`, not to the parent home. `ATLAS_HOME_SKILLS` still
only turns user skills on. The skills and `config.toml` it refers to are
read from that same root. `.agents/skills` does not move with `ATLAS_HOME`.

`ATLAS_WEIGHTS_DIR` overrides the weights location. When a checkout is visible,
`<repo>/weights` is used before the user weights directory.

## Windows

The project root is `.atlas\` in the working directory, for example
`D:\work\atlas\.atlas\data`. The user root is `%USERPROFILE%\.atlas`, for
example `C:\Users\<name>\.atlas`. `ATLAS_HOME` overrides that user root on
Windows the same way it does elsewhere. A value such as `D:\atlas-home` is
the user root itself.

Key files request mode `0600` and the keys directory requests mode `0700`.
Windows does not enforce those Unix bits. The file ACL is what keeps a key
private. Atlas still rejects a key symlink there.
