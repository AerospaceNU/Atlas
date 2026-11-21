# Atlas file layout

Project files live in `.atlas/` under the working directory. User files live in
`~/.atlas/`. Path helpers are in `src/atlas/agent/layout.py`.

These are the directories Atlas already used. Sessions, artifacts, data pulls,
skills, and skill drafts were already under the project `.atlas/` directory.
Keys, weights, and defaults were already under the user directory. Nothing is
read from an older location outside `.atlas/`.

`AGENTS.md` stays at the workspace root. `.agents/skills` stays beside
`.atlas`, not inside it. A data pull writes `manifest.json` in its output
directory, which defaults to `.atlas/data`. One output directory can hold
several sources and windows: each row in the manifest's `sources` list carries
its own `request`, and a pull replaces only the rows for the sources it
searched.

## Project (`.atlas/` in the working directory)

| Path | What it holds |
|---|---|
| `.gitignore` | Written by `ensure_project_layout` with a comment and `*` only when missing. It keeps `.atlas/` out of a foreign repository's git status. Edit or delete it to track files. |
| `sessions/<id>/` | One agent session. `meta.json`: `id`, `model`, `created`, `saved`, `messages`, `prompt`, `interrupted`, `stopped_for_limit`. `state.json`: `id`, `model`, `goal`, `messages`, `totals`, `saved`, `interrupted`, `stopped_for_limit`. No keys. A session with a `state.json` resumes through `--resume ID` or `--resume last` and `/resume`. |
| `artifacts/<id>/` | Files for that session. Removing the session removes this directory. |
| `artifacts/<id>/tool-output/` | Full text of any tool result over 20,000 characters, redacted. It is `.atlas/tool-output/` when the artifact root is the workspace. |
| `artifacts/<id>/jobs/<jobid>.log` | Background bash jobs, capped at 64 MiB each. Removed with the session. |
| `artifacts/<id>/analysis/{stacks,figures,tables,exports,scripts}/` and `analysis/runs.jsonl` | Satellite analysis outputs and the run record. Stacks do not survive a new session. |
| `artifacts/<id>/subagents/<child>/plan.json` | A sub-agent's private plan. |
| `checkpoints/<session>/<n>/` | Pre-edit copies of files for `/undo`. Child agents share the parent's chain. Outside `artifacts/`, so removing a session keeps its undo history. |
| `audit.log` | One JSON line per tool and model event. |
| `plan.json` | The plan written by `update_plan` and read by `read_plan`. |
| `data/` | Data CLI pulls. This is the existing `.atlas/data` default. `manifest.json` is the pull manifest, one row per source, each with its own request. |
| `skills/` | Vetted skills. |
| `skills-drafts/` | Draft skills. They are not loaded until a person enables one. |
| `agent.toml` | Workspace model and tool-call budget. |

## User (`~/.atlas/`)

| Path | What it holds |
|---|---|
| `keys/` | API keys. The directory is mode `0700` and each file is mode `0600`. Keys are not written into the project. |
| `weights/` | Model weights, laid out as `<dir>/<name>/<version>/<file>` plus a `.verified.json` marker. Fetched only by the human-run `atlas models pull`. |
| `defaults/` | User defaults. |
| `skills/` | User skills, loaded only when home skills are on. |
| `skills-drafts/` | User skill drafts. |
| `config.toml` | `home_skills = true` turns user skills on when `ATLAS_HOME_SKILLS` is unset. |

`ATLAS_HOME` overrides this user root. Set it to the directory that would
otherwise be `~/.atlas`, not to the parent home. `ATLAS_HOME_SKILLS` still
only turns user skills on. The skills and `config.toml` it refers to are
read from that same root. `.agents/skills` does not move with `ATLAS_HOME`.

`ATLAS_WEIGHTS_DIR` sets the weights location; otherwise it is `~/.atlas/weights`
(under `ATLAS_HOME` when set). The current directory and the repository are never
probed for weights.

## Windows

The project root is `.atlas\` in the working directory, for example
`D:\work\atlas\.atlas\data`. The user root is `%USERPROFILE%\.atlas`, for
example `C:\Users\<name>\.atlas`. `ATLAS_HOME` overrides that user root on
Windows the same way it does elsewhere. A value such as `D:\atlas-home` is
the user root itself.

Key files request mode `0600` and the keys directory requests mode `0700`.
Windows does not enforce those Unix bits. The file ACL is what keeps a key
private. Atlas still rejects a key symlink there.

The agent (TUI and `atlas run`) needs a POSIX shell and process groups, so on
Windows it runs under WSL2. See
[INSTALL-WINDOWS.md](../install-instructions/INSTALL-WINDOWS.md).
