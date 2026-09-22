# atlas-tui

A small terminal client for the Atlas Python agent session. The screen follows
the Grok Build default: a `#141414` canvas, an open transcript, a rounded
composer, and a one-line status row. It spawns
`uv run python -m atlas.agent` as a child process and speaks the
line-delimited JSON protocol on its stdin/stdout. The Python agent remains the
authority; this crate only renders and forwards messages.

## Run

```bash
atlas
```

That opens the TUI. `atlas data` and `atlas <satellite>` stay on the data CLI.
`~/.local/bin` needs to be on `PATH`, which it is after a normal uv install.
From this checkout, `uv tool install --editable --force .` keeps that `atlas`
command pointed at the current source.
The first launch builds `tui/target/release/atlas-tui` when `cargo` is on
`PATH` or at `~/.cargo/bin`. Later launches use that binary.

`atlas tui --workspace PATH` forwards the extra arguments. `ATLAS_TUI` can
point at a binary you built yourself.

Options:

- `--workspace PATH` — artifact workspace handed to the agent (default: the
  current directory; resolved to an absolute path).
- `--repo PATH` — Atlas repo root used as the child process `current_dir`
  (default: walk the current directory's parents for a `pyproject.toml` whose
  text contains `name = "atlas"`).

`OPENROUTER_API_KEY` is read from the environment, or from `<repo>/.env` when
the repo root is found. The first session writes `<workspace>/.atlas/agent.toml`
when that file is missing, with `model` set to `google/gemini-3.8-flash`.
`ATLAS_MODEL` or `--model` on the Python session overrides the file.

Keys: `enter` sends the input line, `ctrl-r` resumes a turn that stopped at the
tool-call limit, `ctrl-c` / `esc` quits. `PageUp` / `PageDown` scroll.

## Adding a tool

A new safe tool is a concrete `Tool` subclass in a module directly inside
`src/atlas/agent/tools/`. It needs a string `name`, a description, a
Pydantic `input_model`, a `run` method, and a constructor with no arguments.
Discovery picks it up without editing `default_registry()`. The session
advertises it on the `ready` message. `src/atlas/agent/tools/read_file.py` is the
worked example to copy.

Set `trust = "opt_in"` when the tool must stay off unless a caller passes
`include_opt_in=True` to `build_registry`. A duplicate `name` is an error that
names both classes. Discovery does not load tools from the workspace or from
entry points.

The default tool-call budget is 256. A workspace overrides it in
`<workspace>/.atlas/agent.toml` with `max_tool_calls` set to an integer >= 1.
The same file sets `model` to an OpenRouter model id.
