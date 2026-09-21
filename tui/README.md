# atlas-tui

A small terminal client for the Atlas Python agent session. The screen follows
the Grok Build default: a `#141414` canvas, an open transcript, a rounded
composer, and a one-line status row. It spawns
`uv run python -m atlas.agent` as a child process and speaks the
line-delimited JSON protocol on its stdin/stdout. The Python agent remains the
authority; this crate only renders and forwards messages.

## Run

```bash
cargo run --manifest-path tui/Cargo.toml
```

`cargo` comes from a Rust install. A user-local toolchain is at `~/.cargo/bin`.

Options:

- `--workspace PATH` — artifact workspace handed to the agent (default: the
  current directory; resolved to an absolute path).
- `--repo PATH` — Atlas repo root used as the child process `current_dir`
  (default: walk the current directory's parents for a `pyproject.toml` whose
  text contains `name = "atlas"`).

`OPENROUTER_API_KEY` is read from the environment, or from `<repo>/.env` when
the repo root is found. `ATLAS_MODEL` optionally overrides the model id.

Keys: `enter` sends the input line, `ctrl-r` resumes a turn that stopped at the
tool-round limit, `ctrl-c` / `esc` quits. `PageUp` / `PageDown` scroll.

## Adding a tool

A new safe tool is a concrete `Tool` subclass in a module directly inside
`src/atlas/agent/builtins/`. It needs a string `name`, a description, a
Pydantic `input_model`, a `run` method, and a constructor with no arguments.
Discovery picks it up without editing `default_registry()`. The TUI lists it
in the header once the session sends `ready`.

Set `trust = "opt_in"` when the tool must stay off unless a caller asks for
it. Script proposals use that. A duplicate `name` is an error that names both
classes. Discovery does not load tools from the workspace or from entry points.
