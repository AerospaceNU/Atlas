# atlas-tui

A small terminal client for the Atlas Python agent session. The screen follows
the Grok Build default: a `#141414` canvas, an open transcript, a rounded
composer, and a one-line status row. It spawns
`python -P -m atlas.agent --approval ask [--resume ID] --workspace W` as a child
process and speaks the line-delimited JSON protocol on its stdin/stdout. `-P`
keeps modules in the current directory from shadowing `atlas`, so it works from
any repository. The Python agent remains the
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

Usage: `atlas-tui [--workspace PATH] [--repo PATH] [--python PATH] [--model ID] [--resume ID] [--approval MODE]`

Options:

- `--workspace PATH` — artifact workspace handed to the agent (default: the
  current directory; resolved to an absolute path).
- `--python PATH` — interpreter with atlas installed (default: env
  `ATLAS_PYTHON`). The `atlas` launcher sets it.
- `--repo PATH` — optional Atlas checkout used as the child's working
  directory and `.env` source (default: walk the current directory's parents for
  a `pyproject.toml` whose text contains `name = "atlas"`).
- `--model ID` — override the model id for this session.
- `--resume ID` — resume a saved session; `last` is the newest.
- `--approval MODE` — approval mode handed to the agent (default: `ask`;
  `allow`, `ask`, or `deny`).

`OPENROUTER_API_KEY` is read from the environment, or from `<repo>/.env` when
the repo root is found. The first session writes `<workspace>/.atlas/agent.toml`
when that file is missing, with `model` set to `deepseek/deepseek-v4.1-flash`.
`ATLAS_MODEL` or `--model` on the Python session overrides the file.

The agent writes each new session under `<workspace>/.atlas/sessions/<id>/`
and its files under `<workspace>/.atlas/artifacts/<id>/`. The directory list
is `docs/atlas-layout.md`.

Keys: `Enter` sends the draft; while a turn is running, it shows feedback and
keeps the draft without queueing it. `Alt+Enter` inserts a newline (`Shift+Enter`
works where the terminal supports it). `Left` / `Right` move the cursor, and
`Home` / `End` move to the line ends. `Ctrl+A` / `Ctrl+E` move to the start / end
of the current line; `Ctrl+W` deletes the previous word. `Up` / `Down` move
through draft lines, then recall sent prompts at the boundaries. `PageUp` /
`PageDown` scroll the transcript. `Shift+Up` / `Shift+Down` select tool rows.
`Esc` keeps an unfinished ordinary draft. During a running turn the first `Esc`
requests a stop and keeps the draft. A tool that cannot be interrupted keeps
running: the status says so and the turn ends when it returns (`bash` and
`run_tests` report "cancelling"). A second `Esc` prints "Still stopping. Ctrl+C
quits; the conversation is saved up to the last finished step." `Ctrl+R`
resumes a turn that stopped at the tool-call limit or was interrupted, and
`Ctrl+C` copies selected text or quits when nothing is selected. `/new`
starts a fresh conversation and clears the transcript. `/resume [ID]` (default
`last`) loads a saved session and replays its earlier transcript, noting "N
earlier items not shown" when it is cut; the status is Stopped when the session
was interrupted or stopped at the limit. `/undo` restores the files of the last
checkpointed turn. `/compact` shrinks the
model context on demand. `/skill <name>` applies
a workspace skill for the rest of this conversation. The `.atlas` copy wins
over `.agents` when both exist.

Drag across visible text to select it. Releasing the mouse copies the selection
to the clipboard; `Ctrl+C` copies it again. Incoming updates continue rendering
while text is selected; press `Esc` to clear the selection. Paste into the
composer with your terminal's paste shortcut (`Cmd+V` on macOS), `Ctrl+V`, or
`Shift+Insert`. Multiline text stays in the composer until you press `Enter`.
`/copy` copies the last assistant reply.

When a tool needs approval, an **Approval required** panel shows the command
or colored diff preview. Press `y` (**Allow once**), `a` (**Allow for session**),
`n` / `Esc` (**Deny**), `d` (**Deny with reason**), or `Ctrl+Y` to toggle YOLO,
or use the arrow keys / `Tab` + `Enter` or click a choice. **Deny** is selected
initially so bare `Enter` never approves accidentally. Use `PageUp`, `PageDown`,
or the mouse wheel to scroll a long preview. The panel keeps any message you
were composing.

Use `/yolo` (or `Ctrl+Y` in the approval modal) to toggle automatic approval for
this Atlas TUI process. YOLO starts off; while it is on, tool calls proceed
without per-tool prompts, and `/yolo` turns prompting back on. The status row
shows `YOLO on` while enabled.

## Tool rows

A tool row reads `◆ <tool> <summary> (<duration>)` (summary clipped to 60
characters) and streams live output while running. Clicking a row (or pressing
`Ctrl+O` / `Right` / `Enter` on an empty composer) expands arguments, colored
diffs, output, `notes` (harness feedback such as repeated-call or budget notes),
and `artifacts`. An active phase or tool that has run for 5 seconds or more
shows its elapsed time, for example `1m12s`.

## Sessions

A session autosaves during a turn at most every 5 seconds and at round
boundaries. On shutdown the TUI sends `quit`, waits 1 second (20 polls of 50 ms),
then kills the Python process. In-flight tools and jobs are cancelled, but a
kill can orphan background jobs until their lifetime cap.

## Foreign repositories

Run `atlas` inside any repository. `.atlas/` writes its own `.gitignore`, so it
stays out of `git status`. That repository's `AGENTS.md` and `.atlas/skills` are
untrusted guidance, not instructions the agent must obey.

## Windows

The agent needs a POSIX shell and process groups; use WSL2. See
[INSTALL-WINDOWS.md](../install-instructions/INSTALL-WINDOWS.md).

## Adding a tool

A new safe tool is a concrete `Tool` subclass in a module directly inside
`src/atlas/agent/tools/`. It needs a string `name`, a description, a
Pydantic `input_model`, a `run` method, and a constructor with no arguments.
Discovery picks it up without editing `default_registry()`. Tools with
`domain = "satellite"` are dropped under the `coding` tool profile. The session
advertises it on the `ready` message. `src/atlas/agent/tools/read_file.py` is the
worked example to copy.

Set `trust = "opt_in"` when the tool must stay off unless a caller passes
`include_opt_in=True` to `build_registry`. A duplicate `name` is an error that
names both classes. Discovery does not load tools from the workspace or from
entry points.

The default tool-call budget is 256. A workspace overrides it in
`<workspace>/.atlas/agent.toml` with `max_tool_calls` set to an integer >= 1.
The same file sets `model` to an OpenRouter model id.
