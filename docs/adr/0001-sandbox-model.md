# ADR 0001: Sandbox model for agent tools

## Status

Accepted

## Context

`atlas.agent` lets a model call tools (`bash_tool`, `read_file`,
`write_file`, `segment_landcover`, `list_tools`) that read and write the
local filesystem and spawn processes. Those tools are driven by model
output, so the threat is a prompt-injected or misbehaving model turn, not a
human attacker at a shell. This record states how much isolation we provide
against that threat, what we knowingly leave open, and when to revisit, so
the choice isn't relitigated tool by tool.

Three models were considered:

1. **Workspace-scoped path and process enforcement.** Every tool path is
   resolved and checked in-process against one `LocalArtifactStore` root.
   `bash_tool` also gets a stripped environment, cut-off stdin, a bounded
   timeout and a process-group kill. No OS isolation primitive is involved.
2. **OS-level confinement.** Wrap the child process in a kernel sandbox
   (`sandbox-exec` on macOS, Landlock/seccomp/namespaces or bubblewrap on
   Linux) so the shell itself can't reach outside the workspace or the
   network.
3. **Container/VM isolation.** Run each session in a container, so the host
   filesystem and process table are unreachable by construction.

## Decision

We use **option 1, workspace-scoped path and process enforcement**, and
`bash_tool` stays enabled by default (`trust = "default"`).

This is two different guarantees, and they must not be confused:

- **The file tools are confined.** `write_file` and `segment_landcover`
  write only inside the session's artifact folder, and `read_file` and
  `segment_landcover` read only from that folder or the project workspace.
- **`bash_tool` is not confined.** Only its *starting* directory is
  checked, and by default it starts in the project workspace. A command
  runs with the user's full privileges: it can `cd ..`, read `~`, change
  any file the user can, and use the network. We accept that in exchange
  for having a general shell available by default. The measures below
  limit what the process inherits and how long it lives, not what it can
  reach.

### Two roots

A session's `LocalArtifactStore` has two roots:

- **`root`, the write sandbox:** `<workspace>/.atlas/artifacts/<session>/`.
  All writes go here, so removing a session removes everything it wrote,
  and no tool can write into the project tree.
- **`read_root`, the project workspace:** reads try `root` first and fall
  back to the same relative path here, so the agent can inspect source
  files and inputs without being able to change them.

### Path enforcement (`atlas/agent/artifacts.py`)

`_within(base, path)` is the single check, and both roots use it:
`resolve()` for writes and `resolve_read()` for reads, including the
fallback to `read_root`. Every file-touching tool sends its paths through
one of these, and so do the Bash tool's working directory and the audit
writer. It rejects:

- **Absolute paths.**
- **`..` escapes.** The path is fully resolved and must stay under the
  root.
- **Symlink escapes.** Resolution follows symlinks before the containment
  check, so a link that sits inside the workspace but points outside is
  rejected.
- **Hardlinks.** A hardlink has no "real" location to check, because every
  name of the file is equally real. So any existing regular file with more
  than one link is refused, since its other name may be outside the
  workspace.

`list_images()` walks the write root recursively. An entry that fails any
of these checks is skipped on its own, rather than aborting the whole scan.

### Size limits (`read_file`, `write_file`)

`read_file` refuses files over 100,000 bytes, checked on disk before
opening. `write_file` refuses content over 1,000,000 bytes, measured in
encoded bytes rather than characters, before creating any file or folder.
Both fail with a clear error; neither silently truncates.

### Process controls (`atlas/agent/tools/BashTool.py`)

- **Working directory.** `.` means the project workspace (or the write
  root when no workspace is mounted). Any other value goes through
  `resolve_read()`, and an absolute or escaping path raises
  `SandboxDenied`, so it is audited, rather than being returned as plain
  text.
- **Environment.** The child receives only `PATH`, `HOME`, `LANG`,
  `LC_ALL`, `TMPDIR`, `USER` and `SHELL`, so API keys and tokens held by
  the host process aren't inherited.
- **stdin.** The child's stdin is `/dev/null`. The session reads the user's
  protocol messages on its own stdin, and an inherited stdin let a command
  read or swallow them.
- **Timeout.** The model chooses the timeout, so it is bounded to 1–300
  seconds and validated before anything runs.
- **Process lifetime.** Each command starts in a new session and process
  group. The group id is captured right after launch. Every call ends by
  killing the whole group (`os.killpg`), whether the command finished,
  failed or timed out. This matters because a background child that
  redirects its stdio lets the call return normally while it keeps
  running.
- **Output.** Decoded with invalid bytes replaced, so binary output doesn't
  crash the call. CSI escape sequences are stripped, and each stream is
  truncated past 200,000 characters with a visible marker. Truncation is
  deliberate here: this is output we read back, not a workspace file.

### Denials and auditing (`artifacts.py`, `runtime.py`)

Every rejection above raises `SandboxDenied`, a `ValueError` subclass, so
existing callers and tests that expect `ValueError` behave the same.
Enforcement points only raise; they never write audit records.

`Agent._execute()` in `runtime.py` already catches every tool call's
exception to record it as a step. It is the one caller of
`record_denial()`, which produces one audit line per denied tool call. The
line goes to the standard `logging` module and is appended to
`.atlas/sandbox_audit.log` inside the write root (so, per session, at
`<workspace>/.atlas/artifacts/<session>/.atlas/sandbox_audit.log`). A
session can be reviewed afterward without configuring logging. Each part has one job: `artifacts.py` decides
what's allowed, each tool routes its paths through the store, and
`runtime.py` records denials.

The audit write is itself treated as untrusted, because the workspace's own
tools can create files where the log lives:

- The log path goes through `resolve()` like any tool path, and the file
  write is skipped if `resolve()` denies it.
- The file is opened with `O_NOFOLLOW` and refused if it isn't a
  single-link regular file. This catches a link planted after `resolve()`
  returned.
- Each record is escaped onto a single line (`unicode_escape`), so a
  newline in a model-chosen path can't split or forge records.
- If the write fails, `_execute()` logs the failure and still returns the
  step for the original denial, so the turn carries on.

## Rejected alternatives and why

**OS-level confinement** was rejected for now. It is the only option that
would actually confine `bash_tool`. The cost is platform-specific code
(different primitives on macOS and Linux, and CI runners that may not
support them) and the work of writing a policy that still lets ordinary
commands run. We chose to keep Bash available by default without this, and
to record the gap instead of implying it's closed.

**Container/VM isolation** was rejected for now. It is a much larger
operational change (image builds, session lifecycle tied to container
lifecycle, slower tool round-trips) for a tool that runs locally, as the
same user who started it.

## Revisit when

- `bash_tool` is expected to be safe against prompt injection, for example
  when the agent starts reading untrusted content such as downloaded
  files, web pages or third-party metadata. Option 2 is then the right
  fix.
- A tool starts executing content not written by us, such as running a
  model-authored script under its own interpreter and dependencies.
- Atlas serves sessions for anyone other than the person running it
  locally. The trust boundary then moves to "another person's input," and
  option 2 or 3 becomes the default rather than an optional hardening
  step.

## Consequences and known gaps

- **Confinement of the file tools rests on `LocalArtifactStore`** being
  the single path for every file-touching tool. A new tool that builds a
  path without `resolve()` or `resolve_read()` bypasses the sandbox. This
  is a code-review invariant, not something the OS enforces.
- **`bash_tool` is unconfined** beyond its starting directory, as stated
  in the decision. There are no filesystem, network, CPU, memory or disk
  limits on what a command does.
- **A child that calls `setsid()`** leaves the process group we kill and
  can outlive the call. Only OS-level tracking (cgroups, a job object or a
  container) closes this.
- **Bash output is buffered in full before it is truncated.** 300 MB of
  output peaks at about 1 GB of memory, so a command like `yes` can
  exhaust memory before the timeout. Fix: read the pipes incrementally and
  stop keeping output past the cap.
- **Reads reach the whole project workspace,** usually the repo root, so
  `read_file` can read `.env` and anything else in it. Writes are safe
  from this: they stay in the session folder, so the file tools can no
  longer plant `.git/hooks/` or rewrite `.atlas/agent.toml` (Bash still
  can). Fix: have `resolve_read()` refuse protected names (`.env*`, `.git`,
  `.atlas`) in the project workspace.
- **The agent can edit its own audit log.** The log lives inside the write
  root, so `write_file` can empty it. Until it moves somewhere the tools
  can't write, the log is for after-the-fact review, not a defense against
  a session that wants to hide its tracks.
- **Only denials are audited.** Bash commands and successful writes aren't
  persisted, so the log can't show what a command actually did.
- **Terminal escape injection.** Only CSI sequences are stripped from Bash
  output, and `read_file` output isn't sanitized at all. OSC sequences
  (window title, clipboard via OSC 52) reach the TUI, which doesn't filter
  them.
- **Check-then-use.** The file tools check a path, then open it later.
  Exploiting the gap needs a concurrent process, such as a `setsid()`
  escapee, since tool calls otherwise run one at a time.
- **Legitimate hardlinks are refused.** Files in a `.venv` that uv
  hardlinked from its cache can't be read through the file tools when the
  project workspace is the repo root.
- **A failed audit write** is still in the `logging` output, but not in
  the file.

## Verification

Offline tests, run in CI:

- `tests/test_agent_tools.py`: absolute, `..`, symlink and hardlink
  rejection, including for reads that fall back to the project workspace;
  writes never reaching the project workspace; `list_images` skipping
  symlinked and hardlinked entries; oversized and non-UTF-8 reads; and the
  `bash_tool` process controls
  (working directory checks, scrubbed environment, stdin cut off, timeout
  bounds, group kill on timeout and on normal return, output truncation,
  escape stripping and binary output).
- `tests/test_tool_write_file.py`: oversized writes, byte rather than
  character counting, absolute and `..` paths, and not writing through a
  hardlink.
- `tests/test_agent_runtime.py`: a denial recorded end to end through the
  agent loop; the turn surviving a failed audit write; the log refusing a
  symlinked `.atlas`, a symlinked log file and a hardlinked log file; and
  one line per record.
