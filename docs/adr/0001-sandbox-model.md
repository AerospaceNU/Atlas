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

File tools use **option 1, workspace-scoped path enforcement**. `bash_tool`
uses a filesystem slice of **option 2** where the OS provides one, and stays
enabled by default on Linux and macOS (`trust = "default"`). On Windows it
is opt-in and off unless that list is requested.

This is two different guarantees, and they must not be confused:

- **The file tools are confined.** `write_file` and `segment_landcover`
  write only inside the session's artifact folder, and `read_file` and
  `segment_landcover` read only from that folder or the project workspace.
- **`bash_tool` is confined by the OS sandbox, not by the text scan.** The
  scan is best-effort. It resolves paths it can see, and it rejects
  command substitution, parameter expansion, globs, a few `python` / `perl`
  / `node` eval shapes (`-c` / `-e`, combined short flags, an `env`
  prefix, and a program read from stdin), and `exec(open(...))`. An
  interpreter can still build a path the scan never sees. That path is
  stopped by the sandbox, or the shell does not start. On Windows the scan
  is the only layer.

  The shell refuses to start when the user key directory or `ATLAS_HOME`
  resolves inside the workspace. A key denial does not include the path or
  the file contents. On Linux 5.13 or newer the child is placed in a
  Landlock ruleset: the workspace (with execute), read-only `/etc`, a
  private `TMPDIR` inside the workspace, and the toolchain (`/usr`, `/bin`,
  `/lib`, `/lib64`, plus the interpreter prefix). The artifact root and
  `proposals/` are not in that accessible tree. Landlock cannot exclude a
  child of a granted directory, so a rule on the workspace root is applied
  only after those directories are hidden. The usual hide is a private bind
  mount in a user namespace. If user namespaces or the mount are
  unavailable, those directories are moved outside the workspace for the
  command and moved back when it ends, and the root is then granted whole.
  If they cannot be moved outside the workspace, the shell does not start
  and the error says it needs user namespaces or an artifact root outside
  the workspace. Granting the root while they still sit inside it would
  make them readable. If Landlock cannot be applied, the child exits with
  a clear message and bash is not started. On macOS, `sandbox-exec` denies
  the key directory and file access outside the workspace, aside from the
  toolchain and system read paths. If `sandbox-exec` is missing, cannot be
  launched, or rejects the profile, `bash_tool` raises a sandbox-unavailable
  denial rather than returning a normal command result. The network is
  unrestricted on every platform. A child that calls `setsid()` still
  leaves the process group.

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
  text. The same check refuses a directory that resolves into
  `~/.atlas/keys` or `<artifact root>/proposals`.
- **Command paths.** `shell_guard.guard_shell_command()` is a best-effort
  scan. It resolves paths, redirects, and `cd` targets it can see. `..` and
  absolute paths that stay inside the workspace are allowed. Ones that
  leave are refused. Symlinks that already exist are judged after
  resolution. `~/.atlas/keys` is refused when the scan can see it, and the
  error text does not include the key. `<artifact root>/proposals` is not a
  runnable path when the scan can see it, including `python proposals/tool.py`
  and a working directory inside that folder. Command substitution,
  parameter expansion, and globs are refused because they can hide a path.
  `python` / `perl` / `node` eval is refused when the command text shows
  `-c` or `-e` (including a combined short flag such as `-Ic`), an `env`
  prefix, or a program taken from stdin (`python3 -`). That check is
  best-effort and is the only layer on opt-in Windows. A path built inside
  a script is not visible to this scan. A hardlinked regular file is
  refused, matching the file tools. Path comparison folds case on macOS and
  Windows. The OS sandbox is what stops a path the scan misses, including a
  symlink created while the command is running: creating the link inside
  the workspace can succeed, and reading or writing through it to a file
  outside the ruleset is denied.
- **Environment.** The child receives only `PATH`, `HOME`, `LANG`,
  `LC_ALL`, `USER` and `SHELL`, plus a private `TMPDIR` created inside the
  workspace for that command. The host `TMPDIR` is not passed through, so
  API keys and tokens held by the host process aren't inherited and temp
  files are not written to the host temp directory.
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

**OS-level confinement of the whole session** was rejected. The shell does
use a filesystem sandbox. On Linux that is Landlock (kernel 5.13 or newer):
the workspace is executable, `/etc` is read-only, temp files go to a private
`TMPDIR` inside the workspace, and the toolchain paths above are readable
and executable. The interpreter prefix is included so a virtualenv outside
the workspace can still start; it is not writable. The artifact root and
`proposals/` are kept out of that
tree. When user namespaces are unavailable those directories are moved
outside the workspace for the command, or the shell does not start. Devices
a redirect needs (`/dev/null` and the other null devices) are included. On
macOS the same boundary is a `sandbox-exec` profile. If that program is
missing, cannot be launched, or rejects the profile, the tool does not run.
Windows has no such confinement, so `bash_tool` is opt-in and warns, and the
text scan is the only layer. The network is unrestricted. A container
remains out of scope.

**Container/VM isolation** was rejected for now. It is a much larger
operational change (image builds, session lifecycle tied to container
lifecycle, slower tool round-trips) for a tool that runs locally, as the
same user who started it.

## Revisit when

- `bash_tool` is expected to be safe against prompt injection that uses
  the network. The network is unrestricted. Landlock and `sandbox-exec`
  cover host files the text scan cannot see; they do not cover the network.
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
- **`bash_tool` can still use the network.** Nothing in the Landlock
  ruleset or the macOS profile blocks it. A child that calls `setsid()`
  leaves the process group. Landlock and `sandbox-exec` bound the files
  the child can open. They do not bound CPU, memory, or disk. On Windows
  there is no OS file boundary; the tool is off unless it is opted in, and
  it warns. A path the text scan misses can still be opened when it stays
  inside the workspace and is not the masked or parked artifact root or
  `proposals/`.
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
- `tests/test_shell_workspace_guard.py`: the shell's cwd is the workspace;
  `..` and absolute paths that leave it are denied; a symlink escape is
  denied; a user-key read is denied and the error does not contain the
  key; the shell refuses to start when the key directory or `ATLAS_HOME`
  is inside the workspace; a drafted tool under `proposals/` is not
  executed, including an interpreter-built path and a copy of that tree;
  a symlink created during the command cannot read or write a host file;
  without user namespaces a file can still be created at the workspace
  root, or the shell reports that it needs user namespaces or an artifact
  root outside the workspace; Landlock denies an interpreter-built path
  outside the workspace; a failed Landlock syscall exits without running
  the command; `$(...)`, backticks, and `$VAR` are rejected; interpreter
  eval flags combined flags, an `env` prefix, and stdin; macOS builds a
  `sandbox-exec` profile, disables the tool when that program is missing,
  and treats a launch or profile failure as a sandbox denial; Windows
  keeps the tool opt-in and off the default list.
- `tests/test_tool_write_file.py`: oversized writes, byte rather than
  character counting, absolute and `..` paths, and not writing through a
  hardlink.
- `tests/test_agent_runtime.py`: a denial recorded end to end through the
  agent loop; the turn surviving a failed audit write; the log refusing a
  symlinked `.atlas`, a symlinked log file and a hardlinked log file; and
  one line per record.
