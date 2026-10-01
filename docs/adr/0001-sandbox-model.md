# ADR 0001: Sandbox model for agent tools

## Status

Accepted

## Context

`atlas.agent` lets a model call tools (`bash_tool`, `read_file`, `write_file`,
...) that touch the local filesystem and spawn processes. Those tools are
driven by model output, so a prompt-injected or misbehaving model turn is the
threat we're defending against, not a human attacker with a shell. We need to
decide how much isolation that threat model actually calls for, and record it
so the choice doesn't get silently relitigated tool by tool.

Three models were on the table:

1. **Workspace-scoped path and process enforcement** — every tool call is
   confined to one `LocalArtifactStore` root by resolving and checking paths
   in-process; `bash_tool` additionally runs with a stripped environment and
   a process-group timeout. No OS-level isolation primitive is involved.
2. **OS-level confinement** — wrap the same in-process checks with a kernel
   sandbox (`sandbox-exec` on macOS, seccomp/namespaces on Linux) so even a
   bug in our own path logic can't reach outside the workspace.
3. **Container/VM isolation** — run each session inside a container, so the
   entire host filesystem and process table are unreachable by construction.

## Decision

We use **workspace-scoped path and process enforcement** (option 1), as
already implemented in `LocalArtifactStore` and `BashTool`:

- Every path a tool touches is resolved exclusively through
  `LocalArtifactStore.resolve()`/`relative()`, which follow symlinks before
  checking containment, so a path that is lexically under the root but
  resolves elsewhere is rejected — including when reached through
  `list_images()`'s recursive glob, not just direct `resolve()` calls.
- `read_file`/`write_file` enforce byte-level limits and raise before
  reading or writing, rather than silently truncating.
- `bash_tool` runs the child with a minimal explicit environment (no
  inherited API keys/tokens), and a timeout kills the whole process group
  (`os.killpg`) so a backgrounded/disowned child can't outlive it. Command
  output capture is still truncated (not rejected) past a size limit, since
  that's output we read back, not a workspace path.
- Every rejection (escaped path, absolute path, oversized read/write) is
  raised as `atlas.agent.artifacts.SandboxDenied`, a `ValueError` subclass.
  Enforcement points only raise it; they never audit directly.
  `atlas.agent.runtime.Agent._execute()` is the single place that catches
  every tool call's exceptions already (to classify and record a step), so
  it is also the one place that calls `record_denial()` when the exception
  is a `SandboxDenied` — one audit line per denied step, via the standard
  `logging` module and a per-workspace audit log
  (`<root>/.atlas/sandbox_audit.log`), reviewable after the fact without
  needing external logging configuration. Enforcement (`artifacts.py`),
  routing (each tool), and auditing (`runtime.py`) stay three separate
  concerns in three separate files.

## Rejected alternatives and why

**OS-level confinement** was rejected for now: it adds real platform-specific
complexity (different primitives on macOS vs. Linux, CI environments that may
not support them) for a threat model where the "attacker" is model output
constrained to a fixed, explicit set of tools — not arbitrary attacker-chosen
native code. Our enforcement point is a handful of well-tested path/process
checks, not a large or constantly-changing attack surface.

**Container/VM isolation** was rejected for now: it's a much larger
operational change (image builds, session lifecycle tied to container
lifecycle, slower tool round-trips) for a tool set that never runs arbitrary
untrusted binaries supplied by a third party — every tool is one we wrote and
every command runs as the same local user who could already run a shell
directly.

## Revisit when

- A tool starts executing content that isn't fully attributable to our own
  code (e.g. running a model-authored script as a separate interpreter
  invocation with its own dependencies), rather than shell commands scoped to
  the workspace.
- Atlas starts serving sessions for users other than the person running the
  process locally (multi-tenant use), at which point the trust boundary
  moves from "this model's output" to "this other person's input," and
  OS-level or container isolation becomes the right default rather than an
  optional hardening step.

## Consequences

- Isolation correctness rests entirely on `LocalArtifactStore` being the
  single, consistently-used choke point for every filesystem-touching tool.
  Any new tool that resolves a path without going through it bypasses the
  sandbox; this is a code-review invariant, not something enforced by the
  OS.
- No protection against resource exhaustion (CPU/memory/disk quotas) or
  network egress from a spawned command — only path containment, a
  minimal environment, and a process-group timeout.
- The audit log is a plain append-only text file per workspace, not a
  tamper-evident log; it's meant for after-the-fact review, not for
  defending against an attacker who already has write access to the
  workspace root.
