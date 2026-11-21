"""In-container entry point for one benchmark rollout.

The host starts a task container and calls three subcommands through
``docker exec``, in order:

- ``prepare`` records the commit the patch is measured against, keeps
  ``.atlas/`` out of git, and removes refs that point past that commit.
- ``run`` executes one headless agent turn and writes ``trace.json``,
  ``trajectory.md`` and ``meta.json``. The trajectory is appended step by
  step and rewritten in full at the end.
- ``collect`` writes ``patch.diff``. It is separate so the host can still
  collect a patch after it has killed a ``run`` that overran.

The OpenRouter key arrives on stdin (``--api-key-stdin``), never in the
environment, so no ``bash`` child inherits it and it is not readable from
``/proc``. Nothing here reads the task's tests or its reference patch: the
container never holds them.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

from atlas.agent.runtime import AgentStep
from atlas.harness_eval.headless import (
    TOOL_PROFILES,
    HeadlessRun,
    render_step,
    render_trajectory,
    run_headless,
)
from atlas.harness_eval.substrate import exclude_from_git

BASE_REF_FILE = "base_ref"
META_FILE = "meta.json"
_GIT_EXCLUDES = ("__pycache__/", "*.pyc")


def _git(workspace: Path, *args: str, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(workspace), *args],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=timeout,
    )


def prepare(workspace: Path, out: Path) -> int:
    """Record the base commit and hide later history from the agent.

    Tags that are not ancestors of the base commit are deleted and the reflog
    is expired, so ``git log --all``, ``git tag`` and ``git reflog`` show
    nothing newer than the task. Unreachable objects are left in the object
    store: removing them needs a repack that costs more than a rollout.
    """
    out.mkdir(parents=True, exist_ok=True)
    head = _git(workspace, "rev-parse", "HEAD")
    if head.returncode != 0:
        print(f"not a git workspace: {workspace}: {head.stderr.strip()}", file=sys.stderr)
        return 1
    (out / BASE_REF_FILE).write_text(head.stdout.strip() + "\n", encoding="utf-8")
    exclude_from_git(workspace)
    exclude = workspace / ".git" / "info" / "exclude"
    current = exclude.read_text(encoding="utf-8")
    missing = [line for line in _GIT_EXCLUDES if line not in current.splitlines()]
    if missing:
        exclude.write_text(
            current.rstrip("\n") + "\n" + "\n".join(missing) + "\n", encoding="utf-8"
        )
    for tag in _git(workspace, "tag", "--no-merged", "HEAD").stdout.split():
        _git(workspace, "tag", "-d", tag)
    _git(workspace, "reflog", "expire", "--expire=now", "--all")
    for name in ("ORIG_HEAD", "FETCH_HEAD"):
        (workspace / ".git" / name).unlink(missing_ok=True)
    return 0


def collect(workspace: Path, out: Path) -> int:
    """Write every change since the base commit to ``patch.diff``."""
    base_file = out / BASE_REF_FILE
    base = base_file.read_text(encoding="utf-8").strip() if base_file.is_file() else "HEAD"
    _git(workspace, "add", "-A")
    diff = _git(workspace, "diff", "--cached", "--no-color", base)
    if diff.returncode != 0:
        print(f"git diff failed: {diff.stderr.strip()}", file=sys.stderr)
        return 1
    (out / "patch.diff").write_text(diff.stdout, encoding="utf-8")
    return 0


def _meta(
    result: HeadlessRun,
    reasoning_effort: str | None = None,
    provider_order: tuple[str, ...] = (),
    tool_profile: str = "full",
    temperature: float | None = None,
    top_p: float | None = None,
) -> dict[str, object]:
    run = result.run
    costs = [call.usage.cost for call in run.calls if call.usage.cost is not None]
    stopped = next(
        (
            reason
            for flag, reason in (
                (run.stopped_for_limit, "tool_budget"),
                (run.stopped_for_budget, "spend_budget"),
                (run.stopped_for_context, "context"),
                (run.cancelled, "deadline"),
            )
            if flag
        ),
        "",
    )
    return {
        "status": "ok",
        "stopped": stopped,
        "tool_calls": len(run.steps),
        "model_calls": len(run.calls),
        "reasoning_effort": reasoning_effort,
        "provider_order": list(provider_order),
        "tool_profile": tool_profile,
        "temperature": temperature,
        "top_p": top_p,
        "prompt_tokens": sum(call.usage.prompt_tokens for call in run.calls),
        "completion_tokens": sum(call.usage.completion_tokens for call in run.calls),
        "cost_usd": round(sum(costs), 6) if costs else None,
        "wall_seconds": result.wall_seconds,
        "skills": list(result.harness.skills),
        "scripts": list(result.harness.scripts),
        "roles": [role.name for role in result.harness.roles],
        "empty_replies": run.empty_replies,
        "intent_nudges": run.intent_nudges,
        "compacted": run.compaction.occurred,
    }


def _providers(raw: str | None) -> tuple[str, ...]:
    """Split a comma-separated ``--provider`` value into a provider order."""
    return tuple(part.strip() for part in (raw or "").split(",") if part.strip())


def run(args: argparse.Namespace) -> int:
    """Run one agent turn and write its record under ``args.out``."""
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    if args.api_key_stdin:
        api_key = sys.stdin.readline().strip()
    else:
        api_key = os.environ.pop("OPENROUTER_API_KEY", "")
    try:
        if not api_key:
            raise RuntimeError("no OpenRouter key: pass --api-key-stdin or OPENROUTER_API_KEY")
        partial = out / "trajectory.md"
        partial.write_text("", encoding="utf-8")

        def record(step: AgentStep) -> None:
            # Appended as the run goes, so a killed rollout still leaves evidence.
            with partial.open("a", encoding="utf-8") as handle:
                handle.write(render_step(step) + "\n\n")

        result = run_headless(
            workspace=args.workspace,
            prompt=args.prompt_file.read_text(encoding="utf-8"),
            model_id=args.model,
            api_key=api_key,
            harness_dir=args.harness,
            artifact_root=out / "artifacts",
            trace_path=out / "trace.json",
            max_tool_calls=args.max_tool_calls,
            reasoning_effort=args.reasoning_effort,
            provider_order=_providers(args.provider),
            tool_profile=args.tool_profile,
            temperature=args.temperature,
            top_p=args.top_p,
            deadline_seconds=args.deadline,
            on_step=record,
        )
    except Exception as exc:
        meta = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        (out / META_FILE).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        traceback.print_exc()
        return 1
    (out / "trajectory.md").write_text(
        render_trajectory(result.run, notes=result.notes), encoding="utf-8"
    )
    (out / META_FILE).write_text(
        json.dumps(
            _meta(
                result,
                args.reasoning_effort,
                _providers(args.provider),
                args.tool_profile,
                args.temperature,
                args.top_p,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="atlas.harness_eval.rollout", description=__doc__)
    sub = parser.add_subparsers(dest="step", required=True)
    for name in ("prepare", "run", "collect"):
        step = sub.add_parser(name)
        step.add_argument("--workspace", type=Path, required=True)
        step.add_argument("--out", type=Path, required=True)
        if name == "run":
            step.add_argument("--harness", type=Path, default=None)
            step.add_argument("--prompt-file", type=Path, required=True)
            step.add_argument("--model", required=True)
            step.add_argument("--max-tool-calls", type=int, default=None)
            step.add_argument("--reasoning-effort", default=None)
            step.add_argument("--provider", default=None, help="comma-separated provider order")
            step.add_argument("--tool-profile", choices=TOOL_PROFILES, default="full")
            step.add_argument("--temperature", type=float, default=None)
            step.add_argument("--top-p", type=float, default=None)
            step.add_argument("--deadline", type=float, default=None)
            step.add_argument("--api-key-stdin", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Dispatch one rollout step."""
    args = _parser().parse_args(argv)
    if args.step == "prepare":
        return prepare(args.workspace, args.out)
    if args.step == "collect":
        return collect(args.workspace, args.out)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
