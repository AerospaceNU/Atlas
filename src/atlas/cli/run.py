"""``atlas run``: one agent turn, exit codes, optional JSON result."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_EPILOG = """exit codes:
  0 ok   1 error   2 usage   3 tool-call limit   4 spend budget
  5 context stop   6 cancelled or timed out   7 not configured (no API key)

approval defaults to deny: edits and commands are refused. Use --approval allow
to let the agent change the workspace. A prompt starting with '-' needs '--'.
"""


def add_run_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Attach the ``run`` command."""
    parser = sub.add_parser(
        "run",
        help="Run one agent turn and exit",
        description="Run one agent turn over a workspace and exit with a status code.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("prompt", nargs="?", default=None, help="Prompt, or - to read stdin")
    parser.add_argument(
        "--prompt-file", type=Path, default=None, help="Read the prompt from a file"
    )
    parser.add_argument("--workspace", type=Path, default=Path("."), help="Workspace directory")
    parser.add_argument("--model", default=None, help="OpenRouter model id")
    parser.add_argument(
        "--approval",
        choices=("allow", "deny"),
        default="deny",
        help="Mutating tools (default deny)",
    )
    parser.add_argument("--max-tool-calls", type=int, default=None, help="Tool-call budget")
    parser.add_argument("--max-spend-usd", type=float, default=None, help="Spend limit in USD")
    parser.add_argument("--timeout", type=float, default=None, help="Wall-clock limit in seconds")
    parser.add_argument("--resume", default=None, metavar="ID|last", help="Resume a saved session")
    parser.add_argument("--output", choices=("text", "json"), default="text", help="Output format")
    parser.add_argument("--quiet", action="store_true", help="Silence stderr progress and footer")
    parser.add_argument("--trace", type=Path, default=None, help="Append step events (JSONL)")
    parser.set_defaults(_atlas_cmd="run")


def run_run(args: argparse.Namespace) -> int:
    """Build options from parsed arguments and run one turn.

    Returns:
        Process exit code (0-7, see ``atlas run --help``).
    """
    from atlas.agent.oneshot import RunOptions, run_oneshot

    options = RunOptions(
        prompt=args.prompt,
        prompt_file=args.prompt_file,
        workspace=args.workspace,
        model_id=args.model,
        approval=args.approval,
        max_tool_calls=args.max_tool_calls,
        max_spend_usd=args.max_spend_usd,
        timeout=args.timeout,
        resume=args.resume,
        output=args.output,
        quiet=args.quiet,
        trace=args.trace,
    )
    return run_oneshot(options, stdout=sys.stdout, stderr=sys.stderr)
