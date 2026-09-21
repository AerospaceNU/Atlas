from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from atlas.cli.catalog import is_known_target
from atlas.cli.data import add_data_parser, run_data

_ROOT = frozenset({"data", "-h", "--help"})
_TUI_FLAGS = frozenset({"--workspace", "--repo"})
_HELP = """examples:
  atlas                          open the agent TUI
  atlas data                     list satellites
  atlas sentinel2 optical ...    search one collection
"""


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``atlas`` CLI.

    No arguments open the Rust agent TUI. ``atlas data`` and a bare satellite
    name stay on the data commands.

    Args:
        argv: Arguments without the program name. ``None`` reads ``sys.argv``.

    Returns:
        Process exit code.
    """
    tokens = list(argv) if argv is not None else sys.argv[1:]
    if _is_tui_invocation(tokens):
        from atlas.cli.tui import launch_tui

        tui_args = tokens[1:] if tokens and tokens[0] == "tui" else tokens
        return launch_tui(tui_args)
    tokens = _attach_negative_option_values(_inject_data_command(tokens))
    parser = argparse.ArgumentParser(
        prog="atlas",
        description="Open the agent TUI, or search satellite data.",
        epilog=_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="group")
    add_data_parser(sub)
    args = parser.parse_args(tokens)
    if getattr(args, "_atlas_cmd", None) == "data":
        return run_data(args)
    parser.print_help()
    return 0


def _is_tui_invocation(tokens: list[str]) -> bool:
    """True when this command should replace the process with the Rust TUI."""
    if not tokens:
        return True
    head = tokens[0]
    return head == "tui" or head in _TUI_FLAGS


_VALUE_FLAGS = frozenset(
    {
        "--coords",
        "--date",
        "--out",
        "--limit",
        "--min-cloud-cover",
        "--max-cloud-cover",
        "--asset",
    }
)


def _inject_data_command(argv: list[str]) -> list[str]:
    """Treat ``atlas alos`` as ``atlas data alos`` when the token is a satellite."""
    if not argv:
        return argv
    head = argv[0]
    if head in _ROOT or head.startswith("-"):
        return argv
    if is_known_target(head):
        return ["data", *argv]
    return argv


def _attach_negative_option_values(argv: list[str]) -> list[str]:
    """Join ``--coords -71.12,...`` so argparse does not treat the bbox as a flag.

    Argparse sees a token starting with ``-`` as a new option. Western longitudes
    are negative, so ``--coords -71.12,42.32,-71.02,42.40`` otherwise errors with
    ``expected one argument``. ``--coords=-71.12,...`` already works; this makes
    the space-separated form work too.
    """
    out: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token in _VALUE_FLAGS and index + 1 < len(argv):
            nxt = argv[index + 1]
            if nxt.startswith("-") and not nxt.startswith("--"):
                out.append(f"{token}={nxt}")
                index += 2
                continue
        out.append(token)
        index += 1
    return out


__all__ = ["main"]
