from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence

_ROOT = frozenset({"data", "indices", "models", "geo", "run", "-h", "--help"})
_TUI_FLAGS = frozenset({"--workspace", "--repo", "--model", "--resume", "--approval", "--python"})
_HELP = """examples:
  atlas                          open the agent TUI
  atlas data                     list satellites
  atlas run "fix the bug" --approval allow   one-shot turn, exit code
  atlas models list              list model plugins
  atlas geo guide workflow       satellite analysis guide
  atlas sentinel2 optical ...    search one collection
  atlas indices --red red.npy --nir nir.npy --out indices
"""
_VALUE_FLAGS = frozenset(
    {
        "--coords",
        "--polygon",
        "--date",
        "--out",
        "--limit",
        "--min-cloud-cover",
        "--max-cloud-cover",
        "--asset",
        "--nodata",
        "--scale",
        "--offset",
    }
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``atlas`` CLI with lazy subcommand imports for fast TUI startup."""
    tokens = list(argv) if argv is not None else sys.argv[1:]
    if _is_tui_invocation(tokens):
        from atlas.cli.tui import launch_tui

        tui_args = tokens[1:] if tokens and tokens[0] == "tui" else tokens
        return launch_tui(tui_args)

    from atlas.cli.data import add_data_parser, run_data
    from atlas.cli.geo import add_geo_parser, run_geo
    from atlas.cli.indices import add_indices_parser, run_indices
    from atlas.cli.models import add_models_parser, run_models
    from atlas.cli.run import add_run_parser, run_run

    tokens = _attach_negative_option_values(_inject_data_command(tokens))
    parser = argparse.ArgumentParser(
        prog="atlas",
        description="Open the agent TUI, search satellite data, or compute spectral indices.",
        epilog=_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="group")
    add_data_parser(sub)
    add_indices_parser(sub)
    add_models_parser(sub)
    add_geo_parser(sub)
    add_run_parser(sub)
    args = parser.parse_args(tokens)
    handlers: dict[str, Callable[[argparse.Namespace], int]] = {
        "data": run_data,
        "indices": run_indices,
        "models": run_models,
        "geo": run_geo,
        "run": run_run,
    }
    handler = handlers.get(getattr(args, "_atlas_cmd", ""))
    if handler is not None:
        return handler(args)
    parser.print_help()
    return 0


def _is_tui_invocation(tokens: list[str]) -> bool:
    if not tokens:
        return True
    head = tokens[0]
    if head == "tui":
        return True
    return head in _TUI_FLAGS or any(head.startswith(f"{flag}=") for flag in _TUI_FLAGS)


def _inject_data_command(argv: list[str]) -> list[str]:
    if not argv or argv[0] in _ROOT or argv[0].startswith("-"):
        return argv
    from atlas.cli.catalog import is_known_target

    return ["data", *argv] if is_known_target(argv[0]) else argv


def _attach_negative_option_values(argv: list[str]) -> list[str]:
    """Join ``--coords -71.12,...`` so argparse does not treat negative longitudes as flags."""
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
