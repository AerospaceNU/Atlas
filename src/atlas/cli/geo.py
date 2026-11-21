from __future__ import annotations

import argparse


def add_geo_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Attach the ``geo`` command (guide, stacks)."""
    parser = sub.add_parser(
        "geo",
        help="Satellite analysis guide and stacks",
        description="Satellite analysis guide and stack inspection.",
    )
    parser.set_defaults(_atlas_cmd="geo", _geo_parser=parser)


def run_geo(args: argparse.Namespace) -> int:
    """Run ``atlas geo``; the stub prints help."""
    args._geo_parser.print_help()
    return 0
