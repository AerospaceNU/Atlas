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
    geo_sub = parser.add_subparsers(dest="geo_command")
    guide = geo_sub.add_parser("guide", help="Print one guide topic")
    guide.add_argument("topic", nargs="?", default=None, help="workflow, stack, masks, ...")
    geo_sub.add_parser("stacks", help="List stacks under the artifact root")


def run_geo(args: argparse.Namespace) -> int:
    """Run ``atlas geo``; output is byte-identical to ``python -m atlas.geo``."""
    from atlas.geo.__main__ import main

    command = getattr(args, "geo_command", None)
    if command is None:
        args._geo_parser.print_help()
        return 0
    argv = [command] if command == "stacks" else ["guide", *([args.topic] if args.topic else [])]
    return main(argv)
