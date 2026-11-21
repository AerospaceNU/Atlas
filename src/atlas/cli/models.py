from __future__ import annotations

import argparse


def add_models_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Attach the ``models`` command (list, info, pull, verify)."""
    parser = sub.add_parser(
        "models",
        help="List and fetch model weights",
        description="Model-plugin weight management. Human-run; no agent tool downloads weights.",
    )
    parser.set_defaults(_atlas_cmd="models", _models_parser=parser)


def run_models(args: argparse.Namespace) -> int:
    """Run ``atlas models``; the stub prints help."""
    args._models_parser.print_help()
    return 0
