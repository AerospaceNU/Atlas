from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from atlas.models import weights

if TYPE_CHECKING:
    from atlas.models.manifest import ModelManifest

_PULL_HINT = "atlas models pull {name}"


def add_models_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Attach the ``models`` command (list, info, pull, verify)."""
    parser = sub.add_parser(
        "models",
        help="List and fetch model weights",
        description="Model-plugin weight management. Human-run; no agent tool downloads weights.",
    )
    parser.set_defaults(_atlas_cmd="models", _models_parser=parser)
    actions = parser.add_subparsers(dest="models_cmd")
    actions.add_parser("list", help="List models and their weight state")
    info = actions.add_parser("info", help="Show one model")
    info.add_argument("name")
    pull = actions.add_parser("pull", help="Install weights (human-run)")
    pull.add_argument("name")
    source = pull.add_mutually_exclusive_group()
    source.add_argument("--from", dest="src", type=Path, help="local weights file")
    source.add_argument("--url", help="https URL (default: the manifest url)")
    pull.add_argument(
        "--accept-license", action="store_true", help="accept the model licence terms"
    )
    check = actions.add_parser("verify", help="Re-hash installed weights")
    check.add_argument("name", nargs="?")


def _manifests() -> list[ModelManifest]:
    return [m for m in weights.load_manifests() if getattr(m, "schema_version", 2) == 2]


def _find(name: str) -> ModelManifest | None:
    for manifest in _manifests():
        if manifest.name == name:
            return manifest
    known = ", ".join(m.name for m in _manifests())
    print(f"unknown model; known: {known}", file=sys.stderr)
    return None


def _state_note(manifest: ModelManifest, state: weights.WeightState) -> str:
    if state.status == "ready":
        return "ready"
    if state.status == "missing":
        return f"missing: {_PULL_HINT.format(name=manifest.name)}"
    if state.status == "unpublished":
        return "not published; use --from PATH"
    return f"corrupt: {_PULL_HINT.format(name=manifest.name)} --from PATH"


def _list(_: argparse.Namespace) -> int:
    onnx_missing = importlib.util.find_spec("onnxruntime") is None
    for manifest in _manifests():
        state = weights.status(manifest)
        note = _state_note(manifest, state)
        if manifest.tier == "learned" and manifest.extra == "onnx" and onnx_missing:
            note = f"onnx extra missing (uv sync --extra onnx); {note}"
        print(f"{manifest.name}  {manifest.tier}  {manifest.runtime.kind}  {state.status}  {note}")
    return 0


def _info(args: argparse.Namespace) -> int:
    manifest = _find(args.name)
    if manifest is None:
        return 2
    state = weights.status(manifest)
    print(f"name: {manifest.name}  version: {manifest.version}")
    print(f"description: {manifest.description}")
    lic = manifest.license
    print(f"licence: {lic.spdx}  holder: {lic.holder or '-'}  url: {lic.url or '-'}")
    cite = manifest.citation
    print(f"citation: {cite.text or '-'}  doi: {cite.doi or '-'}  url: {cite.url or '-'}")
    spec = manifest.weights
    if spec is not None:
        print(f"weights file: {spec.file}  size_bytes: {spec.size_bytes}")
        print(f"weights sha256: {spec.sha256 or '-'}")
        print(f"weights url: {spec.url or '-'}")
    print(f"state: {state.status} ({state.detail})")
    print("known limits:")
    for limit in manifest.evidence.known_limits:
        print(f"  - {limit}")
    return 0


def _progress(done: int, total: int | None) -> None:
    if total:
        print(f"\r{done / 1e6:.1f}/{total / 1e6:.1f} MB", end="", file=sys.stderr)


def _pull(args: argparse.Namespace) -> int:
    manifest = _find(args.name)
    if manifest is None:
        return 2
    try:
        path = weights.pull(
            manifest,
            url=args.url,
            src=args.src,
            accept_license=args.accept_license,
            progress=_progress,
        )
    except weights.WeightError as exc:
        print(str(exc), file=sys.stderr)
        return exc.exit_code
    state = weights.status(manifest)
    sha = (manifest.weights.sha256 if manifest.weights else "")[:12]
    print(f"installed {path} sha256 {sha}" if state.status == "ready" else state.detail)
    return 0 if state.status == "ready" else 3


def _verify(args: argparse.Namespace) -> int:
    if args.name:
        manifest = _find(args.name)
        if manifest is None:
            return 2
        targets = [manifest]
    else:
        targets = [m for m in _manifests() if m.weights is not None]
    states = []
    for manifest in targets:
        state = weights.verify(manifest)
        states.append(state.status)
        print(f"{manifest.name}  {state.status}  {state.detail}")
    if "corrupt" in states:
        return 3
    if any(s in ("missing", "unpublished") for s in states):
        return 5
    return 0


def run_models(args: argparse.Namespace) -> int:
    """Run ``atlas models``; returns 0 ok, 2 usage, 3 mismatch, 4 network, 5 unpublished."""
    handlers = {"list": _list, "info": _info, "pull": _pull, "verify": _verify}
    handler = handlers.get(getattr(args, "models_cmd", None) or "")
    if handler is None:
        args._models_parser.print_help()
        return 0
    return handler(args)
