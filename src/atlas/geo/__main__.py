"""Guide text and stack listing for ``python -m atlas.geo`` and ``atlas geo``.

Invariant: this module is the single source of the guide text. The satellite_guide tool and
both command lines call these functions, so their output is byte-identical. Read-only: no
network, no execution, no directory creation.
"""

from __future__ import annotations

import importlib.resources
import importlib.util
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path

__all__ = ["TOPICS", "guide_text", "main", "stacks_text"]

TOPICS: tuple[str, ...] = (
    "workflow",
    "stack",
    "masks",
    "areas",
    "models",
    "render",
    "recipes",
    "scripts",
    "limits",
)

MODELS_BLOCK_MAX = 1100
_LINE_MAX = 160
_MAX_STACK_LINES = 40
_HEADER = re.compile(r"^## topic: (\w+)\s*$", re.M)
_SCRATCH = (".tmp-", ".old-")
_EXTRA_MODULES = {"onnx": "onnxruntime", "geo": "rasterio"}
_USAGE = "usage: atlas geo guide TOPIC | atlas geo stacks"


def _bodies() -> dict[str, str]:
    text = importlib.resources.files("atlas.geo").joinpath("guide.md").read_text(encoding="utf-8")
    parts = _HEADER.split(text)
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}


def _clip(line: str) -> str:
    return line if len(line) <= _LINE_MAX else line[: _LINE_MAX - 3] + "..."


def _model_lines() -> list[str]:
    from atlas.models.manifest import iter_model_manifests
    from atlas.models.weights import status

    lines: list[str] = []
    for manifest in sorted(iter_model_manifests(strict=False), key=lambda m: m.name):
        if manifest.tier == "heuristic":
            continue
        label = f"{manifest.name} [{manifest.tier}/{manifest.runtime.kind}]"
        extra = manifest.extra
        module = _EXTRA_MODULES.get(extra or "", extra or "")
        extra_missing = extra is not None and importlib.util.find_spec(module) is None
        state = status(manifest).status if manifest.weights is not None else "ready"
        reasons: list[str] = []
        pulls: list[str] = []
        if extra_missing:
            reasons.append(f"needs the {extra} extra")
            pulls.append(f"uv sync --extra {extra}")
        if state != "ready":
            reasons.append(f"weights {state}")
            if state in ("local", "unpublished"):
                pulls.append(f"atlas models pull {manifest.name} --from PATH")
            else:
                pulls.append(f"atlas models pull {manifest.name}")
        if not reasons:
            lines.append(_clip(f"{label} ready"))
            continue
        lines.append(_clip(f"{label} HIDDEN: {'; '.join(reasons)}"))
        lines.append(_clip("  pull: " + "; ".join(pulls)))
    return lines


def _models_block() -> str:
    try:
        lines = _model_lines()
    except Exception as exc:
        return f"models: unavailable ({type(exc).__name__})"
    if not lines:
        return "models: none registered"
    kept: list[str] = []
    size = 0
    for index, line in enumerate(lines):
        rest = len(lines) - index
        tail = f"+{rest} more" if rest else ""
        if size + len(line) + 1 + len(tail) > MODELS_BLOCK_MAX:
            kept.append(f"+{rest} more")
            break
        kept.append(line)
        size += len(line) + 1
    return "\n".join(kept)


def guide_text(topic: str) -> str:
    """Return one guide topic as text ending in exactly one newline.

    Args:
        topic: One of TOPICS.

    Returns:
        The topic body, or an unknown-topic message with a Fix line.
    """
    if topic not in TOPICS:
        return f"unknown topic {topic!r}. Topics: {', '.join(TOPICS)}. Fix: satellite_guide topic=workflow\n"
    body = _bodies()[topic]
    if topic == "models":
        body = body.replace("{{MODELS}}", _models_block())
    return body.strip() + "\n"


def _stack_line(path: Path) -> str | None:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        grid = doc["grid"]
        names = [str(layer["name"]) for layer in doc["layers"]]
        xres = float(abs(grid["transform"][0]))
        shape = f"{len(names)}x{int(grid['height'])}x{int(grid['width'])}"
        return _clip(f"{doc['id']}  {shape}  {grid['crs']}  {xres:.4g} m  {', '.join(names[:6])}")
    except Exception:
        return None


def stacks_text(root: Path | None = None) -> str:
    """List committed stacks under ``<root>/analysis/stacks``.

    Args:
        root: Artifact root; defaults to ``atlas.geo.paths.workspace()``.

    Returns:
        One line per stack (id, LxHxW, crs, resolution, first layers), or a none message.
    """
    from atlas.geo.paths import workspace

    base = root if root is not None else workspace()
    stacks = base / "analysis" / "stacks"
    lines: list[str] = []
    if stacks.is_dir():
        for child in sorted(stacks.iterdir(), key=lambda p: p.name):
            if any(m in child.name for m in _SCRATCH):
                continue
            line = _stack_line(child / "stack.json")
            if line is not None:
                lines.append(line)
    if not lines:
        return f"no stacks under {stacks}\n"
    if len(lines) > _MAX_STACK_LINES:
        lines = [*lines[:_MAX_STACK_LINES], f"+{len(lines) - _MAX_STACK_LINES} more"]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``guide TOPIC`` or ``stacks``; exit 2 on usage errors."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] not in ("guide", "stacks"):
        sys.stdout.write(_USAGE + "\n")
        return 2
    if args[0] == "stacks":
        sys.stdout.write(stacks_text())
        return 0
    if len(args) < 2:
        sys.stdout.write("Topics: " + ", ".join(TOPICS) + "\n")
        return 2
    text = guide_text(args[1])
    sys.stdout.write(text)
    return 0 if args[1] in TOPICS else 2


if __name__ == "__main__":
    raise SystemExit(main())
