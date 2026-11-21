"""Host helpers for the harness2 SWE loop.

``pull`` fetches task images ahead of a sweep, ``build-runtime`` builds the
Linux Atlas runtime volume, and ``compare`` puts the unmodified harness beside
the adapted one after both have been scored::

    python -m atlas.harness_eval build-runtime
    python -m atlas.harness_eval pull django__django-11099 ...
    python -m atlas.harness_eval compare --k 1

The recursion itself lives in harness2; ``scripts/run_harness2_swe.sh`` drives it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from atlas.harness_eval import swe_docker


def _pull(ids: list[str]) -> int:
    """Pull the image of every instance id; ids come from argv or stdin."""
    todo = ids or [line.strip() for line in sys.stdin if line.strip()]
    if not todo:
        print("error: no instance ids given", file=sys.stderr)
        return 1
    failed = 0
    for index, instance_id in enumerate(todo, 1):
        image = swe_docker.instance_image(instance_id)
        try:
            swe_docker.ensure_image(image)
        except swe_docker.DockerError as exc:
            failed += 1
            print(f"[{index}/{len(todo)}] FAILED {image}: {exc}", flush=True)
        else:
            print(f"[{index}/{len(todo)}] {image}", flush=True)
    return 1 if failed else 0


def _load(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [row for row in data if isinstance(row, dict)] if isinstance(data, list) else []


def _sum_costs(paths: list[Path]) -> tuple[float, int]:
    """Return total ``cost_usd`` over meta files and how many reported one."""
    total, counted = 0.0, 0
    for path in paths:
        try:
            cost = json.loads(path.read_text(encoding="utf-8")).get("cost_usd")
        except (OSError, ValueError):
            continue
        if isinstance(cost, int | float):
            total += float(cost)
            counted += 1
    return total, counted


def _compare(runs_root: Path, k: int, seed: int | None) -> int:
    """Print base and adapted resolve rates over the tasks scored in both."""
    namespace = f"swe-atlas-agg{k}" + (f"-s{seed}" if seed is not None else "")
    ns_dir = runs_root / "swe" / "adapt" / namespace
    base: dict[str, list[bool]] = {}
    for row in _load(ns_dir / "base_judged.json"):
        if isinstance(row.get("score"), dict):
            base.setdefault(str(row["task"]), []).append(bool(row["score"].get("all_pass")))
    adapted = {
        str(row["task_id"]): bool(row.get("all_pass"))
        for row in _load(ns_dir / "judged.json")
        if row.get("graded")
    }
    if not base or not adapted:
        print(
            f"error: need both base_judged.json and judged.json under {ns_dir}; "
            "run the `bases` and `judge` steps first",
            file=sys.stderr,
        )
        return 1
    both = sorted(set(base) & set(adapted))
    if not both:
        print("error: no task is scored for both the base and the adapted harness", file=sys.stderr)
        return 1
    base_rate = sum(sum(base[task]) / len(base[task]) for task in both) / len(both)
    adapted_rate = sum(adapted[task] for task in both) / len(both)
    gained = [task for task in both if adapted[task] and not any(base[task])]
    lost = [task for task in both if not adapted[task] and all(base[task])]
    print(f"{namespace}: {len(both)} task(s) scored for both harnesses")
    print(f"  base harness     {100 * base_rate:5.1f}% resolved")
    print(f"  adapted harness  {100 * adapted_rate:5.1f}% resolved")
    print(f"  gained {len(gained)}, lost {len(lost)}")
    for label, tasks in (("gained", gained), ("lost", lost)):
        for task in tasks:
            print(f"    {label}: {task}")
    solver, solver_n = _sum_costs(sorted((runs_root / "swe" / "executions").glob("*/meta.json")))
    improver, improver_n = _sum_costs(
        sorted((runs_root / "swe" / "adapt" / "_audit").glob("**/*.meta.json"))
    )
    print(f"  solver spend    ${solver:.2f} over {solver_n} rollout(s)")
    print(f"  improver spend  ${improver:.2f} over {improver_n} session(s)")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run one helper."""
    parser = argparse.ArgumentParser(prog="atlas.harness_eval", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build-runtime", help="build the Linux Atlas runtime volume")
    pull = sub.add_parser("pull", help="pull task images (ids from argv or stdin)")
    pull.add_argument("instance_ids", nargs="*")
    compare = sub.add_parser("compare", help="base vs adapted resolve rate")
    compare.add_argument("--k", type=int, default=1)
    compare.add_argument("--seed", type=int, default=None)
    compare.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help="harness2 output root; defaults to HARNESS2_RUNS_DIR",
    )
    args = parser.parse_args(argv)
    if args.command == "build-runtime":
        swe_docker.build_runtime()
        problem = swe_docker.runtime_error()
        print(f"error: {problem}" if problem else f"runtime ready: {swe_docker.RUNTIME_VOLUME}")
        return 1 if problem else 0
    if args.command == "pull":
        return _pull(list(args.instance_ids))
    root = args.runs_root or Path(os.environ.get("HARNESS2_RUNS_DIR", "runs"))
    return _compare(root.expanduser().resolve(), args.k, args.seed)


if __name__ == "__main__":
    raise SystemExit(main())
