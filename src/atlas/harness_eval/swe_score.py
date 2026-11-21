"""Score one SWE-bench patch with the benchmark's own tests.

Run as a script in an environment that has ``swebench`` (see
:data:`SWEBENCH_REQUIREMENT`); Atlas itself does not depend on it::

    uv run --no-project --with swebench==4.1.0 python swe_score.py --eval-jsonl rows.jsonl --instance-id <id> --patch patch.diff --out score.json

It starts a fresh container from the task's instance image, applies the patch,
runs the evaluation script the ``swebench`` package builds for that instance,
and grades the log with the package's own parser. The verdict is the
benchmark's: ``resolved`` means every FAIL_TO_PASS and PASS_TO_PASS test
passed. This module is the only place the task's tests are read. It imports
nothing from Atlas, and the rollout container never sees its inputs.

Deviation from stock SWE-bench scoring: files the agent patch creates that the
hidden test patch also creates are stripped from the agent patch first, so the
test patch applies instead of failing with "already exists".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from itertools import pairwise
from pathlib import Path
from typing import Any

SWEBENCH_REQUIREMENT = "swebench==4.1.0"
PLATFORM = "linux/amd64"
WORKSPACE = "/testbed"
_APPLY_COMMANDS = (
    "git apply --verbose",
    "git apply --verbose --reject",
    "patch --batch --fuzz=5 -p1 -i",
)


_DIFF_HEADER = "diff --git "
_NEW_FILE = re.compile(r"^(new file mode\b|--- /dev/null)", re.MULTILINE)
_TARGET = re.compile(r"^\+\+\+ b/(.+?)\s*$", re.MULTILINE)
_HEADER_PATHS = re.compile(r"^diff --git a/(.+) b/(.+?)\s*$")


def _split_sections(patch: str) -> tuple[str, list[str]]:
    """Split a git patch into (preamble, sections); each section starts at ``diff --git ``."""
    lines = patch.splitlines(keepends=True)
    starts = [i for i, line in enumerate(lines) if line.startswith(_DIFF_HEADER)]
    if not starts:
        return patch, []
    bounds = [*starts, len(lines)]
    sections = ["".join(lines[a:b]) for a, b in pairwise(bounds)]
    return "".join(lines[: starts[0]]), sections


def _section_target(section: str) -> str | None:
    match = _TARGET.search(section)
    if match:
        return match.group(1)
    header = _HEADER_PATHS.match(section.splitlines()[0]) if section else None
    return header.group(2) if header else None


def strip_test_created_files(patch: str, test_patch: str) -> tuple[str, list[str]]:
    """Drop patch sections that create a file the test patch also creates.

    Returns the remaining patch (byte-identical apart from removed sections)
    and the stripped paths. A patch without ``diff --git`` headers is returned
    unchanged.
    """
    _, test_sections = _split_sections(test_patch)
    created = {
        target
        for section in test_sections
        if _NEW_FILE.search(section) and (target := _section_target(section))
    }
    preamble, sections = _split_sections(patch)
    if not sections or not created:
        return patch, []
    kept: list[str] = []
    stripped: list[str] = []
    for section in sections:
        target = _section_target(section)
        if target in created:
            stripped.append(target)
        else:
            kept.append(section)
    if not stripped:
        return patch, []
    return (preamble + "".join(kept) if kept else ""), stripped


def _docker(*args: str, timeout: float = 300) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args], capture_output=True, text=True, errors="replace", timeout=timeout
    )


def load_instance(eval_jsonl: Path, instance_id: str) -> dict[str, Any]:
    """Return the full dataset row for ``instance_id``."""
    with eval_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row: dict[str, Any] = json.loads(line)
                if row.get("instance_id") == instance_id:
                    return row
    raise KeyError(f"{instance_id} is not in {eval_jsonl}")


def _verdict(report: dict[str, Any], note: str = "") -> dict[str, Any]:
    """Reduce a swebench report to the fields the adapter reads."""
    status = report.get("tests_status") or {}

    def count(key: str) -> tuple[int, int]:
        group = status.get(key) or {}
        passed = len(group.get("success") or [])
        return passed, passed + len(group.get("failure") or [])

    f2p_ok, f2p_all = count("FAIL_TO_PASS")
    p2p_ok, p2p_all = count("PASS_TO_PASS")
    return {
        "resolved": bool(report.get("resolved")),
        "patch_applied": bool(report.get("patch_successfully_applied")),
        "fail_to_pass": [f2p_ok, f2p_all],
        "pass_to_pass": [p2p_ok, p2p_all],
        "note": note,
    }


def score(instance: dict[str, Any], patch: str, log_dir: Path, *, timeout: int) -> dict[str, Any]:
    """Apply ``patch`` in a fresh task container, run the tests and grade them.

    Raises:
        RuntimeError: The container could not be started, or the tests timed
            out. Neither is a verdict on the patch.
    """
    from swebench.harness.grading import get_eval_report  # type: ignore[import-not-found]
    from swebench.harness.test_spec.test_spec import (  # type: ignore[import-not-found]
        make_test_spec,
    )

    instance_id = str(instance["instance_id"])
    spec = make_test_spec(instance, namespace="swebench", arch="x86_64")
    total = len(spec.FAIL_TO_PASS) + len(spec.PASS_TO_PASS)
    empty = {"resolved": False, "patch_applied": False}
    patch, stripped = strip_test_created_files(patch, str(instance.get("test_patch") or ""))
    stripped_note = f"stripped test-created files: {', '.join(stripped)}" if stripped else ""
    if stripped:
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "stripped_files.txt").write_text("\n".join(stripped) + "\n", encoding="utf-8")
    if not patch.strip():
        return _verdict(empty, "; ".join(n for n in ("empty patch", stripped_note) if n)) | {
            "fail_to_pass": [0, len(spec.FAIL_TO_PASS)],
            "pass_to_pass": [0, total - len(spec.FAIL_TO_PASS)],
        }
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "patch.diff").write_text(patch, encoding="utf-8")
    (log_dir / "eval.sh").write_text(spec.eval_script, encoding="utf-8")
    digest = hashlib.sha1(str(log_dir.resolve()).encode()).hexdigest()[:12]
    name = f"atlas-swe-eval-{digest}"
    _docker("rm", "-f", name)
    started = _docker(
        "run", "-d", "--name", name, "--platform", PLATFORM, "-w", WORKSPACE,
        "--entrypoint", "sleep", spec.instance_image_key, "infinity",
        timeout=3600,
    )  # fmt: skip
    if started.returncode != 0:
        raise RuntimeError(f"could not start {spec.instance_image_key}: {started.stderr[-400:]}")
    try:
        for source, target in (("patch.diff", "/tmp/patch.diff"), ("eval.sh", "/eval.sh")):
            copied = _docker("cp", str(log_dir / source), f"{name}:{target}")
            if copied.returncode != 0:
                raise RuntimeError(f"docker cp {source} failed: {copied.stderr[-300:]}")
        applied = None
        for command in _APPLY_COMMANDS:
            applied = _docker("exec", name, "bash", "-c", f"{command} /tmp/patch.diff")
            if applied.returncode == 0:
                break
        if applied is None or applied.returncode != 0:
            detail = "" if applied is None else (applied.stdout + applied.stderr)[-600:]
            (log_dir / "apply_error.txt").write_text(detail, encoding="utf-8")
            return _verdict(
                empty, "; ".join(n for n in ("patch did not apply", stripped_note) if n)
            ) | {
                "fail_to_pass": [0, len(spec.FAIL_TO_PASS)],
                "pass_to_pass": [0, total - len(spec.FAIL_TO_PASS)],
            }
        try:
            ran = _docker("exec", name, "/bin/bash", "/eval.sh", timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"tests timed out after {timeout}s") from exc
        log_path = log_dir / "test_output.txt"
        log_path.write_text(ran.stdout + ran.stderr, encoding="utf-8")
    finally:
        _docker("rm", "-f", name)
    prediction = {
        "instance_id": instance_id,
        "model_patch": patch,
        "model_name_or_path": "atlas",
    }
    report = get_eval_report(
        test_spec=spec,
        prediction=prediction,
        test_log_path=str(log_path),
        include_tests_status=True,
    )[instance_id]
    (log_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    verdict = _verdict(report, stripped_note)
    if not verdict["patch_applied"]:
        # The model patch applied above, so this is the evaluation script failing.
        raise RuntimeError("evaluation log has no test output; see test_output.txt")
    return verdict


def main(argv: list[str] | None = None) -> int:
    """Score one patch and write the verdict as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-jsonl", type=Path, required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--patch", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args(argv)
    if not args.patch.is_file():
        print(
            f"swe_score: patch file not found: {args.patch}. Collection failed or the "
            "rollout did not finish; see collect.json beside it.",
            file=sys.stderr,
        )
        return 1
    try:
        instance = load_instance(args.eval_jsonl, args.instance_id)
        patch = args.patch.read_text(encoding="utf-8")
        verdict = score(
            instance, patch, args.log_dir or args.out.parent / "eval", timeout=args.timeout
        )
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    args.out.write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verdict))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
