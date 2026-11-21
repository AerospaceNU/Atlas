"""SWE-bench Lite adapter for harness2-style improvement loops.

Duck-typed against harness2's ``BenchmarkAdapter`` protocol
(``harness2/core/spec.py``). Importing this module never imports harness2, so
default tests stay offline.

Dataset layout: one JSONL on disk with fields ``instance_id, repo,
problem_statement, base_commit`` (+ optional ``hints_text`` which is *not*
shown to the improver). Point ``SWE_BENCH_JSONL`` at it, or pass a path to
:class:`SWEAdapter`. Get Lite via::

    curl -L https://huggingface.co/datasets/SWE-bench/SWE-bench/blob/main/data/lite/test.jsonl

Each task maps to one directory under ``swe_data/<instance_id>/`` holding
``task.json`` (the record above) and ``repo/`` (a checkout at ``base_commit``).
``execute`` checks the repo out fresh per ``run_id``, runs the Atlas headless
runner against the issue text only (never ``test_patch``), and writes
``trace.json``, ``patch.diff`` and ``trajectory.md`` beside the run.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from atlas.harness_eval.headless import collect_patch, render_trajectory, run_headless
from atlas.harness_eval.substrate import AtlasSubstrate
from atlas.harness_eval.surface import ATLAS_SURFACE

# BenchPrompts vocabulary: must describe the WORK, not measurement. No
# benchmark/grader/rubric/score words.
SWE_WORK = (
    "Fix a GitHub issue in a Python repository. The agent receives the issue "
    "text and a checkout at a base commit, then produces a patch that "
    "addresses the reported behaviour."
)
SWE_INV_LEVERS = (
    "- reproduce the issue first: find the relevant module, write the "
    "smallest reproduction, and confirm it fails before editing\n"
    "- make the minimal change that matches existing style; keep the public "
    "API stable and add no new dependencies\n"
    "- rerun the reproduction plus the surrounding test module; inspect the "
    "result and iterate until the behaviour matches the issue text"
)
SWE_DOM_LEVERS = (
    "- use the repository's own conventions for the touched package; follow "
    "its error types, naming and docstring style\n"
    "- keep the change scoped to the described symptom; do not refactor "
    "unrelated files or change unrelated behaviour"
)
SWE_YARDSTICK = (
    "Did the patch address the issue as written, keep the reproduction "
    "green, and leave unrelated files untouched?"
)

ENV_JSONL = "SWE_BENCH_JSONL"
ENV_DATA_ROOT = "SWE_BENCH_DATA_ROOT"


@dataclass
class TaskRecord:
    """One SWE-bench Lite instance without test hints."""

    instance_id: str
    repo: str
    problem_statement: str
    base_commit: str
    fail_to_pass: tuple[str, ...] = ()
    pass_to_pass: tuple[str, ...] = ()


def _load_jsonl(path: Path) -> list[TaskRecord]:
    records: list[TaskRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        fail = raw.get("fail_to_pass", raw.get("FAIL_TO_PASS", []))
        passed = raw.get("pass_to_pass", raw.get("PASS_TO_PASS", []))
        records.append(
            TaskRecord(
                instance_id=str(raw["instance_id"]),
                repo=str(raw["repo"]),
                problem_statement=str(raw["problem_statement"]),
                base_commit=str(raw["base_commit"]),
                fail_to_pass=tuple(str(x) for x in (fail or [])),
                pass_to_pass=tuple(str(x) for x in (passed or [])),
            )
        )
    return records


class SWEAdapter:
    """harness2-shaped adapter: tasks, execute, trajectory, score."""

    def __init__(
        self,
        *,
        jsonl: Path | str | None = None,
        data_root: Path | str | None = None,
        runs_root: Path | str | None = None,
        runs_dir: Path | str | None = None,
        model_id: str = "",
        mechanism: str = "opencode",
    ) -> None:
        self._jsonl = Path(jsonl or os.environ.get(ENV_JSONL, ""))
        self._data_root = Path(data_root or os.environ.get(ENV_DATA_ROOT, "swe_data"))
        self.runs_root = Path(runs_dir or runs_root or "runs/swe")
        self.substrate = AtlasSubstrate()
        self._records: dict[str, TaskRecord] = {}
        # BenchSpec-compatible surface (duck-typed; harness2 validates the real one).
        self.spec = type(
            "Spec",
            (),
            {
                "name": "swe",
                "install_mode": "workspace_agents_md",
                "obs_trunc": 50_000,
                "leak_globs": ("test_patch", "test.patch"),
            },
        )()
        self._mechanism_value = mechanism

    # --- lazy index ----------------------------------------------------
    @property
    def _index(self) -> dict[str, TaskRecord]:
        if not self._records:
            if not self._jsonl.is_file():
                raise FileNotFoundError(f"SWE-bench JSONL not found: {self._jsonl}")
            for rec in _load_jsonl(self._jsonl):
                self._records[rec.instance_id] = rec
        return self._records

    def tasks(self, split: str = "all") -> list[str]:
        """Task ids in stable order (harness2 requires reproducible streams)."""
        if split not in ("all", "train", "test"):
            raise ValueError(f"unknown split {split!r}")
        return sorted(self._index)

    def domain_of(self, task_id: str) -> str:
        """Repo is the grouping unit, so lessons never leak across repos."""
        return self._index[task_id].repo

    def task_prompt(self, task_id: str) -> str:
        """Issue text only; hints and test patch are never shown to the solver."""
        rec = self._index[task_id]
        return f"Fix this issue in {rec.repo}:\n\n{rec.problem_statement}\n"

    def task_files(self, task_id: str) -> Path | None:
        """Return the dataset task dir holding task.json + repo/, or None."""
        base = self._data_root / task_id
        return base if base.is_dir() else None

    # --- harness -------------------------------------------------------
    def base_harness(self) -> Path:
        """Materialise the base Atlas tree under runs_root/base."""
        from atlas.harness_eval.surface import base_tree, write_tree

        dest = self.runs_root / "base"
        write_tree(ATLAS_SURFACE, base_tree(), dest)
        return dest

    def validate_tree(self, tree: dict[str, str]) -> None:
        """Reject off-surface trees; message is fed back to the improver."""
        from atlas.harness_eval.surface import validate

        validate(ATLAS_SURFACE, tree)

    # --- rollout -------------------------------------------------------
    def _run_workspace(self, task_id: str, run_id: str) -> Path:
        run_dir = self.runs_root / "executions" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        record = self._index[task_id]
        src_repo = self._data_root / task_id / "repo"
        ws = run_dir / "workspace"
        if not ws.exists():
            if src_repo.is_dir():
                shutil.copytree(src_repo, ws)
            elif record.repo.startswith("file://"):
                shutil.copytree(record.repo[len("file://") :], ws)
            else:
                ws.mkdir(parents=True, exist_ok=True)
                (ws / "README.md").write_text(f"# {record.repo}\n", encoding="utf-8")
            if (ws / ".git").exists():
                subprocess.run(
                    ["git", "-C", str(ws), "checkout", "--detach", record.base_commit],
                    capture_output=True,
                    timeout=30,
                )
        return ws

    def execute(
        self,
        task_id: str,
        harness_dir: Path | None,
        run_id: str,
        *,
        timeout: int | None = None,
        model: Any = None,
        model_id: str = "",
        api_key: str = "",
    ) -> Path:
        """Run one Atlas rollout and return its run directory.

        The workspace is checked out at ``base_commit``; the agent sees only
        the issue text. ``harness_dir=None`` means the unmodified base.
        """
        run_dir = self.runs_root / "executions" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        ws = self._run_workspace(task_id, run_id)
        active_harness = harness_dir or self.base_harness()
        headless = run_headless(
            workspace=ws,
            prompt=self.task_prompt(task_id),
            model_id=model_id,
            api_key=api_key,
            harness_dir=Path(active_harness),
            trace_path=run_dir / "trace.json",
            model=model,
        )
        patch = collect_patch(ws)
        (run_dir / "patch.diff").write_text(patch, encoding="utf-8")
        (run_dir / "trajectory.md").write_text(render_trajectory(headless.run), encoding="utf-8")
        return run_dir

    def delivered(self, run_dir: Path) -> bool:
        """Did the rollout produce a non-empty patch?"""
        patch = Path(run_dir) / "patch.diff"
        return patch.is_file() and bool(patch.read_text(encoding="utf-8").strip())

    def trajectory(self, run_dir: Path, *, obs_trunc: int = 50_000) -> str:
        """Return the shared-grammar trajectory, never including score."""
        p = Path(run_dir) / "trajectory.md"
        if p.is_file():
            return p.read_text(encoding="utf-8")
        from atlas.agent.trace import read_trace

        try:
            return render_trajectory(
                read_trace(Path(run_dir) / "trace.json").run, obs_trunc=obs_trunc
            )
        except Exception:
            return ""

    def deliverable_text(self, run_dir: Path) -> str:
        """Patch diff as the judged artifact."""
        return (
            (Path(run_dir) / "patch.diff").read_text(encoding="utf-8")
            if (Path(run_dir) / "patch.diff").exists()
            else ""
        )

    def deliverable_files(self, run_dir: Path) -> dict[str, str]:
        """One entry per artifact produced (here just the patch)."""
        text = self.deliverable_text(run_dir)
        return {"patch.diff": text} if text else {}

    # --- score ---------------------------------------------------------
    def score(
        self,
        run_dir: Path,
        task_id: str,
        *,
        run_pytest: bool = False,
        fail_to_pass: list[str] | None = None,
    ) -> dict[str, Any] | None:
        """Deterministic score; None means unscored (not zero).

        ``run_pytest`` False (default) keeps tests offline: resolved is True
        iff the patch exists. Pass True on Linux/Docker to run the instance's
        ``FAIL_TO_PASS`` list for real.
        """
        run_dir = Path(run_dir)
        if not self.delivered(run_dir):
            return {"resolved": False, "reason": "no patch"}
        if not run_pytest:
            return {"resolved": True, "reason": "patch present (pytest not run)"}
        ws = run_dir / "workspace"
        tests = fail_to_pass or []
        if not tests:
            try:
                rec = self._data_root / task_id / "task.json"
                data = json.loads(rec.read_text(encoding="utf-8"))
                tests = list(data.get("fail_to_pass", data.get("FAIL_TO_PASS", [])))
            except Exception:
                tests = []
            if not tests:
                try:
                    rec2 = self._index.get(task_id)
                    if rec2 is not None:
                        tests = list(rec2.fail_to_pass)
                except Exception:
                    tests = []
        if not tests:
            return {"resolved": False, "reason": "no fail_to_pass list"}
        cmd = ["python", "-m", "pytest", *tests, "-q"]
        proc = subprocess.run(cmd, cwd=str(ws), capture_output=True, text=True, timeout=120)
        return {"resolved": proc.returncode == 0, "pytest": proc.stdout[-2000:]}


def swe_prompts() -> dict[str, str]:
    """Return the harness2 BenchPrompts text for SWE (for harness2 wiring)."""
    return {
        "work": SWE_WORK,
        "inv": SWE_INV_LEVERS,
        "dom": SWE_DOM_LEVERS,
        "yardstick": SWE_YARDSTICK,
    }
