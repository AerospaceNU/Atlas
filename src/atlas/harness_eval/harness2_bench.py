"""harness2-native SWE adapter (installed as ``harness2/benchmarks/swebench.py``).

This file is the source of truth for ``--bench swe --substrate atlas``.
``scripts/setup_harness2_swe.sh`` copies it into a local harness2 checkout. It
imports harness2 inside functions so Atlas unit tests can collect it without
harness2 present; the adapter body only runs inside harness2.

The solver is Atlas, the improver is whichever channel harness2 is configured
with (OpenCode by default). One rollout is one container started from the
task's SWE-bench instance image (:mod:`atlas.harness_eval.swe_docker`).
Scoring runs the benchmark's own tests in a second, fresh container
(:mod:`atlas.harness_eval.swe_score`), so there is no model judge. The
improver and the solver see the issue text and nothing else: the tests and
the reference patch live in a separate file that only the scorer opens.
"""

# Bridge file: needs harness2 installed to type-check; ignore when absent.
# mypy: ignore-errors

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from pathlib import Path

ENV_JSONL = "SWE_BENCH_JSONL"
ENV_EVAL_JSONL = "SWE_BENCH_EVAL_JSONL"
ENV_MAX_TOOL_CALLS = "ATLAS_SWE_MAX_TOOL_CALLS"
ENV_REASONING_EFFORT = "ATLAS_SWE_REASONING_EFFORT"
ENV_TOOL_PROFILE = "ATLAS_TOOL_PROFILE"
ENV_TEMPERATURE = "ATLAS_TEMPERATURE"
ENV_TOP_P = "ATLAS_TOP_P"
ENV_ROLLOUT_MINUTES = "ATLAS_SWE_ROLLOUT_MINUTES"
ENV_ROLLOUT_ATTEMPTS = "ATLAS_SWE_ROLLOUT_ATTEMPTS"
ENV_SCORE_PARALLEL = "ATLAS_SWE_SCORE_PARALLEL"
ENV_IMPROVER_EFFORT = "HARNESS2_IMPROVER_EFFORT"
# Files a rollout writes; cleared before a rerun so a stale verdict cannot survive it.
_RUN_FILES = (
    "meta.json",
    "trace.json",
    "trajectory.md",
    "patch.diff",
    "rollout.log",
    "base_ref",
    "score.json",
)

_ATLAS_SURFACE_MD = """\
# WHAT YOU CAN CHANGE

The harness is a tree of components around the Atlas agent. Each has a different runtime cost
and a different reach:

| path | what it is | when it is in context |
|---|---|---|
| `systemprompt.md` | role and procedure | ALWAYS ON -- every token competes with the agent's own output budget |
| `guardrails.md` | standing rules | ALWAYS ON -- appended to the system prompt |
| `LongTermMEMORY.md` | durable domain facts | ALWAYS ON -- appended to the system prompt |
| `tool_descriptions/tool_guidance.md` | how to use the tools | ALWAYS ON -- appended to the system prompt |
| `skills/<name>/SKILL.md` | a reusable workflow | name and description always on; the body loads ON DEMAND through the `use_skill` tool |
| `skills/<name>/scripts/*` | a helper carried with its skill | run via `bash` |
| `sub_agents/<name>/AGENT.md` | a delegated role | runs in its OWN context through `spawn_agent`, returns text |
| `scripts/<name>.py` | a deterministic helper | run on demand via `bash` |

Nothing else is editable. In particular you cannot add, remove or change a TOOL: the agent's
tools are fixed, and the harness only changes how the agent is told to use them. They are
`read_files`, `read_file`, `ls`, `glob`, `grep`, `outline`, `find_references`, `write_file`,
`edit_file`, `patch_file`, `apply_patch`, `bash` (one shell command, 30s by default and
900s at most per call), `run_tests` (an argv list with no shell, 900s at most), `diagnostics`,
`git_status`, `git_diff`, `git_commit`, `web_fetch`, `update_plan`, `read_plan`, `spawn_agent`
and `use_skill`. A run ends when the agent answers without a tool call or spends its tool-call
budget, so a harness that makes the agent spend calls on ceremony costs it the work.

The components that are easy to get wrong:

`skills/` -- frontmatter needs `name:` equal to the directory name and a `description:`, or the
skill is dropped silently at load. It must also be named in the always-on text, and any helper
it calls must travel with it; both rules are below.

`sub_agents/` -- frontmatter needs a `description:`. The agent delegates by calling
`spawn_agent` with a task whose first line is `agent: <name>`; the role's file becomes part of
the child's system prompt. The child sees only that file and the task text, never the main
conversation, and it returns text. It is worth it for exactly one shape of problem: a bounded
sub-task whose INPUT is large and whose ANSWER is small. Writes are refused unless the call
sets `allow_writes`.

`scripts/` -- a script runs under the REPOSITORY'S interpreter, which can be as old as
Python 3.6: standard library only, no modern syntax. Name it in a skill or the always-on text
as `python3 scripts/<name>.py`; the runtime installs it and rewrites that path.

Change only what a trajectory shows needs changing -- but WHERE you change something, the FORM
matters as much as the content, and the cheap form is not the obvious one.

ALWAYS-ON PROSE IS THE EXPENSIVE OPTION. Every word of it is in context on every run. An
on-demand component costs nothing until the moment it is invoked.

So prefer, in this order:
  - a SKILL for anything with a recognisable trigger -- a procedure for a kind of change, a
    framework, a recurring situation. This is the default form for new instruction.
  - a SCRIPT for a deterministic operation the runs hand-rolled and got wrong, or did slowly.
  - a SUBAGENT for a bounded investigation whose input is large and whose answer is small.
  - ALWAYS-ON TEXT last, and only for a short rule that genuinely applies to every run. If you
    add always-on text, cut something else to pay for it.

THREE RULES THAT DISCARD THE WHOLE PROPOSAL. Validation runs before any rollout, and a harness
that fails it is thrown away entire.

1. EVERY SKILL NEEDS A TRIGGER LINE. If you create or keep `skills/<name>/SKILL.md`, its
   directory name must appear VERBATIM in `systemprompt.md` or `guardrails.md`, in the SAME edit:

       When <the situation this skill handles>, use the `<name>` skill.

2. `systemprompt.md` KEEPS ITS FIXED SENTENCES. It is the product's own prompt and a set of
   its sentences is required verbatim. Add to the file and reword around them; do not delete or
   rephrase them. Validation names any you removed.

3. EVERY SCRIPT YOU NAME MUST EXIST. A `scripts/<file>.py` invoked from a SKILL.md or from
   always-on text must ship in the same proposal, a skill's own helper at
   `skills/<skill-name>/scripts/<file>.py`; one dropped loose in the skill directory is OFF THE
   EDITABLE SURFACE and silently discarded.

HOW THE RESULT IS TAKEN. When the agent stops, every change in the repository working tree --
edited, added and deleted files alike -- is collected as its patch. A scratch file the agent
leaves behind is part of that patch, and work it did not write to the working tree is not."""

_LOOP_LEVER = (
    "`scripts/<name>.py` check that the always-on text makes the last step before finishing "
    "-- for example, confirming a {deliverable} exists"
)
_TRIGGER_SENTENCE = (
    "A skill exists only once its name appears in the always-on text: write that trigger line "
    "in the same edit."
)


def register_surface_text() -> None:
    """Teach harness2's prompt builder the Atlas surface.

    harness2 ships improver text for the OpenCode and Codex surfaces only and
    refuses any other surface name. This adds the Atlas description, which
    names the Atlas tools and the components Atlas can actually load.
    """
    from harness2.core import prompts

    if "atlas" in prompts._SURFACE_TEXTS:
        return
    how = prompts._surface_part(prompts._SURFACE_OPENCODE, "# HOW TO MAKE A CHANGE")
    surface_md = f"{_ATLAS_SURFACE_MD}\n\n{how}"
    prompts._SURFACE_TEXTS["atlas"] = prompts._SurfaceText(
        surface_md=surface_md,
        inv=prompts._INV_TEMPLATE.replace("__LOOP_LEVER__", _LOOP_LEVER),
        dom=prompts._DOM_TEMPLATE.replace("__TRIGGER_SENTENCE__", _TRIGGER_SENTENCE),
        tts_surface=surface_md,
    )


def _bench_spec(substrate: str) -> object:
    """Build the harness2 BenchSpec for (swe, atlas)."""
    from harness2 import config
    from harness2.core.spec import BenchPrompts, BenchSpec, resolve_channel_model

    if substrate != "atlas":
        raise ValueError(f"--bench swe runs the Atlas solver only; got --substrate {substrate!r}")
    ch = resolve_channel_model(None, None, effort=os.environ.get(ENV_IMPROVER_EFFORT, "high"))
    return BenchSpec(
        name="swe",
        runs_root=config.RUNS / "swe",
        install_mode="workspace_agents_md",
        prompts=BenchPrompts(
            work=(
                "Fix a GitHub issue in a Python repository. The agent receives "
                "the issue text and a checkout at a base commit, then produces "
                "a patch that addresses the reported behaviour."
            ),
            domain_word="repository",
            deliverable_word="patch",
            inv_levers=(
                "- reproduce the issue first: find the relevant module, write "
                "the smallest reproduction, and confirm it fails before editing\n"
                "- make the minimal change that matches existing style; keep "
                "the public API stable and add no new dependencies\n"
                "- rerun the reproduction plus the surrounding test module"
            ),
            dom_levers=(
                "- use the repository's own conventions for the touched "
                "package; follow its error types, naming and docstring style\n"
                "- keep the change scoped to the described symptom; do not "
                "refactor unrelated files"
            ),
            yardstick=(
                "Did the patch address the issue as written, keep the "
                "reproduction green, and leave unrelated files untouched?"
            ),
            selection_checks=(
                "Prefer the patch that touches the fewest files, follows "
                "repo conventions, and matches the issue text exactly."
            ),
            selection_lead=(
                "COVERAGE OF WHAT WAS NAMED. Count what the issue explicitly "
                "asks to be fixed and pick the candidate that addresses the "
                "most of it. A specific fix the source supports is worth more "
                "than a vague change."
            ),
        ),
        channels={"proposer": ch, "selector": ch},
        domain_word="repository",
        leak_globs=("test_patch", "test.patch", "score.json", "*.eval.jsonl"),
        allow_subagents=True,
        obs_trunc=50_000,
        proposer_deliverables="contents",
    )


class _AtlasSubstrateLike:
    """Minimal SubstrateLike for core (only surface() is read by core)."""

    name = "atlas"

    def surface(self) -> object:
        """Return the harness2-native Atlas surface."""
        from atlas.harness_eval.surface import to_harness2_surface

        return to_harness2_surface()


def _load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _atlas_root() -> Path:
    """Return the Atlas checkout this module runs from."""
    import atlas

    return Path(atlas.__file__).resolve().parents[2]


def _api_key() -> str:
    """Return the OpenRouter key from the environment or the Atlas ``.env``."""
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if key:
        return key
    from dotenv import dotenv_values

    return str(dotenv_values(_atlas_root() / ".env").get("OPENROUTER_API_KEY") or "")


def _non_empty(tree: dict[str, str]) -> dict[str, str]:
    """Drop empty files, which a tree may list or omit without meaning anything."""
    return {rel: text for rel, text in tree.items() if text}


def _env_effort() -> str | None:
    """Return the validated reasoning effort from the environment, or ``None``."""
    from atlas.agent.model import validate_reasoning_effort

    raw = os.environ.get(ENV_REASONING_EFFORT, "").strip()
    return validate_reasoning_effort(raw or None)


def _env_profile() -> str:
    """Return the validated tool profile from the environment (default full)."""
    from atlas.harness_eval.headless import TOOL_PROFILES

    raw = os.environ.get(ENV_TOOL_PROFILE, "").strip() or "full"
    if raw not in TOOL_PROFILES:
        raise ValueError(f"{ENV_TOOL_PROFILE} must be one of {TOOL_PROFILES}, got {raw!r}")
    return raw


def _env_float(name: str) -> float | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


def _env_int(name: str, default: int | None) -> int | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


class SWEBenchAdapter:
    """SWE-bench behind the harness2 BenchmarkAdapter protocol, solved by Atlas."""

    def __init__(
        self,
        substrate: str = "atlas",
        *,
        split_file: Path | None = None,
        jsonl: Path | None = None,
        eval_jsonl: Path | None = None,
        model_id: str = "",
    ) -> None:
        register_surface_text()
        self.split_file = split_file
        self.spec = _bench_spec(substrate)
        self.substrate = _AtlasSubstrateLike()
        self._jsonl = Path(jsonl or os.environ.get(ENV_JSONL, ""))
        explicit = eval_jsonl or os.environ.get(ENV_EVAL_JSONL, "")
        self._eval_jsonl = (
            Path(explicit) if explicit else self._jsonl.with_name(f"{self._jsonl.stem}.eval.jsonl")
        )
        self._model_id = model_id
        self._index: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._score_gate = threading.BoundedSemaphore(_env_int(ENV_SCORE_PARALLEL, 2) or 2)

    def _records(self) -> dict[str, dict]:
        if not self._index:
            if not self._jsonl.is_file():
                raise FileNotFoundError(
                    f"SWE-bench JSONL not found: {self._jsonl} (set {ENV_JSONL}; build one "
                    f"with scripts/fetch_swe_dataset.py)"
                )
            # Assigned whole: domains run on threads and must not see a partial index.
            self._index = {str(rec["instance_id"]): rec for rec in _load_jsonl(self._jsonl)}
        return self._index

    def _model(self) -> str:
        """Return the OpenRouter id the solver runs on."""
        from harness2 import config

        return (
            self._model_id
            or os.environ.get("OPENROUTER_MODEL", "")
            or config.SOLVER_MODEL_OC.removeprefix("openrouter/")
        )

    # --- protocol ------------------------------------------------------
    def tasks(self, split: str = "all") -> list[str]:
        """Sorted instance ids; split file support mirrors harness2/select."""
        from harness2.benchmarks.tasks import select

        return select(
            {t: self.domain_of(t) for t in sorted(self._records())},
            split,
            self.split_file,
            benchmark="swe",
        )

    def domain_of(self, task_id: str) -> str:
        """Repo groups lessons so they never leak across repos."""
        return str(self._records()[task_id]["repo"])

    def task_prompt(self, task_id: str) -> str:
        """Issue text only; hints and the test patch never reach a model."""
        rec = self._records()[task_id]
        return (
            f"Fix this issue in {rec['repo']}. The repository is checked out in the "
            f"working directory.\n\n{rec['problem_statement']}\n"
        )

    def task_files(self, task_id: str) -> Path | None:
        """No source material: the repository lives in the task image, not on the host."""
        return None

    def base_harness(self) -> Path:
        """Materialise the base Atlas tree under spec.runs_root/base.

        Written only when the directory does not already hold that tree.
        Domains run on threads and rollouts mount this directory, so a
        rewrite per call would delete files under a running rollout.
        """
        from harness2.core import tree as HT

        from atlas.harness_eval.surface import base_tree

        dest = self.spec.runs_root / "base"
        surface = self.substrate.surface()
        with self._lock:
            want = base_tree()
            have = HT.read_tree(surface, dest) if dest.is_dir() else None
            if have is None or _non_empty(have) != _non_empty(want):
                HT.write_tree(surface, want, dest)
        return dest

    def validate_tree(self, tree: dict[str, str]) -> None:
        """Keep the system prompt shippable: every sentence Atlas requires stays.

        ``atlas.agent.system_prompt`` refuses a package prompt that lacks its
        required rules, so a tree that dropped one could never be adopted back
        into Atlas. The message is returned to the improver as retry feedback.
        """
        from atlas.agent.system_prompt import _REQUIRED_RULES

        prompt = tree.get("systemprompt.md", "")
        missing = [rule for rule in _REQUIRED_RULES if rule not in prompt]
        if prompt.strip() and missing:
            quoted = "; ".join(f'"{rule}"' for rule in missing)
            raise ValueError(
                "systemprompt.md must keep these sentences verbatim (add text around them, "
                f"do not delete or reword them): {quoted}"
            )

    def run_dir(self, run_id: str) -> Path:
        """Return the directory one rollout writes to."""
        return self.spec.runs_root / "executions" / run_id.replace("/", "__")

    def execute(
        self, task_id: str, harness_dir: Path | None, run_id: str, *, timeout: int | None = None
    ) -> Path:
        """Run one Atlas rollout in the task's container, idempotently by ``run_id``.

        A finished run directory is returned as-is, so a resumed sweep does
        not pay for a rollout twice. A rollout that errored before producing
        a trajectory is retried, then raised.
        """
        from atlas.harness_eval.swe_docker import RolloutSpec, run_rollout

        self._records()[task_id]
        run_dir = self.run_dir(run_id)
        meta_path = run_dir / "meta.json"
        finished = meta_path.is_file() and (run_dir / "patch.diff").is_file()
        if finished and json.loads(meta_path.read_text(encoding="utf-8")).get("status") != "error":
            return run_dir
        minutes = timeout / 60 if timeout else (_env_int(ENV_ROLLOUT_MINUTES, 45) or 45)
        spec = RolloutSpec(
            instance_id=task_id,
            run_dir=run_dir,
            harness_dir=Path(harness_dir) if harness_dir else self.base_harness(),
            prompt=self.task_prompt(task_id),
            model=self._model(),
            api_key=_api_key(),
            max_tool_calls=_env_int(ENV_MAX_TOOL_CALLS, None),
            reasoning_effort=_env_effort(),
            tool_profile=_env_profile(),
            temperature=_env_float(ENV_TEMPERATURE),
            top_p=_env_float(ENV_TOP_P),
            deadline_seconds=minutes * 60,
        )
        meta: dict = {}
        for _ in range(max(1, _env_int(ENV_ROLLOUT_ATTEMPTS, 2) or 2)):
            for name in _RUN_FILES:
                (run_dir / name).unlink(missing_ok=True)
            shutil.rmtree(run_dir / "eval", ignore_errors=True)
            meta = run_rollout(spec)
            if meta.get("status") != "error":
                return run_dir
        raise RuntimeError(f"rollout failed: {meta.get('error', 'unknown error')}")

    def delivered(self, run_dir: Path) -> bool:
        """Non-empty patch file."""
        p = Path(run_dir) / "patch.diff"
        return p.is_file() and bool(p.read_text(encoding="utf-8", errors="replace").strip())

    def trajectory(self, run_dir: Path, *, obs_trunc: int = 50_000) -> str:
        """Shared grammar, rendered from the trace; never includes a verdict."""
        from atlas.agent.trace import read_trace
        from atlas.harness_eval.headless import render_trajectory

        run_dir = Path(run_dir)
        rendered = run_dir / "trajectory.md"
        trace = run_dir / "trace.json"
        # A killed rollout has no trace, only the steps it appended as it ran.
        if rendered.is_file() and (obs_trunc >= 50_000 or not trace.is_file()):
            text = rendered.read_text(encoding="utf-8", errors="replace")
            if text.strip():
                return text
        return render_trajectory(read_trace(trace).run, obs_trunc=obs_trunc)

    def deliverable_text(self, run_dir: Path) -> str:
        """Patch diff as the judged artifact."""
        p = Path(run_dir) / "patch.diff"
        return p.read_text(encoding="utf-8", errors="replace") if p.is_file() else ""

    def deliverable_files(self, run_dir: Path) -> dict[str, str]:
        """One entry for the patch."""
        text = self.deliverable_text(run_dir)
        return {"patch.diff": text} if text else {}

    def score(self, run_dir: Path, task_id: str) -> dict | None:
        """Run the task's tests against the patch; ``None`` when they could not run.

        The verdict is binary, as the benchmark defines it: one criterion,
        passed only when every FAIL_TO_PASS and PASS_TO_PASS test passes. The
        per-group counts ride along in the raw record. A verdict is cached in
        the run directory; a scorer failure is never cached and never a zero.
        """
        run_dir = Path(run_dir)
        cached = run_dir / "score.json"
        if not cached.is_file():
            from atlas.harness_eval import swe_score

            command = [
                "uv", "run", "--no-project", "--python", "3.12",
                "--with", swe_score.SWEBENCH_REQUIREMENT,
                "python", str(Path(swe_score.__file__).resolve()),
                "--eval-jsonl", str(self._eval_jsonl),
                "--instance-id", task_id,
                "--patch", str(run_dir / "patch.diff"),
                "--out", str(cached),
                "--log-dir", str(run_dir / "eval"),
            ]  # fmt: skip
            with self._score_gate:
                proc = subprocess.run(command, capture_output=True, text=True, timeout=5400)
            if proc.returncode != 0 or not cached.is_file():
                print(f"  ! scoring {task_id} failed: {proc.stderr.strip()[-300:]}", flush=True)
                return None
        verdict = json.loads(cached.read_text(encoding="utf-8"))
        resolved = bool(verdict.get("resolved"))
        return {
            "n_passed": int(resolved),
            "n_criteria": 1,
            "all_pass": resolved,
            "fail_to_pass": verdict.get("fail_to_pass"),
            "pass_to_pass": verdict.get("pass_to_pass"),
            "note": verdict.get("note", ""),
        }

    def preflight(self) -> None:
        """Everything a paid run needs, checked without a model call."""
        from harness2 import config

        from atlas.agent.system_prompt import load_system_prompt
        from atlas.harness_eval import swe_docker

        problems: list[str] = []
        load_system_prompt()
        try:
            records = self._records()
        except FileNotFoundError as exc:
            raise RuntimeError(str(exc)) from exc
        if not self._eval_jsonl.is_file():
            problems.append(
                f"scoring rows not found: {self._eval_jsonl} (set {ENV_EVAL_JSONL}; "
                "scripts/fetch_swe_dataset.py writes it beside the task file)"
            )
        else:
            known = {str(rec["instance_id"]) for rec in _load_jsonl(self._eval_jsonl)}
            absent = sorted(set(records) - known)
            if absent:
                problems.append(f"{len(absent)} task(s) have no scoring row, e.g. {absent[:3]}")
        if not _api_key():
            problems.append("OPENROUTER_API_KEY is not set (shell or the Atlas .env)")
        if not self._model():
            problems.append("no solver model: set HARNESS2_SOLVER_MODEL=openrouter/<id>")
        if shutil.which("uv") is None:
            problems.append("uv is not on PATH; scoring runs the swebench package through it")
        if self.spec.channels["proposer"].kind == "opencode":
            entry = config.OPENCODE_DIR / "packages" / "opencode" / "src" / "index.ts"
            if not entry.is_file() or not Path(config.BUN).exists():
                problems.append(
                    "the OpenCode improver is not installed; run scripts/setup_harness2_swe.sh"
                )
        if not swe_docker.daemon_ready():
            problems.append("the Docker daemon is not reachable; start Docker and retry")
        else:
            broken = swe_docker.runtime_error()
            if broken:
                problems.append(
                    f"the Atlas container runtime is not usable ({broken}); "
                    "run scripts/setup_harness2_swe.sh"
                )
            try:
                swe_docker.ensure_proxy()
            except swe_docker.DockerError as exc:
                problems.append(f"the rollout egress proxy did not start: {exc}")
        if problems:
            raise RuntimeError("; ".join(problems))


__all__ = ["SWEBenchAdapter", "_bench_spec", "register_surface_text"]
