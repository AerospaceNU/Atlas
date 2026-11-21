# Atlas + harness2 (SWE-bench)

[harness2](https://github.com/google-research/harness2) improves an agent's
harness by recursion: for each task it runs the current harness, has an
improver propose edits, probes those edits on the same task, and composes a
refined harness. Lessons and adopted harnesses accumulate per repository.

Here the **solver is Atlas** and the **improver is OpenCode**. The benchmark is
SWE-bench, scored by its own tests, so there is no model judge.

## Setup (once)

Needs `git`, `uv`, `npm` and a running Docker daemon. It calls no model.

```bash
scripts/setup_harness2_swe.sh
```

It clones harness2 at a pinned commit into `local/harness2` (gitignored),
applies `scripts/harness2_swe.patch`, installs the Atlas adapter and the
OpenCode checkout harness2's improver sessions run on, builds the Linux Atlas
runtime volume, and writes the task sets to `local/swe/`.

## Run

```bash
scripts/run_harness2_swe.sh check     # readiness, no model calls
scripts/run_harness2_swe.sh dry       # the task stream
scripts/run_harness2_swe.sh pull      # pre-pull task images (about 1 GB each)
scripts/run_harness2_swe.sh run       # the sweep: spends money
scripts/run_harness2_swe.sh judge     # score the run harness2 selected per task
scripts/run_harness2_swe.sh bases     # score the unmodified-harness run per task
scripts/run_harness2_swe.sh compare   # base vs adapted resolve rate, and spend
```

Start small: `MAX=1 DOMAINS=django/django scripts/run_harness2_swe.sh run` is
one task, which is four Atlas rollouts and three improver sessions at `K=1`.
Repeat a command to resume; finished rollouts are not rerun. Use a new `RUNS`
directory when you change models or tasks, or harness2 refuses to mix them.

| Variable | Default | Meaning |
|---|---|---|
| `TASKS` | `lite100` | task file stem in `local/swe` (`lite-smoke` is 20 tasks) |
| `SPLIT` | `train` | `all`, `train` or `test` of `<TASKS>.split.json` |
| `K`, `MODE` | `1`, `parallel` | recursion budget; `parallel` or `sequential` |
| `MAX` | unset | per-repository task cap |
| `DOMAINS` | unset | comma list of repositories |
| `WORKERS` | `2` | repositories run concurrently |
| `MODEL` | `deepseek/deepseek-v4.1-flash` | OpenRouter id for the Atlas solver |
| `IMPROVER` | `MODEL` | OpenRouter id for the OpenCode improver |
| `RUNS` | `local/runs/<TASKS>` | output root |
| `ATLAS_SWE_MAX_TOOL_CALLS` | Atlas default | per-rollout tool budget |
| `ATLAS_SWE_ROLLOUT_MINUTES` | `45` | per-rollout wall clock |
| `ATLAS_TOOL_PROFILE` | `full` | `full` or `coding` (drops satellite tools) |
| `ATLAS_TEMPERATURE`, `ATLAS_TOP_P` | unset | sampling sent to the provider |

Another task set: `uv run --with pyarrow python scripts/fetch_swe_dataset.py
--dataset lite|verified|full --n <count>`.

## What the improver can change

The editable tree (`atlas.harness_eval.surface.ATLAS_SURFACE`) and where
`atlas.harness_eval.substrate.install_harness` puts each part:

| Tree path | In a rollout |
|---|---|
| `systemprompt.md` | the system prompt base |
| `guardrails.md`, `LongTermMEMORY.md`, `tool_descriptions/tool_guidance.md` | appended to the system prompt |
| `skills/<name>/SKILL.md` | `.atlas/skills/<name>/`, loaded by `use_skill` |
| `scripts/<name>.py` | `.atlas/scripts/`, run through `bash` |
| `sub_agents/<name>/AGENT.md` | a named role: `spawn_agent` with a task starting `agent: <name>` |

The base tree is the package prompt as shipped. Tools are not editable, and
there is no plugin component because Atlas has no hook loader. A proposed
`systemprompt.md` must keep every sentence `atlas.agent.system_prompt`
requires, so an adopted prompt can go back into the package unchanged.

## How a rollout runs

`atlas.harness_eval.swe_docker` starts one container per rollout from the
task's published image (`swebench/sweb.eval.x86_64.<id>`, amd64, emulated on
Apple Silicon). Atlas is not baked into those images: a Docker volume
(`atlas-swe-rt`) holds a Linux Python and the dependencies, and `src/` is
mounted read-only, so rollouts always run the working tree. Rebuild the volume
with `REBUILD_RUNTIME=1 scripts/setup_harness2_swe.sh` after a dependency
change.

Inside the container `atlas.harness_eval.rollout` records the base commit,
removes tags and reflog entries newer than it, runs one headless turn, and
collects every working-tree change as `patch.diff`. Each run directory under
`<RUNS>/swe/executions/` holds `trace.json`, `trajectory.md`, `meta.json`
(tool calls, tokens, cost) and the patch.

`atlas.harness_eval.swe_score` scores a patch in a second, fresh container
with the pinned `swebench` package: apply the patch, run the instance's
evaluation script, grade the log. A task counts only when every FAIL_TO_PASS
and PASS_TO_PASS test passes.

## Rollout flags

`rollout run` also takes `--tool-profile {full,coding}` (`coding` withholds the
satellite and model tools and `web_fetch`), `--temperature` and `--top-p`
(sent to the provider only when set). `RolloutSpec` carries the same fields, and
`meta.json` records them.

## Isolation

- **Network.** Rollout containers sit on an internal Docker network. Their
  only route out is `atlas-swe-proxy`, which tunnels to `openrouter.ai` and
  refuses everything else. This is required, not a precaution: on its first
  unrestricted run the agent downloaded the fixed release from PyPI. As a
  result `web_fetch` and `pip install` fail inside a rollout.
  `ATLAS_SWE_ALLOWED_HOSTS` replaces the allowlist.
- **Answers.** The solver and improver see the issue text only. Tests and the
  reference patch are in `<TASKS>.eval.jsonl`, which only the scorer opens.
- **Key.** The OpenRouter key reaches the rollout on stdin, not in the
  environment. Improver sessions get that key and no other variable from
  `.env`.
- **Limit.** Unreachable git objects newer than the base commit stay in the
  image's object store; removing them needs a repack per rollout.

## Layout

- `src/atlas/harness_eval/harness2_bench.py` — the harness2 adapter; copied to
  `local/harness2/harness2/benchmarks/swebench.py` by setup. Rerun setup after
  editing it.
- `headless.py`, `substrate.py`, `surface.py` — one headless turn, the harness
  install, the editable surface.
- `rollout.py`, `swe_docker.py`, `egress_proxy.py`, `swe_score.py` — the
  container entry point, the host driver, the proxy, the scorer.
- `swe_adapter.py` — an offline shim for unit tests. Its `score` only checks
  that a patch exists; do not report results from it.
- `scripts/harness2_swe.patch` — the changes to harness2: the `swe` bench and
  `atlas` substrate, a preflight that skips the OpenCode executor probe, and
  two improver-session fixes (closed stdin, isolated OpenCode config).
