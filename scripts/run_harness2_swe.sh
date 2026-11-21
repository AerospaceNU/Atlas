#!/usr/bin/env bash
set -euo pipefail
# Drive the harness2 loop with Atlas as the solver and OpenCode as the improver.
# Run scripts/setup_harness2_swe.sh once first.
#
#   scripts/run_harness2_swe.sh check              # readiness, no model calls
#   scripts/run_harness2_swe.sh dry                # print the task stream
#   scripts/run_harness2_swe.sh pull               # pre-pull the task images
#   scripts/run_harness2_swe.sh run                # the improvement sweep (spends money)
#   scripts/run_harness2_swe.sh judge              # score the selected runs
#   scripts/run_harness2_swe.sh bases              # score the unmodified-harness runs
#   scripts/run_harness2_swe.sh report             # adapted results table
#   scripts/run_harness2_swe.sh compare            # base vs adapted resolve rate and spend
#
# Knobs (environment):
#   TASKS     task file stem under local/swe (default lite100; lite-smoke is 20 tasks)
#   SPLIT     all | train | test (default train)
#   K, MODE   recursion budget and parallel | sequential (default 1, parallel)
#   MAX       per-repository task cap (default unset: every task of the split)
#   DOMAINS   comma list of repositories, e.g. django/django
#   WORKERS   repositories run concurrently (default 2)
#   MODEL     OpenRouter id for the Atlas solver (default deepseek/deepseek-v4.1-flash)
#   IMPROVER  OpenRouter id for the OpenCode improver (default: MODEL)
#   RUNS      output root (default local/runs/<TASKS>); use a new one per experiment
#   ATLAS_SWE_MAX_TOOL_CALLS, ATLAS_SWE_ROLLOUT_MINUTES   per-rollout limits

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HARNESS2_DIR="${ROOT}/local/harness2"
TASKS="${TASKS:-lite100}"
SWE_DIR="${SWE_DIR:-${ROOT}/local/swe}"
MODEL="${MODEL:-deepseek/deepseek-v4.1-flash}"
IMPROVER="${IMPROVER:-${MODEL}}"
SPLIT="${SPLIT:-train}"
K="${K:-1}"
MODE="${MODE:-parallel}"
WORKERS="${WORKERS:-2}"

export PYTHONPATH="${HARNESS2_DIR}${PYTHONPATH:+:$PYTHONPATH}"
export SWE_BENCH_JSONL="${SWE_DIR}/${TASKS}.jsonl"
export SWE_BENCH_EVAL_JSONL="${SWE_DIR}/${TASKS}.eval.jsonl"
export HARNESS2_RUNS_DIR="${RUNS:-${ROOT}/local/runs/${TASKS}}"
export HARNESS2_SOLVER_MODEL="openrouter/${MODEL}"
export HARNESS2_IMPROVER_MODEL="openrouter/${IMPROVER}"
# Scoring is the benchmark's own tests; harness2 only needs the slot to be named.
export HARNESS2_JUDGE_MODEL="swebench-tests"
export OPENROUTER_MODEL="${MODEL}"

# Only the OpenRouter key leaves .env: improver sessions inherit this environment.
if [ -z "${OPENROUTER_API_KEY:-}" ] && [ -f "${ROOT}/.env" ]; then
  OPENROUTER_API_KEY="$(grep -E '^(export )?OPENROUTER_API_KEY=' "${ROOT}/.env" | tail -1 | cut -d= -f2- | tr -d "\"'")"
  export OPENROUTER_API_KEY
fi

if [ ! -d "${HARNESS2_DIR}/harness2" ]; then
  echo "error: ${HARNESS2_DIR} is missing; run scripts/setup_harness2_swe.sh" >&2
  exit 1
fi

cell=(--bench swe --substrate atlas)
identity=(--k "${K}" --mode "${MODE}")
stream=(--split "${SPLIT}" --split-file "${SWE_DIR}/${TASKS}.split.json" --workers "${WORKERS}")
[ -n "${MAX:-}" ] && stream+=(--max-tasks "${MAX}")
[ -n "${DOMAINS:-}" ] && stream+=(--only-domains "${DOMAINS}")

h2() { uv run --project "${ROOT}" python -m "harness2.cli.$1" "${@:2}"; }

case "${1:-check}" in
  check) h2 check "${cell[@]}" ;;
  dry) h2 run_sweep "${cell[@]}" "${identity[@]}" "${stream[@]}" --dry-run ;;
  run) h2 run_sweep "${cell[@]}" "${identity[@]}" "${stream[@]}" ;;
  judge) h2 judge "${cell[@]}" "${identity[@]}" ;;
  bases) h2 judge "${cell[@]}" "${identity[@]}" --bases ;;
  report) h2 report "${cell[@]}" --k "${K}" --mode "${MODE}" ;;
  compare) uv run --project "${ROOT}" python -m atlas.harness_eval compare --k "${K}" ;;
  pull)
    h2 run_sweep "${cell[@]}" "${identity[@]}" "${stream[@]}" --dry-run \
      | awk '$1 ~ /^[0-9]+$/ && NF == 3 {print $3}' \
      | uv run --project "${ROOT}" python -m atlas.harness_eval pull
    ;;
  *)
    sed -n '3,25p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
    ;;
esac
