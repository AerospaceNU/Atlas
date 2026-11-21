#!/usr/bin/env bash
set -euo pipefail
# One-time setup for the harness2 loop: Atlas is the solver, OpenCode the improver.
# Idempotent; every step skips what is already in place. Needs git, uv, npm and a
# running Docker daemon. Nothing here calls a model.
#
#   1. harness2 at a pinned commit under local/harness2 (gitignored), plus our patch
#   2. the Atlas SWE adapter copied into it
#   3. the pinned OpenCode checkout and Bun that harness2's improver sessions run on
#   4. the Linux Atlas runtime volume that task containers mount
#   5. a 100-task SWE-bench Lite set and the scoring rows for the 20-task smoke set

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HARNESS2_DIR="${ROOT}/local/harness2"
HARNESS2_PIN="${HARNESS2_PIN:-6868817518876de8fd170c6a8354daca273b6191}"
PATCH="${ROOT}/scripts/harness2_swe.patch"
ADAPTER_SRC="${ROOT}/src/atlas/harness_eval/harness2_bench.py"
ADAPTER_DEST="${HARNESS2_DIR}/harness2/benchmarks/swebench.py"
SWE_DIR="${ROOT}/local/swe"

for tool in git uv npm docker; do
  command -v "${tool}" >/dev/null || { echo "error: ${tool} is not on PATH" >&2; exit 1; }
done
docker info >/dev/null 2>&1 || { echo "error: the Docker daemon is not running" >&2; exit 1; }

echo "== harness2 @ ${HARNESS2_PIN:0:12}"
if [ ! -d "${HARNESS2_DIR}/.git" ]; then
  git init -q "${HARNESS2_DIR}"
  git -C "${HARNESS2_DIR}" remote add origin https://github.com/google-research/harness2
fi
if [ "$(git -C "${HARNESS2_DIR}" rev-parse HEAD 2>/dev/null || true)" != "${HARNESS2_PIN}" ]; then
  if [ -n "$(git -C "${HARNESS2_DIR}" status --porcelain --untracked-files=no 2>/dev/null)" ]; then
    echo "error: ${HARNESS2_DIR} has local changes at another commit; move it aside" >&2
    exit 1
  fi
  git -C "${HARNESS2_DIR}" fetch -q --depth 1 origin "${HARNESS2_PIN}"
  git -C "${HARNESS2_DIR}" checkout -q --detach FETCH_HEAD
fi
if git -C "${HARNESS2_DIR}" apply --reverse --check "${PATCH}" 2>/dev/null; then
  echo "patch already applied"
else
  git -C "${HARNESS2_DIR}" apply --check "${PATCH}"
  git -C "${HARNESS2_DIR}" apply "${PATCH}"
  echo "applied ${PATCH}"
fi
cp "${ADAPTER_SRC}" "${ADAPTER_DEST}"
echo "installed adapter -> ${ADAPTER_DEST}"

echo "== OpenCode improver (pinned checkout + Bun)"
PYTHONPATH="${HARNESS2_DIR}" uv run --project "${ROOT}" python - <<'PY'
from harness2 import config, install

install.clone("opencode")
install.patch("opencode")
if not config.BUN.exists():
    install.run(["npm", "install", "--prefix", config.DEPENDENCIES / "tools", "bun@1.3.11"])
if not (config.OPENCODE_DIR / "node_modules").is_dir():
    install.run(
        [config.BUN, "install", "--frozen-lockfile", "--ignore-scripts"], cwd=config.OPENCODE_DIR
    )
print("opencode ready")
PY

echo "== Atlas container runtime"
if docker volume inspect atlas-swe-rt >/dev/null 2>&1 && [ -z "${REBUILD_RUNTIME:-}" ]; then
  echo "volume atlas-swe-rt exists (REBUILD_RUNTIME=1 rebuilds it after a dependency change)"
else
  uv run --project "${ROOT}" python -m atlas.harness_eval build-runtime
fi

echo "== task sets"
if [ ! -f "${SWE_DIR}/lite100.jsonl" ]; then
  uv run --no-project --python 3.12 --with pyarrow python "${ROOT}/scripts/fetch_swe_dataset.py" \
    --dataset lite --n 100 --out-dir "${SWE_DIR}"
fi
if [ ! -f "${SWE_DIR}/lite-smoke.eval.jsonl" ]; then
  uv run --no-project --python 3.12 --with pyarrow python "${ROOT}/scripts/fetch_swe_dataset.py" \
    --dataset full --ids-from "${ROOT}/evals/swe/lite-smoke.jsonl" --out-dir "${SWE_DIR}"
fi

echo "== readiness"
"${ROOT}/scripts/run_harness2_swe.sh" check
echo "next: scripts/run_harness2_swe.sh dry   (then pull, run, judge, bases, compare)"
