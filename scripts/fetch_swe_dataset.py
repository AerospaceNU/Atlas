"""Build a SWE-bench task set for the harness2 loop.

Downloads one public SWE-bench parquet from Hugging Face, takes a sample that
is stratified by repository, and writes three files under ``--out-dir``::

    <name>.jsonl        what the solver and improver may see (issue text only)
    <name>.eval.jsonl   full rows for scoring: tests and the reference patch
    <name>.split.json   stratified train/test partition in harness2's format

Run with pyarrow available::

    uv run --with pyarrow python scripts/fetch_swe_dataset.py --dataset lite --n 100

``--ids-from`` rebuilds the eval rows for an existing task list instead of
sampling, e.g. the bundled 20-task smoke set (drawn from ``--dataset full``).
"""

from __future__ import annotations

import argparse
import json
import random
import urllib.request
from collections import defaultdict
from pathlib import Path

DATASETS = {
    "lite": "SWE-bench/SWE-bench_Lite",
    "verified": "SWE-bench/SWE-bench_Verified",
    "full": "SWE-bench/SWE-bench",
}
PARQUET = "https://huggingface.co/datasets/{repo}/resolve/main/data/test-00000-of-00001.parquet"
PUBLIC_FIELDS = ("instance_id", "repo", "base_commit", "problem_statement")
ROOT = Path(__file__).resolve().parents[1]


def load_rows(dataset: str, cache_dir: Path) -> list[dict[str, object]]:
    """Download (once) and read the dataset's test split."""
    import pyarrow.parquet as pq

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{dataset}-test.parquet"
    if not path.is_file():
        url = PARQUET.format(repo=DATASETS[dataset])
        print(f"downloading {url}")
        request = urllib.request.Request(url, headers={"User-Agent": "swe-dataset-fetch"})
        with urllib.request.urlopen(request, timeout=300) as response:
            path.write_bytes(response.read())
    rows: list[dict[str, object]] = pq.read_table(path).to_pylist()
    return rows


def stratified_sample(rows: list[dict[str, object]], n: int, seed: int) -> list[dict[str, object]]:
    """Take ``n`` rows, each repository in proportion, at least two where it has them."""
    by_repo: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_repo[str(row["repo"])].append(row)
    if n >= len(rows):
        return rows
    quota = {repo: min(len(items), 2) for repo, items in by_repo.items()}
    remaining = n - sum(quota.values())
    if remaining < 0:
        raise ValueError(f"--n {n} is below two tasks per repository ({sum(quota.values())})")
    spare = {repo: len(items) - quota[repo] for repo, items in by_repo.items()}
    total_spare = sum(spare.values())
    shares = {repo: remaining * spare[repo] / total_spare for repo in spare}
    for repo in spare:
        quota[repo] += int(shares[repo])
    leftover = n - sum(quota.values())
    for repo in sorted(spare, key=lambda r: (shares[r] - int(shares[r]), r), reverse=True):
        if leftover <= 0:
            break
        if quota[repo] < len(by_repo[repo]):
            quota[repo] += 1
            leftover -= 1
    picked: list[dict[str, object]] = []
    for repo in sorted(by_repo):
        items = sorted(by_repo[repo], key=lambda row: str(row["instance_id"]))
        random.Random(f"{seed}:{repo}").shuffle(items)
        picked.extend(items[: quota[repo]])
    return picked


def split(rows: list[dict[str, object]], seed: int, train_fraction: float) -> dict[str, object]:
    """Partition by repository the way harness2's ``split_tasks.py`` does."""
    by_repo: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        by_repo[str(row["repo"])].append(str(row["instance_id"]))
    train: list[str] = []
    test: list[str] = []
    for repo, ids in sorted(by_repo.items()):
        ids.sort()
        random.Random(f"{seed}:{repo}").shuffle(ids)
        cut = int(len(ids) * train_fraction)
        train.extend(ids[:cut])
        test.extend(ids[cut:])
    return {
        "benchmark": "swe",
        "seed": seed,
        "train_fraction": train_fraction,
        "train": sorted(train),
        "test": sorted(test),
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Fetch, sample and write the task set."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="lite")
    parser.add_argument("--n", type=int, default=100, help="tasks to sample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "local" / "swe")
    parser.add_argument("--name", default=None, help="file stem; default <dataset><n>")
    parser.add_argument(
        "--ids-from", type=Path, default=None, help="JSONL of instance_id rows to keep as-is"
    )
    args = parser.parse_args(argv)
    rows = load_rows(args.dataset, args.out_dir / "cache")
    if args.ids_from is not None:
        wanted = [
            json.loads(line)["instance_id"]
            for line in args.ids_from.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        index = {str(row["instance_id"]): row for row in rows}
        missing = [task for task in wanted if task not in index]
        if missing:
            parser.exit(1, f"error: not in {args.dataset}: {missing[:5]}\n")
        picked = [index[task] for task in wanted]
        name = args.name or args.ids_from.stem
    else:
        picked = stratified_sample(rows, args.n, args.seed)
        name = args.name or f"{args.dataset}{len(picked)}"
    picked.sort(key=lambda row: str(row["instance_id"]))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    public = [{key: row[key] for key in PUBLIC_FIELDS} for row in picked]
    _write_jsonl(args.out_dir / f"{name}.jsonl", public)
    _write_jsonl(args.out_dir / f"{name}.eval.jsonl", picked)
    partition = split(picked, args.seed, args.train_fraction)
    (args.out_dir / f"{name}.split.json").write_text(
        json.dumps(partition, indent=2) + "\n", encoding="utf-8"
    )
    counts: dict[str, int] = defaultdict(int)
    for row in picked:
        counts[str(row["repo"])] += 1
    print(f"{args.out_dir / name}.jsonl: {len(picked)} tasks over {len(counts)} repositories")
    for repo, count in sorted(counts.items()):
        print(f"{count:>5}  {repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
