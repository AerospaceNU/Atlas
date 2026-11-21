# evals/swe — bundled smoke set for Atlas + harness2

`lite-smoke.jsonl` is 20 real SWE-bench `test` instances (seed-42 shuffle of
`SWE-bench/SWE-bench`, 2294 rows) over 7 repositories: django 9, sympy 4,
scikit-learn 3, and astropy, matplotlib, xarray, sphinx 1 each. It is from
the full set, not Lite. `splits/swe.json` is its 7 train / 13 test partition.

The working task sets are generated, not tracked. `scripts/setup_harness2_swe.sh`
writes them to `local/swe/` (gitignored):

| File | Contents |
|---|---|
| `lite100.jsonl` | 100 SWE-bench Lite tasks, stratified over 12 repositories; issue text only |
| `lite100.eval.jsonl` | the same tasks with tests and reference patch; read only by the scorer |
| `lite100.split.json` | 47 train / 53 test, stratified by repository |
| `lite-smoke.*` | the same three files for the 20 tasks above |

Select a set with `TASKS=lite100` (default) or `TASKS=lite-smoke` on
`scripts/run_harness2_swe.sh`. Build another with
`scripts/fetch_swe_dataset.py`. See `docs/harness2-swe.md` for the loop.
