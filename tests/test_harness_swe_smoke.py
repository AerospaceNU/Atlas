from __future__ import annotations

import json
from pathlib import Path


def test_smoke_dataset_is_real_and_hidden_hints() -> None:
    path = Path("evals/swe/lite-smoke.jsonl")
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert len(rows) == 20
    assert all("instance_id" in row and "repo" in row for row in rows)
    assert all(row["base_commit"] for row in rows)
    splits = json.loads(Path("evals/swe/splits/swe.json").read_text(encoding="utf-8"))
    assert splits["benchmark"] == "swe"
    assert sorted(splits["train"] + splits["test"]) == sorted(row["instance_id"] for row in rows)
