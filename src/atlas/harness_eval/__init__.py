"""Atlas side of harness2-style harness improvement.

Submodules:

- :mod:`headless` -- one non-interactive agent turn + trace emission.
- :mod:`surface` -- the editable harness tree harness2 may change.
- :mod:`substrate` -- install a tree into a rollout (prompt, skills, scripts, roles).
- :mod:`harness2_bench` -- the harness2 adapter for SWE-bench (run inside harness2).
- :mod:`rollout`, :mod:`swe_docker`, :mod:`egress_proxy` -- one rollout in a
  task container on an allowlisted network.
- :mod:`swe_score` -- score a patch with the benchmark's own tests.
- :mod:`swe_adapter` -- offline shim for unit tests; its score is not a result.

harness2 drives the recursion; see ``docs/harness2-swe.md``.
"""

from __future__ import annotations

from atlas.harness_eval.headless import HeadlessRun, run_headless
from atlas.harness_eval.substrate import AtlasSubstrate, Install
from atlas.harness_eval.surface import ATLAS_SURFACE
from atlas.harness_eval.swe_adapter import SWEAdapter

__all__ = [
    "ATLAS_SURFACE",
    "AtlasSubstrate",
    "HeadlessRun",
    "Install",
    "SWEAdapter",
    "run_headless",
]
