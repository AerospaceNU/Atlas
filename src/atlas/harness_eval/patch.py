"""Build the patch for a workspace without touching its real git index.

Invariant: ``ok=True`` means the diff was computed; an empty patch is never
returned for a failure. The module imports only the standard library so every
collector (rollout, headless) can share it.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

# Ignore rules passed through a temporary core.excludesFile. Pathspec excludes would make
# `add` fail when the same path is already ignored (rollout.prepare ignores .atlas/).
_EXCLUDES = ".atlas/\n__pycache__/\n*.pyc\n"


@dataclass(frozen=True)
class PatchResult:
    ok: bool
    patch: str
    error: str = ""


def _git(
    workspace: Path, *args: str, env: dict[str, str], timeout: int
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(workspace), *args],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=timeout,
        env=env,
    )


def _failure(name: str, proc: subprocess.CompletedProcess[str]) -> PatchResult:
    detail = proc.stderr.strip()[:500]
    return PatchResult(False, "", f"git {name} failed (rc {proc.returncode}): {detail}")


def build_patch(workspace: Path, base: str, timeout: int = 120) -> PatchResult:
    """Return the diff of the workspace tree (untracked included) against ``base``.

    The diff is computed through a throwaway ``GIT_INDEX_FILE``, so the real
    index and any ``index.lock`` are never touched. The fresh index has no stat
    cache, so ``add -A`` re-hashes every non-ignored file once per call.
    `.atlas`, ``__pycache__`` and ``*.pyc`` are excluded. Never raises.

    Args:
        workspace: Git working tree.
        base: Commit-ish the patch is measured against.
        timeout: Seconds allowed per git command.

    Returns:
        A ``PatchResult``; ``ok`` is False with ``error`` set when any git step fails.
    """
    steps: tuple[tuple[str, ...], ...] = (
        ("read-tree", base),
        ("-c", "core.excludesFile={exclude}", "add", "-A"),
        ("diff", "--cached", "--no-color", base),
    )
    try:
        with tempfile.TemporaryDirectory() as tmp:
            env = {
                **os.environ,
                "GIT_INDEX_FILE": str(Path(tmp) / "index"),
                "GIT_OPTIONAL_LOCKS": "0",
            }
            exclude = Path(tmp) / "exclude"
            exclude.write_text(_EXCLUDES, encoding="utf-8")
            proc: subprocess.CompletedProcess[str] | None = None
            for args in steps:
                name = args[2] if args[0] == "-c" else args[0]
                real = tuple(a.replace("{exclude}", str(exclude)) for a in args)
                try:
                    proc = _git(workspace, *real, env=env, timeout=timeout)
                except subprocess.TimeoutExpired:
                    return PatchResult(False, "", f"git {name} timed out after {timeout}s")
                if proc.returncode != 0:
                    return _failure(name, proc)
            return PatchResult(True, proc.stdout if proc else "")
    except FileNotFoundError:
        return PatchResult(False, "", "git not found")
    except OSError as exc:
        return PatchResult(False, "", f"patch build failed: {exc}")
