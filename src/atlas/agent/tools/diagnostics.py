from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import Tool, ToolResult
from atlas.agent.tools._process import run_captured
from atlas.agent.tools._project_env import project_bin
from atlas.agent.tools._walk import SKIP_DIR_NAMES

DiagnosticName = Literal["ruff", "mypy", "cargo"]
_DIAGNOSTIC_RE = re.compile(r"^(?P<path>.+?):(?P<line>\d+):(?:(?P<column>\d+):)? (?P<message>.+)$")
_MAX_DIAGNOSTICS = 100
_MAX_OUTPUT_CHARS = 100_000
_CRASH_TAIL_LINES = 20
_PYTHON_SUFFIXES = {".py", ".pyi"}


@dataclass
class _Check:
    """One resolved static-analysis command."""

    name: str
    argv: list[str]
    cwd: Path


class DiagnosticsInput(BaseModel):
    """Arguments for ``diagnostics``.

    ``tools`` narrows the run to specific checks; the default runs every check
    whose binary is installed in the project.
    """

    path: str = Field(
        default=".", description="File or directory to check, relative to the workspace."
    )
    tools: list[DiagnosticName] | None = Field(
        default=None,
        description="Checks to run: 'ruff', 'mypy', 'cargo'. Default is every available check.",
    )
    timeout: int = Field(
        default=120,
        ge=1,
        le=600,
        description="Seconds per check (1-600).",
    )


class DiagnosticsTool(Tool):
    """Run the project's installed static checks and summarise diagnostics.

    Only binaries already present are used: ``ruff`` and ``mypy`` from PATH or
    ``.venv/bin``, and ``cargo`` for a Rust workspace. Nothing is installed and
    no check writes to the workspace. A non-zero exit is a result, not an
    exception.
    """

    name = "diagnostics"
    description = (
        "Run installed static checks (ruff, mypy, cargo check) and return compact "
        "file:line:column diagnostics with exit codes. Missing tools are reported, never installed."
    )
    trust: ClassVar[Literal["default", "opt_in"]] = "default"

    @property
    def input_model(self) -> type[BaseModel]:
        return DiagnosticsInput

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        request = DiagnosticsInput.model_validate(arguments)
        root = store.read_root or store.root
        target = store.resolve_read(request.path)
        if not target.exists():
            raise FileNotFoundError(f"Path does not exist: {request.path}")
        wanted: list[DiagnosticName] = list(request.tools or ["ruff", "mypy", "cargo"])
        skipped: list[str] = []
        if target.is_file() and target.suffix not in _PYTHON_SUFFIXES:
            skipped = [n for n in wanted if n in ("ruff", "mypy")]
            wanted = [n for n in wanted if n not in ("ruff", "mypy")]
        checks, missing = self._resolve(wanted, root, target, request.path) if wanted else ([], [])
        rows: list[str] = []
        if missing:
            rows.append("missing: " + ", ".join(missing))
        if skipped:
            rows.append(f"skipped {', '.join(skipped)}: {request.path} is not a Python file")
        if not checks:
            rows.append("no diagnostics tools available")
            return ToolResult(text="\n".join(rows))
        for check in checks:
            rows.append(self._run_check(check, request.timeout))
        ran_python = any(check.name in ("ruff", "mypy") for check in checks)
        if target.is_dir() and ran_python and not _has_python_files(target):
            rows.append(f"note: no Python files under {request.path}")
        text = "\n".join(rows)
        if len(text) > _MAX_OUTPUT_CHARS:
            text = text[:_MAX_OUTPUT_CHARS] + "\n... [output truncated]"
        return ToolResult(text=text)

    def _resolve(
        self, wanted: list[DiagnosticName], root: Path, target: Path, path: str
    ) -> tuple[list[_Check], list[str]]:
        """Return runnable checks and the names that are unavailable."""
        checks: list[_Check] = []
        missing: list[str] = []
        for name in wanted:
            if name == "ruff":
                binary = _find_binary(root, "ruff")
                if binary is None:
                    missing.append("ruff")
                    continue
                checks.append(
                    _Check(
                        name="ruff",
                        argv=[
                            binary,
                            "check",
                            "--no-cache",
                            "--output-format=concise",
                            path,
                        ],
                        cwd=root,
                    )
                )
            elif name == "mypy":
                binary = _find_binary(root, "mypy")
                if binary is None:
                    missing.append("mypy")
                    continue
                checks.append(
                    _Check(
                        name="mypy",
                        argv=[binary, "--no-error-summary", "--show-column-numbers", path],
                        cwd=root,
                    )
                )
            else:
                binary = shutil.which("cargo")
                cargo_dir = _find_cargo_dir(target, root)
                if binary is None or cargo_dir is None:
                    missing.append("cargo")
                    continue
                checks.append(
                    _Check(
                        name="cargo",
                        argv=[binary, "check", "--quiet", "--message-format=short"],
                        cwd=cargo_dir,
                    )
                )
        return checks, missing

    def _run_check(self, check: _Check, timeout: int) -> str:
        """Run one check and return its report section."""
        try:
            result = run_captured(check.argv, cwd=check.cwd, timeout=timeout)
        except OSError as exc:
            return f"== {check.name}\n[error: {type(exc).__name__}: {exc}]"
        if result.timed_out:
            return f"== {check.name}\n[timed out after {timeout}s]"
        output = f"{result.stdout}\n{result.stderr}".strip()
        diagnostics = [
            line
            for line in output.splitlines()
            if _DIAGNOSTIC_RE.match(line) or line.startswith("error")
        ]
        rows = [f"== {check.name} (exit {result.exit_code}, {len(diagnostics)} diagnostics)"]
        if not diagnostics and result.exit_code not in (0, None):
            return "\n".join(
                [
                    f"== {check.name} (exit {result.exit_code}, 0 diagnostics)",
                    f"[{check.name} failed: exit {result.exit_code}]",
                    *output.splitlines()[-_CRASH_TAIL_LINES:],
                ]
            )
        if not diagnostics:
            rows.append("no diagnostics")
        else:
            rows.extend(diagnostics[:_MAX_DIAGNOSTICS])
            if len(diagnostics) > _MAX_DIAGNOSTICS:
                rows.append(f"... [{len(diagnostics) - _MAX_DIAGNOSTICS} more diagnostics]")
        return "\n".join(rows)


def _find_cargo_dir(target: Path, root: Path) -> Path | None:
    """Return the nearest directory containing ``Cargo.toml`` for ``target``."""
    current = target if target.is_dir() else target.parent
    while True:
        if (current / "Cargo.toml").is_file():
            return current
        if current == root or current.parent == current:
            return None
        current = current.parent


def _find_binary(root: Path, name: str) -> str | None:
    """Return an installed tool binary: the project venv first, then PATH."""
    return project_bin(root, name) or shutil.which(name)


def _has_python_files(path: Path, limit: int = 20_000) -> bool:
    """True when a Python file exists under ``path``, or the tree is too big to say."""
    seen = 0
    for _, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIR_NAMES]
        for name in files:
            if name.endswith((".py", ".pyi")):
                return True
            seen += 1
        seen += len(dirs)
        if seen >= limit:
            return True
    return False
