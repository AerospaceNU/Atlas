from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import Tool, ToolResult

DiagnosticName = Literal["ruff", "mypy", "cargo"]
_DIAGNOSTIC_RE = re.compile(r"^(?P<path>.+?):(?P<line>\d+):(?:(?P<column>\d+):)? (?P<message>.+)$")
_MAX_DIAGNOSTICS = 100
_MAX_OUTPUT_CHARS = 100_000


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
        checks, missing = self._resolve(request.tools, root, request.path)
        rows: list[str] = []
        if missing:
            rows.append("missing: " + ", ".join(missing))
        if not checks:
            rows.append("no diagnostics tools available")
            return ToolResult(text="\n".join(rows))
        for check in checks:
            rows.append(self._run_check(check, request.timeout))
        text = "\n".join(rows)
        if len(text) > _MAX_OUTPUT_CHARS:
            text = text[:_MAX_OUTPUT_CHARS] + "\n... [output truncated]"
        return ToolResult(text=text)

    def _resolve(
        self, requested: list[DiagnosticName] | None, root: Path, path: str
    ) -> tuple[list[_Check], list[str]]:
        """Return runnable checks and the names that are unavailable."""
        wanted = requested or ["ruff", "mypy", "cargo"]
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
                if binary is None or not (root / "Cargo.toml").is_file():
                    missing.append("cargo")
                    continue
                checks.append(
                    _Check(
                        name="cargo",
                        argv=[binary, "check", "--quiet", "--message-format=short"],
                        cwd=root,
                    )
                )
        return checks, missing

    def _run_check(self, check: _Check, timeout: int) -> str:
        """Run one check and return its report section."""
        try:
            result = subprocess.run(
                check.argv,
                cwd=check.cwd,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return f"== {check.name}\n[timed out after {timeout}s]"
        except OSError as exc:
            return f"== {check.name}\n[error: {type(exc).__name__}: {exc}]"
        output = f"{result.stdout}\n{result.stderr}".strip()
        diagnostics = [
            line
            for line in output.splitlines()
            if _DIAGNOSTIC_RE.match(line) or line.startswith("error")
        ]
        rows = [f"== {check.name} (exit {result.returncode}, {len(diagnostics)} diagnostics)"]
        if not diagnostics:
            rows.append("no diagnostics")
        else:
            rows.extend(diagnostics[:_MAX_DIAGNOSTICS])
            if len(diagnostics) > _MAX_DIAGNOSTICS:
                rows.append(f"... [{len(diagnostics) - _MAX_DIAGNOSTICS} more diagnostics]")
        return "\n".join(rows)


def _find_binary(root: Path, name: str) -> str | None:
    """Return an installed tool binary from PATH or the project's venv."""
    found = shutil.which(name)
    if found is not None:
        return found
    for candidate in (root / ".venv" / "bin" / name, root / "venv" / "bin" / name):
        if candidate.is_file():
            return str(candidate)
    return None
