from __future__ import annotations

import contextlib
import os
import re
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import Tool, ToolResult

_PYTEST_COUNT_RE = re.compile(
    r"(\d+) (passed|failed|error|errors|skipped|xfailed|xpassed|warning|warnings)\b"
)
_CARGO_RESULT_RE = re.compile(
    r"test result: (ok|FAILED)\. (\d+) passed; (\d+) failed; (\d+) ignored"
)
_TAIL_CHARS = 4000
_MAX_FAILURES = 20


class RunTestsInput(BaseModel):
    """Arguments for ``run_tests``.

    With no ``command``, the tool detects the project's test command. An
    explicit ``command`` is an argv list, never a shell string.
    """

    command: list[str] | None = Field(
        default=None,
        min_length=1,
        description="Explicit argv to run, e.g. ['uv', 'run', 'pytest', '-q']. No shell.",
    )
    path: str = Field(default=".", description="Working directory relative to the workspace.")
    timeout: int = Field(
        default=300,
        ge=1,
        le=900,
        description="Seconds to allow the command to run (1-900).",
    )


class RunTestsTool(Tool):
    """Run the project's test command and summarise the result.

    The command is detected from the workspace (pyproject/pytest config,
    ``Cargo.toml``, then ``package.json``) unless an explicit argv is given.
    A non-zero exit is a result, not an exception. Output is capped, and the
    command runs without a shell.
    """

    name = "run_tests"
    description = (
        "Detect and run the project's test command, returning exit code, pass/fail counts, "
        "failing test names, and the tail of the output. Non-zero exit is reported, not raised."
    )
    trust: ClassVar[Literal["default", "opt_in"]] = "default"
    max_output_chars: ClassVar[int] = 20_000

    @property
    def input_model(self) -> type[BaseModel]:
        return RunTestsInput

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        request = RunTestsInput.model_validate(arguments)
        root = store.read_root or store.root
        cwd = store.resolve_read(request.path) if request.path not in ("", ".") else root
        if not cwd.is_dir():
            return ToolResult(text=f"Working directory not found: {request.path}")
        if request.command is not None:
            argv = list(request.command)
            source = "explicit"
        else:
            detected = self._detect(cwd)
            if detected is None:
                return ToolResult(
                    text=(
                        "no test command detected; pass command=[...] "
                        "(looked for pyproject.toml, Cargo.toml, package.json)"
                    )
                )
            argv, source = detected
        binary = shutil.which(argv[0])
        if binary is None:
            return ToolResult(text=f"command not found: {argv[0]}")
        argv = [binary, *argv[1:]]

        process = subprocess.Popen(
            argv,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            start_new_session=True,
        )
        # The shell-equivalent leader pid is the process group id while it lives.
        pgid = process.pid
        try:
            stdout, stderr = process.communicate(timeout=request.timeout)
        except subprocess.TimeoutExpired:
            self._kill(process, pgid)
            return ToolResult(
                text=f"test command timed out after {request.timeout}s: {' '.join(argv)}"
            )
        finally:
            self._kill(process, pgid)

        combined = f"{stdout}\n{stderr}".strip()
        return ToolResult(
            text=self._report(
                argv=argv,
                source=source,
                exit_code=process.returncode,
                output=combined,
            )
        )

    def _detect(self, root: Path) -> tuple[list[str], str] | None:
        """Return the test argv and its source, or None when nothing matches."""
        python_config = (
            (root / "pytest.ini").is_file()
            or (root / "tox.ini").is_file()
            or (root / "pyproject.toml").is_file()
        )
        if python_config:
            if shutil.which("uv") is not None:
                return (["uv", "run", "pytest", "-q"], "pyproject.toml/pytest")
            return ([sys.executable, "-m", "pytest", "-q"], "pyproject.toml/pytest")
        if (root / "Cargo.toml").is_file():
            return (["cargo", "test", "--quiet"], "Cargo.toml")
        if (root / "package.json").is_file():
            return (["npm", "test"], "package.json")
        return None

    def _kill(self, process: subprocess.Popen[str], pgid: int) -> None:
        """Kill the command's process group after every exit path."""
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pgid, signal.SIGKILL)
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=2)

    def _report(self, *, argv: list[str], source: str, exit_code: int, output: str) -> str:
        """Build the structured summary for one test run."""
        rows = [f"command: {' '.join(argv)} ({source})", f"exit_code: {exit_code}"]
        counts = self._counts(output)
        if counts:
            rows.append(counts)
        failures = self._failures(output)
        if failures:
            rows.append("failures:")
            rows.extend(f"  {line}" for line in failures)
        tail = output[-_TAIL_CHARS:] if output else "(no output)"
        rows.append("tail:")
        rows.append(tail)
        text = "\n".join(rows)
        if len(text) > self.max_output_chars:
            text = text[: self.max_output_chars] + "\n... [output truncated]"
        return text

    def _counts(self, output: str) -> str:
        """Return a compact counts line for pytest or cargo output."""
        cargo = _CARGO_RESULT_RE.search(output)
        if cargo is not None:
            return f"passed={cargo.group(2)} failed={cargo.group(3)} ignored={cargo.group(4)}"
        totals: dict[str, int] = {}
        for match in _PYTEST_COUNT_RE.finditer(output):
            key = match.group(2).removesuffix("s")
            totals[key] = int(match.group(1))
        if not totals:
            return ""
        order = ("passed", "failed", "error", "skipped", "xfailed", "xpassed", "warning")
        parts = [f"{name}={totals[name]}" for name in order if name in totals]
        return " ".join(parts)

    def _failures(self, output: str) -> list[str]:
        """Return identifiable failing test names, capped."""
        lines = output.splitlines()
        failures = [
            line.strip()
            for line in lines
            if line.startswith("FAILED ") or line.startswith("ERROR ")
        ][:_MAX_FAILURES]
        if failures:
            return failures
        for index, line in enumerate(lines):
            if line.strip() == "failures:":
                collected = []
                for candidate in lines[index + 1 :]:
                    stripped = candidate.strip()
                    if not stripped:
                        break
                    collected.append(stripped)
                    if len(collected) >= _MAX_FAILURES:
                        break
                return collected
        return []
