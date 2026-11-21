from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import Tool, ToolResult
from atlas.agent.tools._process import OUTPUT_LIMIT_NOTE, run_captured
from atlas.agent.tools._project_env import project_python

MAX_TIMEOUT = 900
_PYTEST_COUNT_RE = re.compile(
    r"(\d+) (passed|failed|error|errors|skipped|xfailed|xpassed|warning|warnings)\b"
)
_CARGO_RESULT_RE = re.compile(
    r"test result: (?:ok|FAILED)\. (\d+) passed; (\d+) failed; (\d+) ignored"
)
_GO_PKG_RE = re.compile(r"^(ok|FAIL)[ \t]+(\S+)[ \t]+(?:[\d.]+s|\(cached\)|\[)", re.MULTILINE)
_GO_FAIL_RE = re.compile(r"^\s*--- FAIL: (\S+)")
_UNITTEST_RAN_RE = re.compile(r"^Ran (\d+) tests? in ", re.MULTILINE)
_UNITTEST_FAILED_RE = re.compile(r"^FAILED \((.*)\)\s*$", re.MULTILINE)
_UNITTEST_OK_RE = re.compile(r"^OK(?: \((.*)\))?\s*$", re.MULTILINE)
_UNITTEST_NAME_RE = re.compile(r"^(?:FAIL|ERROR): \S+ \(.+\)\s*$")
_SYMPY_RE = re.compile(r"tests finished: (.*)")
_KV_RE = re.compile(r"^([a-z ]+)=(\d+)$")
_FAILURE_LINE_RE = re.compile(r"^(?:FAILED|ERROR) (\S+)(.*)$")
_SUMMARY_HEADER_RE = re.compile(r"^=+ short test summary info =+\s*$")
_SYMPY_PART_RE = re.compile(
    r"(\d+) (expected to fail but passed|expected to fail|passed|failed|skipped|deselected"
    r"|exceptions)\b"
)
_SYMPY_LABELS = {"expected to fail": "xfailed", "expected to fail but passed": "xpassed"}
_PYTEST_MISSING_HINT = (
    "hint: pytest is not available to this interpreter; pass command=[...] with the "
    'project\'s own runner, for example ["python", "tests/runtests.py", "<label>"] '
    'or ["bin/test", "<path>"]'
)
_TAIL_CHARS = 4000
_MAX_FAILURES = 20


def _resolve_executable(name: str, cwd: Path) -> str | None:
    """Resolve ``name`` to an executable path, or None when it does not exist.

    A name with a path separator is resolved against ``cwd`` (the directory the
    command runs in), never the process working directory; absolute paths are
    used as given. A path must be an existing executable file; a bare name is looked up on PATH.
    """
    if "/" not in name and os.sep not in name:
        return shutil.which(name)
    candidate = Path(name)
    if not candidate.is_absolute():
        candidate = cwd / candidate
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    return None


def _node_test_script(root: Path) -> bool:
    """True when package.json defines a real ``scripts.test`` (not the npm-init stub)."""
    try:
        data = json.loads((root / "package.json").read_text(errors="ignore"))
    except (OSError, ValueError):
        return False
    scripts = data.get("scripts") if isinstance(data, dict) else None
    script = scripts.get("test") if isinstance(scripts, dict) else None
    return isinstance(script, str) and bool(script.strip()) and "no test specified" not in script


def _mentions_pytest(path: Path) -> bool:
    """True when the file text contains "pytest"; False when unreadable."""
    try:
        return "pytest" in path.read_text(errors="ignore")
    except OSError:
        return False


def _node_runner(root: Path) -> list[str]:
    """Pick the package-manager test argv from the first lockfile present."""
    if (root / "pnpm-lock.yaml").is_file():
        return ["pnpm", "test"]
    if (root / "yarn.lock").is_file():
        return ["yarn", "test"]
    if (root / "bun.lock").is_file() or (root / "bun.lockb").is_file():
        return ["bun", "run", "test"]  # not bare "bun test": that is bun's own runner
    return ["npm", "test"]


def _unittest_stats(summary: str) -> dict[str, int]:
    """Parse the ``key=N`` items of a unittest result line; keys may hold spaces."""
    stats: dict[str, int] = {}
    for item in summary.split(","):
        match = _KV_RE.match(item.strip())
        if match is not None:
            stats[match.group(1)] = int(match.group(2))
    return stats


def _summary_section(lines: list[str]) -> list[str]:
    """Return the lines after pytest's "short test summary info" header, or []."""
    for index, line in enumerate(lines):
        if _SUMMARY_HEADER_RE.match(line):
            return lines[index + 1 :]
    return []


def _is_failure_line(line: str) -> bool:
    """True for ``FAILED <id>`` or ``ERROR <id>`` where the id looks like a test id."""
    match = _FAILURE_LINE_RE.match(line)
    if match is None:
        return False
    token, rest = match.group(1), match.group(2)
    return "::" in token or token.endswith(".py") or rest.startswith(" - ")


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
        description="Seconds to allow the command to run (values above 900 are clamped to 900).",
    )


class RunTestsTool(Tool):
    """Run the project's test command and summarise the result.

    The command is detected from the workspace (pytest config or ``pyproject.toml``,
    ``go.mod``, ``Cargo.toml``, then a ``package.json`` test script) unless an explicit
    argv is given.
    A non-zero exit is a result, not an exception. Output is capped, and the
    command runs without a shell.
    """

    name = "run_tests"
    description = (
        "Run tests and summarise the result: exit code, pass/fail counts, failing test names "
        "and the tail of the output. With no command it detects pytest (via uv, the project .venv "
        "or PATH python), go test, cargo test or the package.json test script; "
        "pass command=[...] (an argv list, no shell) for any other runner or to select tests. "
        "timeout up to 900 seconds."
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
                        "no test command detected; pass command=[...] (looked for pytest "
                        "config, go.mod, Cargo.toml, package.json with a test script)"
                    )
                )
            argv, source = detected
        binary = _resolve_executable(argv[0], cwd)
        if binary is None:
            return ToolResult(text=f"command not found: {argv[0]}")
        argv = [binary, *argv[1:]]

        timeout = min(request.timeout, MAX_TIMEOUT)
        note = f"note: timeout clamped to {MAX_TIMEOUT}s" if timeout != request.timeout else None
        started = time.monotonic()
        captured = run_captured(argv, cwd=cwd, timeout=timeout)
        combined = f"{captured.stdout}\n{captured.stderr}".strip()
        if captured.timed_out:
            report = self._timeout_report(argv=argv, timeout=timeout, output=combined)
            return ToolResult(text=f"{note}\n{report}" if note else report)
        if captured.cancelled:
            elapsed = int(time.monotonic() - started)
            tail = combined[-_TAIL_CHARS:] if combined else "(no output)"
            return ToolResult(
                text=f"test command cancelled after {elapsed}s: {' '.join(argv)}\ntail:\n{tail}"
            )
        report = self._report(
            argv=argv,
            source=source,
            exit_code=captured.exit_code if captured.exit_code is not None else -1,
            output=combined,
            output_limited=captured.output_limited,
        )
        if note:
            first, _, rest = report.partition("\n")
            report = f"{first}\n{note}\n{rest}"
        return ToolResult(text=report)

    def _detect(self, root: Path) -> tuple[list[str], str] | None:
        """Return the test argv and its source, or None when nothing matches."""
        has_ini = (root / "pytest.ini").is_file() or (root / "tox.ini").is_file()
        pyproject = root / "pyproject.toml"
        other = (
            (root / "go.mod").is_file()
            or (root / "Cargo.toml").is_file()
            or _node_test_script(root)
        )
        if has_ini or (pyproject.is_file() and (not other or _mentions_pytest(pyproject))):
            return self._detect_python(root)
        if (root / "go.mod").is_file():
            return (["go", "test", "./..."], "go.mod")
        if (root / "Cargo.toml").is_file():
            return (["cargo", "test", "--quiet"], "Cargo.toml")
        if _node_test_script(root):
            argv = _node_runner(root)
            return (argv, f"package.json ({argv[0]})")
        # No make/gradle/maven/ctest by design: they run un-gated build scripts (owner decision).
        return None

    def _detect_python(self, root: Path) -> tuple[list[str], str]:
        """Pick the pytest argv: uv, then the project venv, then PATH python."""
        if (root / "uv.lock").is_file() and shutil.which("uv") is not None:
            return (["uv", "run", "pytest", "-q"], "pyproject.toml/pytest")
        venv_python = project_python(root)
        if venv_python is not None:
            return ([venv_python, "-m", "pytest", "-q"], "pyproject.toml/pytest (project venv)")
        python = shutil.which("python") or shutil.which("python3") or sys.executable
        return ([python, "-m", "pytest", "-q"], "pyproject.toml/pytest")

    def _timeout_report(self, *, argv: list[str], timeout: int, output: str) -> str:
        """Report a timeout together with whatever was recorded before the kill."""
        rows = [f"test command timed out after {timeout}s: {' '.join(argv)}"]
        counts = self._counts(output)
        if counts:
            rows.append(counts)
        failures = self._failures(output)
        if failures:
            rows.append("failures:")
            rows.extend(f"  {line}" for line in failures)
        rows.append("tail:")
        rows.append(output[-_TAIL_CHARS:] if output else "(no output)")
        return "\n".join(rows)

    def _report(
        self,
        *,
        argv: list[str],
        source: str,
        exit_code: int,
        output: str,
        output_limited: bool = False,
    ) -> str:
        """Build the structured summary for one test run."""
        rows = [f"command: {' '.join(argv)} ({source})", f"exit_code: {exit_code}"]
        if output_limited:
            rows.append(OUTPUT_LIMIT_NOTE)
        counts = self._counts(output)
        if counts:
            rows.append(counts)
        failures = self._failures(output)
        if failures:
            rows.append("failures:")
            rows.extend(f"  {line}" for line in failures)
        if "No module named pytest" in output:
            rows.append(_PYTEST_MISSING_HINT)
        tail = output[-_TAIL_CHARS:] if output else "(no output)"
        rows.append("tail:")
        rows.append(tail)
        text = "\n".join(rows)
        if len(text) > self.max_output_chars:
            text = text[: self.max_output_chars] + "\n... [output truncated]"
        return text

    def _counts(self, output: str) -> str:
        """Return a compact counts line for pytest or cargo output."""
        cargo = _CARGO_RESULT_RE.findall(output)
        if cargo:
            passed, failed, ignored = (sum(int(row[i]) for row in cargo) for i in range(3))
            return f"passed={passed} failed={failed} ignored={ignored}"
        packages = _GO_PKG_RE.findall(output)
        if packages:
            failed_pkgs = sum(1 for status, _ in packages if status == "FAIL")
            return f"packages={len(packages)} failed_packages={failed_pkgs}"
        unittest = self._unittest_counts(output)
        if unittest:
            return unittest
        sympy = _SYMPY_RE.search(output)
        if sympy is not None:
            summary = sympy.group(1).split(", in ", 1)[0]
            found = {_SYMPY_LABELS.get(w, w): int(n) for n, w in _SYMPY_PART_RE.findall(summary)}
            return " ".join(f"{name}={count}" for name, count in found.items())
        totals: dict[str, int] = {}
        for match in _PYTEST_COUNT_RE.finditer(output):
            key = match.group(2).removesuffix("s")
            totals[key] = int(match.group(1))
        if not totals:
            return ""
        order = ("passed", "failed", "error", "skipped", "xfailed", "xpassed", "warning")
        parts = [f"{name}={totals[name]}" for name in order if name in totals]
        return " ".join(parts)

    def _unittest_counts(self, output: str) -> str:
        """Return counts for unittest/Django output, or an empty string."""
        ran = _UNITTEST_RAN_RE.search(output)
        if ran is None:
            return ""
        total = int(ran.group(1))
        failed = _UNITTEST_FAILED_RE.search(output)
        ok = _UNITTEST_OK_RE.search(output)
        summary = failed or ok
        if summary is None:
            return f"ran={total}"
        stats = _unittest_stats(summary.group(1) or "")
        skipped = stats.get("skipped", 0)
        xfailed = stats.get("expected failures", 0)
        xpassed = stats.get("unexpected successes", 0)
        bad = stats.get("failures", 0) + stats.get("errors", 0)
        parts = [f"passed={max(total - bad - skipped - xfailed - xpassed, 0)}"]
        for label, value in (
            ("failed", stats.get("failures", 0)),
            ("error", stats.get("errors", 0)),
            ("skipped", skipped),
            ("xfailed", xfailed),
            ("xpassed", xpassed),
        ):
            if value:
                parts.append(f"{label}={value}")
        return " ".join(parts)

    def _failures(self, output: str) -> list[str]:
        """Return identifiable failing test names, capped."""
        lines = output.splitlines()
        section = _summary_section(lines)
        failures = [
            line.strip()
            for line in (section or lines)
            if _is_failure_line(line) or _UNITTEST_NAME_RE.match(line) or _GO_FAIL_RE.match(line)
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
