from __future__ import annotations

import ast
import fnmatch
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import Tool, ToolResult
from atlas.agent.tools._walk import load_ignore_set, prune_dirs

_EXTRA_SKIP_DIRS = frozenset({"build", "dist", "target", ".atlas"})
_DEFINITION_KINDS = frozenset({"class", "def", "assign", "param"})
_PYTHON_SUFFIXES = (".py", ".pyi")
_EXT_GLOB = re.compile(r"\*\.(\w+)")


@dataclass
class _ScanStats:
    """Counts of Python files that could not be searched."""

    unparsed: int = 0
    skipped: int = 0


@dataclass
class _Match:
    """One definition or reference found in a Python file."""

    path: str
    line: int
    column: int
    kind: str
    text: str


class FindReferencesInput(BaseModel):
    """Arguments for ``find_references``.

    ``name`` is one Python identifier. ``glob`` filters which files are parsed;
    ``include_definition`` keeps or drops the binding sites.
    """

    name: str = Field(
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
        description="Python identifier to find, e.g. 'apply_patch' or 'Agent'.",
    )
    path: str = Field(default=".", description="File or directory relative to the workspace.")
    glob: str | None = Field(
        default="*.py",
        description="Only parse files whose name matches this shell-style pattern.",
    )
    include_definition: bool = Field(
        default=True,
        description="Include the definition and assignment sites for the name.",
    )
    max_matches: int = Field(
        default=200,
        ge=1,
        le=1000,
        description="Maximum number of matches to return (1-1000).",
    )


class FindReferencesTool(Tool):
    """Find every use of a Python name without importing or executing the file.

    The scan is AST-based, so names inside comments and string literals are not
    matches. Attribute access ``obj.name`` is reported as ``attribute``, and a
    binding site as ``assign``, ``class``, ``def``, or ``param``. Every path is
    resolved through :class:`~atlas.agent.artifacts.LocalArtifactStore`.
    """

    name = "find_references"
    description = (
        "Python only (.py/.pyi): find where a name is defined and used (AST-based, so comments "
        "and strings are ignored). Returns path:line:column with the source line. A directory "
        "scan skips git-ignored paths, .git, virtualenvs, caches, build output (build, dist, "
        "target) and .atlas. For other languages use grep."
    )
    trust: ClassVar[Literal["default", "opt_in"]] = "default"
    max_file_bytes: ClassVar[int] = 1_000_000
    max_scan_bytes: ClassVar[int] = 20_000_000
    max_line_chars: ClassVar[int] = 200

    @property
    def input_model(self) -> type[BaseModel]:
        return FindReferencesInput

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        request = FindReferencesInput.model_validate(arguments)
        base = store.resolve_read(request.path)
        if not base.exists():
            raise FileNotFoundError(f"Path does not exist: {request.path}")
        refusal = self._non_python_refusal(request, base, store)
        if refusal is not None:
            return ToolResult(text=refusal)
        found: list[_Match] = []
        overflow = 0
        scanned = 0
        stats = _ScanStats()
        stopped = False
        for path, text in self._iter_python(base, request.glob, store.root, stats):
            scanned += len(text.encode("utf-8", errors="replace"))
            if scanned > self.max_scan_bytes:
                stopped = True
                break
            matches = self._scan_file(path, text, request.name, store, stats)
            for match in matches:
                if not request.include_definition and match.kind in _DEFINITION_KINDS:
                    continue
                if len(found) < request.max_matches:
                    found.append(match)
                else:
                    overflow += 1
        found.sort(key=lambda match: (match.path, match.line, match.column))
        rows = [
            f"{match.path}:{match.line}:{match.column}: {match.kind}: {match.text}"
            for match in found
        ]
        if overflow:
            rows.append(f"... [{overflow} more matches]")
        if not found and not overflow:
            rows = [f"no references to {request.name}"]
        rows.extend(self._notes(stats, stopped))
        return ToolResult(text="\n".join(rows))

    def _non_python_refusal(
        self, request: FindReferencesInput, base: Path, store: LocalArtifactStore
    ) -> str | None:
        """Return a refusal text when the target is explicitly not Python."""
        ext: str | None = None
        if base.is_file() and base.suffix.lower() not in _PYTHON_SUFFIXES:
            ext = base.suffix
        elif request.glob is not None:
            found = _EXT_GLOB.fullmatch(request.glob)
            if found is not None and f".{found.group(1).lower()}" not in _PYTHON_SUFFIXES:
                ext = found.group(1)
        if ext is None:
            return None
        return (
            f"find_references reads Python (.py/.pyi) only; {store.relative(base)} is not "
            f"Python. Use grep -w {request.name} for other languages."
        )

    def _notes(self, stats: _ScanStats, stopped: bool) -> list[str]:
        """Return note lines for files that were not searched."""
        notes: list[str] = []
        if stats.unparsed:
            notes.append(
                f"note: {stats.unparsed} Python file(s) could not be parsed (syntax error) "
                "and were not searched"
            )
        if stats.skipped:
            notes.append(
                f"note: {stats.skipped} Python file(s) over 1 MB, binary or non-UTF-8 were skipped"
            )
        if stopped:
            notes.append("note: scan stopped after 20 MB; narrow path")
        return notes

    def _iter_python(
        self, base: Path, pattern: str | None, store_root: Path, stats: _ScanStats
    ) -> Iterator[tuple[Path, str]]:
        """Yield readable Python files under ``base`` with their text.

        Only ``.py``/``.pyi`` files are candidates, so ``stats.skipped`` counts Python only.
        """
        for path in self._candidates(base, store_root):
            if path.suffix.lower() not in _PYTHON_SUFFIXES:
                continue
            if pattern is not None and not fnmatch.fnmatch(path.name, pattern):
                continue
            try:
                data = path.read_bytes()
            except OSError:
                continue
            if len(data) > self.max_file_bytes or b"\x00" in data:
                stats.skipped += 1
                continue
            try:
                yield path, data.decode("utf-8")
            except UnicodeDecodeError:
                stats.skipped += 1

    def _candidates(self, base: Path, store_root: Path) -> Iterator[Path]:
        """Yield regular files under ``base`` in sorted order, honouring skips and gitignore."""
        if base.is_file():
            yield base
            return
        ignore = load_ignore_set(base, root=store_root)
        for root, dir_names, file_names in os.walk(base, followlinks=False):
            current = Path(root)
            prune_dirs(
                current, dir_names, root=store_root, extra_skip=_EXTRA_SKIP_DIRS, ignore=ignore
            )
            for name in sorted(file_names):
                if ignore is not None and ignore.hides_file(current, name):
                    continue
                path = current / name
                if path.is_file():
                    yield path

    def _scan_file(
        self, path: Path, text: str, name: str, store: LocalArtifactStore, stats: _ScanStats
    ) -> list[_Match]:
        """Parse one file and return its definition and reference matches."""
        try:
            tree = ast.parse(text)
        except SyntaxError:
            stats.unparsed += 1
            return []
        lines = text.splitlines()
        display = store.relative(path)
        found: dict[tuple[int, int], _Match] = {}

        def add(node: ast.AST, kind: str) -> None:
            line = getattr(node, "lineno", 0)
            column = getattr(node, "col_offset", 0) + 1
            key = (line, column)
            existing = found.get(key)
            if existing is not None and existing.kind not in ("reference", "attribute"):
                return
            preview = lines[line - 1].strip() if 0 < line <= len(lines) else ""
            found[key] = _Match(
                path=display,
                line=line,
                column=column,
                kind=kind,
                text=preview[: self.max_line_chars],
            )

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                add(node, "def")
            elif isinstance(node, ast.ClassDef) and node.name == name:
                add(node, "class")
            elif isinstance(node, ast.Name) and node.id == name:
                add(node, "reference")
            elif isinstance(node, ast.Attribute) and node.attr == name:
                add(node, "attribute")
            elif isinstance(node, ast.arg) and node.arg == name:
                add(node, "param")
            elif isinstance(node, (ast.Assign, ast.AnnAssign)) and name in _assigned_names(node):
                add(node, "assign")
        return sorted(found.values(), key=lambda match: (match.line, match.column))


def _assigned_names(node: ast.Assign | ast.AnnAssign) -> list[str]:
    """Return simple names bound by an assignment."""
    targets: list[ast.expr] = list(node.targets) if isinstance(node, ast.Assign) else [node.target]
    names: list[str] = []
    for target in targets:
        if isinstance(target, ast.Name):
            names.append(target.id)
    return names
