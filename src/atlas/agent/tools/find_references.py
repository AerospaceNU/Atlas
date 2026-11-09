from __future__ import annotations

import ast
import fnmatch
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import Tool, ToolResult

_SKIP_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".atlas",
        ".venv",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "target",
        "venv",
    }
)
_DEFINITION_KINDS = frozenset({"class", "def", "assign", "param"})


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
        "Find definitions and uses of a Python identifier across the workspace using an AST "
        "scan. Comments and string literals are excluded; every hit is path:line:column."
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
        found: list[_Match] = []
        overflow = 0
        scanned = 0
        for path, text in self._iter_python(base, request.glob):
            scanned += len(text.encode("utf-8", errors="replace"))
            if scanned > self.max_scan_bytes:
                break
            matches = self._scan_file(path, text, request.name, store)
            for match in matches:
                if not request.include_definition and match.kind in _DEFINITION_KINDS:
                    continue
                if len(found) < request.max_matches:
                    found.append(match)
                else:
                    overflow += 1
        if not found and not overflow:
            return ToolResult(text=f"no references to {request.name}")
        found.sort(key=lambda match: (match.path, match.line, match.column))
        rows = [
            f"{match.path}:{match.line}:{match.column}: {match.kind}: {match.text}"
            for match in found
        ]
        if overflow:
            rows.append(f"... [{overflow} more matches]")
        return ToolResult(text="\n".join(rows))

    def _iter_python(self, base: Path, pattern: str | None) -> list[tuple[Path, str]]:
        """Return readable Python files under ``base`` with their text."""
        if base.is_file():
            candidates = [base]
        else:
            candidates = []
            for root, dir_names, file_names in os.walk(base, followlinks=False):
                dir_names[:] = sorted(name for name in dir_names if name not in _SKIP_DIRS)
                for name in sorted(file_names):
                    candidates.append(Path(root) / name)
        files: list[tuple[Path, str]] = []
        for path in candidates:
            if not path.is_file():
                continue
            if pattern is not None and not fnmatch.fnmatch(path.name, pattern):
                continue
            try:
                data = path.read_bytes()
            except OSError:
                continue
            if len(data) > self.max_file_bytes or b"\x00" in data:
                continue
            try:
                files.append((path, data.decode("utf-8")))
            except UnicodeDecodeError:
                continue
        return files

    def _scan_file(
        self, path: Path, text: str, name: str, store: LocalArtifactStore
    ) -> list[_Match]:
        """Parse one file and return its definition and reference matches."""
        try:
            tree = ast.parse(text)
        except SyntaxError:
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
