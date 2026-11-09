from __future__ import annotations

import ast
import fnmatch
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from atlas.agent.artifacts import ArtifactError, LocalArtifactStore
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
_SOURCE_SUFFIXES = frozenset(
    {
        ".c",
        ".cc",
        ".cpp",
        ".cs",
        ".go",
        ".h",
        ".hpp",
        ".java",
        ".js",
        ".jsx",
        ".kt",
        ".m",
        ".php",
        ".py",
        ".pyi",
        ".rb",
        ".rs",
        ".scala",
        ".swift",
        ".ts",
        ".tsx",
    }
)
_FALLBACK_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^\s*(?:export\s+)?(?:default\s+)?class\s+([A-Za-z_]\w*)"), "class"),
    (re.compile(r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+([A-Za-z_]\w*)"), "fn"),
    (re.compile(r"^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)"), "def"),
    (re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_]\w*)"), "function"),
    (
        re.compile(
            r"^\s*(?:export\s+)?(?:pub(?:\([^)]*\))?\s+)?(?P<kw>interface|struct|trait|enum|type)\s+(?P<name>[A-Za-z_]\w*)"
        ),
        "type",
    ),
    (re.compile(r"^\s*(?:pub\s+)?impl\s+(?:<[^>]*>\s*)?([A-Za-z_]\w*)"), "impl"),
    (
        re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:const|static|let|var)\s+([A-Za-z_]\w*)"),
        "value",
    ),
)


@dataclass
class _Symbol:
    """One named definition found in a source file."""

    line: int
    end_line: int
    depth: int
    kind: str
    name: str
    detail: str = ""


class OutlineInput(BaseModel):
    """Arguments for ``outline``.

    ``path`` may be a file or a directory. ``glob`` filters file names, and a
    bare directory with no glob reads the common source suffixes.
    """

    path: str = Field(default=".", description="File or directory relative to the workspace.")
    glob: str | None = Field(
        default=None,
        description="Only outline files whose name matches this shell-style pattern, e.g. '*.py'.",
    )
    max_files: int = Field(
        default=50,
        ge=1,
        le=200,
        description="Maximum number of files to outline (1-200).",
    )


class OutlineTool(Tool):
    """Return the structural outline of source files.

    Python files are parsed with :mod:`ast`, so comments and string literals
    never produce symbols. Other source files fall back to a line-based scan.
    Every path resolves through :class:`~atlas.agent.artifacts.LocalArtifactStore`.
    Files are read as text, never imported or executed.
    """

    name = "outline"
    description = (
        "List the classes, functions, methods, and module constants of source files with "
        "line ranges. Python uses an AST parse; other languages use a line-based scan."
    )
    trust: ClassVar[Literal["default", "opt_in"]] = "default"
    max_file_bytes: ClassVar[int] = 1_000_000
    max_output_chars: ClassVar[int] = 200_000
    max_symbols_per_file: ClassVar[int] = 200

    @property
    def input_model(self) -> type[BaseModel]:
        return OutlineInput

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        request = OutlineInput.model_validate(arguments)
        base = store.resolve_read(request.path)
        if not base.exists():
            raise FileNotFoundError(f"Path does not exist: {request.path}")
        if base.is_file():
            files = [base]
        else:
            files = list(self._iter_files(base, request.glob, request.max_files))
        if not files:
            return ToolResult(text="no source files found")

        sections: list[str] = []
        used = 0
        for path in files:
            section = self._outline_file(path, store)
            if used + len(section) > self.max_output_chars:
                sections.append("... [output limit reached]")
                break
            sections.append(section)
            used += len(section)
        return ToolResult(text="\n\n".join(sections))

    def _iter_files(self, base: Path, pattern: str | None, limit: int) -> list[Path]:
        """Return up to ``limit`` source files under ``base`` in sorted order."""
        found: list[Path] = []
        for root, dir_names, file_names in os.walk(base, followlinks=False):
            dir_names[:] = sorted(name for name in dir_names if name not in _SKIP_DIRS)
            for name in sorted(file_names):
                candidate = Path(root) / name
                if not candidate.is_file():
                    continue
                if pattern is not None:
                    if not fnmatch.fnmatch(name, pattern):
                        continue
                elif candidate.suffix.lower() not in _SOURCE_SUFFIXES:
                    continue
                found.append(candidate)
                if len(found) >= limit:
                    return found
        return found

    def _outline_file(self, path: Path, store: LocalArtifactStore) -> str:
        """Return one file's outline section, or a short error section."""
        display = store.relative(path)
        try:
            data = store.read_bytes(display, max_bytes=self.max_file_bytes)
        except (ArtifactError, FileNotFoundError, OSError) as exc:
            return f"== {display}\n[error: {type(exc).__name__}: {exc}]"
        if b"\x00" in data:
            return f"== {display}\n[skipped: binary file]"
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            return f"== {display}\n[skipped: not valid UTF-8]"
        lines = text.splitlines()
        symbols: list[_Symbol] = []
        note = ""
        if path.suffix.lower() in (".py", ".pyi"):
            try:
                symbols = self._python_symbols(ast.parse(text))
            except SyntaxError as exc:
                note = f" [parse error at line {exc.lineno}; line scan used]"
                symbols = self._scan_symbols(lines)
        else:
            symbols = self._scan_symbols(lines)
        symbols = symbols[: self.max_symbols_per_file]
        header = f"== {display} ({len(lines)} lines, {len(symbols)} symbols){note}"
        if not symbols:
            return f"{header}\n(no symbols found)"
        rows = [header]
        for symbol in symbols:
            indent = "  " * symbol.depth
            span = (
                f"L{symbol.line}"
                if symbol.end_line == symbol.line
                else f"L{symbol.line}-L{symbol.end_line}"
            )
            rows.append(f"{span:>14}  {indent}{symbol.kind} {symbol.name}{symbol.detail}")
        return "\n".join(rows)

    def _python_symbols(self, tree: ast.Module) -> list[_Symbol]:
        """Collect module-level and class-level definitions in source order."""
        symbols: list[_Symbol] = []

        def visit_body(body: list[ast.stmt], depth: int) -> None:
            for node in body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    kind = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
                    symbols.append(
                        _Symbol(
                            line=node.lineno,
                            end_line=node.end_lineno or node.lineno,
                            depth=depth,
                            kind=kind,
                            name=node.name,
                            detail=_signature(node),
                        )
                    )
                elif isinstance(node, ast.ClassDef):
                    bases = ", ".join(ast.unparse(base) for base in node.bases)
                    symbols.append(
                        _Symbol(
                            line=node.lineno,
                            end_line=node.end_lineno or node.lineno,
                            depth=depth,
                            kind="class",
                            name=node.name,
                            detail=f"({bases})" if bases else "",
                        )
                    )
                    visit_body(node.body, depth + 1)
                elif depth == 0:
                    names = _assigned_names(node)
                    if names:
                        value = _assigned_value(node)
                        symbols.append(
                            _Symbol(
                                line=node.lineno,
                                end_line=node.end_lineno or node.lineno,
                                depth=0,
                                kind="value",
                                name=", ".join(names),
                                detail=value,
                            )
                        )

        visit_body(tree.body, 0)
        return sorted(symbols, key=lambda symbol: symbol.line)

    def _scan_symbols(self, lines: list[str]) -> list[_Symbol]:
        """Line-scan definitions for a non-Python source file."""
        symbols: list[_Symbol] = []
        for number, line in enumerate(lines, 1):
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            for pattern, kind in _FALLBACK_PATTERNS:
                match = pattern.match(line)
                if match is None:
                    continue
                groups = match.groupdict()
                symbols.append(
                    _Symbol(
                        line=number,
                        end_line=number,
                        depth=min(indent // 4, 6),
                        kind=groups.get("kw") or kind,
                        name=groups.get("name") or match.group(1),
                    )
                )
                break
        return symbols


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Render a compact ``(args) -> return`` signature."""
    try:
        arguments = ast.unparse(node.args)
    except Exception:  # pragma: no cover - ast.unparse is stable on valid trees
        arguments = "..."
    returns = f" -> {ast.unparse(node.returns)}" if node.returns is not None else ""
    return f"({arguments}){returns}"


def _assigned_names(node: ast.stmt) -> list[str]:
    """Return simple names bound by a module-level assignment."""
    targets: list[ast.expr] = []
    if isinstance(node, ast.Assign):
        targets = list(node.targets)
    elif isinstance(node, ast.AnnAssign):
        targets = [node.target]
    names: list[str] = []
    for target in targets:
        if isinstance(target, ast.Name):
            names.append(target.id)
    return names


def _assigned_value(node: ast.stmt) -> str:
    """Return a short rendering of an assignment's value."""
    if not isinstance(node, (ast.Assign, ast.AnnAssign)):
        return ""
    value = node.value
    if value is None:
        return ""
    try:
        text = ast.unparse(value)
    except Exception:  # pragma: no cover - ast.unparse is stable on valid trees
        return ""
    return f" = {text[:60]}"
