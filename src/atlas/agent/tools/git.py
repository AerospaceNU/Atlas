from __future__ import annotations

import shutil
import subprocess
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import Tool, ToolResult

_TIMEOUT_SECONDS = 30.0


def _git_binary() -> str:
    """Return the git binary path, refusing to guess when it is absent."""
    binary = shutil.which("git")
    if binary is None:
        raise ValueError("git binary not found on PATH")
    return binary


def _run(store: LocalArtifactStore, arguments: list[str]) -> subprocess.CompletedProcess[str]:
    """Run one git subcommand in the workspace without a shell.

    ``arguments`` is always an argument list, so no model-supplied string is
    interpreted by a shell. The current directory is the resolved workspace
    root, keeping every pathspec inside the sandbox.
    """
    try:
        return subprocess.run(
            [_git_binary(), *arguments],
            cwd=store.resolve("."),
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError("git command timed out") from exc


def _is_work_tree(store: LocalArtifactStore) -> bool:
    """Return whether the workspace root is inside a git work tree."""
    result = _run(store, ["rev-parse", "--is-inside-work-tree"])
    return result.returncode == 0 and result.stdout.strip() == "true"


def _display_path(store: LocalArtifactStore, path: str) -> str:
    """Resolve ``path`` through the store and return its workspace-relative form."""
    return store.relative(store.resolve(path))


def _repo_prefix(store: LocalArtifactStore) -> str:
    """Return the workspace root's prefix inside the repository root, if any."""
    result = _run(store, ["rev-parse", "--show-prefix"])
    return result.stdout.strip() if result.returncode == 0 else ""


def _strip_prefix(path: str, prefix: str) -> str:
    """Strip the repository prefix from one status path, quoted or not."""
    if not prefix:
        return path
    if path.startswith(prefix):
        return path[len(prefix) :]
    if path.startswith(f'"{prefix}'):
        return f'"{path[1 + len(prefix) :]}'
    return path


def _workspace_status_line(line: str, prefix: str) -> str:
    """Rewrite one porcelain status entry as a workspace-relative path."""
    if not prefix or len(line) <= 3 or line.startswith("##"):
        return line
    head, body = line[:3], line[3:]
    old, arrow, new = body.partition(" -> ")
    if arrow:
        body = f"{_strip_prefix(old, prefix)} -> {_strip_prefix(new, prefix)}"
    else:
        body = _strip_prefix(body, prefix)
    return f"{head}{body}"


def _failure(action: str, result: subprocess.CompletedProcess[str]) -> ValueError:
    """Build a workspace-relative error for a failed git invocation."""
    detail = result.stderr.strip() or result.stdout.strip() or f"exit status {result.returncode}"
    return ValueError(f"git {action} failed: {detail}")


class GitStatusInput(BaseModel):
    """Arguments for ``git_status``.

    ``path`` is a workspace-relative directory or file used to narrow the
    reported changes; the default reports the whole workspace.
    """

    path: str = Field(default=".", description="Path relative to the local artifact workspace.")


class GitDiffInput(BaseModel):
    """Arguments for ``git_diff``.

    ``path`` is workspace-relative, ``staged`` reads the index instead of the
    working tree, and ``max_bytes`` caps the returned diff text.
    """

    path: str = Field(default=".", description="Path relative to the local artifact workspace.")
    staged: bool = Field(
        default=False, description="Show staged changes (git diff --cached) instead of unstaged."
    )
    max_bytes: int = Field(
        default=100_000,
        ge=1,
        le=1_000_000,
        description="Maximum number of diff bytes to return (1-1000000).",
    )


class GitCommitInput(BaseModel):
    """Arguments for ``git_commit``.

    ``message`` is passed as a single commit message. ``paths`` lists
    workspace-relative files or directories to stage first; an empty list
    stages every change in the workspace.
    """

    message: str = Field(
        min_length=1,
        max_length=2000,
        description="Commit message, 1-2000 characters.",
    )
    paths: list[str] = Field(
        default_factory=list,
        description=(
            "Workspace-relative files or directories to stage before committing. "
            "Empty stages all changes."
        ),
    )


class GitStatusTool(Tool):
    """Show the current branch and changed files in the workspace.

    The optional path and every reported name are workspace-relative. A
    workspace that is not a git work tree returns ``not a git work tree``
    instead of raising.
    """

    name = "git_status"
    description = (
        "Show the current git branch and changed files in the local artifact workspace. "
        "Returns the ## branch line followed by porcelain status entries."
    )
    trust: ClassVar[Literal["default", "opt_in"]] = "default"

    @property
    def input_model(self) -> type[BaseModel]:
        return GitStatusInput

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        request = GitStatusInput.model_validate(arguments)
        display = _display_path(store, request.path)
        if not _is_work_tree(store):
            return ToolResult(text="not a git work tree")
        prefix = _repo_prefix(store)
        result = _run(store, ["status", "--porcelain=v1", "--branch", "--", display])
        if result.returncode != 0:
            raise _failure("status", result)
        lines = [_workspace_status_line(line, prefix) for line in result.stdout.splitlines()]
        return ToolResult(text="\n".join(lines))


class GitDiffTool(Tool):
    """Show a unified diff for the workspace or one workspace-relative path.

    The path is resolved only through :class:`~atlas.agent.artifacts.LocalArtifactStore`,
    so a diff cannot leave the sandbox. Output is truncated at ``max_bytes``
    with a visible marker.
    """

    name = "git_diff"
    description = (
        "Show a unified git diff in the local artifact workspace, optionally staged. "
        "Long diffs are truncated with a visible marker."
    )
    trust: ClassVar[Literal["default", "opt_in"]] = "default"

    @property
    def input_model(self) -> type[BaseModel]:
        return GitDiffInput

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        request = GitDiffInput.model_validate(arguments)
        display = _display_path(store, request.path)
        if not _is_work_tree(store):
            return ToolResult(text="not a git work tree")
        command = ["diff", "--relative", "--cached"] if request.staged else ["diff", "--relative"]
        command.extend(["--", display])
        result = _run(store, command)
        if result.returncode != 0:
            raise _failure("diff", result)
        data = result.stdout.encode("utf-8")
        if not data:
            return ToolResult(text="no changes")
        if len(data) <= request.max_bytes:
            return ToolResult(text=result.stdout)
        kept = data[: request.max_bytes].decode("utf-8", errors="ignore")
        removed = len(data) - request.max_bytes
        return ToolResult(text=f"{kept}\n... [truncated {removed} bytes]")


class GitCommitTool(Tool):
    """Stage workspace changes and create one commit.

    Only ``git add`` and ``git commit`` are run; amend, force, and push are not
    expressible through this tool. Every path is resolved through
    :class:`~atlas.agent.artifacts.LocalArtifactStore` before staging, so a
    commit cannot reach outside the sandbox. With nothing staged after the add,
    no empty commit is created.
    """

    name = "git_commit"
    description = (
        "Stage workspace-relative paths (or all changes) and create one git commit. "
        "Returns the first line of the commit output, or 'nothing to commit'."
    )
    trust: ClassVar[Literal["default", "opt_in"]] = "default"

    @property
    def input_model(self) -> type[BaseModel]:
        return GitCommitInput

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        request = GitCommitInput.model_validate(arguments)
        if not request.message.strip() or "\x00" in request.message:
            raise ValueError("Commit message must not be empty or contain NUL bytes")
        displays = [_display_path(store, path) for path in request.paths]
        if not _is_work_tree(store):
            raise ValueError("not a git work tree")

        add_command = ["add", "--", *displays] if displays else ["add", "-A", "--", "."]
        added = _run(store, add_command)
        if added.returncode != 0:
            raise _failure("add", added)

        staged = _run(store, ["diff", "--cached", "--quiet"])
        if staged.returncode == 0:
            return ToolResult(text="nothing to commit")
        if staged.returncode != 1:
            raise _failure("diff", staged)

        committed = _run(store, ["commit", "-m", request.message])
        if committed.returncode != 0:
            raise _failure("commit", committed)
        lines = committed.stdout.strip().splitlines()
        return ToolResult(text=lines[0] if lines else "committed")
