"""Atlas substrate: turn a harness tree into what one rollout runs under.

:func:`install_harness` is the mapping from the editable surface to Atlas. It
composes the system prompt from the always-on files, writes skills and helper
scripts under the workspace ``.atlas`` directory where ``load_skills`` and
``bash`` find them, and returns the delegated roles for ``spawn_agent``.
It writes files and reads text; it never runs anything from the tree.

Benchmark launchers own executing rollouts (see
:mod:`atlas.harness_eval.harness2_bench`). When harness2 is installed the
``Install`` value is convertible to its ``harness2.substrate.base.Install``.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from atlas.agent.system_prompt import load_system_prompt
from atlas.harness_eval.surface import ATLAS_SURFACE, SurfaceSpec, read_tree

# Always-on files after ``systemprompt.md``, with the heading each is appended under.
ALWAYS_ON_SECTIONS: tuple[tuple[str, str], ...] = (
    ("guardrails.md", "Guardrails"),
    ("LongTermMEMORY.md", "Long-term memory"),
    ("tool_descriptions/tool_guidance.md", "Tool guidance"),
)
# Workspace directory the on-demand components are installed under. Atlas
# discovers skills at ``<workspace>/.atlas/skills``.
WORKSPACE_DIR = ".atlas"

_SCRIPT_REF = re.compile(r"(?<![\w./-])((?:skills/[\w\-]+/)?scripts/[\w.\-]+)")
_SKILL_FILE = re.compile(r"skills/([a-z0-9][a-z0-9_\-]*)/(.+)")
_SUBAGENT_FILE = re.compile(r"sub_agents/([a-z0-9][a-z0-9_\-]*)/AGENT\.md")
_SCRIPT_FILE = re.compile(r"scripts/[a-z0-9_]+\.py")
_DESCRIPTION = re.compile(r"^description:\s*(.+)$", re.M)


@dataclass(frozen=True)
class SubagentRole:
    """One ``sub_agents/<name>/AGENT.md`` the agent can delegate to by name."""

    name: str
    description: str
    body: str


@dataclass(frozen=True)
class InstalledHarness:
    """What a harness tree contributed to one rollout.

    ``system_prompt`` is ``None`` when the tree has no always-on text, which
    leaves the package prompt in force.
    """

    system_prompt: str | None = None
    roles: tuple[SubagentRole, ...] = ()
    skills: tuple[str, ...] = ()
    scripts: tuple[str, ...] = ()


def _split_frontmatter(text: str) -> tuple[str, str]:
    """Return ``(frontmatter, body)``; the frontmatter is empty when there is none."""
    if not text.startswith("---"):
        return "", text
    end = text.find("\n---", 3)
    if end == -1:
        return "", text
    return text[3:end], text[end + 4 :].lstrip("\n")


def _map_script_paths(text: str, shipped: frozenset[str]) -> str:
    """Point a tree-relative script path at its installed workspace location.

    Only a path the tree ships is rewritten, so a repository's own
    ``scripts/`` directory is left alone.
    """

    def swap(match: re.Match[str]) -> str:
        rel = match.group(1)
        return f"{WORKSPACE_DIR}/{rel}" if rel in shipped else rel

    return _SCRIPT_REF.sub(swap, text)


def compose_system_prompt(
    tree: Mapping[str, str], roles: tuple[SubagentRole, ...] = ()
) -> str | None:
    """Return the system prompt a tree asks for, or ``None`` for the package prompt.

    ``systemprompt.md`` is the base; an empty or missing one falls back to the
    package prompt so the other always-on files still apply. Skills are not
    listed here: ``render_system_prompt`` adds the catalog at run time.
    """
    sections = [
        (title, tree.get(rel, "").strip())
        for rel, title in ALWAYS_ON_SECTIONS
        if tree.get(rel, "").strip()
    ]
    base = tree.get("systemprompt.md", "").strip()
    if not base and not sections and not roles:
        return None
    parts = [base or load_system_prompt()]
    parts += [f"## {title}\n\n{body}" for title, body in sections]
    if roles:
        listing = "\n".join(f"- `{role.name}`: {role.description}" for role in roles)
        parts.append(
            "## Delegated roles\n\n"
            "spawn_agent can run a task under one of these roles. Start the task with "
            "the line `agent: <name>`, then give the self-contained instruction.\n\n" + listing
        )
    return "\n\n".join(parts)


def install_harness(harness_dir: Path | None, workspace: Path) -> InstalledHarness:
    """Install a harness tree into ``workspace`` and return what it contributes.

    Skills land in ``.atlas/skills/<name>/`` and helper scripts in
    ``.atlas/scripts/``. Script paths written tree-relative in the always-on
    text and in skill bodies are rewritten to those locations. In a git
    workspace ``.atlas/`` is added to ``.git/info/exclude`` so the harness
    never appears in the patch. ``harness_dir=None`` installs nothing.
    """
    if harness_dir is None or not Path(harness_dir).is_dir():
        return InstalledHarness()
    tree = read_tree(ATLAS_SURFACE, harness_dir)
    shipped = frozenset(
        rel
        for rel in tree
        if _SCRIPT_FILE.fullmatch(rel) or (rel.startswith("skills/") and "/scripts/" in rel)
    )
    home = workspace / WORKSPACE_DIR
    skills: set[str] = set()
    roles: list[SubagentRole] = []
    for rel, content in sorted(tree.items()):
        skill = _SKILL_FILE.fullmatch(rel)
        if skill is not None:
            skills.add(skill.group(1))
            if skill.group(2) == "SKILL.md":
                content = _map_script_paths(content, shipped)
        elif not _SCRIPT_FILE.fullmatch(rel):
            role = _SUBAGENT_FILE.fullmatch(rel)
            if role is not None:
                frontmatter, body = _split_frontmatter(content)
                found = _DESCRIPTION.search(frontmatter)
                roles.append(
                    SubagentRole(
                        name=role.group(1),
                        description=found.group(1).strip() if found else role.group(1),
                        body=_map_script_paths(body, shipped),
                    )
                )
            continue
        target = home / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    exclude_from_git(workspace)
    always_on = {
        rel: _map_script_paths(tree.get(rel, ""), shipped) for rel in ATLAS_SURFACE.always_on
    }
    return InstalledHarness(
        system_prompt=compose_system_prompt(always_on, tuple(roles)),
        roles=tuple(roles),
        skills=tuple(sorted(skills)),
        scripts=tuple(sorted(shipped)),
    )


def exclude_from_git(workspace: Path) -> None:
    """Keep ``.atlas/`` (harness and agent state) out of ``git status`` and the patch."""
    info = workspace / ".git" / "info"
    if not (workspace / ".git").is_dir():
        return
    info.mkdir(exist_ok=True)
    exclude = info / "exclude"
    line = f"{WORKSPACE_DIR}/"
    current = exclude.read_text(encoding="utf-8") if exclude.is_file() else ""
    if line not in current.splitlines():
        exclude.write_text(current.rstrip("\n") + f"\n{line}\n", encoding="utf-8")


@dataclass(frozen=True)
class Install:
    """A materialised harness tree ready for a launcher."""

    staging_dir: Path
    fingerprint: str
    extras: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RunRequest:
    """One execution request handed to a benchmark launcher body."""

    task_ref: str
    run_ids: tuple[str, ...]
    timeout: int


def _fingerprint(tree_dir: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(tree_dir.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            h.update(str(p.relative_to(tree_dir)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


class AtlasSubstrate:
    """Atlas behind the small substrate-agnostic interface."""

    name = "atlas"

    def surface(self) -> SurfaceSpec:
        """Return the Atlas editable surface."""
        return ATLAS_SURFACE

    def materialise(self, tree_dir: Path, spec: object, ctx: Mapping[str, str]) -> Install:
        """Stage ``tree_dir`` into a runnable harness layout.

        ``tree_dir`` is the proposed tree root. ``spec`` is a BenchSpec-like
        value with ``install_mode`` (accepted but advisory: Atlas always
        installs the same layout). ``ctx`` carries ``file:<rel>`` entries
        copied alongside the tree and ``{{token}}`` substitutions in
        ``systemprompt.md``.

        Returns an :class:`Install` whose ``staging_dir`` is passed as
        ``harness_dir`` to :func:`atlas.harness_eval.headless.run_headless`.
        """
        tree_dir = Path(tree_dir)
        if not tree_dir.is_dir():
            raise ValueError(f"harness tree not found: {tree_dir}")
        mode = getattr(spec, "install_mode", "workspace_agents_md")
        if mode not in ("workspace_agents_md", "workspace_config", "home_dir_jinja", "codex_root"):
            raise ValueError(f"unsupported install_mode for Atlas: {mode!r}")
        staged = Path(tempfile.mkdtemp(prefix="atlas-harness-"))
        try:
            shutil.copytree(tree_dir, staged, dirs_exist_ok=True)
            for key, body in sorted(ctx.items()):
                if key.startswith("file:"):
                    rel = key[len("file:") :]
                    target = staged / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(body, encoding="utf-8")
                elif (
                    "{{"
                    in (staged / "systemprompt.md").read_text(encoding="utf-8", errors="ignore")
                    if (staged / "systemprompt.md").exists()
                    else False
                ):
                    # Cheap {{token}} substitution in the prompt only.
                    p = staged / "systemprompt.md"
                    p.write_text(
                        p.read_text(encoding="utf-8").replace(f"{{{{{key}}}}}", body),
                        encoding="utf-8",
                    )
            return Install(staging_dir=staged, fingerprint=_fingerprint(staged))
        except Exception:
            shutil.rmtree(staged, ignore_errors=True)
            raise

    def preflight(self) -> None:
        """Check the harness is importable and the prompt loads."""
        load_system_prompt()
        # Importing default_registry proves tools still discover; no model call.
        from atlas.agent.contracts import default_registry

        if not default_registry().definitions:
            raise RuntimeError("default tool registry is empty")

    def to_harness2_install(self, install: Install) -> object:
        """Convert to harness2 ``Install`` when that package is available."""
        from harness2.substrate.base import Install as H2Install  # type: ignore[import-not-found]

        return H2Install(
            staging_dir=install.staging_dir,
            fingerprint=install.fingerprint,
            extras=dict(install.extras),
        )
