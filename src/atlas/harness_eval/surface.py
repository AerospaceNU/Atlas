"""Atlas harness tree: the file surface harness2 may improve.

This mirrors harness2's ``SurfaceSpec`` (``harness2/core/tree.py``) so the
same Propose-Probe-Compose recursion can operate on Atlas without importing
harness2 at test time. When harness2 is installed you can hand
:data:`ATLAS_SURFACE` to its validation by constructing its ``SurfaceSpec``
with the same fields (see :func:`to_harness2_surface`).

Paths use harness2's portable names so trees are diffable across substrates.
:mod:`atlas.harness_eval.substrate` installs them into a rollout:

- ``systemprompt.md`` -> system prompt base (c1)
- ``guardrails.md`` -> appended to the system prompt (c2)
- ``LongTermMEMORY.md`` -> appended to the system prompt (c3)
- ``tool_descriptions/tool_guidance.md`` -> appended to the system prompt (c4)
- ``sub_agents/<name>/AGENT.md`` -> named role for ``spawn_agent`` (c5)
- ``skills/<name>/SKILL.md`` -> workspace skill, loaded by ``use_skill`` (c6)
- ``scripts/<name>.py`` -> helper run through ``bash`` (c7)

Atlas has no hook or plugin loader, so c8 is off the surface: a component
that cannot change a rollout is not offered to the improver.

Nothing here executes code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_NAME = r"[a-z0-9][a-z0-9_\-]*"


@dataclass(frozen=True)
class SurfaceSpec:
    """The editable surface of one agent framework."""

    name: str
    always_on: tuple[str, ...]
    always_on_max: int
    always_on_cap_reason: str
    skill_dir: str
    skill_name_rule: str
    trigger_hint: str
    autotrigger_file: str
    trigger_line: str
    subagent_requires_mode: bool
    plugin_dir: str | None
    hooks_file: str | None
    stubs: tuple[tuple[str, str], ...] = ()


ATLAS_SURFACE = SurfaceSpec(
    name="atlas",
    always_on=(
        "systemprompt.md",
        "guardrails.md",
        "LongTermMEMORY.md",
        "tool_descriptions/tool_guidance.md",
    ),
    always_on_max=24_000,
    always_on_cap_reason="every token here competes with the agent's own output budget",
    skill_dir="skills",
    skill_name_rule="must_match",
    trigger_hint=(
        "Fix it by adding a trigger line naming `{name}` to systemprompt.md or "
        'guardrails.md, e.g. "When <the situation this skill handles>, use the '
        '`{name}` skill."'
    ),
    autotrigger_file="guardrails.md",
    trigger_line="- When {desc}, use the `{name}` skill.\n",
    subagent_requires_mode=False,
    plugin_dir=None,
    hooks_file=None,
    stubs=(
        (
            "scripts/README.md",
            "<!-- c7: deterministic helpers. Add one by emitting scripts/<name>.py and name "
            "it in a skill or the always-on text as `python3 scripts/<name>.py`; the runtime "
            "installs it in the workspace and rewrites that path. It runs under the "
            "repository's own interpreter, which can be as old as Python 3.6: standard "
            "library only. -->\n",
        ),
        (
            "sub_agents/README.md",
            "<!-- c5: delegated roles. Add one as sub_agents/<name>/AGENT.md with a "
            "`description:` in its frontmatter. The agent delegates with spawn_agent, "
            "starting the task with the line `agent: <name>`. -->\n",
        ),
        (
            "skills/README.md",
            "<!-- c6: reusable workflows. Add one as skills/<name>/SKILL.md with "
            "frontmatter `name: <name>` and `description:`. Loaded on demand by use_skill. -->\n",
        ),
    ),
)

Tree = dict[str, str]


def _rx(surface: SurfaceSpec) -> dict[str, re.Pattern[str]]:
    sd = re.escape(surface.skill_dir)
    out = {
        "skill": re.compile(rf"{sd}/({_NAME})/SKILL\.md"),
        "skill_script": re.compile(rf"{sd}/{_NAME}/scripts/[A-Za-z0-9_.\-]+"),
        "subagent": re.compile(rf"sub_agents/({_NAME})/AGENT\.md"),
        "script": re.compile(r"scripts/[a-z0-9_][a-z0-9_]*\.py"),
    }
    if surface.plugin_dir is not None:
        out["plugin"] = re.compile(rf"{re.escape(surface.plugin_dir)}/({_NAME})\.ts")
    return out


def normalize_rel(rel: str) -> str:
    """Normalise a relative path, rejecting escapes."""
    r = str(rel).strip().replace("\\", "/")
    if r.startswith("/") or ".." in r.split("/"):
        raise ValueError(f"path escapes the harness: {rel!r}")
    while r.startswith("./"):
        r = r[2:]
    return r.strip("/")


def is_allowed(surface: SurfaceSpec, rel: str) -> bool:
    """Check whether a path is on the editable surface."""
    try:
        rel = normalize_rel(rel)
    except ValueError:
        return False
    if rel in surface.always_on:
        return True
    if surface.hooks_file is not None and rel == surface.hooks_file:
        return True
    r = _rx(surface)
    if (
        bool(r["skill"].fullmatch(rel))
        or bool(r["skill_script"].fullmatch(rel))
        or bool(r["subagent"].fullmatch(rel))
        or bool(r["script"].fullmatch(rel))
    ):
        return True
    return "plugin" in r and bool(r["plugin"].fullmatch(rel))


def read_tree(surface: SurfaceSpec, d: Path | str, *, rejected: list[str] | None = None) -> Tree:
    """Read every on-surface file under ``d`` into ``rel -> content``."""
    d = Path(d)
    out: Tree = {}
    stubs = frozenset(rel for rel, _ in surface.stubs)
    if not d.is_dir():
        return out
    for p in sorted(d.rglob("*")):
        if not p.is_file() or "__pycache__" in p.parts:
            continue
        rel = str(p.relative_to(d))
        if rel in stubs:
            continue
        if not is_allowed(surface, rel):
            if rejected is not None:
                rejected.append(rel)
            continue
        out[rel] = p.read_text(encoding="utf-8", errors="replace")
    return out


def write_tree(surface: SurfaceSpec, tree: Tree, dest: Path | str, *, stubs: bool = True) -> Path:
    """Write ``tree`` under ``dest``, adding README stubs when asked."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for rel, content in tree.items():
        if not is_allowed(surface, rel):
            raise ValueError(f"path is not on the {surface.name} surface: {rel}")
        target = dest / normalize_rel(rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    if stubs:
        for rel, content in surface.stubs:
            target = dest / rel
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
    return dest


def ff_diff(surface: SurfaceSpec, before: Tree, after: Tree) -> dict[str, list[str]]:
    """Summarise added/removed/modified paths between two trees."""
    added = [k for k in after if k not in before]
    removed = [k for k in before if k not in after]
    modified = [k for k in before if k in after and before[k] != after[k]]
    return {"added": sorted(added), "removed": sorted(removed), "modified": sorted(modified)}


# Alias kept for readability at call sites.
diff_summary = ff_diff


def base_tree() -> Tree:
    """Return the minimal base harness seeded from package prompt + README stubs."""
    from atlas.agent.system_prompt import load_system_prompt

    base: Tree = {name: "" for name in ATLAS_SURFACE.always_on}
    base["systemprompt.md"] = load_system_prompt() + "\n"
    base["tool_descriptions/tool_guidance.md"] = (
        "Use grep and glob to locate code, read_file (offset and limit) or read_files to "
        "inspect it, edit_file or patch_file to change it, run_tests to verify, and "
        "bash for other shell commands.\n"
    )
    return base


def validate(surface: SurfaceSpec, tree: Tree, *, allow_subagents: bool = True) -> None:
    """Validate a proposed tree; raise ``ValueError`` with all problems."""
    errs: list[str] = []
    for rel in tree:
        if not is_allowed(surface, rel):
            errs.append(f"off surface: {rel}")
    total = sum(len(v) for k, v in tree.items() if k in surface.always_on)
    if total > surface.always_on_max:
        errs.append(
            f"always-on text {total} chars exceeds {surface.always_on_max} ({surface.always_on_cap_reason})"
        )
    for rel, body in tree.items():
        if (
            re.match(rf"{re.escape(surface.skill_dir)}/({_NAME})/SKILL\.md", rel)
            and "name:" not in body[:800]
        ):
            errs.append(f"{rel}: missing frontmatter name")
    if not allow_subagents:
        for rel in tree:
            if rel.startswith("sub_agents/"):
                errs.append(f"subagents disabled but found {rel}")
    if errs:
        raise ValueError("; ".join(errs))


def to_harness2_surface() -> object:
    """Build a harness2-native ``SurfaceSpec`` from :data:`ATLAS_SURFACE`.

    Requires harness2 installed (``uv pip install -e .`` from its tree). The
    conversion is a plain dataclass copy so the recursion engine can drive
    Atlas trees without a second schema.
    """
    from harness2.core.tree import SurfaceSpec as H2Spec  # type: ignore[import-not-found]

    s = ATLAS_SURFACE
    return H2Spec(
        name=s.name,
        always_on=s.always_on,
        always_on_max=s.always_on_max,
        always_on_cap_reason=s.always_on_cap_reason,
        skill_dir=s.skill_dir,
        skill_name_rule=s.skill_name_rule,
        trigger_hint=s.trigger_hint,
        autotrigger_file=s.autotrigger_file,
        trigger_line=s.trigger_line,
        subagent_requires_mode=s.subagent_requires_mode,
        plugin_dir=s.plugin_dir,
        hooks_file=s.hooks_file,
        stubs=s.stubs,
    )
