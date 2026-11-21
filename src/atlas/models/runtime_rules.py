"""Rules runtime: classify pixels with band-math rules from a model manifest.

Invariant: rule text is parsed and interpreted by `atlas.geo.expr`; nothing is handed to the
Python interpreter. Names a rule may read are the input bands, the manifest parameters and
earlier `define` entries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from atlas.geo._lazy import check_cancel
from atlas.geo.errors import ExprError, GeoError, fix
from atlas.geo.expr import eval_expr, expr_names
from atlas.models.runtime_api import RuntimeContext, RuntimeOutput

if TYPE_CHECKING:
    from atlas.models.manifest import ModelManifest

__all__ = ["RulesRuntime", "rule_problems"]

_ROLES = ("pre", "post")


def _allowed_names(manifest: ModelManifest, defined: set[str]) -> set[str]:
    bands = manifest.input.bands
    names = set(manifest.parameters) | defined
    if manifest.input.temporal.max > 1:
        names |= {f"{role}.{band}" for role in _ROLES for band in bands}
    else:
        names |= set(bands)
    return names


def rule_problems(manifest: ModelManifest) -> list[str]:
    """Return why the rules of `manifest` cannot run (empty when they can).

    Checks that every expression parses and reads only known names; used at registration so
    a broken manifest is refused instead of failing on first use.
    """
    rules = manifest.runtime.rules
    if rules is None:
        return ["runtime.rules is missing"]
    if manifest.input.mode == "series":
        return ["rules runtime supports modes 'single' and 'pair' only"]
    problems: list[str] = []
    defined: set[str] = set()
    steps = [(f"define.{k}", v) for k, v in rules.define.items()]
    steps += [(f"class {c.name!r}", c.when) for c in rules.classes]
    for index, (label, text) in enumerate(steps):
        try:
            names = expr_names(text)
        except ExprError as exc:
            problems.append(f"{label}: {str(exc).split(' Fix:')[0]}")
            continue
        unknown = sorted(names - _allowed_names(manifest, defined))
        if unknown:
            problems.append(f"{label} reads unknown names {unknown}")
        if index < len(rules.define):
            defined.add(label.removeprefix("define."))
    return problems


def _environment(ctx: RuntimeContext) -> dict[str, Any]:
    manifest = ctx.manifest
    env: dict[str, Any] = {k: float(v) for k, v in ctx.params.items() if not isinstance(v, str)}
    for role, names in ctx.layers_by_role.items():
        if manifest.input.temporal.max > 1:
            env[f"pre.{role}"] = ctx.stack.band(names[0])
            env[f"post.{role}"] = ctx.stack.band(names[-1])
        else:
            env[role] = ctx.stack.band(names[0])
    return env


def _broadcast(value: NDArray[Any], shape: tuple[int, int], label: str) -> NDArray[np.bool_]:
    if value.dtype != np.bool_:
        raise GeoError(fix(f"rule {label} is not a comparison.", "write 'when' as a comparison"))
    return np.broadcast_to(value, shape)


class RulesRuntime:
    """Pure-numpy classifier driven by `[runtime.rules]`."""

    def run(self, ctx: RuntimeContext) -> RuntimeOutput:
        """Evaluate defines, then classes in order; the first matching class wins.

        Pixels outside `ctx.valid` and valid pixels no class matches stay NaN; the latter are
        reported in `extras["unmatched_px"]`.
        """
        rules = ctx.manifest.runtime.rules
        if rules is None:
            raise GeoError(fix("manifest has no [runtime.rules].", "use a rules manifest"))
        shape = ctx.stack.grid.shape
        env = _environment(ctx)
        for key, text in rules.define.items():
            check_cancel(ctx.deadline)
            env[key] = eval_expr(text, env)
        result = np.full(shape, np.nan, dtype=np.float32)
        todo = np.array(ctx.valid, dtype=bool)
        planes: dict[str, NDArray[np.bool_]] = {}
        for cls in rules.classes:
            check_cancel(ctx.deadline)
            hit = _broadcast(eval_expr(cls.when, env), shape, repr(cls.name)) & todo
            result[hit] = cls.code
            todo &= ~hit
        names = {c.code: c.name for c in rules.classes}
        output = ctx.manifest.output
        for semantic, members in output.qa_planes.items():
            codes = [c for c, n in names.items() if n in members]
            planes[semantic] = np.isin(result, codes)
        return RuntimeOutput(
            layers={output.layer: result},
            qa_planes=planes,
            extras={"unmatched_px": int(todo.sum())},
        )
