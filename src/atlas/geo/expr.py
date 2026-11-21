"""Band-math expression grammar: a whitelisted AST walked node by node over numpy arrays.

Invariant: expression text is parsed and interpreted, never handed to the Python
interpreter for execution. Only the node kinds handled by _Walk and _Interp are
evaluated; every other node, name or call is an ExprError.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from atlas.geo.errors import ExprError, fix

__all__ = ["FUNCTIONS", "ExprError", "eval_expr", "expr_names", "mangle"]

FUNCTIONS = ("abs", "sqrt", "log", "log10", "exp", "min", "max", "clip", "where", "isnan")
_ARITY = {
    "abs": 1,
    "sqrt": 1,
    "log": 1,
    "log10": 1,
    "exp": 1,
    "isnan": 1,
    "min": 2,
    "max": 2,
    "clip": 3,
    "where": 3,
}
_DATE_PATTERN = r"([A-Za-z_]\w*)@(\d{8})"
_DATE_REPLACEMENT = r"\1__at__\2"
_MAX_DEPTH = 20
_MAX_EXPONENT = 64
_MAX_MESSAGE = 300
_MAX_LISTED_NAMES = 24
_LIST_CHARS = 150
_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.BitAnd, ast.BitOr)
_UNARYOPS = (ast.USub, ast.UAdd, ast.Invert)
_CMPOPS = (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq)
_PASSIVE = (ast.operator, ast.unaryop, ast.cmpop, ast.expr_context)
_HINT = "use & | ~ and where(cond,a,b)"

Value = Any


def mangle(name: str) -> str:
    """Make a dated layer name a legal identifier: red@20240601 -> red__at__20240601."""
    return re.sub(_DATE_PATTERN, _DATE_REPLACEMENT, name)


def _unmangle(name: str) -> str:
    return re.sub(r"__at__(\d{8})", r"@\1", name)


def _error(message: str) -> ExprError:
    text = message if len(message) <= _MAX_MESSAGE else message[: _MAX_MESSAGE - 3] + "..."
    return ExprError(text)


def _clip_text(text: str, limit: int = 40) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _parse(expression: str, max_chars: int) -> ast.Expression:
    if not isinstance(expression, str):
        raise _error(fix("expression must be a string.", "pass expression='(nir-red)/(nir+red)'"))
    if len(expression) > max_chars:
        raise _error(
            fix(f"expression has {len(expression)} chars (limit {max_chars}).", "shorten it")
        )
    text = re.sub(_DATE_PATTERN, _DATE_REPLACEMENT, expression.strip())
    try:
        return ast.parse(text, mode="eval")
    except SyntaxError as exc:
        reason = _clip_text(str(exc.msg), 80)
        raise _error(fix(f"syntax error: {reason}.", "fix the expression")) from exc
    except (RecursionError, MemoryError):
        raise _error(fix("expression is nested too deeply.", "simplify it")) from None
    except ValueError as exc:
        raise _error(fix(f"invalid text: {_clip_text(str(exc), 80)}.", "remove it")) from exc


def _allowed() -> str:
    return "allowed functions: " + ", ".join(FUNCTIONS)


def _kind_error(node: ast.AST) -> ExprError:
    kind = type(node).__name__
    return _error(fix(f"{kind} is not allowed in band math.", f"{_HINT} where relevant"))


class _Walk:
    """One pass over the tree: whitelist, node budget, depth budget."""

    def __init__(self, max_nodes: int) -> None:
        self.max_nodes = max_nodes
        self.count = 0

    def check(self, node: ast.AST, depth: int) -> None:
        if isinstance(node, _PASSIVE):
            return
        self.count += 1
        if self.count > self.max_nodes:
            raise _error(fix(f"expression has over {self.max_nodes} nodes.", "simplify it"))
        if depth > _MAX_DEPTH:
            raise _error(fix(f"expression nests deeper than {_MAX_DEPTH}.", "simplify it"))
        if isinstance(node, ast.Expression):
            self.check(node.body, depth + 1)
        elif isinstance(node, ast.Constant):
            self._constant(node)
        elif isinstance(node, ast.Name):
            return
        elif isinstance(node, ast.Attribute):
            self._attribute(node)
        elif isinstance(node, ast.BinOp):
            self._binop(node, depth)
        elif isinstance(node, ast.UnaryOp):
            self._unary(node, depth)
        elif isinstance(node, ast.Compare):
            self._compare(node, depth)
        elif isinstance(node, ast.Call):
            self._call(node, depth)
        else:
            raise _kind_error(node)

    def _constant(self, node: ast.Constant) -> None:
        value = node.value
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise _error(
                fix(f"constant {_clip_text(repr(value))} is not a number.", "use int or float")
            )

    def _attribute(self, node: ast.Attribute) -> None:
        if not isinstance(node.value, ast.Name) or node.attr.startswith("_"):
            raise _error(fix("attribute access is only alias.layer.", "use names like a.red"))

    def _binop(self, node: ast.BinOp, depth: int) -> None:
        if not isinstance(node.op, _BINOPS):
            raise _error(fix(f"operator {type(node.op).__name__} is not allowed.", _HINT))
        self.check(node.left, depth + 1)
        self.check(node.right, depth + 1)

    def _unary(self, node: ast.UnaryOp, depth: int) -> None:
        if not isinstance(node.op, _UNARYOPS):
            raise _error(fix("operator not allowed.", "use - + ~ (for not, use ~)"))
        self.check(node.operand, depth + 1)

    def _compare(self, node: ast.Compare, depth: int) -> None:
        if not all(isinstance(op, _CMPOPS) for op in node.ops):
            raise _error(fix("comparison not allowed.", "use < <= > >= == !="))
        self.check(node.left, depth + 1)
        for item in node.comparators:
            self.check(item, depth + 1)

    def _call(self, node: ast.Call, depth: int) -> None:
        if not isinstance(node.func, ast.Name):
            raise _error(fix("only plain function calls are allowed.", _allowed()))
        name = node.func.id
        if name not in FUNCTIONS:
            raise _error(
                f"unknown function '{_clip_text(name)}'; allowed: {', '.join(FUNCTIONS)}. "
                "Fix: use one of them"
            )
        if node.keywords:
            raise _error(fix(f"{name}() takes no keyword arguments.", f"call {name} positionally"))
        if len(node.args) != _ARITY[name]:
            raise _error(
                fix(f"{name}() takes {_ARITY[name]} argument(s), got {len(node.args)}.", "fix it")
            )
        for arg in node.args:
            self.check(arg, depth + 1)


def _collect(node: ast.AST, out: set[str]) -> None:
    if isinstance(node, ast.Name):
        out.add(node.id)
    elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        out.add(f"{node.value.id}.{node.attr}")
    elif isinstance(node, ast.Call):
        for arg in node.args:
            _collect(arg, out)
    else:
        for child in ast.iter_child_nodes(node):
            _collect(child, out)


def expr_names(expression: str) -> set[str]:
    """Return the free names an expression reads, un-mangled (alias.layer joined).

    Args:
        expression: Band-math text.

    Returns:
        Names such as nir, red@20240601, a.red; function names are excluded.

    Raises:
        ExprError: The text is not a valid expression.
    """
    tree = _parse(expression, 10_000)
    _Walk(10_000).check(tree, 0)
    found: set[str] = set()
    _collect(tree, found)
    return {_unmangle(name) for name in found}


def _is_bool(value: Value) -> bool:
    return bool(value.dtype == np.bool_)


def _is_array(value: Value) -> bool:
    return bool(value.ndim > 0)


def _list_names(names: list[str]) -> str:
    """List up to 24 names within a character budget, then '+N more'."""
    shown: list[str] = []
    used = 0
    for name in names[:_MAX_LISTED_NAMES]:
        piece = _clip_text(name, 30)
        if used + len(piece) > _LIST_CHARS:
            break
        shown.append(piece)
        used += len(piece) + 2
    more = len(names) - len(shown)
    return (", ".join(shown) or "none") + (f", +{more} more" if more > 0 else "")


class _Interp:
    """Evaluate a validated tree over numpy values.

    Scalars are np.float64; arrays are float32 (or bool for comparisons). Every
    non-finite result of division, log, log10 and sqrt becomes NaN; no Python exception
    escapes for arithmetic.
    """

    def __init__(self, env: Mapping[str, Any], max_elements: int) -> None:
        self.raw = {mangle(str(key)): (str(key), val) for key, val in env.items()}
        self.max_elements = max_elements
        self.bound: dict[str, Value] = {}

    # -- names ----------------------------------------------------------------------

    def _bind(self, key: str) -> Value:
        if key in self.bound:
            return self.bound[key]
        if key not in self.raw:
            raise self._unknown(key)
        arr = np.asarray(self.raw[key][1])
        if arr.ndim == 0:
            try:
                out: Value = np.float64(float(arr))
            except (TypeError, ValueError):
                shown = _clip_text(key)
                raise _error(fix(f"name '{shown}' is not numeric.", "pass numbers")) from None
        else:
            self._guard_size(arr.size)
            out = arr.astype(np.float32, copy=False)
        self.bound[key] = out
        return out

    def _unknown(self, key: str) -> ExprError:
        names = sorted(orig for orig, _ in self.raw.values())
        return _error(
            f"unknown name '{_clip_text(_unmangle(key))}'; available: {_list_names(names)}. "
            "Fix: pick names from inspect_stack"
        )

    def _guard_size(self, size: int) -> None:
        if size > self.max_elements:
            raise _error(
                fix(
                    f"array of {size:,} elements exceeds {self.max_elements:,}.",
                    "use a smaller window",
                )
            )

    # -- dispatch -------------------------------------------------------------------

    def run(self, node: ast.AST) -> Value:
        if isinstance(node, ast.Expression):
            return self.run(node.body)
        if isinstance(node, ast.Constant):
            return self._constant(node)
        if isinstance(node, ast.Name):
            return self._bind(node.id)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            return self._bind(f"{node.value.id}.{node.attr}")
        if isinstance(node, ast.BinOp):
            return self._binop(node)
        if isinstance(node, ast.UnaryOp):
            return self._unary(node)
        if isinstance(node, ast.Compare):
            return self._compare(node)
        if isinstance(node, ast.Call):
            return self._call(node)
        raise _kind_error(node)

    def _constant(self, node: ast.Constant) -> Value:
        value = node.value
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise _kind_error(node)
        try:
            return np.float64(float(value))
        except OverflowError:
            raise _error(fix("number too large.", "use a smaller constant")) from None

    # -- helpers --------------------------------------------------------------------

    def _shape(self, values: tuple[Value, ...] | list[Value]) -> tuple[int, ...] | None:
        shape: tuple[int, ...] | None = None
        for v in values:
            if not _is_array(v):
                continue
            if shape is None:
                shape = tuple(v.shape)
            elif tuple(v.shape) != shape:
                raise _error(
                    fix(
                        f"shape mismatch {shape} vs {tuple(v.shape)}.",
                        "use layers from the same stack",
                    )
                )
        return shape

    def _numeric(self, *values: Value) -> list[Value]:
        """Cast bool to float; scalars take the dtype that keeps arrays float32."""
        arrays = any(_is_array(v) for v in values)
        self._shape(values)
        out: list[Value] = []
        for v in values:
            if _is_array(v):
                out.append(v.astype(np.float32, copy=False))
            else:
                out.append(np.float32(v) if arrays else np.float64(v))
        return out

    def _finish(self, value: Value, *, finite: bool = False) -> Value:
        out = np.asarray(value)
        if finite and out.dtype.kind == "f":
            out = np.where(np.isfinite(out), out, np.nan).astype(out.dtype, copy=False)
        if out.ndim == 0:
            return np.bool_(out) if out.dtype == np.bool_ else np.float64(out)
        self._guard_size(out.size)
        return out if out.dtype == np.bool_ else out.astype(np.float32, copy=False)

    def _require_bool(self, *values: Value) -> None:
        if not all(_is_bool(v) for v in values):
            raise _error(
                fix("& | ~ need boolean operands.", "use comparisons, e.g. (red>0.1)&(nir>0.2)")
            )

    # -- operators ------------------------------------------------------------------

    def _binop(self, node: ast.BinOp) -> Value:
        left, right = self.run(node.left), self.run(node.right)
        op = node.op
        if isinstance(op, ast.BitAnd | ast.BitOr):
            self._require_bool(left, right)
            self._shape((left, right))
            fn = np.logical_and if isinstance(op, ast.BitAnd) else np.logical_or
            return self._finish(fn(left, right))
        a, b = self._numeric(left, right)
        with np.errstate(all="ignore"):
            if isinstance(op, ast.Add):
                return self._finish(a + b)
            if isinstance(op, ast.Sub):
                return self._finish(a - b)
            if isinstance(op, ast.Mult):
                return self._finish(a * b)
            if isinstance(op, ast.Div):
                return self._finish(a / b, finite=True)
            return self._finish(self._power(a, b))

    def _power(self, base: Value, exponent: Value) -> Value:
        peak = float(np.max(np.abs(exponent))) if exponent.size else 0.0
        if not peak <= _MAX_EXPONENT:
            raise _error(
                fix(f"exponent too large (limit {_MAX_EXPONENT}).", "use a smaller exponent")
            )
        return np.power(base, exponent)

    def _unary(self, node: ast.UnaryOp) -> Value:
        value = self.run(node.operand)
        if isinstance(node.op, ast.Invert):
            self._require_bool(value)
            return self._finish(np.logical_not(value))
        (num,) = self._numeric(value)
        return self._finish(-num if isinstance(node.op, ast.USub) else num)

    def _compare(self, node: ast.Compare) -> Value:
        left = self.run(node.left)
        result: Value = None
        for op, comparator in zip(node.ops, node.comparators, strict=True):
            right = self.run(comparator)
            part = self._compare_pair(op, left, right)
            result = part if result is None else self._finish(np.logical_and(result, part))
            left = right
        return result

    def _compare_pair(self, op: ast.cmpop, left: Value, right: Value) -> Value:
        a, b = self._numeric(left, right)
        shape = self._shape((a, b))
        valid = ~(np.isnan(a) | np.isnan(b))
        table: dict[type, Any] = {
            ast.Lt: np.less,
            ast.LtE: np.less_equal,
            ast.Gt: np.greater,
            ast.GtE: np.greater_equal,
            ast.Eq: np.equal,
            ast.NotEq: np.not_equal,
        }
        out = np.logical_and(table[type(op)](a, b), valid)
        if shape is not None and out.shape != shape:
            out = np.broadcast_to(out, shape).copy()
        return self._finish(out)

    # -- calls ----------------------------------------------------------------------

    def _call(self, node: ast.Call) -> Value:
        func = node.func
        if not isinstance(func, ast.Name) or func.id not in FUNCTIONS:
            raise _error(fix("unknown function.", _allowed()))
        args = [self.run(a) for a in node.args]
        name = func.id
        if name == "where":
            return self._where(*args)
        nums = self._numeric(*args)
        with np.errstate(all="ignore"):
            if name == "abs":
                return self._finish(np.abs(nums[0]))
            if name == "sqrt":
                return self._finish(np.sqrt(nums[0]), finite=True)
            if name == "log":
                return self._finish(np.log(nums[0]), finite=True)
            if name == "log10":
                return self._finish(np.log10(nums[0]), finite=True)
            if name == "exp":
                return self._finish(np.exp(nums[0]))
            if name == "isnan":
                return self._finish(np.isnan(nums[0]))
            if name == "min":
                return self._finish(np.minimum(nums[0], nums[1]))
            if name == "max":
                return self._finish(np.maximum(nums[0], nums[1]))
            return self._finish(np.minimum(np.maximum(nums[0], nums[1]), nums[2]))

    def _where(self, cond: Value, a: Value, b: Value) -> Value:
        if not _is_bool(cond):
            raise _error(fix("where() needs a boolean condition.", "use where(red>0.1, a, b)"))
        self._shape((cond, a, b))
        x, y = self._numeric(a, b)
        return self._finish(np.where(cond, x, y))


def eval_expr(
    expression: str,
    env: Mapping[str, NDArray[np.generic] | float],
    *,
    max_chars: int = 500,
    max_nodes: int = 200,
    max_elements: int = 50_000_000,
) -> NDArray[np.generic]:
    """Evaluate a band-math expression over named arrays without running Python code.

    Allowed: numbers, names (nir, a.red, red@20240601), + - * / **, comparisons, & | ~ on
    booleans, and the functions in FUNCTIONS.

    Args:
        expression: Band-math text, at most max_chars characters.
        env: Name to array (cast to float32) or number. Dated names use their original
            name@YYYYMMDD key.
        max_chars: Longest accepted expression.
        max_nodes: Most syntax nodes accepted.
        max_elements: Largest array allowed anywhere in the evaluation.

    Returns:
        float32 array (bool for comparisons); scalar-only input gives a 0-d array.

    Raises:
        ExprError: Syntax, unsupported construct, unknown name, shape mismatch or size guard.
    """
    tree = _parse(expression, max_chars)
    _Walk(max_nodes).check(tree, 0)
    try:
        value = _Interp(env, max_elements).run(tree)
    except (RecursionError, MemoryError):
        raise _error(fix("expression is too large to evaluate.", "simplify it")) from None
    out = np.asarray(value)
    if out.ndim == 0 and out.dtype != np.bool_:
        out = out.astype(np.float32)
    return out
