"""Restricted, safe expression language used for derived columns and row-level constraints.

Nothing here calls ``eval`` or ``exec``. User text is parsed with :mod:`ast` and only a small
whitelist of node types is interpreted by our own vectorised evaluator.

Supported syntax
----------------
* Numbers, strings (``'Pass'``), ``True``/``False``/``None``, list/tuple literals.
* Arithmetic: ``+ - * / // % **`` and unary ``-``.
* Comparisons: ``== != < <= > >=`` (chained comparisons such as ``0 <= x <= 100`` work),
  ``in`` / ``not in`` against a list literal.
* Logic: ``and``, ``or``, ``not``.
* Conditional: ``a if cond else b``.
* Functions: abs, round, min, max, sqrt, log, exp, floor, ceil, clip, where, isnull, notnull,
  year, month, days_between, date.
* Column names (must be valid identifiers such as ``unit_price``).
"""
from __future__ import annotations

import ast
import operator
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .exceptions import ExpressionError

MAX_EXPR_LENGTH = 500
MAX_NODES = 200

_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_CMP_OPS = {
    ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le,
    ast.Gt: operator.gt, ast.GtE: operator.ge,
}


def _where(cond, a, b):
    return np.where(np.asarray(cond, dtype=bool), a, b)


def _round(x, digits=0):
    return np.round(x, int(digits))


def _clip(x, lo, hi):
    return np.clip(x, lo, hi)


def _year(x):
    return pd.Series(pd.to_datetime(x)).dt.year


def _month(x):
    return pd.Series(pd.to_datetime(x)).dt.month


def _days_between(a, b):
    return (pd.to_datetime(b) - pd.to_datetime(a)).dt.days


FUNCTIONS = {
    "abs": np.abs, "round": _round, "min": np.minimum, "max": np.maximum, "sqrt": np.sqrt,
    "log": np.log, "exp": np.exp, "floor": np.floor, "ceil": np.ceil, "clip": _clip,
    "where": _where, "isnull": pd.isna, "notnull": pd.notna, "year": _year, "month": _month,
    "days_between": _days_between, "date": pd.Timestamp,
}


def parse_expression(expr: str) -> ast.Expression:
    """Parse and structurally validate ``expr``. Raises :class:`ExpressionError`."""
    if not isinstance(expr, str) or not expr.strip():
        raise ExpressionError("Expression must be a non-empty string.")
    if len(expr) > MAX_EXPR_LENGTH:
        raise ExpressionError(f"Expression is longer than {MAX_EXPR_LENGTH} characters.")
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"Invalid expression syntax in {expr!r}: {exc.msg}") from exc
    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_NODES:
        raise ExpressionError("Expression is too complex.")
    allowed = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp, ast.Call,
        ast.Name, ast.Constant, ast.List, ast.Tuple, ast.Load, ast.And, ast.Or, ast.Not,
        ast.USub, ast.UAdd, *_BIN_OPS, *_CMP_OPS, ast.In, ast.NotIn,
    )
    for node in nodes:
        if not isinstance(node, allowed):
            raise ExpressionError(
                f"Unsupported syntax '{type(node).__name__}' in expression {expr!r}. "
                "See dataseed.expressions for the supported syntax.")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
                raise ExpressionError(
                    f"Unknown or unsupported function in {expr!r}. Supported: {sorted(FUNCTIONS)}")
            if node.keywords:
                raise ExpressionError("Keyword arguments are not supported in expressions.")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float, str, bool, type(None))):
            raise ExpressionError(f"Unsupported constant {node.value!r}.")
    return tree


def referenced_columns(expr: str) -> list[str]:
    """Return column names referenced by ``expr`` (function names excluded), in order of appearance."""
    tree = parse_expression(expr)
    func_nodes = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and id(node) not in func_nodes and node.id not in names:
            names.append(node.id)
    return sorted(names, key=lambda n: expr.find(n))


def evaluate_expression(expr: str, data: Mapping[str, Any]) -> Any:
    """Evaluate ``expr`` against ``data`` (a DataFrame or mapping of column -> array/Series).

    Returns a Series/array (or a scalar for constant expressions).
    """
    tree = parse_expression(expr)

    def ev(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant):
            v = node.value
            if isinstance(v, bool) or v is None or isinstance(v, str):
                return v
            return np.float64(v)
        if isinstance(node, ast.Name):
            if node.id not in data:
                raise ExpressionError(f"Unknown column '{node.id}' in expression {expr!r}.")
            return data[node.id]
        if isinstance(node, (ast.List, ast.Tuple)):
            return [ev(e) for e in node.elts]
        if isinstance(node, ast.UnaryOp):
            v = ev(node.operand)
            if isinstance(node.op, ast.Not):
                return ~v if isinstance(v, (pd.Series, np.ndarray)) else (not v)
            return -v if isinstance(node.op, ast.USub) else +v
        if isinstance(node, ast.BinOp):
            return _BIN_OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.BoolOp):
            vals = [ev(v) for v in node.values]
            fn = np.logical_and if isinstance(node.op, ast.And) else np.logical_or
            out = vals[0]
            for v in vals[1:]:
                out = fn(out, v)
            return out
        if isinstance(node, ast.Compare):
            left = ev(node.left)
            result = None
            for op, comp in zip(node.ops, node.comparators):
                right = ev(comp)
                if isinstance(op, (ast.In, ast.NotIn)):
                    if not isinstance(right, list):
                        raise ExpressionError("'in' requires a list literal on the right-hand side.")
                    r = left.isin(right) if isinstance(left, pd.Series) else np.isin(left, right)
                    if isinstance(op, ast.NotIn):
                        r = ~r
                else:
                    r = _CMP_OPS[type(op)](left, right)
                result = r if result is None else np.logical_and(result, r)
                left = right
            return result
        if isinstance(node, ast.IfExp):
            return _where(ev(node.test), ev(node.body), ev(node.orelse))
        if isinstance(node, ast.Call):
            return FUNCTIONS[node.func.id](*[ev(a) for a in node.args])
        raise ExpressionError(f"Unsupported node {type(node).__name__}")

    try:
        with np.errstate(all="ignore"):
            return ev(tree)
    except ExpressionError:
        raise
    except (TypeError, ValueError, OverflowError, ZeroDivisionError, AttributeError) as exc:
        raise ExpressionError(f"Could not evaluate {expr!r}: {exc}") from exc
