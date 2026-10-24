"""Post-generation (or stand-alone) validation of a dataset against a schema.

Validation never modifies or drops rows: it only reports. Every check ends as ``passed``,
``failed`` or ``not_evaluated`` (e.g. a needed column is absent). Checks with severity
``error`` are hard constraints; ``warning`` checks (missing-rate drift, correlation drift)
are informative statistical checks.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from .exceptions import ExpressionError
from .expressions import evaluate_expression, referenced_columns
from .models import ValidationCheck, ValidationResult
from .schema import ColumnSpec, Schema
from .inference import BOOL_FALSE, BOOL_TRUE

MAX_EXAMPLES = 5


def _blank_to_na(s: pd.Series) -> pd.Series:
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        return s.map(lambda v: None if isinstance(v, str) and v.strip() == "" else v)
    return s


def coerce_series(series: pd.Series, spec: ColumnSpec) -> tuple[pd.Series, pd.Series]:
    """Convert ``series`` to the type of ``spec`` without modifying the input.

    Returns ``(coerced, invalid_mask)``; values that cannot be converted become missing in
    ``coerced`` and are flagged in ``invalid_mask``.
    """
    s = _blank_to_na(series)
    present = s.notna()
    t = spec.type
    if t in ("integer", "float"):
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            num = s.astype(float)
        else:
            num = pd.to_numeric(s.astype(object).where(present, None).map(
                lambda v: v.strip().replace(",", "") if isinstance(v, str) else v), errors="coerce").astype(float)
        invalid = present & num.isna()
        if t == "integer":
            invalid = invalid | (num.notna() & (num % 1 != 0))
        return num, invalid
    if t == "boolean":
        if pd.api.types.is_bool_dtype(s):
            return s.astype("boolean"), pd.Series(False, index=s.index)
        toks = s.astype(str).str.strip().str.lower()
        out = pd.Series(pd.NA, index=s.index, dtype="boolean")
        out[present & toks.isin(BOOL_TRUE)] = True
        out[present & toks.isin(BOOL_FALSE)] = False
        return out, present & out.isna()
    if t in ("date", "datetime"):
        if pd.api.types.is_datetime64_any_dtype(s):
            return s.astype("datetime64[ns]"), pd.Series(False, index=s.index)
        txt = s.astype(object).where(present, None)
        fmt = spec.effective_date_format()
        parsed = pd.to_datetime(txt, format=fmt, errors="coerce")
        if parsed[present].isna().any():
            alt = pd.to_datetime(txt, errors="coerce", format="mixed")
            parsed = parsed.fillna(alt)
        return parsed.astype("datetime64[ns]"), present & parsed.isna()
    # categorical / string / identifier
    out = s.map(lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA else str(v))
    return out, pd.Series(False, index=s.index)


def _examples(df: pd.DataFrame, mask: pd.Series, col: str | None = None, n: int = MAX_EXAMPLES) -> list:
    idx = np.flatnonzero(np.asarray(mask))[:n]
    out = []
    for i in idx:
        item: dict[str, Any] = {"row": int(i)}
        if col is not None:
            v = df[col].iloc[i]
            item["value"] = None if pd.isna(v) else (v.isoformat() if hasattr(v, "isoformat") else
                                                     v.item() if hasattr(v, "item") else v)
        out.append(item)
    return out


def _identifier_regex(spec: ColumnSpec) -> re.Pattern | None:
    prefix, suffix = spec.prefix or "", spec.suffix or ""
    if not prefix and not suffix and not spec.width:
        return None
    return re.compile(f"^{re.escape(prefix)}\\d+{re.escape(suffix)}$")


def _template_regex(pattern: str) -> re.Pattern:
    return re.compile("^" + "".join("\\d" if ch == "#" else "[A-Z]" if ch == "?" else re.escape(ch)
                                    for ch in pattern) + "$")


def validate_dataset(df: pd.DataFrame, schema: Schema, expected_rows: int | None = None,
                     max_examples: int = MAX_EXAMPLES) -> ValidationResult:
    """Validate ``df`` against ``schema`` and return every check outcome (no data is changed)."""
    res = ValidationResult(n_rows=len(df))
    add = res.checks.append
    n = len(df)

    if expected_rows is not None:
        ok = n == expected_rows
        add(ValidationCheck("row_count", "passed" if ok else "failed",
                            f"{n} rows (expected {expected_rows})" if not ok else f"exactly {n} rows", count=n))
    expected_cols = schema.column_names
    missing = [c for c in expected_cols if c not in df.columns]
    extra = [c for c in df.columns if c not in expected_cols]
    if missing or extra:
        add(ValidationCheck("columns", "failed", f"missing columns: {missing}; unexpected columns: {extra}",
                            count=len(missing) + len(extra)))
    elif list(df.columns) != expected_cols:
        add(ValidationCheck("column_order", "failed",
                            f"column order differs: expected {expected_cols}, found {list(df.columns)}"))
    else:
        add(ValidationCheck("columns", "passed", "column names and order match the schema"))

    coerced: dict[str, pd.Series] = {}
    nullmask: dict[str, pd.Series] = {}
    for spec in schema.columns:
        if spec.name not in df.columns:
            add(ValidationCheck("column_checks", "not_evaluated", "column is absent", column=spec.name))
            continue
        raw = df[spec.name]
        c, invalid = coerce_series(raw, spec)
        coerced[spec.name] = c
        nullmask[spec.name] = c.isna()
        _check_column(res, df, spec, raw, c, invalid, n, max_examples)

    for con in schema.constraints:
        if con["type"] == "unique_together":
            cols = con["columns"]
            if any(x not in coerced for x in cols):
                add(ValidationCheck("unique_together", "not_evaluated", f"columns {cols} missing"))
                continue
            sub = pd.DataFrame({x: coerced[x] for x in cols})
            dup = sub.duplicated(keep="first") & ~sub.isna().any(axis=1)
            add(ValidationCheck(f"unique_together({', '.join(cols)})", "failed" if dup.any() else "passed",
                                f"{int(dup.sum())} duplicate combination(s)" if dup.any() else "all combinations unique",
                                count=int(dup.sum()), examples=_examples(df, dup)[:max_examples]))
    frame = _expression_frame(coerced)
    for spec in schema.columns:
        if spec.is_derived and spec.name in coerced:
            _check_derived(res, df, spec, coerced, nullmask, frame, max_examples)
    for con in schema.constraints:
        if con["type"] == "expression":
            _check_expression(res, df, con, coerced, nullmask, frame, max_examples)
    for spec in schema.columns:
        if spec.missing_model and spec.name in coerced:
            _check_missing_model(res, spec, coerced, max_examples)
    dups = int(df.duplicated().sum()) if n else 0
    add(ValidationCheck("duplicate_rows", "passed", f"{dups} fully duplicated row(s) ({dups / n:.2%})" if n else "no rows",
                        severity="info", count=dups))
    _check_correlations(res, schema, coerced)
    return res


def _expression_frame(coerced: dict[str, pd.Series]) -> dict[str, Any]:
    frame: dict[str, Any] = {}
    for k, v in coerced.items():
        frame[k] = v.fillna(False).astype(bool) if str(v.dtype) == "boolean" else v
    return frame


def _check_column(res: ValidationResult, df, spec: ColumnSpec, raw, c: pd.Series, invalid: pd.Series,
                  n: int, max_ex: int) -> None:
    add, name = res.checks.append, spec.name
    inv = int(invalid.sum())
    add(ValidationCheck("type", "failed" if inv else "passed",
                        f"{inv} value(s) cannot be read as {spec.type}" if inv else f"all values readable as {spec.type}",
                        column=name, count=inv, examples=_examples(df, invalid, name)[:max_ex]))
    nulls = c.isna()
    if not spec.is_nullable:
        k = int(nulls.sum())
        add(ValidationCheck("not_null", "failed" if k else "passed",
                            f"{k} missing value(s) in a non-nullable column" if k else "no missing values",
                            column=name, count=k, examples=_examples(df, nulls)[:max_ex]))
    elif spec.missing_rate > 0 and not spec.missing_model and not spec.is_derived and n:
        rate = nulls.mean()
        tol = max(0.01, 4 * np.sqrt(spec.missing_rate * (1 - spec.missing_rate) / n))
        ok = abs(rate - spec.missing_rate) <= tol
        add(ValidationCheck("missing_rate", "passed" if ok else "failed",
                            f"observed {rate:.2%}, configured {spec.missing_rate:.2%}", column=name,
                            severity="warning", count=int(nulls.sum())))
    valid = c.notna() & ~invalid
    if spec.type in ("integer", "float", "date", "datetime") and (spec.min is not None or spec.max is not None):
        lo = pd.Timestamp(spec.min) if (spec.type in ("date", "datetime") and spec.min is not None) else spec.min
        hi = pd.Timestamp(spec.max) if (spec.type in ("date", "datetime") and spec.max is not None) else spec.max
        out = pd.Series(False, index=c.index)
        if lo is not None:
            out |= valid & (c < lo)
        if hi is not None:
            out |= valid & (c > hi)
        k = int(out.sum())
        add(ValidationCheck("bounds", "failed" if k else "passed",
                            f"{k} value(s) outside [{spec.min}, {spec.max}]" if k else f"all values within [{spec.min}, {spec.max}]",
                            column=name, count=k, examples=_examples(df, out, name)[:max_ex]))
    if spec.categories is not None and spec.type in ("categorical", "integer", "float"):
        if spec.type == "categorical":
            allowed = {str(x) for x in spec.categories}
            bad = valid & ~c.isin(allowed)
        else:
            bad = valid & ~c.isin([float(x) for x in spec.categories])
        k = int(bad.sum())
        add(ValidationCheck("allowed_values", "failed" if k else "passed",
                            f"{k} value(s) not in the allowed set" if k else "all values are allowed",
                            column=name, count=k, examples=_examples(df, bad, name)[:max_ex]))
    if spec.is_unique:
        dup = c.duplicated(keep="first") & c.notna()
        k = int(dup.sum())
        add(ValidationCheck("unique", "failed" if k else "passed",
                            f"{k} duplicate value(s) in a unique column" if k else "all values unique",
                            column=name, count=k, examples=_examples(df, dup, name)[:max_ex]))
    if spec.type == "identifier":
        rx = _identifier_regex(spec)
        if rx is not None:
            bad = c.notna() & ~c.astype(str).map(lambda v: bool(rx.match(v)))
            k = int(bad.sum())
            add(ValidationCheck("identifier_format", "failed" if k else "passed",
                                f"{k} identifier(s) do not match prefix/suffix format" if k else "identifier format matches",
                                column=name, count=k, examples=_examples(df, bad, name)[:max_ex]))
    if spec.type == "string" and spec.pattern:
        rx = _template_regex(spec.pattern)
        bad = c.notna() & ~c.astype(str).map(lambda v: bool(rx.match(v)))
        k = int(bad.sum())
        add(ValidationCheck("pattern", "failed" if k else "passed",
                            f"{k} value(s) do not match pattern {spec.pattern!r}" if k else "values match the pattern",
                            column=name, count=k, examples=_examples(df, bad, name)[:max_ex]))


def _inputs_null(expr: str, nullmask: dict[str, pd.Series], n: int) -> pd.Series:
    m = pd.Series(False, index=range(n))
    for r in referenced_columns(expr):
        if r in nullmask:
            m = m | nullmask[r].reset_index(drop=True)
    return m


def _check_derived(res, df, spec: ColumnSpec, coerced, nullmask, frame, max_ex) -> None:
    name = f"derived({spec.name})"
    try:
        refs = referenced_columns(spec.expression)
        absent = [r for r in refs if r not in coerced]
        if absent:
            res.checks.append(ValidationCheck(name, "not_evaluated", f"needs missing column(s) {absent}", column=spec.name))
            return
        arr = np.asarray(evaluate_expression(spec.expression, frame))
        if arr.ndim == 0:
            arr = np.full(len(df), arr.item(), dtype=object)
        expected = pd.Series(arr)
    except ExpressionError as exc:
        res.checks.append(ValidationCheck(name, "not_evaluated", str(exc), column=spec.name))
        return
    actual = coerced[spec.name].reset_index(drop=True)
    skip = _inputs_null(spec.expression, nullmask, len(df)).to_numpy()
    if spec.type in ("integer", "float"):
        tol = 0.5001 if spec.type == "integer" else (10 ** -spec.decimals * 0.5001 if spec.decimals else 1e-6)
        e = pd.to_numeric(expected, errors="coerce").to_numpy(float)
        a = actual.to_numpy(float)
        ok = np.isclose(a, e, atol=tol + 1e-9, rtol=1e-6) | (np.isnan(a) & np.isnan(e))
    elif spec.type == "boolean":
        ok = (actual.astype("boolean").fillna(False).to_numpy(bool) == expected.astype(bool).to_numpy())
    elif spec.type in ("date", "datetime"):
        ok = (actual.to_numpy() == pd.to_datetime(expected).to_numpy())
    else:
        ok = (actual.astype(str).to_numpy() == expected.astype(str).to_numpy())
    bad = pd.Series(~ok & ~skip)
    k = int(bad.sum())
    res.checks.append(ValidationCheck(name, "failed" if k else "passed",
                                      f"{k} row(s) differ from '{spec.expression}'" if k else f"matches '{spec.expression}'",
                                      column=spec.name, count=k, examples=_examples(df, bad, spec.name)[:max_ex]))


def _check_expression(res, df, con: dict, coerced, nullmask, frame, max_ex) -> None:
    expr = con["expression"]
    name = f"constraint({expr})"
    try:
        refs = referenced_columns(expr)
        absent = [r for r in refs if r not in coerced]
        if absent:
            res.checks.append(ValidationCheck(name, "not_evaluated", f"needs missing column(s) {absent}"))
            return
        out = evaluate_expression(expr, frame)
        arr = np.asarray(out)
        if arr.dtype != bool:
            res.checks.append(ValidationCheck(name, "not_evaluated", "expression does not produce True/False"))
            return
        arr = np.broadcast_to(arr, (len(df),))
    except ExpressionError as exc:
        res.checks.append(ValidationCheck(name, "not_evaluated", str(exc)))
        return
    skip = _inputs_null(expr, nullmask, len(df)).to_numpy()
    bad = pd.Series(~arr & ~skip)
    k, ns = int(bad.sum()), int(skip.sum())
    msg = f"{k} row(s) violate the constraint" if k else "all evaluated rows satisfy the constraint"
    if ns:
        msg += f" ({ns} row(s) skipped because an input is missing)"
    res.checks.append(ValidationCheck(name, "failed" if k else "passed", msg, count=k,
                                      examples=_examples(df, bad)[:max_ex]))


def _check_missing_model(res: ValidationResult, spec: ColumnSpec, coerced: dict[str, pd.Series], max_ex: int) -> None:
    """Compare observed missing rates with a missing-at-random model (statistical, warning severity)."""
    m = spec.missing_model
    name = f"missing_model({spec.name})"
    ctrl = coerced.get(m["column"])
    if ctrl is None:
        res.checks.append(ValidationCheck(name, "not_evaluated", f"control column {m['column']!r} is absent",
                                          column=spec.name, severity="warning"))
        return
    miss = coerced[spec.name].isna().reset_index(drop=True)
    ctrl = ctrl.reset_index(drop=True)
    groups: dict[str, tuple[np.ndarray, float]] = {}
    if m["type"] == "group":
        keys = ctrl.astype(str).str.lower().where(ctrl.notna())
        known = {str(k).lower() for k in m["rates"]}
        for k, rate in m["rates"].items():
            groups[str(k)] = (((keys == str(k).lower()).to_numpy()), rate)
        groups["(other groups)"] = ((keys.notna() & ~keys.isin(known)).to_numpy(), m.get("default", spec.missing_rate))
    else:
        x = pd.to_numeric(ctrl, errors="coerce")
        cond = {">": x > m["value"], ">=": x >= m["value"], "<": x < m["value"], "<=": x <= m["value"]}[m["op"]]
        ok = x.notna()
        groups[f"{m['column']} {m['op']} {m['value']:g}"] = ((cond & ok).to_numpy(), m["rate_if_true"])
        groups["otherwise"] = ((~cond & ok).to_numpy(), m["rate_if_false"])
    bad, details = [], []
    for label, (mask, rate) in groups.items():
        k = int(mask.sum())
        if k < 20:
            continue
        obs = float(miss.to_numpy()[mask].mean())
        details.append(f"{label}: {obs:.1%} (configured {rate:.1%}, n={k})")
        if abs(obs - rate) > max(0.02, 4 * np.sqrt(rate * (1 - rate) / k)):
            bad.append(label)
    if not details:
        res.checks.append(ValidationCheck(name, "not_evaluated", "no group has >= 20 rows", column=spec.name, severity="warning"))
        return
    res.checks.append(ValidationCheck(name, "failed" if bad else "passed", "; ".join(details), column=spec.name,
                                      severity="warning", count=len(bad)))


def _check_correlations(res: ValidationResult, schema: Schema, coerced) -> None:
    for rel in schema.relationships:
        a, b = rel["columns"]
        label = f"correlation({a}, {b})"
        if a not in coerced or b not in coerced:
            res.checks.append(ValidationCheck(label, "not_evaluated", "column missing", severity="warning"))
            continue
        x, y = (pd.to_numeric(coerced[k].astype("int64") if str(coerced[k].dtype).startswith("datetime")
                              and coerced[k].notna().all() else coerced[k], errors="coerce") for k in (a, b))
        both = pd.concat([x, y], axis=1).dropna()
        if len(both) < 100 or both.iloc[:, 0].nunique() < 2 or both.iloc[:, 1].nunique() < 2:
            res.checks.append(ValidationCheck(label, "not_evaluated", "fewer than 100 usable rows", severity="warning"))
            continue
        r = float(np.corrcoef(both.iloc[:, 0], both.iloc[:, 1])[0, 1])
        ok = abs(r - rel["value"]) <= 0.12
        res.checks.append(ValidationCheck(label, "passed" if ok else "failed",
                                          f"observed Pearson {r:.2f} vs requested {rel['value']:.2f} "
                                          "(rounding/truncation can shift this slightly)", severity="warning"))
