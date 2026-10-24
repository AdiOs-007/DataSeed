"""Relationship modelling: Gaussian copula, conditional categories and rule detection.

Gaussian copula in one paragraph: each numeric column keeps its own marginal distribution
(see ``distributions.py``). To correlate columns we draw correlated standard-normal numbers
(Cholesky factor of a correlation matrix), turn them into uniforms with the normal CDF and push
those uniforms through every column's quantile function. The correlation matrix is estimated
from normal scores of the sample ranks (robust to non-normal marginals) or taken from the
user's schema. Small correlations that a small sample cannot distinguish from noise are dropped.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .schema import NUMERIC_LIKE, ColumnSpec, Schema

MAX_ABS_CORR = 0.95


def normal_scores(series: pd.Series) -> pd.Series:
    """Rank-based normal scores (NaN preserved)."""
    ranks = series.rank(method="average")
    n = series.notna().sum()
    return pd.Series(stats.norm.ppf(ranks / (n + 1)), index=series.index)


def correlation_candidates(schema: Schema) -> list[ColumnSpec]:
    """Columns that may take part in a Gaussian-copula correlation model."""
    return [c for c in schema.columns if c.type in NUMERIC_LIKE and not c.is_derived
            and not c.is_unique and not c.categories and c.distribution != "constant"]


def estimate_correlations(typed: pd.DataFrame, schema: Schema, min_rows: int = 30
                          ) -> tuple[list[dict], list[str]]:
    """Estimate significant pairwise correlations from the sample. Returns (pairs, notes)."""
    notes: list[str] = []
    cands = [c for c in correlation_candidates(schema) if c.name in typed.columns]
    pairs: list[dict] = []
    if len(cands) < 2:
        return pairs, notes
    num = pd.DataFrame({c.name: pd.to_numeric(
        typed[c.name].astype("int64") if c.type in ("date", "datetime") and typed[c.name].notna().all()
        else typed[c.name].map(lambda v: v.timestamp() if hasattr(v, "timestamp") else v),
        errors="coerce") for c in cands})
    num = num.loc[:, num.nunique() > 1]
    if num.shape[1] < 2:
        return pairs, notes
    scores = num.apply(normal_scores)
    for i, a in enumerate(scores.columns):
        for b in scores.columns[i + 1:]:
            both = scores[[a, b]].dropna()
            n = len(both)
            if n < min_rows:
                notes.append(f"Correlation {a}~{b}: only {n} complete rows (< {min_rows}); not estimated.")
                continue
            r = float(np.corrcoef(both[a], both[b])[0, 1])
            threshold = max(0.1, 2.0 / math.sqrt(n))
            if abs(r) < threshold:
                continue
            pairs.append({"type": "correlation", "columns": [a, b],
                          "value": round(float(np.clip(r, -MAX_ABS_CORR, MAX_ABS_CORR)), 3),
                          "source": "inferred", "n_observations": int(n)})
    if pairs:
        notes.append(f"{len(pairs)} correlation(s) above the noise threshold (~2/sqrt(n)) will be modelled "
                     "with a Gaussian copula; weaker ones are treated as independent.")
    return pairs, notes


def nearest_psd(matrix: np.ndarray) -> np.ndarray:
    """Project a symmetric matrix onto the closest positive-definite correlation matrix."""
    a = (matrix + matrix.T) / 2.0
    vals, vecs = np.linalg.eigh(a)
    vals = np.clip(vals, 1e-6, None)
    b = vecs @ np.diag(vals) @ vecs.T
    d = np.sqrt(np.diag(b))
    b = b / np.outer(d, d)
    np.fill_diagonal(b, 1.0)
    return (b + b.T) / 2.0


def build_correlation_matrix(columns: Sequence[str], pairs: Sequence[dict]
                             ) -> tuple[np.ndarray, dict]:
    """Assemble a correlation matrix; repair it if it is not positive definite.

    Returns ``(matrix, info)`` where ``info`` has ``repaired`` (bool) and ``max_adjustment``.
    """
    idx = {c: i for i, c in enumerate(columns)}
    m = np.eye(len(columns))
    for p in pairs:
        a, b = p["columns"]
        if a in idx and b in idx:
            m[idx[a], idx[b]] = m[idx[b], idx[a]] = p["value"]
    min_eig = float(np.linalg.eigvalsh(m).min())
    if min_eig > 1e-6:
        return m, {"repaired": False, "max_adjustment": 0.0, "min_eigenvalue": min_eig}
    fixed = nearest_psd(m)
    return fixed, {"repaired": True, "max_adjustment": float(np.abs(fixed - m).max()), "min_eigenvalue": min_eig}


def correlated_normals(matrix: np.ndarray, m: int, rng: np.random.Generator) -> np.ndarray:
    """Draw ``m`` rows of correlated standard normals (shape ``m x k``)."""
    try:
        chol = np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:  # nearly singular: add jitter
        chol = np.linalg.cholesky(nearest_psd(matrix) + 1e-6 * np.eye(len(matrix)))
    return rng.standard_normal((m, len(matrix))) @ chol.T


def cramers_v(table: pd.DataFrame) -> float:
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.values.sum()
    k = min(table.shape) - 1
    return float(math.sqrt(chi2 / (n * k))) if n and k else 0.0


def estimate_conditionals(typed: pd.DataFrame, schema: Schema, min_rows: int = 60,
                          min_group: int = 10) -> tuple[list[dict], list[str]]:
    """Find categorical columns whose distribution clearly depends on an earlier categorical column.

    Evidence required: chi-square p < 0.01, Cramer's V >= 0.25, >= ``min_rows`` complete rows and
    at least two parent groups with >= ``min_group`` rows. Returns (models, notes).
    """
    cats = [c for c in schema.columns if c.type in ("categorical", "boolean") and not c.is_derived
            and c.name in typed.columns]
    models: list[dict] = []
    notes: list[str] = []
    for j, child in enumerate(cats):
        if child.type != "categorical":
            continue
        best = None
        for parent in cats[:j]:
            both = typed[[parent.name, child.name]].dropna()
            if len(both) < min_rows:
                if len(both) and parent is cats[0]:
                    notes.append(f"{child.name}: only {len(both)} rows (< {min_rows}); conditional models not attempted.")
                continue
            table = pd.crosstab(both[parent.name].astype(str), both[child.name].astype(str))
            big = table[table.sum(axis=1) >= min_group]
            if big.shape[0] < 2 or table.shape[1] < 2:
                continue
            p = stats.chi2_contingency(big, correction=False)[1]
            v = cramers_v(big)
            if p < 0.01 and v >= 0.25 and (best is None or v > best[0]):
                best = (v, p, parent, big)
        if best:
            v, p, parent, big = best
            probs = {str(idx): {str(k): float(x) for k, x in (row / row.sum()).items() if x > 0}
                     for idx, row in big.iterrows()}
            models.append({"child": child.name, "parent": parent.name, "probabilities": probs,
                           "cramers_v": round(v, 3), "p_value": float(p)})
            notes.append(f"{child.name} depends on {parent.name} (Cramer's V={v:.2f}, p={p:.3g}); "
                         "conditional probabilities estimated per group.")
    return models, notes


def _round_threshold(lo_max: float, hi_min: float) -> float:
    """Pick a tidy threshold t with lo_max < t <= hi_min (round numbers preferred)."""
    for step in (100, 50, 25, 10, 5, 1, 0.5, 0.1):
        cands = [k * step for k in range(math.floor(lo_max / step), math.ceil(hi_min / step) + 1)
                 if lo_max < k * step <= hi_min]
        if cands:
            mid = (lo_max + hi_min) / 2
            return float(min(cands, key=lambda t: abs(t - mid)))
    return float((lo_max + hi_min) / 2)


def detect_threshold_rules(typed: pd.DataFrame, schema: Schema, min_rows: int = 20,
                           min_class: int = 3) -> list[dict]:
    """Find binary columns perfectly separated by a threshold on a numeric column.

    Example: ``passed`` is True exactly when ``marks >= 40`` for every row. Only *perfect*
    separation with at least ``min_class`` rows per class is reported.
    """
    rules: list[dict] = []
    nums = [c for c in schema.columns if c.type in ("integer", "float") and not c.is_unique
            and c.name in typed.columns and typed[c.name].nunique() > 2]
    for tgt in schema.columns:
        if tgt.name not in typed.columns or tgt.is_derived:
            continue
        if tgt.type == "boolean":
            labels = (True, False)
        elif tgt.type == "categorical" and tgt.categories and len(tgt.categories) == 2:
            labels = tuple(tgt.categories)
        else:
            continue
        for num in nums:
            if num.name == tgt.name:
                continue
            both = typed[[num.name, tgt.name]].dropna()
            if len(both) < min_rows:
                continue
            x = pd.to_numeric(both[num.name], errors="coerce")
            y = both[tgt.name]
            for hi_label, lo_label in (labels, labels[::-1]):
                hi_x, lo_x = x[y == hi_label], x[y == lo_label]
                if len(hi_x) < min_class or len(lo_x) < min_class:
                    continue
                if lo_x.max() < hi_x.min():
                    t = _round_threshold(float(lo_x.max()), float(hi_x.min()))
                    tname = int(t) if float(t).is_integer() else t
                    cond = f"{num.name} >= {tname}"
                    expr = cond if tgt.type == "boolean" else \
                        f"where({cond}, {hi_label!r}, {lo_label!r})"
                    rules.append({"kind": "threshold", "column": tgt.name, "expression": expr,
                                  "inputs": [num.name], "n_rows": int(len(both)),
                                  "confidence": "medium" if len(both) < 30 else "high",
                                  "evidence": f"{tgt.name} is {hi_label!r} for every row with {cond} "
                                              f"and {lo_label!r} otherwise ({len(both)} rows)."})
                    break
    return _dedupe_by_column(rules)


def detect_arithmetic_rules(typed: pd.DataFrame, schema: Schema, min_rows: int = 10) -> list[dict]:
    """Find numeric columns that equal ``a * b`` or ``a + b`` of two other columns in every row."""
    nums = [c for c in schema.columns if c.type in ("integer", "float") and not c.is_unique
            and c.name in typed.columns and typed[c.name].nunique() > 2]
    rules: list[dict] = []
    for tgt in nums:
        for i, a in enumerate(nums):
            for b in nums[i + 1:]:
                if tgt.name in (a.name, b.name):
                    continue
                df = typed[[tgt.name, a.name, b.name]].dropna().astype(float)
                if len(df) < min_rows:
                    continue
                tol = 10 ** -(tgt.decimals or 0) * 0.5001 + 1e-9
                for sym, vals in (("*", df[a.name] * df[b.name]), ("+", df[a.name] + df[b.name])):
                    if (df[a.name].nunique() > 1 and df[b.name].nunique() > 1
                            and np.allclose(df[tgt.name], vals, atol=tol, rtol=1e-9)):
                        rules.append({"kind": "arithmetic", "column": tgt.name,
                                      "expression": f"{a.name} {sym} {b.name}", "inputs": [a.name, b.name],
                                      "n_rows": int(len(df)), "confidence": "medium" if len(df) < 30 else "high",
                                      "evidence": f"{tgt.name} == {a.name} {sym} {b.name} in all {len(df)} rows."})
    return _dedupe_by_column(rules)


def _dedupe_by_column(rules: list[dict]) -> list[dict]:
    seen, out = set(), []
    for r in rules:
        if r["column"] not in seen:
            seen.add(r["column"])
            out.append(r)
    return out


def estimate_numeric_conditionals(typed: pd.DataFrame, schema: Schema, min_rows: int = 60, min_group: int = 10
                                  ) -> tuple[list[dict], list[str]]:
    """Find numeric columns whose distribution clearly differs between groups of a categorical column.

    Evidence required: Kruskal-Wallis p < 0.01 and epsilon-squared effect size >= 0.06 over groups with
    at least ``min_group`` rows (>= 2 such groups, >= ``min_rows`` rows in total). Per-group parameters
    (mean/std for normal/lognormal columns, quantiles for empirical ones) are returned.
    """
    parents = [c for c in schema.columns if c.type in ("categorical", "boolean") and not c.is_derived and c.name in typed]
    models: list[dict] = []
    notes: list[str] = []
    for child in schema.columns:
        if (child.type not in ("integer", "float") or child.is_derived or child.is_unique or child.categories
                or child.distribution not in ("normal", "lognormal", "empirical") or child.name not in typed):
            continue
        best = None
        for parent in parents:
            both = typed[[parent.name, child.name]].dropna()
            both = both.assign(**{child.name: pd.to_numeric(both[child.name], errors="coerce")}).dropna()
            if len(both) < min_rows:
                continue
            groups = {str(k): g[child.name].to_numpy(float) for k, g in both.groupby(both[parent.name].astype(str))
                      if len(g) >= min_group}
            if len(groups) < 2:
                continue
            h, p = stats.kruskal(*groups.values())
            n = sum(len(g) for g in groups.values())
            eps = (h - len(groups) + 1) / (n - len(groups)) if n > len(groups) else 0.0
            if p < 0.01 and eps >= 0.06 and (best is None or eps > best[0]):
                best = (eps, p, parent, groups)
        if best is None:
            continue
        eps, p, parent, groups = best
        params: dict[str, dict] = {}
        for key, v in groups.items():
            if child.distribution == "empirical":
                k = min(len(v), 15)
                params[key] = {"quantiles": [round(float(q), 6) for q in np.quantile(v, np.linspace(0, 1, k))]}
            elif np.std(v, ddof=1) > 0 and (child.distribution == "normal" or v.mean() > 0):
                params[key] = {"mean": round(float(v.mean()), 6), "std": round(float(v.std(ddof=1)), 6)}
        if len(params) >= 2:
            models.append({"child": child.name, "parent": parent.name, "params": params, "effect": round(float(eps), 3),
                           "p_value": float(p)})
            notes.append(f"{child.name} differs between {parent.name} groups (Kruskal-Wallis p={p:.3g}, effect size "
                         f"{eps:.2f}); group-specific distribution parameters estimated.")
    return models, notes


def estimate_missing_models(typed: pd.DataFrame, schema: Schema, min_rows: int = 60, min_missing: int = 8,
                            min_group: int = 10) -> tuple[list[dict], list[str]]:
    """Detect missing-at-random patterns: missingness that depends on a categorical column.

    Evidence required: >= ``min_rows`` rows, >= ``min_missing`` missing values, chi-square p < 0.01 and
    Cramer's V >= 0.2 across groups with >= ``min_group`` rows. Missingness that merely *correlates* with
    a group in a small sample is NOT reported (MCAR is kept).
    """
    parents = [c for c in schema.columns if c.type in ("categorical", "boolean") and not c.is_derived and c.name in typed]
    models: list[dict] = []
    notes: list[str] = []
    for col in schema.columns:
        if col.is_derived or col.type == "identifier" or col.name not in typed:
            continue
        miss = typed[col.name].isna()
        if len(typed) < min_rows or miss.sum() < min_missing:
            continue
        best = None
        for parent in parents:
            if parent.name == col.name:
                continue
            ok = typed[parent.name].notna()
            groups = typed.loc[ok, parent.name].astype(str)
            counts = groups.value_counts()
            keep = counts[counts >= min_group].index
            if len(keep) < 2:
                continue
            sel = groups.isin(keep)
            table = pd.crosstab(groups[sel], miss[ok][sel])
            if table.shape[1] < 2:
                continue
            p = stats.chi2_contingency(table, correction=False)[1]
            v = cramers_v(table)
            if p < 0.01 and v >= 0.2 and (best is None or v > best[0]):
                rates = {str(k): round(float(miss[ok][groups == k].mean()), 4) for k in keep}
                best = (v, p, parent, rates)
        if best:
            v, p, parent, rates = best
            models.append({"column": col.name, "parent": parent.name, "rates": rates, "cramers_v": round(v, 3),
                           "p_value": float(p), "default": round(float(miss.mean()), 6)})
            notes.append(f"Missing values of {col.name} depend on {parent.name} (Cramer's V={v:.2f}, p={p:.3g}); "
                         "modelled as missing-at-random by group.")
    return models, notes
