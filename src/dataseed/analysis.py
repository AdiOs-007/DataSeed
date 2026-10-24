"""Statistical quality analysis of a generated dataset (optionally compared with its source).

IMPORTANT: none of these metrics proves that synthetic data is realistic. They describe how
closely marginal distributions, category proportions and pairwise linear relationships agree
with a *small* sample. ``metric_notes`` in the report explains each metric and its limits.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .inference import infer_schema
from .models import QualityReport, ValidationResult
from .schema import Schema
from .validation import coerce_series, validate_dataset

METRIC_NOTES = {
    "ks_statistic": "Two-sample Kolmogorov-Smirnov distance (0 = identical empirical CDFs, 1 = disjoint). Sensitive "
                    "to any shape difference but has little power when the source sample is small.",
    "ks_exceeds_critical": "True if the KS distance exceeds the approximate 5% critical value. False means 'no detectable "
                           "difference', NOT 'proven identical'. With 20 source rows only large differences are detectable.",
    "wasserstein_normalized": "Average horizontal distance between the two CDFs divided by the source standard deviation "
                              "(0 = identical; 0.1 is roughly 10% of a standard deviation).",
    "mean_diff_in_source_std": "(generated mean - source mean) / source std.",
    "total_variation_distance": "Half the sum of absolute differences between category proportions (0 = identical, 1 = no overlap).",
    "correlation_difference": "Mean absolute difference between Pearson correlations of source and generated numeric columns. "
                              "Correlations from small samples are very noisy.",
    "exact_row_overlap": "Share of generated rows identical to a source row. High values mean generation largely reproduces "
                         "source rows. This is a sanity check, NOT a privacy guarantee.",
    "general": "Agreement on these statistics is necessary but not sufficient for realism; they cannot detect missing "
               "real-world structure that the sample or schema never contained.",
}


def _coerce_frame(df: pd.DataFrame, schema: Schema) -> dict[str, pd.Series]:
    return {c.name: coerce_series(df[c.name], c)[0] for c in schema.columns if c.name in df.columns}


def _num(s: pd.Series) -> np.ndarray:
    if str(s.dtype).startswith("datetime"):
        s = s.dropna().astype("datetime64[ns]").astype("int64") / 8.64e13   # days
    return pd.to_numeric(s, errors="coerce").dropna().to_numpy(dtype=float)


def _summary(s: pd.Series, spec, n: int, max_categories: int) -> dict[str, Any]:
    out: dict[str, Any] = {"type": spec.type, "n_missing": int(s.isna().sum()),
                           "missing_pct": round(float(s.isna().mean() * 100), 3) if n else 0.0,
                           "n_unique": int(s.nunique())}
    if spec.type in ("integer", "float"):
        v = _num(s)
        if len(v):
            q = np.quantile(v, [0.05, 0.25, 0.5, 0.75, 0.95])
            out.update(mean=float(v.mean()), std=float(v.std(ddof=1)) if len(v) > 1 else 0.0, min=float(v.min()),
                       q05=float(q[0]), q25=float(q[1]), median=float(q[2]), q75=float(q[3]), q95=float(q[4]),
                       max=float(v.max()))
    elif spec.type in ("date", "datetime"):
        d = s.dropna()
        if len(d):
            out.update(min=d.min().isoformat(), max=d.max().isoformat())
    elif spec.type in ("categorical", "boolean"):
        vc = s.dropna().astype(str).value_counts(normalize=True)
        out["frequencies"] = {str(k): round(float(v), 6) for k, v in vc.head(max_categories).items()}
        if len(vc) > max_categories:
            out["frequencies_truncated"] = True
    return out


def _compare_numeric(src: pd.Series, gen: pd.Series, warnings: list[str], name: str) -> dict[str, Any]:
    a, b = _num(src), _num(gen)
    out: dict[str, Any] = {"n_source": len(a), "n_generated": len(b)}
    if len(a) < 2 or len(b) < 2:
        out["note"] = "too few values to compare"
        return out
    sd = float(a.std(ddof=1))
    ks = stats.ks_2samp(a, b)
    crit = 1.358 * math.sqrt((len(a) + len(b)) / (len(a) * len(b)))
    out.update(ks_statistic=float(ks.statistic), ks_p_value=float(ks.pvalue), ks_critical_5pct=crit,
               ks_exceeds_critical=bool(ks.statistic > crit),
               source_mean=float(a.mean()), generated_mean=float(b.mean()),
               source_std=sd, generated_std=float(b.std(ddof=1)))
    if sd > 0:
        out["wasserstein_normalized"] = float(stats.wasserstein_distance(a, b) / sd)
        out["mean_diff_in_source_std"] = float((b.mean() - a.mean()) / sd)
    else:
        out["note"] = "source column is constant; normalised metrics not defined"
    out["interpretation"] = ("difference detectable at the 5% level" if out["ks_exceeds_critical"]
                             else "no detectable difference (low power if the source sample is small)")
    if len(a) < 30:
        out["reliability"] = f"source sample has only {len(a)} values; this comparison is weak evidence"
        warnings.append(f"Column '{name}': compared with only {len(a)} source values - results are unreliable.")
    return out


def _compare_categorical(src: pd.Series, gen: pd.Series, warnings: list[str], name: str) -> dict[str, Any]:
    ps = src.dropna().astype(str).value_counts(normalize=True)
    pg = gen.dropna().astype(str).value_counts(normalize=True)
    cats = ps.index.union(pg.index)
    diff = (ps.reindex(cats, fill_value=0) - pg.reindex(cats, fill_value=0)).abs()
    out: dict[str, Any] = {"n_source": int(src.notna().sum()), "n_generated": int(gen.notna().sum()),
                           "total_variation_distance": float(diff.sum() / 2), "max_abs_difference": float(diff.max()),
                           "categories_only_in_generated": sorted(set(pg.index) - set(ps.index)),
                           "categories_missing_in_generated": sorted(set(ps.index) - set(pg.index))}
    if out["n_source"] < 30:
        out["reliability"] = f"source sample has only {out['n_source']} values; proportions are unreliable"
        warnings.append(f"Column '{name}': category proportions compared against only {out['n_source']} source values.")
    return out


def _corr_matrix(cols: dict[str, pd.Series], names: list[str]) -> pd.DataFrame:
    df = pd.DataFrame({k: pd.Series(_num_aligned(cols[k])) for k in names})
    return df.corr(method="pearson")


def _num_aligned(s: pd.Series) -> pd.Series:
    if str(s.dtype).startswith("datetime"):
        return pd.Series(s.astype("datetime64[ns]").astype("int64") / 8.64e13).where(s.notna().to_numpy())
    return pd.to_numeric(s, errors="coerce")


def analyze_quality(generated: pd.DataFrame, source: pd.DataFrame | None = None, schema: Schema | None = None,
                    validation: ValidationResult | None = None, max_categories: int = 20) -> QualityReport:
    """Summarise ``generated`` and, if ``source`` is given, compare the two.

    Args:
        generated: The synthetic data.
        source: Optional original sample (raw ``load_sample`` output or typed frame).
        schema: Schema to interpret types; inferred from ``generated`` if omitted.
        validation: Pre-computed validation (computed here if ``schema`` is given and this is None).
    """
    warnings: list[str] = []
    if generated is None or len(generated.columns) == 0:
        raise ValueError("analyze_quality needs a non-empty generated DataFrame.")
    if schema is None:
        schema = infer_schema(generated).schema
        warnings.append("No schema supplied; column types were inferred from the generated data.")
    gen = _coerce_frame(generated, schema)
    n = len(generated)
    dup = int(generated.duplicated().sum()) if n else 0
    overview = {"n_rows": n, "n_columns": generated.shape[1], "duplicate_rows": dup,
                "duplicate_rows_pct": round(dup / n * 100, 3) if n else 0.0,
                "missing_cells": int(generated.isna().sum().sum())}
    if n < 30:
        warnings.append(f"Generated data has only {n} rows; statistics are very noisy.")
    columns = {c.name: _summary(gen[c.name], c, n, max_categories) for c in schema.columns if c.name in gen}
    num_names = [c.name for c in schema.columns if c.type in ("integer", "float") and c.name in gen
                 and gen[c.name].nunique() > 1]
    correlations: dict[str, Any] = {}
    if len(num_names) >= 2 and n >= 3:
        correlations["generated"] = _corr_matrix(gen, num_names).round(4).replace({np.nan: None}).to_dict()
    comparisons: dict[str, Any] = {}
    if source is not None:
        src = _coerce_frame(source, schema)
        common = [c for c in schema.columns if c.name in src and c.name in gen]
        for c in schema.columns:
            if c.name not in src:
                warnings.append(f"Column '{c.name}' is not in the source; no comparison made.")
        col_cmp: dict[str, Any] = {}
        for c in common:
            entry: dict[str, Any] = {"source_missing_pct": round(float(src[c.name].isna().mean() * 100), 3),
                                     "generated_missing_pct": columns[c.name]["missing_pct"]}
            if c.type in ("integer", "float", "date", "datetime"):
                entry.update(_compare_numeric(src[c.name], gen[c.name], warnings, c.name))
            elif c.type in ("categorical", "boolean"):
                entry.update(_compare_categorical(src[c.name], gen[c.name], warnings, c.name))
            else:
                s_vals = set(src[c.name].dropna().astype(str))
                overlap = gen[c.name].dropna().astype(str).isin(s_vals).mean() * 100 if n else 0.0
                entry["note"] = "identifiers and free text are not compared statistically"
                entry["generated_values_also_in_source_pct"] = round(float(overlap), 3)
            col_cmp[c.name] = entry
        comparisons["columns"] = col_cmp
        cnames = [x for x in num_names if x in src and src[x].nunique() > 1]
        if len(cnames) >= 2:
            cs, cg = _corr_matrix(src, cnames), _corr_matrix(gen, cnames)
            iu = np.triu_indices(len(cnames), 1)
            d = (cs.values - cg.values)[iu]
            d = np.abs(d[~np.isnan(d)])
            pairs = sorted(((cnames[i], cnames[j], float(cs.values[i, j]), float(cg.values[i, j]))
                            for i, j in zip(*iu) if not np.isnan(cs.values[i, j] - cg.values[i, j])),
                           key=lambda t: -abs(t[2] - t[3]))[:5]
            correlations["source"] = cs.round(4).replace({np.nan: None}).to_dict()
            comparisons["correlation_difference"] = {
                "mean_abs_difference": float(d.mean()) if len(d) else None,
                "largest_differences": [{"columns": [a, b], "source": s, "generated": g} for a, b, s, g in pairs]}
            if len(source) < 30:
                warnings.append("Correlations were compared against fewer than 30 source rows; treat as indicative only.")
        shared = [c.name for c in common if c.type not in ("identifier",)]
        if shared and n:
            h_g = pd.util.hash_pandas_object(generated[shared].astype(str), index=False)
            h_s = pd.util.hash_pandas_object(source[shared].astype(str), index=False)
            ov = int(h_g.isin(set(h_s)).sum())
            comparisons["exact_row_overlap"] = {"rows": ov, "pct": round(ov / n * 100, 3),
                                                "columns_compared": shared}
    compliance = None
    if validation is None and schema is not None:
        validation = validate_dataset(generated, schema)
    if validation is not None:
        compliance = {"is_valid": validation.is_valid, **validation.to_dict()["summary"],
                      "failed_checks": [f"{c.name}" + (f"[{c.column}]" if c.column else "") + f": {c.message}"
                                        for c in validation.failed],
                      "not_evaluated_checks": [f"{c.name}: {c.message}" for c in validation.not_evaluated]}
    return QualityReport(overview=overview, columns=columns, correlations=correlations, comparisons=comparisons,
                         constraint_compliance=compliance, warnings=list(dict.fromkeys(warnings)),
                         metric_notes=METRIC_NOTES)
