"""Profiling a sample dataset and inferring a :class:`~dataseed.schema.Schema` from it.

Inference never silently discards information: every type conversion, ambiguity, excluded
column and low-confidence estimate is recorded in the returned :class:`InferenceResult`.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .exceptions import CircularDependencyError, InferenceError, SchemaError
from .models import ColumnProfile, GenerationSettings, InferenceResult
from .relationships import (detect_arithmetic_rules, detect_threshold_rules, estimate_conditionals,
                            estimate_correlations, estimate_missing_models, estimate_numeric_conditionals)
from .schema import ColumnSpec, Schema

BOOL_TRUE = {"true", "yes", "y", "t", "1"}
BOOL_FALSE = {"false", "no", "n", "f", "0"}
ID_TOKENS = {"id", "uuid", "guid", "key", "code", "number", "num", "no", "sku", "roll"}
DATE_FORMATS = ["%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d", "%d.%m.%Y",
                "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%d/%m/%Y %H:%M:%S", "%Y-%m-%d %H:%M"]
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
PHONE_RE = re.compile(r"^\+?[\d\s\-().]{7,}$")
ID_PATTERN_RE = re.compile(r"^(\D*?)(\d+)(\D*)$")
N_QUANTILES = 25


def tokens(name: str) -> list[str]:
    s = re.sub(r"([a-z])([A-Z])", r"\1_\2", str(name))
    return [t for t in re.split(r"[^A-Za-z0-9]+", s.lower()) if t]


def is_id_name(name: str) -> bool:
    t = tokens(name)
    return bool(t) and (t[-1] in ID_TOKENS or t[0] == "id")


def text_kind_from_name(name: str) -> str | None:
    t = set(tokens(name))
    joined = "".join(tokens(name))
    if t & {"email", "mail"}:
        return "email"
    if t & {"phone", "mobile", "tel", "telephone"}:
        return "phone"
    if "address" in t:
        return "address"
    if t & {"company", "organisation", "organization", "employer"}:
        return "company"
    if t & {"city", "town"}:
        return "city"
    if "username" in t or joined == "username":
        return "username"
    if joined in ("firstname", "givenname") or ("first" in t and "name" in t):
        return "first_name"
    if joined in ("lastname", "surname", "familyname") or "surname" in t or ("last" in t and "name" in t):
        return "last_name"
    if "name" in t:
        return "name"
    return None


def _clean(s: pd.Series) -> pd.Series:
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        s = s.map(lambda v: v.strip() if isinstance(v, str) else v)
        s = s.map(lambda v: None if isinstance(v, str) and v == "" else v)
    return s


def _decimals(values: pd.Series, source_is_text: bool) -> int:
    if source_is_text:
        d = 0
        for tok in values.astype(str):
            if "." in tok and "e" not in tok.lower():
                d = max(d, len(tok.split(".")[1].rstrip()))
        return min(d, 8)
    x = values.astype(float).to_numpy()
    for d in range(0, 9):
        if np.allclose(np.round(x, d), x, atol=1e-9, rtol=0):
            return d
    return 8


def _try_bool(nonnull: pd.Series) -> pd.Series | None:
    if pd.api.types.is_bool_dtype(nonnull):
        return nonnull.astype(bool)
    toks = nonnull.astype(str).str.strip().str.lower()
    allowed = BOOL_TRUE | BOOL_FALSE
    if toks.isin(allowed).all() and toks.nunique() <= 2:
        return toks.isin(BOOL_TRUE)
    return None


def _try_numeric(nonnull: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return (numeric values (NaN where unparseable), mask of unparseable values)."""
    if pd.api.types.is_numeric_dtype(nonnull) and not pd.api.types.is_bool_dtype(nonnull):
        return nonnull.astype(float), pd.Series(False, index=nonnull.index)
    cleaned = nonnull.astype(str).str.replace(",", "", regex=False).str.strip()
    num = pd.to_numeric(cleaned, errors="coerce")
    return num.astype(float), num.isna()


def _detect_date_format(nonnull: pd.Series) -> tuple[pd.Series | None, str | None, bool]:
    """Try to parse all values as dates. Returns (parsed or None, strptime format or None, ambiguous)."""
    if pd.api.types.is_datetime64_any_dtype(nonnull):
        return nonnull, None, False
    sample = nonnull.astype(str).str.strip()
    if not sample.str.contains(r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}|\d{1,2}\s+[A-Za-z]{3}", regex=True).mean() > 0.9:
        return None, None, False
    ok = []
    for fmt in DATE_FORMATS:
        parsed = pd.to_datetime(sample, format=fmt, errors="coerce")
        if parsed.notna().all():
            ok.append((fmt, parsed))
    if ok:
        ambiguous = len(ok) > 1 and {f for f, _ in ok} >= {"%d/%m/%Y", "%m/%d/%Y"}
        fmt, parsed = ok[0]
        if ambiguous:
            fmt, parsed = next((f, p) for f, p in ok if f == "%d/%m/%Y")
        return parsed, fmt, ambiguous
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    if parsed.notna().mean() >= 0.9:
        return parsed, None, False
    return None, None, False


def _id_pattern(values: pd.Series) -> dict | None:
    """Detect a common ``prefix + digits + suffix`` pattern. Returns its parts or None."""
    parts = [ID_PATTERN_RE.match(str(v)) for v in values]
    if not parts or any(m is None for m in parts):
        return None
    prefixes, digits, suffixes = zip(*[m.groups() for m in parts])
    if len(set(prefixes)) != 1 or len(set(suffixes)) != 1:
        return None
    lens = {len(d) for d in digits}
    nums = [int(d) for d in digits]
    padded = len(lens) == 1 and any(d.startswith("0") for d in digits)
    width = max(lens) if (padded or (len(lens) == 1 and prefixes[0] != "")) else 0
    if len(lens) > 1 and prefixes[0] != "":
        width = 0
    return {"prefix": prefixes[0], "suffix": suffixes[0], "width": width, "start": min(nums), "nums": nums}


def _confidence(n: int) -> str:
    return "low" if n < 8 else "medium" if n < 30 else "high"


def _shapiro_p(x: np.ndarray) -> float | None:
    if len(x) < 8 or np.ptp(x) == 0:
        return None
    try:
        return float(stats.shapiro(x[:5000]).pvalue)
    except Exception:  # pragma: no cover - defensive
        return None


def _fit_numeric(spec: ColumnSpec, x: pd.Series, integer: bool, notes: list[str]) -> None:
    """Choose a marginal model for a numeric column and write it into ``spec``."""
    v = x.to_numpy(dtype=float)
    n = len(v)
    spec.min, spec.max = (int(v.min()), int(v.max())) if integer else (float(v.min()), float(v.max()))
    if np.ptp(v) == 0:
        spec.distribution, spec.default = "constant", (int(v[0]) if integer else float(v[0]))
        notes.append("column is constant in the sample; generated as a constant")
        return
    uniq = np.unique(v)
    if integer and len(uniq) <= 6 and n >= 20:
        counts = pd.Series(v).value_counts()
        spec.categories = [int(k) for k in counts.index]
        spec.probabilities = [round(float(c) / n, 6) for c in counts.values]
        spec.min = spec.max = None
        notes.append("few distinct integer values; modelled as a discrete distribution with observed frequencies")
        return
    mean, std = float(v.mean()), float(v.std(ddof=1))
    p = _shapiro_p(v)
    skew = float(stats.skew(v)) if n >= 3 and np.ptp(v) > 0 else 0.0
    if n < 8:
        spec.distribution, spec.mean, spec.std = "normal", round(mean, 6), round(std, 6)
        notes.append(f"only {n} values: assumed a normal distribution truncated to the observed range (low confidence)")
    elif p is not None and p > 0.05:
        spec.distribution, spec.mean, spec.std = "normal", round(mean, 6), round(std, 6)
        notes.append(f"normality not rejected (Shapiro p={p:.2f}); normal truncated to the observed range")
    elif n >= 30 and v.min() > 0 and skew > 1:
        lp = _shapiro_p(np.log(v))
        if lp is not None and lp > 0.05:
            spec.distribution, spec.mean, spec.std = "lognormal", round(mean, 6), round(std, 6)
            notes.append(f"right-skewed and log-normal looking (Shapiro on logs p={lp:.2f}); lognormal")
            return
        spec.distribution = "empirical"
    else:
        spec.distribution = "empirical"
    if spec.distribution == "empirical":
        k = min(n, N_QUANTILES)
        q = np.quantile(v, np.linspace(0, 1, k))
        spec.quantiles = [int(round(a)) if integer else round(float(a), 6) for a in q]
        notes.append("not clearly normal; modelled with a smoothed empirical distribution (interpolated quantiles)")


def _infer_column(name: str, raw: pd.Series, n_rows: int, conv: list[str], amb: list[str],
                  warn: list[str]) -> tuple[ColumnProfile, ColumnSpec | None, pd.Series]:
    s = _clean(raw)
    nonnull = s.dropna()
    n_nn = len(nonnull)
    n_missing = n_rows - n_nn
    text_source = not (pd.api.types.is_numeric_dtype(raw) or pd.api.types.is_bool_dtype(raw)
                       or pd.api.types.is_datetime64_any_dtype(raw))
    prof = ColumnProfile(name, "string", str(raw.dtype), n_rows, n_missing,
                         int(nonnull.nunique()), float(nonnull.nunique() / n_nn) if n_nn else 0.0)
    if n_nn == 0:
        prof.confidence = "low"
        prof.notes.append("column is entirely empty; excluded from the schema")
        amb.append(f"Column '{name}' is entirely empty and was excluded from the schema.")
        return prof, None, s
    spec = ColumnSpec(name=name, type="string", source="inferred", n_observed=n_nn,
                      missing_rate=round(n_missing / n_rows, 6), nullable=n_missing > 0 or None)
    if n_missing > 0 and n_rows < 30:
        prof.notes.append(f"missing rate {n_missing}/{n_rows} is from a tiny sample (unreliable)")
    typed = s
    ratio = prof.unique_ratio
    kind = None   # type decision

    # ---- type decision -------------------------------------------------------------
    boolean = _try_bool(nonnull) if (not pd.api.types.is_datetime64_any_dtype(raw)) else None
    numeric = None
    if boolean is not None:
        kind = "boolean"
        if text_source or pd.api.types.is_numeric_dtype(raw):
            conv.append(f"Column '{name}': values {sorted(nonnull.astype(str).str.lower().unique())} converted to True/False.")
        if pd.api.types.is_numeric_dtype(raw):
            amb.append(f"Column '{name}' contains only 0/1; treated as boolean (change to integer if it is a count).")
    elif pd.api.types.is_datetime64_any_dtype(raw):
        kind = "datetime"
    else:
        num, bad = _try_numeric(nonnull)
        if bad.mean() <= 0.1 and (~bad).sum() > 0 and not _looks_like_date(nonnull):
            kind, numeric = "numeric", num
            if bad.any():
                examples = list(nonnull[bad].astype(str).unique()[:5])
                warn.append(f"Column '{name}': {int(bad.sum())} value(s) are not numeric {examples} and were "
                            "treated as missing in the model. Review the source data.")
                prof.notes.append(f"{int(bad.sum())} non-numeric value(s) treated as missing")
            if text_source:
                conv.append(f"Column '{name}': numbers stored as text were converted to numeric.")
        else:
            parsed, fmt, ambiguous = _detect_date_format(nonnull)
            if parsed is not None:
                kind = "datetime"
                spec.date_format = fmt
                conv.append(f"Column '{name}': text parsed as dates" + (f" using format {fmt}." if fmt else " (mixed formats)."))
                if ambiguous:
                    amb.append(f"Column '{name}': dates like 01/02/2020 are ambiguous (day-first vs month-first); day-first assumed.")
                nonnull = parsed
                typed = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
                typed.loc[parsed.index] = parsed
    # ---- identifier check ----------------------------------------------------------
    idname = is_id_name(name)
    if kind in ("numeric", None, "boolean") and n_nn >= 5 and kind != "boolean":
        raw_vals = nonnull if kind is None else nonnull.astype(str).str.strip()
        pat = _id_pattern(raw_vals) if kind in (None, "numeric") else None
        whole = kind == "numeric" and bool((numeric.dropna() % 1 == 0).all())
        if pat and idname and ratio >= 0.9 and (kind is None or whole):
            return _make_identifier(prof, spec, s, pat, n_nn, amb)
        if kind is None and pat and ratio >= 0.95 and n_nn >= 10 and pat["width"] >= 2:
            amb.append(f"Column '{name}' does not have an id-like name but its values follow a "
                       f"'{pat['prefix']}<number>{pat['suffix']}' pattern; treated as an identifier.")
            return _make_identifier(prof, spec, s, pat, n_nn, amb)
        if kind == "numeric" and whole and ratio == 1.0 and n_nn >= 10 and not idname:
            nums = np.sort(numeric.dropna().to_numpy())
            if np.all(np.diff(nums) == 1):
                amb.append(f"Column '{name}' is a consecutive integer sequence; treated as an identifier (row index).")
                return _make_identifier(prof, spec, s, {"prefix": "", "suffix": "", "width": 0, "start": int(nums[0])}, n_nn, amb)
        if idname and kind in (None, "numeric") and ratio < 0.9 and pat:
            spec.type, spec.text_kind = "string", "template"
            spec.pattern = pat["prefix"] + "#" * max(len(str(max(pat["nums"]))), pat["width"]) + pat["suffix"]
            prof.inferred_type = "string"
            amb.append(f"Column '{name}' looks like a repeated identifier/foreign key ({prof.n_unique} distinct values in "
                       f"{n_nn} rows). Generated as random codes from the pattern '{spec.pattern}', so relationships to other tables are not preserved.")
            prof.confidence = "low"
            return prof, spec, s
    # ---- build spec by type --------------------------------------------------------
    prof.confidence = _confidence(n_nn)
    if kind == "boolean":
        spec.type = "boolean"
        spec.probability = round(float(boolean.mean()), 6)
        typed = pd.Series(pd.NA, index=s.index, dtype="boolean")
        typed.loc[nonnull.index] = boolean.astype(bool)
        prof.stats = {"p_true": spec.probability}
        if n_nn < 30:
            prof.notes.append(f"P(True) estimated from only {n_nn} values")
    elif kind == "numeric":
        vals = numeric.dropna()
        integer = bool((vals % 1 == 0).all())
        if integer and text_source and nonnull.astype(str).str.contains(".", regex=False).any():
            integer = False
            amb.append(f"Column '{name}': values such as '72.0' were found; treated as float.")
        spec.type = "integer" if integer else "float"
        if not integer:
            spec.decimals = _decimals(vals, text_source)
        _fit_numeric(spec, vals, integer, prof.notes)
        typed = pd.Series(np.nan, index=s.index, dtype="float64")
        typed.loc[vals.index] = vals
        q = np.quantile(vals, [0.25, 0.5, 0.75])
        prof.stats = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                      "min": float(vals.min()), "max": float(vals.max()), "q25": float(q[0]),
                      "median": float(q[1]), "q75": float(q[2]),
                      "skew": float(stats.skew(vals)) if len(vals) > 2 and np.ptp(vals) > 0 else 0.0}
        if len(vals) < 8 and spec.distribution != "constant":
            warn.append(f"Column '{name}': only {len(vals)} values - the fitted distribution is a rough guess.")
    elif kind == "datetime":
        d = pd.to_datetime(nonnull)
        has_time = bool(((d - d.dt.normalize()) != pd.Timedelta(0)).any())
        spec.type = "datetime" if has_time else "date"
        iso = (lambda t: t.isoformat(sep=" ") if has_time else t.date().isoformat())
        spec.min, spec.max = iso(d.min()), iso(d.max())
        if len(d) >= 8 and d.nunique() > 1:
            ns = d.astype("datetime64[ns]").astype("int64").to_numpy()
            q = pd.to_datetime(np.quantile(ns, np.linspace(0, 1, min(len(d), N_QUANTILES))).astype("int64"), unit="ns")
            spec.distribution, spec.quantiles = "empirical", [iso(t) for t in q]
        else:
            spec.distribution = "uniform"
            prof.notes.append("few dates in the sample; generated uniformly between min and max")
        if d.min() == d.max():
            spec.distribution, spec.default, spec.quantiles = "constant", spec.min, None
        typed = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
        typed.loc[nonnull.index] = d.dt.normalize() if not has_time else d
        prof.stats = {"min": spec.min, "max": spec.max}
    else:
        vals = nonnull.astype(str)
        k = vals.nunique()
        typed = s.map(lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))
        email = vals.map(lambda v: bool(EMAIL_RE.match(v))).mean() > 0.9
        phone = (not email) and vals.map(lambda v: bool(PHONE_RE.match(v)) and sum(c.isdigit() for c in v) >= 7).mean() > 0.9
        name_kind = text_kind_from_name(name)
        cat_ok = (k <= 12 and k / n_nn <= 0.5) or (k / n_nn <= 0.2 and k <= 200) or (n_nn < 10 and k <= 4)
        if cat_ok and not email and not phone and not (name_kind in ("name", "first_name", "last_name", "email", "phone", "address") and k / n_nn > 0.5):
            spec.type = "categorical"
            counts = vals.value_counts()
            spec.categories = [str(c) for c in counts.index]
            spec.probabilities = [round(float(c) / n_nn, 6) for c in counts.values]
            prof.stats = {"frequencies": {str(c): int(v) for c, v in counts.items()}}
            rare = [c for c, v in counts.items() if v < 3]
            if n_nn < 30:
                prof.notes.append(f"category proportions come from only {n_nn} values (unreliable)")
            elif rare:
                prof.notes.append(f"rare categories (<3 observations): {rare[:5]}")
        else:
            spec.type = "string"
            spec.text_kind = "email" if email else "phone" if phone else name_kind or (
                "sentence" if vals.str.split().str.len().mean() >= 4 else "word")
            spec.unique = bool(ratio >= 0.95 and n_nn >= 5) or None
            prof.confidence = "low"
            prof.notes.append(f"free text / high-cardinality values are not reproduced; synthetic '{spec.text_kind}' values will be generated")
            amb.append(f"Column '{name}' is treated as free text ('{spec.text_kind}'): its source values are not reused and "
                       "its semantics are not modelled.")
    prof.inferred_type = spec.type
    if spec.type in ("integer", "float", "date", "datetime") and spec.distribution == "constant":
        prof.notes.append("constant column")
    return prof, spec, typed


def _looks_like_date(nonnull: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(nonnull):
        return False
    return bool(nonnull.astype(str).str.fullmatch(r"\s*\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}.*").mean() > 0.9)


def _make_identifier(prof, spec, s, pat, n_nn, amb):
    spec.type = "identifier"
    spec.prefix, spec.suffix, spec.width, spec.start = pat["prefix"], pat["suffix"], pat["width"], pat["start"]
    spec.unique, spec.nullable, spec.missing_rate = True, None, 0.0
    spec.distribution = "auto"
    prof.inferred_type, prof.confidence = "identifier", "high"
    prof.notes.append("identifier: new sequential ids are generated, not sampled or modelled as a measurement")
    if prof.n_missing:
        amb.append(f"Column '{prof.name}' has {prof.n_missing} missing identifier(s); generated identifiers are never missing.")
    typed = s.copy()
    return prof, spec, typed


def _apply_rules(schema: Schema, rules: list[dict], typed: pd.DataFrame) -> list[dict]:
    """Turn detected rules into derived columns. Returns the rules that were applied."""
    applied = []
    for r in rules:
        col = schema.column(r["column"])
        backup = col.to_dict()
        inputs_nullable = any(schema.column(i).is_nullable for i in r["inputs"])
        derived = ColumnSpec(name=col.name, type=col.type, expression=r["expression"], source="inferred",
                             decimals=col.decimals if col.type == "float" else None,
                             categories=col.categories if col.type == "categorical" else None,
                             nullable=True if inputs_nullable else None, confidence=r["confidence"],
                             n_observed=r["n_rows"])
        idx = schema.columns.index(col)
        schema.columns[idx] = derived
        try:
            schema.generation_order()
        except CircularDependencyError:
            schema.columns[idx] = ColumnSpec.from_dict(backup)
            continue
        applied.append(r)
    return applied


def infer_schema(df: pd.DataFrame, dataset_name: str = "dataset", settings: GenerationSettings | None = None
                 ) -> InferenceResult:
    """Profile ``df`` and infer a schema, relationships and diagnostics.

    Works on frames from :func:`load_sample` (text columns) or any typed DataFrame.

    Raises:
        InferenceError: empty input or no usable columns.
    """
    if not isinstance(df, pd.DataFrame) or df.empty or df.shape[1] == 0:
        raise InferenceError("The sample dataset is empty; nothing to infer.")
    settings = settings or GenerationSettings()
    n_rows = len(df)
    conv: list[str] = list(df.attrs.get("load_notes", []))
    amb: list[str] = []
    warn: list[str] = []
    profiles, specs, typed_cols = [], [], {}
    for name in df.columns:
        prof, spec, typed = _infer_column(str(name), df[name], n_rows, conv, amb, warn)
        profiles.append(prof)
        if spec is not None:
            specs.append(spec)
            typed_cols[str(name)] = typed
    if not specs:
        raise InferenceError("No usable columns were found in the sample.")
    typed_frame = pd.DataFrame(typed_cols)
    dups = int(typed_frame.duplicated().sum())
    if dups:
        warn.append(f"The sample contains {dups} duplicate row(s). They are kept in the profile; generated rows are not copies of sample rows.")
    if n_rows < 10:
        warn.append(f"The sample has only {n_rows} rows. Distributions, proportions and relationships inferred from so few "
                    "rows are unreliable; consider specifying ranges/categories in a schema instead.")
    elif n_rows < 30:
        warn.append(f"The sample has only {n_rows} rows: estimates are rough, and correlations/conditional models "
                    "will not be inferred from fewer than 30 rows.")
    schema = Schema(dataset_name=dataset_name, columns=specs,
                    description="Schema inferred from a sample dataset. Review before relying on it.")
    # relationships
    candidates: list[dict] = []
    applied: list[dict] = []
    rel_notes: list[str] = []
    if settings.detect_rules:
        candidates = detect_threshold_rules(typed_frame, schema) + detect_arithmetic_rules(typed_frame, schema)
        applied = _apply_rules(schema, candidates, typed_frame)
        for r in applied:
            rel_notes.append(f"Rule detected and applied: {r['column']} = {r['expression']} ({r['evidence']}) "
                             f"Confidence {r['confidence']}; verify that this is a real business rule.")
        for r in candidates:
            if r not in applied:
                rel_notes.append(f"Rule for {r['column']} detected but not applied (it would create a circular dependency).")
    corr: list[dict] = []
    if settings.infer_correlations:
        corr, notes = estimate_correlations(typed_frame, schema, settings.min_rows_for_correlation)
        rel_notes += notes
        schema.relationships = corr
    if settings.infer_conditionals:
        models, notes = estimate_conditionals(typed_frame, schema, settings.min_rows_for_conditional)
        rel_notes += notes
        for m in models:
            child = schema.column(m["child"])
            child.conditional_on, child.conditional_probabilities = m["parent"], m["probabilities"]
        nmodels, notes = estimate_numeric_conditionals(typed_frame, schema, settings.min_rows_for_conditional)
        rel_notes += notes
        for m in nmodels:
            child = schema.column(m["child"])
            child.conditional_on, child.conditional_params = m["parent"], m["params"]
    if settings.infer_missing_patterns:
        mmodels, notes = estimate_missing_models(typed_frame, schema, settings.min_rows_for_conditional)
        rel_notes += notes
        for m in mmodels:
            col = schema.column(m["column"])
            col.missing_model = {"type": "group", "column": m["parent"], "rates": m["rates"], "default": m["default"]}
    if n_rows >= 30 and not corr:
        rel_notes.append("No numeric correlation was strong enough to be distinguished from noise; columns are generated independently.")
    for c in schema.columns:
        if c.is_nullable and c.missing_rate == 0 and not c.is_derived:
            c.nullable = None
    try:
        schema.validate()
    except SchemaError as exc:
        raise InferenceError(f"Inferred schema is invalid (this is a bug): {exc}") from exc
    return InferenceResult(schema=schema, profiles=profiles, typed_frame=typed_frame, n_rows=n_rows,
                           duplicate_rows=dups, conversions=conv, ambiguities=amb, warnings=warn,
                           candidate_rules=candidates, relationship_notes=rel_notes, correlations=corr)


def profile_dataframe(df: pd.DataFrame) -> InferenceResult:
    """Alias of :func:`infer_schema` emphasising the profile part of the result."""
    return infer_schema(df)
