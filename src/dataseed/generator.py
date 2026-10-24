"""The synthetic data generation engine.

Pipeline for one run
--------------------
1. Validate the request and the schema; check that uniqueness is even possible for ``n_rows``.
2. Build a marginal distribution for every numeric/date column (``distributions.py``).
3. If correlations exist, build a Gaussian copula (``relationships.py``).
4. Generate a *batch* of complete rows column by column in dependency order: copula/numeric,
   categorical (optionally conditional on another column), boolean, text, then derived columns.
5. Constraint satisfaction by targeted re-sampling: rows that violate a row-level constraint or
   a uniqueness rule are replaced by freshly generated rows that satisfy it (rejection sampling).
   Rows are replaced, never deleted, so the row count is exact. If a constraint cannot be met
   a :class:`ConstraintConflictError` explains why.
6. Assign identifiers, then apply missing values (exact counts) and propagate them to derived
   columns, then convert to final pandas dtypes.
7. Validate the result and attach a report of every decision.
"""
from __future__ import annotations

import dataclasses
import secrets
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from . import textgen
from .distributions import Constant, Marginal, build_marginal, from_internal
from .exceptions import ConstraintConflictError, ExpressionError, GenerationError
from .expressions import evaluate_expression, referenced_columns
from .inference import infer_schema
from .models import (GenerationReport, GenerationResult, GenerationSettings, InferenceResult)
from .relationships import build_correlation_matrix, correlated_normals
from .schema import ColumnSpec, Schema
from .validation import validate_dataset

MAX_CANDIDATES = 200_000


def _apply_overrides(schema: Schema, overrides: dict[str, dict], report: GenerationReport) -> Schema:
    if not overrides:
        return schema
    data = schema.to_dict()
    cols = {c["name"]: c for c in data["columns"]}
    for name, props in overrides.items():
        if name not in cols:
            raise GenerationError(f"Override refers to unknown column {name!r}. Known columns: {list(cols)}")
        if not isinstance(props, dict):
            raise GenerationError(f"Override for {name!r} must be a dict of column properties.")
        cols[name].update(props)
        report.add("info", f"user override applied: {props}", name, origin="user")
    return Schema.from_dict(data)


class _Engine:
    def __init__(self, schema: Schema, n: int, rng: np.random.Generator, settings: GenerationSettings,
                 report: GenerationReport):
        self.schema, self.n, self.rng, self.settings, self.report = schema, n, rng, settings, report
        self.order = schema.generation_order()
        self.spec = {c.name: c for c in schema.columns}
        self.marginals: dict[str, Marginal] = {}
        self.group_marginals: dict[str, dict[str, Marginal]] = {}
        self.copula_cols: list[str] = []
        self.corr: np.ndarray | None = None
        self.unique_cols = [c.name for c in schema.columns if c.is_unique and c.type not in ("identifier", "string")]
        self.unique_strings = [c.name for c in schema.columns if c.is_unique and c.type == "string"]
        self.unique_together = [con["columns"] for con in schema.constraints if con["type"] == "unique_together"]
        self.row_constraints = [con for con in schema.constraints if con["type"] == "expression"]

    # ------------------------------------------------------------------ setup
    def prepare(self) -> None:
        self._check_feasibility()
        for c in self.schema.columns:
            origin = "user" if c.source in ("user", "parsed") else "inferred"
            if c.is_derived:
                self.report.add("method", f"derived from expression: {c.expression}", c.name, origin)
            elif c.type == "identifier":
                self.report.add("method", self._identifier_description(c), c.name, origin)
            elif c.type in ("integer", "float", "date", "datetime"):
                m = build_marginal(c)
                self.marginals[c.name] = m
                extra = f"; rounded to {c.decimals} decimal(s)" if c.decimals is not None else ""
                desc = (f"uniform between {c.min} and {c.max}" if c.type in ("date", "datetime") and m.kind == "uniform"
                        else m.describe())
                self.report.add("method", desc + extra, c.name, origin, c.confidence)
                for note in m.notes:
                    self.report.add("fallback", note, c.name, "assumed")
                    self.report.warn(f"Column '{c.name}': {note}.")
                if c.conditional_on:
                    self.group_marginals[c.name] = self._build_group_marginals(c)
                    self.report.add("relationship", f"distribution depends on {c.conditional_on}: separate parameters for "
                                    f"{len(c.conditional_params)} group(s); other groups use the overall distribution",
                                    c.name, origin)
            elif c.type == "categorical":
                self.report.add("method", self._categorical_description(c), c.name, origin, c.confidence)
            elif c.type == "boolean":
                p = c.probability if c.probability is not None else 0.5
                if c.probability is None:
                    self.report.add("assumption", "no probability given; assumed P(True)=0.5", c.name, "assumed")
                    self.report.warn(f"Column '{c.name}': no probability given; assumed 50% True.")
                self.report.add("method", f"Bernoulli with P(True)={p:g}", c.name, origin, c.confidence)
            else:
                kind = c.text_kind or ("template" if c.pattern else "word")
                self.report.add("method", f"synthetic text generator '{kind}'" +
                                (f" with pattern {c.pattern!r}" if c.pattern else ""), c.name, origin)
                if kind in ("word", "sentence"):
                    self.report.warn(f"Column '{c.name}': free text is generated from a tiny made-up vocabulary; it has no semantic realism.")
            if c.n_observed is not None and c.n_observed < 30 and not c.is_derived and c.type != "identifier":
                self.report.warn(f"Column '{c.name}': parameters were inferred from only {c.n_observed} observations; treat them as low-confidence.")
            if c.missing_model and not c.is_derived:
                self.report.add("missing", self._missing_model_description(c), c.name, origin)
            elif c.is_nullable and c.missing_rate > 0 and not c.is_derived:
                self.report.add("missing", f"{c.missing_rate:.2%} of values will be missing (completely at random)", c.name, origin)
            elif c.is_derived and any(self.spec[r].has_missing for r in referenced_columns(c.expression)):
                self.report.add("missing", "missing wherever an input column is missing", c.name, "default")
        self._prepare_copula()
        for con in self.row_constraints:
            self.report.add("constraint", f"row constraint: {con['expression']}"
                            + (f" ({con['description']})" if con.get("description") else ""), origin="user")
        for cols in self.unique_together:
            self.report.add("constraint", f"unique combination of {cols}", origin="user")
        for name in self.unique_cols:
            self.report.add("constraint", "values are unique", name, "user")
        for c in self.schema.columns:
            if c.conditional_on and c.type == "categorical":
                self.report.add("relationship", f"distribution conditional on {c.conditional_on} "
                                f"({len(c.conditional_probabilities)} group(s))", c.name,
                                "inferred" if c.source == "inferred" else "user")

    @staticmethod
    def _missing_model_description(c: ColumnSpec) -> str:
        m = c.missing_model
        if m["type"] == "group":
            rates = ", ".join(f"{k}: {v:.1%}" for k, v in m["rates"].items())
            return (f"missing at random depending on {m['column']} ({rates}; other groups "
                    f"{m.get('default', c.missing_rate):.1%})")
        return (f"missing at random depending on {m['column']} {m['op']} {m['value']:g}: "
                f"{m['rate_if_true']:.1%} if true, {m['rate_if_false']:.1%} otherwise")

    def _build_group_marginals(self, c: ColumnSpec) -> dict[str, Marginal]:
        out: dict[str, Marginal] = {}
        for key, params in c.conditional_params.items():
            if "quantiles" in params:
                over = dict(distribution="empirical", quantiles=params["quantiles"], mean=None, std=None)
            else:
                dist = c.distribution if c.distribution in ("normal", "lognormal") else "normal"
                over = dict(distribution=dist, mean=params["mean"], std=params["std"], quantiles=None)
            try:
                out[str(key).lower()] = build_marginal(dataclasses.replace(
                    c, conditional_on=None, conditional_params=None, **over))
            except (GenerationError, ValueError, ZeroDivisionError) as exc:
                raise GenerationError(f"Column '{c.name}': invalid conditional parameters for group {key!r}: {exc}") from exc
        return out

    @staticmethod
    def _identifier_description(c: ColumnSpec) -> str:
        w = c.width if c.width is not None else (6 if (c.prefix or c.suffix) else 0)
        return (f"{c.sequence or 'sequential'} identifiers {c.prefix or ''}<number zero-padded to {w}>{c.suffix or ''} "
                f"starting at {c.start if c.start is not None else 1}")

    @staticmethod
    def _categorical_description(c: ColumnSpec) -> str:
        if not c.categories:
            return f"constant value {c.default!r}"
        if c.conditional_on:
            return f"categorical, conditional on {c.conditional_on}"
        if c.probabilities:
            return f"categorical with given/observed probabilities over {len(c.categories)} categories"
        return f"categorical, equal probabilities over {len(c.categories)} categories (assumed)"

    def _prepare_copula(self) -> None:
        pairs = self.schema.relationships
        if not pairs:
            return
        cols = []
        for p in pairs:
            for c in p["columns"]:
                if c not in cols:
                    cols.append(c)
        matrix, info = build_correlation_matrix(cols, pairs)
        user_pairs = [p for p in pairs if p.get("source", "user") != "inferred"]
        if info["repaired"]:
            msg = (f"The requested correlations are not mutually consistent (smallest eigenvalue "
                   f"{info['min_eigenvalue']:.2f}); the matrix was adjusted by up to {info['max_adjustment']:.2f}.")
            if user_pairs and info["max_adjustment"] > 0.05:
                raise ConstraintConflictError(
                    msg + " Choose correlations that can coexist (e.g. if A~B=0.9 and B~C=0.9 then A~C cannot be -0.9).")
            self.report.warn(msg)
            self.report.add("fallback", msg, origin="assumed")
        self.copula_cols, self.corr = cols, matrix
        for p in pairs:
            origin = "inferred" if p.get("source") == "inferred" else "user"
            n_obs = f" (from {p['n_observations']} rows)" if p.get("n_observations") else ""
            self.report.add("relationship", f"Gaussian-copula correlation {p['value']:+.2f} between "
                            f"{p['columns'][0]} and {p['columns'][1]}{n_obs}", None, origin)

    def _check_feasibility(self) -> None:
        n, problems = self.n, []
        for c in self.schema.columns:
            if c.is_unique and c.type != "identifier":
                size = self._domain_size(c)
                if size is not None and size < n:
                    problems.append(f"Column '{c.name}' must be unique, but only {size:,} distinct values are possible "
                                    f"for the configured range/categories while {n:,} rows were requested.")
        for cols in self.unique_together:
            sizes = [self._domain_size(self.spec[x]) for x in cols]
            if all(s is not None for s in sizes):
                total = int(np.prod([float(s) for s in sizes]))
                if total < n:
                    problems.append(f"unique_together {cols} allows only {total:,} distinct combinations, fewer than the {n:,} rows requested.")
        if problems:
            raise ConstraintConflictError("Constraints cannot be satisfied:\n" + "\n".join(f"  - {p}" for p in problems))

    def _domain_size(self, c: ColumnSpec) -> int | None:
        """Number of distinct values a column can take, or None if effectively unlimited."""
        if c.is_derived:
            return None
        if c.type == "boolean":
            return 2
        if c.type == "categorical":
            return len(c.categories) if c.categories else 1
        if c.categories:
            return len(c.categories)
        if c.distribution == "constant" or (c.default is not None and c.min is None and c.max is None):
            return 1
        if c.type == "integer" and c.min is not None and c.max is not None:
            return int(np.floor(c.max) - np.ceil(c.min) + 1)
        if c.type == "date" and c.min is not None and c.max is not None:
            return int((pd.Timestamp(c.max) - pd.Timestamp(c.min)).days + 1)
        if c.type == "string" and (c.text_kind == "template" or c.pattern):
            return 10 ** c.pattern.count("#") * 26 ** c.pattern.count("?")
        return None

    # ------------------------------------------------------------------ batches
    def batch(self, m: int) -> dict[str, np.ndarray]:
        """Generate ``m`` complete rows (no identifiers, no missing values)."""
        rng = self.rng
        cols: dict[str, np.ndarray] = {}
        z_u = None
        if self.copula_cols:
            z_u = stats.norm.cdf(correlated_normals(self.corr, m, rng))
        for name in self.order:
            c = self.spec[name]
            if c.type == "identifier":
                continue
            if c.is_derived:
                cols[name] = self._derive(c, cols, m)
            elif c.type in ("integer", "float", "date", "datetime"):
                cols[name] = self._numeric(c, m, z_u, cols)
            elif c.type == "categorical":
                cols[name] = self._categorical(c, m, cols)
            elif c.type == "boolean":
                p = c.probability if c.probability is not None else 0.5
                cols[name] = rng.random(m) < p
            else:
                cols[name] = textgen.generate_text(c.text_kind, m, rng, c.pattern, c.domain)
        return cols

    def _numeric(self, c: ColumnSpec, m: int, z_u, cols: dict[str, np.ndarray]) -> np.ndarray:
        marg = self.marginals[c.name]
        if c.is_unique:
            internal = self._unique_numeric(c, marg, m)
        else:
            u = z_u[:, self.copula_cols.index(c.name)] if c.name in self.copula_cols else self.rng.random(m)
            internal = marg.values(u)
            groups = self.group_marginals.get(c.name)
            if groups:
                parent = pd.Series(cols[c.conditional_on]).astype(str).str.lower().to_numpy()
                for key, gm in groups.items():
                    mask = parent == key
                    if mask.any():
                        internal[mask] = gm.values(u[mask])
        return self._from_internal(internal, c)

    @staticmethod
    def _from_internal(internal: np.ndarray, c: ColumnSpec) -> np.ndarray:
        if c.type == "integer":
            return internal.astype("int64")
        if c.type == "float":
            return internal.astype("float64")
        return from_internal(internal, c.type).to_numpy(dtype="datetime64[ns]")

    def _unique_numeric(self, c: ColumnSpec, marg: Marginal, m: int) -> np.ndarray:
        vals = marg.values(self.rng.random(m))
        for _ in range(60):
            dup = pd.Series(vals).duplicated().to_numpy()
            if not dup.any():
                return vals
            vals[dup] = marg.values(self.rng.random(int(dup.sum())))
        dup = pd.Series(vals).duplicated().to_numpy()
        if not dup.any():
            return vals
        if marg.integer and marg.lo is not None and marg.hi is not None:
            domain = np.arange(np.ceil(marg.lo), np.floor(marg.hi) + 1)
            unused = np.setdiff1d(domain, vals[~dup])
            if len(unused) >= dup.sum():
                vals[dup] = self.rng.choice(unused, int(dup.sum()), replace=False)
                self.report.add("fallback", "distribution too concentrated to give unique values; remaining duplicates were "
                                "replaced by random unused values from the allowed range", c.name, "assumed")
                self.report.warn(f"Column '{c.name}': unique values required filling gaps uniformly; its distribution is flatter than configured.")
                return vals
        raise ConstraintConflictError(f"Could not generate {m} unique values for column '{c.name}' with its distribution.")

    def _categorical(self, c: ColumnSpec, m: int, cols: dict[str, np.ndarray]) -> np.ndarray:
        if not c.categories:
            return np.full(m, c.default, dtype=object)
        cats = np.asarray(c.categories, dtype=object)
        probs = np.asarray(c.probabilities if c.probabilities else [1 / len(cats)] * len(cats), dtype=float)
        probs = probs / probs.sum()
        if c.is_unique:
            return self.rng.choice(cats, size=m, replace=False)
        if c.conditional_on:
            parent = np.asarray(cols[c.conditional_on]).astype(str)
            out = np.empty(m, dtype=object)
            u = self.rng.random(m)
            handled = np.zeros(m, dtype=bool)
            for key, dist in c.conditional_probabilities.items():
                mask = parent == str(key)
                cat_list = [k for k in dist]
                p = np.array([dist[k] for k in cat_list], dtype=float)
                idx = np.minimum(np.searchsorted(np.cumsum(p / p.sum()), u[mask], side="right"), len(cat_list) - 1)
                out[mask] = np.asarray(cat_list, dtype=object)[idx]
                handled |= mask
            rest = ~handled
            if rest.any():
                self.report.warn(f"Column '{c.name}': parent values without a conditional model fell back to the overall distribution.")
                idx = np.minimum(np.searchsorted(np.cumsum(probs), u[rest], side="right"), len(cats) - 1)
                out[rest] = cats[idx]
            return out
        return cats[self.rng.choice(len(cats), size=m, p=probs)]

    def finalize_unique_strings(self, cols: dict[str, np.ndarray]) -> None:
        """Make unique text columns unique over the *whole* dataset (after constraint resampling)."""
        for name in self.unique_strings:
            c = self.spec[name]
            s = pd.Series(cols[name])
            dup = s.duplicated(keep="first")
            if not dup.any():
                continue
            if c.pattern:   # pattern codes are redrawn so they keep their format
                arr = cols[name].copy()
                for _ in range(100):
                    idx = np.flatnonzero(pd.Series(arr).duplicated(keep="first").to_numpy())
                    if len(idx) == 0:
                        break
                    arr[idx] = textgen.generate_text(c.text_kind, len(idx), self.rng, c.pattern, c.domain)
                else:
                    raise ConstraintConflictError(f"Could not make column '{name}' unique with pattern {c.pattern!r}.")
                cols[name] = arr
                self.report.add("fallback", f"{int(dup.sum())} duplicate code(s) were redrawn to keep the column unique", name, "assumed")
                continue
            counts = s.groupby(s).cumcount()[dup]
            out = s.copy()
            if c.text_kind == "email":
                parts = s[dup].str.rsplit("@", n=1, expand=True)
                out[dup] = parts[0] + counts.astype(str) + "@" + parts[1]
            else:
                out[dup] = s[dup] + " " + (counts + 1).astype(str)
            cols[name] = out.to_numpy(dtype=object)
            self.report.add("fallback", f"{int(dup.sum())} duplicate text value(s) made unique by appending a counter "
                            "(the synthetic vocabulary is small)", name, "assumed")

    def _derive(self, c: ColumnSpec, cols: dict[str, np.ndarray], m: int) -> np.ndarray:
        frame = pd.DataFrame({k: v for k, v in cols.items()})
        try:
            out = evaluate_expression(c.expression, frame)
        except ExpressionError as exc:
            raise GenerationError(f"Derived column '{c.name}': {exc}") from exc
        arr = np.asarray(out)
        if arr.ndim == 0:
            arr = np.full(m, arr.item(), dtype=arr.dtype if arr.dtype != object else object)
        if len(arr) != m:
            raise GenerationError(f"Derived column '{c.name}' produced {len(arr)} values for {m} rows.")
        if c.type == "boolean":
            return arr.astype(bool)
        if c.type in ("integer", "float"):
            arr = arr.astype(float)
            if not np.isfinite(arr).all():
                raise GenerationError(f"Derived column '{c.name}' produced non-finite values (division by zero or overflow?).")
            if c.type == "integer":
                return np.rint(arr).astype("int64")
            return np.round(arr, c.decimals) if c.decimals is not None else arr
        if c.type in ("date", "datetime"):
            return pd.to_datetime(pd.Series(arr)).to_numpy(dtype="datetime64[ns]")
        return arr.astype(str).astype(object)

    # ------------------------------------------------------------------ constraints
    def _frame(self, cols: dict[str, np.ndarray]) -> pd.DataFrame:
        return pd.DataFrame(cols)

    def violations(self, cols: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, int]]:
        """Boolean mask of rows that must be replaced, and counts per reason."""
        if not cols:
            return np.zeros(self.n, dtype=bool), {}
        m = len(next(iter(cols.values())))
        bad = np.zeros(m, dtype=bool)
        reasons: dict[str, int] = {}
        frame = self._frame(cols) if (self.row_constraints or self.unique_together or self.unique_cols) else None
        for con in self.row_constraints:
            ok = np.asarray(evaluate_expression(con["expression"], frame))
            ok = np.broadcast_to(ok.astype(bool), (m,))
            reasons[f"constraint: {con['expression']}"] = int((~ok).sum())
            bad |= ~ok
        for c in self.schema.columns:
            if c.is_derived:
                bad |= self._derived_limits(c, cols[c.name], reasons)
        for name in self.unique_cols:
            d = frame[name].duplicated(keep="first").to_numpy()
            reasons[f"unique: {name}"] = int(d.sum())
            bad |= d
        for cs in self.unique_together:
            d = frame.duplicated(subset=cs, keep="first").to_numpy()
            reasons[f"unique_together: {cs}"] = int(d.sum())
            bad |= d
        return bad, reasons

    def _derived_limits(self, c: ColumnSpec, arr: np.ndarray, reasons: dict[str, int]) -> np.ndarray:
        bad = np.zeros(len(arr), dtype=bool)
        if c.type in ("integer", "float"):
            if c.min is not None:
                bad |= arr < c.min
            if c.max is not None:
                bad |= arr > c.max
        elif c.type in ("date", "datetime"):
            if c.min is not None:
                bad |= arr < np.datetime64(pd.Timestamp(c.min))
            if c.max is not None:
                bad |= arr > np.datetime64(pd.Timestamp(c.max))
        if c.categories is not None and c.type == "categorical":
            bad |= ~np.isin(arr.astype(str), [str(x) for x in c.categories])
        if bad.any():
            reasons[f"derived limits: {c.name}"] = int(bad.sum())
        return bad

    def satisfy(self, cols: dict[str, np.ndarray]) -> None:
        """Replace violating rows with fresh, valid rows (targeted rejection sampling)."""
        attempts, stall, factor = 0, 0, 2
        replaced_total = 0
        initial_bad = None
        while True:
            bad, reasons = self.violations(cols)
            nbad = int(bad.sum())
            if initial_bad is None:
                initial_bad = nbad
            if nbad == 0:
                break
            if attempts >= self.settings.max_resample_attempts:
                raise ConstraintConflictError(self._explain(reasons, nbad, attempts))
            attempts += 1
            m = int(min(max(nbad * factor, 200), MAX_CANDIDATES))
            cand = self.batch(m)
            cand_bad, _ = self.violations(cand)
            ok = ~cand_bad
            ok &= ~self._collides_with_existing(cand, cols, bad)
            good = np.flatnonzero(ok)
            take = min(len(good), nbad)
            if take:
                rows = np.flatnonzero(bad)[:take]
                for k, arr in cand.items():
                    cols[k][rows] = arr[good[:take]]
                replaced_total += take
                stall = 0
            else:
                stall += 1
            factor = min(factor * 4, 100) if len(good) < nbad else factor
            if stall >= 6:
                raise ConstraintConflictError(self._explain(reasons, nbad, attempts, stalled=True))
        if initial_bad:
            self.report.add("constraint", f"{initial_bad} of {self.n} initially generated rows violated a constraint or "
                            f"uniqueness rule and were replaced by freshly generated valid rows ({attempts} round(s)). "
                            "Values are therefore drawn from the configured model conditioned on the constraints.",
                            origin="default")

    def _collides_with_existing(self, cand, cols, bad) -> np.ndarray:
        hit = np.zeros(len(next(iter(cand.values()), np.empty(0))), dtype=bool)
        keep = ~bad
        for name in self.unique_cols:
            hit |= pd.Series(cand[name]).isin(set(cols[name][keep].tolist())).to_numpy()
        for cs in self.unique_together:
            existing = pd.MultiIndex.from_arrays([cols[c][keep] for c in cs])
            mine = pd.MultiIndex.from_arrays([cand[c] for c in cs])
            hit |= mine.isin(existing)
        return hit

    def _explain(self, reasons: dict[str, int], nbad: int, attempts: int, stalled: bool = False) -> str:
        probe = self.batch(2000)
        lines = []
        for con in self.row_constraints:
            ok = np.broadcast_to(np.asarray(evaluate_expression(con["expression"], self._frame(probe))).astype(bool), (2000,))
            lines.append(f"  - constraint '{con['expression']}' is satisfied by only {ok.mean():.2%} of freshly generated rows")
        for k, v in reasons.items():
            if v and not k.startswith("constraint:"):
                lines.append(f"  - {k}: {v} violating row(s)")
        why = "no valid row could be generated" if stalled else "the maximum number of re-sampling rounds was reached"
        return (f"Could not satisfy all constraints for {self.n} rows: {nbad} row(s) still invalid after {attempts} "
                f"round(s); {why}.\n" + "\n".join(lines) +
                "\nCheck for contradictory ranges (e.g. a rule that the configured min/max can never meet), widen the "
                "ranges, relax the constraint, or reduce the number of rows if uniqueness is the problem.")

    # ------------------------------------------------------------------ finishing
    def assign_identifiers(self, cols: dict[str, np.ndarray]) -> None:
        for c in self.schema.columns:
            if c.type != "identifier":
                continue
            start = c.start if c.start is not None else 1
            nums = start + np.arange(self.n, dtype="int64")
            if (c.sequence or "sequential") == "shuffled":
                nums = self.rng.permutation(nums)
            prefix, suffix = c.prefix or "", c.suffix or ""
            width = c.width if c.width is not None else (6 if (prefix or suffix) else 0)
            if not prefix and not suffix and width == 0:
                cols[c.name] = nums
            else:
                s = pd.Series(nums.astype(str)).str.zfill(width)
                cols[c.name] = (prefix + s + suffix).to_numpy(dtype=object)

    def _missing_mask(self, c: ColumnSpec, cols: dict[str, np.ndarray]) -> np.ndarray:
        """Exact-count missing mask: completely at random, or at random given another column (MAR)."""
        n, rng = self.n, self.rng
        mask = np.zeros(n, dtype=bool)

        def pick(idx: np.ndarray, rate: float) -> None:
            k = int(round(rate * len(idx)))
            if k:
                mask[rng.choice(idx, size=k, replace=False)] = True

        m = c.missing_model
        if not m:
            pick(np.arange(n), c.missing_rate)
            if c.missing_rate > 0 and not mask.any():
                self.report.warn(f"Column '{c.name}': missing rate {c.missing_rate:.2%} of {n} rows rounds to 0 missing values.")
        elif m["type"] == "group":
            ctrl = pd.Series(cols[m["column"]]).astype(str).str.lower().to_numpy()
            default = m.get("default", c.missing_rate)
            known = {str(k).lower() for k in m["rates"]}
            for key, rate in m["rates"].items():
                pick(np.flatnonzero(ctrl == str(key).lower()), rate)
            pick(np.flatnonzero(~np.isin(ctrl, list(known))), default)
        else:
            x = np.asarray(cols[m["column"]], dtype=float)
            cond = {">": x > m["value"], ">=": x >= m["value"], "<": x < m["value"], "<=": x <= m["value"]}[m["op"]]
            pick(np.flatnonzero(cond), m["rate_if_true"])
            pick(np.flatnonzero(~cond), m["rate_if_false"])
        return mask

    def apply_missing(self, cols: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        masks: dict[str, np.ndarray] = {}
        for name in self.order:
            c = self.spec[name]
            if c.is_derived:
                m = np.zeros(self.n, dtype=bool)
                for r in referenced_columns(c.expression):
                    m |= masks.get(r, False)
                masks[name] = m
            elif c.is_nullable and c.has_missing and self.settings.missing_policy == "schema" and c.type != "identifier":
                masks[name] = self._missing_mask(c, cols)
            else:
                masks[name] = np.zeros(self.n, dtype=bool)
        if self.settings.missing_policy == "none" and any(c.has_missing for c in self.schema.columns):
            self.report.add("missing", "missing_policy='none': all configured missing values were ignored", origin="user")
        return masks

    def to_frame(self, cols: dict[str, np.ndarray], masks: dict[str, np.ndarray]) -> pd.DataFrame:
        data: dict[str, Any] = {}
        for c in self.schema.columns:
            arr, mask = cols[c.name], masks.get(c.name, np.zeros(self.n, dtype=bool))
            has = bool(mask.any())
            if c.type == "integer":
                data[c.name] = pd.Series(pd.array(arr, dtype="Int64")).mask(mask) if has else arr
            elif c.type == "float":
                a = arr.astype("float64").copy()
                a[mask] = np.nan
                data[c.name] = a
            elif c.type == "boolean":
                data[c.name] = pd.Series(pd.array(arr, dtype="boolean")).mask(mask) if has else arr
            elif c.type == "identifier" and arr.dtype.kind in "iu":
                data[c.name] = arr
            elif c.type in ("date", "datetime"):
                s = pd.Series(arr.astype("datetime64[ns]"))
                data[c.name] = s.mask(mask)
            else:
                a = arr.astype(object).copy()
                a[mask] = None
                data[c.name] = a
        df = pd.DataFrame(data)
        df.attrs["dataseed_date_formats"] = {c.name: c.effective_date_format() for c in self.schema.columns
                                             if c.effective_date_format()}
        return df


def generate_from_schema(schema: Schema, n_rows: int | None = None, seed: int | None = None,
                         settings: GenerationSettings | None = None, *,
                         _mode: str = "schema", _inference: InferenceResult | None = None) -> GenerationResult:
    """Generate exactly ``n_rows`` synthetic rows from ``schema``.

    Args:
        schema: A validated :class:`Schema` (see :func:`dataseed.schema_from_dict`).
        n_rows: Number of rows; defaults to ``schema.generation['n_rows']`` if present.
        seed: Random seed. ``None`` picks a random seed (it is recorded in the report).
        settings: Optional :class:`GenerationSettings` (max rows, strictness, overrides...).

    Returns:
        :class:`GenerationResult` with the DataFrame, the report of decisions, and validation.

    Raises:
        GenerationError / ConstraintConflictError / SchemaError: invalid request or impossible constraints.
    """
    settings = settings or GenerationSettings()
    if n_rows is None:
        n_rows = schema.n_rows
    if n_rows is None:
        raise GenerationError("The number of rows is required (pass n_rows or set generation.n_rows in the schema).")
    if isinstance(n_rows, bool) or not isinstance(n_rows, (int, np.integer)) or n_rows < 1:
        raise GenerationError(f"n_rows must be a positive integer, got {n_rows!r}.")
    if n_rows > settings.max_rows:
        raise GenerationError(f"Requested {n_rows:,} rows, above the configured maximum of {settings.max_rows:,}. "
                              "Raise GenerationSettings.max_rows (mind memory use) or request fewer rows.")
    n_rows = int(n_rows)
    if seed is None:
        seed = secrets.randbits(32)
    report = GenerationReport(mode=_mode, dataset_name=schema.dataset_name, n_rows=n_rows, seed=int(seed))
    schema = _apply_overrides(schema, settings.overrides, report)
    schema.validate()
    if _inference is not None:
        report.warnings.extend(w for w in _inference.warnings if w not in report.warnings)
        for a in _inference.ambiguities:
            report.warn(a)
        for note in _inference.relationship_notes:
            report.add("relationship", note, origin="inferred")
        for conv in _inference.conversions:
            report.add("info", conv, origin="inferred")
    rng = np.random.default_rng(int(seed))
    eng = _Engine(schema, n_rows, rng, settings, report)
    eng.prepare()
    cols = eng.batch(n_rows)
    eng.satisfy(cols)
    eng.finalize_unique_strings(cols)
    eng.assign_identifiers(cols)
    masks = eng.apply_missing(cols)
    df = eng.to_frame(cols, masks)[schema.column_names]
    if len(df) != n_rows:  # pragma: no cover - internal invariant
        raise GenerationError(f"Internal error: produced {len(df)} rows instead of {n_rows}.")
    validation = validate_dataset(df, schema, expected_rows=n_rows)
    if not validation.is_valid:
        failures = "; ".join(f"{c.name}[{c.column or ''}]: {c.message}" for c in validation.errors[:5])
        if settings.strict:
            raise ConstraintConflictError(f"The generated data failed validation: {failures}")
        report.warn(f"Generated data failed validation (strict=False): {failures}")
    for c in validation.failed:
        if c.severity == "warning":
            report.warn(f"{c.name}: {c.message}")
    return GenerationResult(dataframe=df, schema=schema, report=report, validation=validation, inference=_inference)


def generate_from_sample(sample: pd.DataFrame, n_rows: int, seed: int | None = None,
                         settings: GenerationSettings | None = None, dataset_name: str = "dataset"
                         ) -> GenerationResult:
    """Infer a schema from ``sample`` and generate ``n_rows`` new synthetic rows.

    Distributions, category frequencies, missing rates, correlations (>= 30 sample rows) and
    perfectly-fitting rules (e.g. ``passed = marks >= 40``) are estimated from the sample; see
    ``result.inference`` and ``result.report`` for every assumption and warning. Rows of the
    sample are never copied into the output.
    """
    settings = settings or GenerationSettings()
    inference = infer_schema(sample, dataset_name=dataset_name, settings=settings)
    return generate_from_schema(inference.schema, n_rows, seed, settings, _mode="sample", _inference=inference)
