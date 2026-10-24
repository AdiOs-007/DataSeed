"""Schema definition, validation and (de)serialisation.

A :class:`Schema` describes a dataset: columns, relationships (correlations), constraints and
optional generation settings. It can be built in Python, loaded from JSON, produced by
:func:`dataseed.infer_schema` (from a sample) or by :func:`dataseed.parse_description`.
"""
from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .exceptions import CircularDependencyError, ExpressionError, SchemaError
from .expressions import referenced_columns

VALID_TYPES = ("integer", "float", "categorical", "boolean", "date", "datetime", "identifier", "string")
TYPE_ALIASES = {
    "int": "integer", "number": "float", "numeric": "float", "double": "float",
    "category": "categorical", "bool": "boolean", "text": "string", "str": "string",
    "id": "identifier", "timestamp": "datetime",
}
NUMERIC_TYPES = ("integer", "float")
TIME_TYPES = ("date", "datetime")
NUMERIC_LIKE = NUMERIC_TYPES + TIME_TYPES
DISTRIBUTIONS = ("auto", "normal", "uniform", "lognormal", "empirical", "constant")
TEXT_KINDS = ("word", "sentence", "name", "first_name", "last_name", "email", "phone", "address",
              "company", "city", "username", "template")
SEQUENCES = ("sequential", "shuffled")
DEFAULT_DATE_FORMAT = "%Y-%m-%d"
DEFAULT_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
RELATIONSHIP_TYPES = ("correlation", "derived", "dependency")
MISSING_MODEL_TYPES = ("group", "threshold")
THRESHOLD_OPS = (">", ">=", "<", "<=")
CONDITIONAL_PARAM_KEYS = ("mean", "std", "quantiles")
CONSTRAINT_TYPES = ("expression", "unique_together")
GENERATION_KEYS = ("n_rows", "seed")

_COMMON = {"name", "type", "description", "nullable", "missing_rate", "missing_model", "unique", "default",
           "expression", "depends_on", "source", "confidence", "n_observed"}
_ALLOWED = {
    "integer": _COMMON | {"min", "max", "distribution", "mean", "std", "categories",
                          "probabilities", "quantiles", "conditional_on", "conditional_params"},
    "float": _COMMON | {"min", "max", "distribution", "mean", "std", "categories",
                        "probabilities", "quantiles", "decimals", "conditional_on", "conditional_params"},
    "categorical": _COMMON | {"categories", "probabilities", "conditional_on",
                              "conditional_probabilities"},
    "boolean": _COMMON | {"probability"},
    "date": _COMMON | {"min", "max", "distribution", "quantiles", "date_format"},
    "datetime": _COMMON | {"min", "max", "distribution", "quantiles", "date_format"},
    "identifier": _COMMON | {"prefix", "suffix", "width", "start", "sequence"},
    "string": _COMMON | {"text_kind", "pattern", "domain"},
}


@dataclass
class ColumnSpec:
    """Definition of one column. Only properties relevant to ``type`` may be set."""

    name: str
    type: str
    description: str = ""
    nullable: bool | None = None          # None -> True iff missing_rate > 0
    missing_rate: float = 0.0
    missing_model: dict | None = None     # missing-at-random pattern (see MISSING_MODEL_TYPES)
    unique: bool | None = None            # None -> True for identifiers, else False
    default: Any = None                   # constant value when nothing else defines the column
    expression: str | None = None         # makes the column derived (see dataseed.expressions)
    depends_on: list[str] | None = None   # extra ordering dependencies
    # numeric / time
    min: Any = None
    max: Any = None
    distribution: str = "auto"
    mean: float | None = None
    std: float | None = None
    decimals: int | None = None
    quantiles: list | None = None         # fitted quantile function (from sample inference)
    date_format: str | None = None
    # categorical (also discrete numeric)
    categories: list | None = None
    probabilities: list[float] | None = None
    conditional_on: str | None = None
    conditional_probabilities: dict | None = None   # categorical: {parent_value: {category: p}}
    conditional_params: dict | None = None          # numeric: {parent_value: {"mean":..,"std":..} | {"quantiles":[..]}}
    # boolean
    probability: float | None = None      # P(True)
    # identifier
    prefix: str | None = None
    suffix: str | None = None
    width: int | None = None
    start: int | None = None
    sequence: str | None = None
    # string
    text_kind: str | None = None
    pattern: str | None = None
    domain: str | None = None
    # provenance
    source: str = "user"                  # user | inferred | parsed
    confidence: str | None = None
    n_observed: int | None = None

    # ------------------------------------------------------------------ helpers
    @property
    def is_derived(self) -> bool:
        return self.expression is not None

    @property
    def is_numeric_like(self) -> bool:
        return self.type in NUMERIC_LIKE

    @property
    def is_unique(self) -> bool:
        return bool(self.unique) if self.unique is not None else self.type == "identifier"

    @property
    def has_missing(self) -> bool:
        """True if values can be missing (constant rate or a missing_model)."""
        return self.missing_rate > 0 or bool(self.missing_model)

    @property
    def is_nullable(self) -> bool:
        return bool(self.nullable) if self.nullable is not None else self.has_missing

    def effective_date_format(self) -> str | None:
        if self.type == "date":
            return self.date_format or DEFAULT_DATE_FORMAT
        if self.type == "datetime":
            return self.date_format or DEFAULT_DATETIME_FORMAT
        return None

    def to_dict(self) -> dict:
        default = {f.name: f.default for f in dataclasses.fields(ColumnSpec) if f.default is not dataclasses.MISSING}
        out: dict[str, Any] = {"name": self.name, "type": self.type}
        for f in dataclasses.fields(self):
            if f.name in ("name", "type"):
                continue
            v = getattr(self, f.name)
            if v == default.get(f.name) or (f.name == "source" and v == "user"):
                continue
            out[f.name] = v
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "ColumnSpec":
        if not isinstance(d, dict):
            raise SchemaError("Each column definition must be a JSON object.")
        d = dict(d)
        name = d.get("name")
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(d) - known)
        if unknown:
            raise SchemaError(f"Column {name!r} has unknown properties: {unknown}. Allowed: {sorted(known)}")
        if "type" not in d:
            raise SchemaError(f"Column {name!r} is missing required property 'type'.")
        t = str(d["type"]).strip().lower()
        d["type"] = TYPE_ALIASES.get(t, t)
        if "name" not in d:
            raise SchemaError("A column definition is missing required property 'name'.")
        probs = d.get("probabilities")
        if isinstance(probs, dict):          # {"A": 0.5, "B": 0.5}
            cats = d.get("categories")
            if cats is None:
                d["categories"] = list(probs)
            elif set(map(str, cats)) != set(map(str, probs)):
                raise SchemaError(f"Column {name!r}: probability keys do not match categories.")
            d["probabilities"] = [probs[k] if k in probs else probs[str(k)] for k in d["categories"]]
        return cls(**d)


@dataclass
class Schema:
    """A validated description of a dataset."""

    dataset_name: str
    columns: list[ColumnSpec]
    relationships: list[dict] = field(default_factory=list)
    constraints: list[dict] = field(default_factory=list)
    generation: dict = field(default_factory=dict)
    description: str = ""

    # ------------------------------------------------------------------ access
    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    def column(self, name: str) -> ColumnSpec:
        for c in self.columns:
            if c.name == name:
                return c
        raise KeyError(f"No column named {name!r} in schema {self.dataset_name!r}.")

    @property
    def n_rows(self) -> int | None:
        return self.generation.get("n_rows")

    def copy(self) -> "Schema":
        return Schema.from_dict(self.to_dict(), validate=False)

    # ------------------------------------------------------------------ validation
    def validate(self) -> None:
        """Raise :class:`SchemaError` listing every problem found."""
        problems = collect_schema_problems(self)
        if problems:
            raise SchemaError(f"Schema {self.dataset_name!r} is invalid:", problems)
        self.generation_order()   # raises CircularDependencyError

    def dependencies(self, col: ColumnSpec) -> list[str]:
        deps: list[str] = []
        if col.expression:
            deps += referenced_columns(col.expression)
        deps += col.depends_on or []
        if col.conditional_on:
            deps.append(col.conditional_on)
        return list(dict.fromkeys(deps))

    def generation_order(self) -> list[str]:
        """Column names ordered so each column comes after the columns it depends on."""
        names = self.column_names
        index = {n: i for i, n in enumerate(names)}
        deps = {c.name: [d for d in self.dependencies(c) if d in index] for c in self.columns}
        done: list[str] = []
        remaining = list(names)
        while remaining:
            ready = [n for n in remaining if all(d in done for d in deps[n])]
            if not ready:
                raise CircularDependencyError(
                    "Circular dependency between columns: " + _find_cycle(deps, remaining))
            n = min(ready, key=index.get)
            done.append(n)
            remaining.remove(n)
        return done

    # ------------------------------------------------------------------ serialisation
    def to_dict(self) -> dict:
        out: dict[str, Any] = {"dataset_name": self.dataset_name}
        if self.description:
            out["description"] = self.description
        if self.generation:
            out["generation"] = dict(self.generation)
        out["columns"] = [c.to_dict() for c in self.columns]
        out["relationships"] = [dict(r) for r in self.relationships]
        if self.constraints:
            out["constraints"] = [dict(c) for c in self.constraints]
        return out

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, default=_json_default)

    @classmethod
    def from_dict(cls, d: dict, validate: bool = True) -> "Schema":
        if not isinstance(d, dict):
            raise SchemaError("A schema must be a JSON object.")
        allowed = {"dataset_name", "description", "columns", "relationships", "constraints", "generation"}
        unknown = sorted(set(d) - allowed)
        if unknown:
            raise SchemaError(f"Unknown top-level schema properties: {unknown}. Allowed: {sorted(allowed)}")
        if not isinstance(d.get("columns"), list) or not d["columns"]:
            raise SchemaError("A schema needs a non-empty 'columns' list.")
        cols = [ColumnSpec.from_dict(c) for c in d["columns"]]
        schema = cls(
            dataset_name=str(d.get("dataset_name") or "dataset"),
            columns=cols,
            relationships=[],
            constraints=[dict(c) for c in (d.get("constraints") or [])],
            generation=dict(d.get("generation") or {}),
            description=str(d.get("description") or ""),
        )
        problems: list[str] = []
        for rel in d.get("relationships") or []:
            _merge_relationship(schema, rel, problems)
        if problems:
            raise SchemaError(f"Schema {schema.dataset_name!r} is invalid:", problems)
        if validate:
            schema.validate()
        return schema


def schema_from_dict(data: dict) -> Schema:
    """Build and validate a :class:`Schema` from a plain dict (e.g. parsed JSON)."""
    return Schema.from_dict(data)


def schema_to_dict(schema: Schema) -> dict:
    """Return the JSON-serialisable dict form of ``schema``."""
    return schema.to_dict()


# ---------------------------------------------------------------------- internals
def _json_default(o: Any):
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    raise TypeError(f"Not JSON serialisable: {type(o)}")


def _find_cycle(deps: dict[str, list[str]], remaining: list[str]) -> str:
    rem = set(remaining)
    for start in remaining:
        path, seen = [start], {start}
        cur = start
        while True:
            nxt = next((d for d in deps[cur] if d in rem), None)
            if nxt is None:
                break
            if nxt in seen:
                cyc = path[path.index(nxt):] + [nxt]
                return " -> ".join(cyc)
            path.append(nxt)
            seen.add(nxt)
            cur = nxt
    return ", ".join(remaining)


def _merge_relationship(schema: Schema, rel: dict, problems: list[str]) -> None:
    """Normalise a relationship: ``derived``/``dependency`` merge into columns; others are kept."""
    if not isinstance(rel, dict) or "type" not in rel:
        problems.append(f"Relationship {rel!r} must be an object with a 'type'.")
        return
    t = rel["type"]
    if t not in RELATIONSHIP_TYPES:
        problems.append(f"Unknown relationship type {t!r}. Supported: {list(RELATIONSHIP_TYPES)}")
        return
    if t == "correlation":
        extra = set(rel) - {"type", "columns", "value", "source", "n_observations", "note"}
        if extra:
            problems.append(f"Correlation relationship has unknown properties: {sorted(extra)}")
        schema.relationships.append(dict(rel))
        return
    target = rel.get("column")
    try:
        col = schema.column(target)
    except KeyError:
        problems.append(f"Relationship of type '{t}' refers to unknown column {target!r}.")
        return
    if t == "derived":
        expr = rel.get("expression")
        if col.expression not in (None, expr):
            problems.append(f"Column {target!r} has two different derivation rules.")
        col.expression = expr
    else:
        deps = rel.get("depends_on")
        if not isinstance(deps, list):
            problems.append(f"Dependency relationship for {target!r} needs a 'depends_on' list.")
            return
        col.depends_on = list(dict.fromkeys((col.depends_on or []) + deps))


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def parse_bound(value: Any, col_type: str) -> float | None:
    """Convert a min/max bound to a number (numeric types) or epoch seconds (time types)."""
    if value is None:
        return None
    if col_type in TIME_TYPES:
        return pd.Timestamp(value).timestamp() if not isinstance(value, pd.Timestamp) else value.timestamp()
    return float(value)


def _check_probabilities(c: ColumnSpec, p: list[str]) -> None:
    probs = c.probabilities
    if probs is None:
        return
    if not isinstance(probs, list) or c.categories is None or len(probs) != len(c.categories):
        p.append(f"Column {c.name!r}: 'probabilities' must be a list with one value per category.")
        return
    if any(not _is_num(x) or x < 0 for x in probs):
        p.append(f"Column {c.name!r}: probabilities must be non-negative numbers.")
    elif abs(sum(probs) - 1.0) > 1e-3:
        p.append(f"Column {c.name!r}: probabilities sum to {sum(probs):.4f}, expected 1.0.")


def _column_problems(c: ColumnSpec, by_name: dict[str, ColumnSpec]) -> list[str]:
    p: list[str] = []
    label = f"Column {c.name!r}"
    if not isinstance(c.name, str) or not c.name.strip():
        return ["A column has an empty or non-string name."]
    if c.type not in VALID_TYPES:
        return [f"{label}: invalid type {c.type!r}. Valid types: {list(VALID_TYPES)}"]
    # properties that do not belong to this type
    defaults = {f.name: f.default for f in dataclasses.fields(ColumnSpec) if f.default is not dataclasses.MISSING}
    for f in dataclasses.fields(c):
        if f.name not in _ALLOWED[c.type] and getattr(c, f.name) != defaults.get(f.name):
            p.append(f"{label}: property '{f.name}' is not valid for type '{c.type}'.")
    if not isinstance(c.missing_rate, (int, float)) or isinstance(c.missing_rate, bool) or not 0 <= c.missing_rate < 1:
        p.append(f"{label}: missing_rate must be a number in [0, 1).")
        return p
    if c.nullable is False and c.has_missing:
        p.append(f"{label}: nullable is false but the column is configured to have missing values.")
    if c.distribution not in DISTRIBUTIONS:
        p.append(f"{label}: unknown distribution {c.distribution!r}. Valid: {list(DISTRIBUTIONS)}")
    # derived columns
    if c.expression is not None:
        try:
            refs = referenced_columns(c.expression)
        except ExpressionError as exc:
            p.append(f"{label}: {exc}")
            refs = []
        for r in refs:
            if r == c.name:
                p.append(f"{label}: expression refers to itself.")
            elif r not in by_name:
                p.append(f"{label}: expression refers to unknown column {r!r}.")
            elif by_name[r].type == "identifier":
                p.append(f"{label}: expressions cannot use identifier column {r!r}.")
            elif c.nullable is False and by_name[r].has_missing and not by_name[r].is_derived:
                p.append(f"{label}: is declared not nullable but is derived from {r!r}, which can have "
                         f"missing values. Allow nulls on {c.name!r} or remove the missing configuration of {r!r}.")
        if c.type == "identifier":
            p.append(f"{label}: identifier columns cannot be derived.")
        if c.missing_rate > 0 or c.missing_model:
            p.append(f"{label}: derived columns inherit missing values from their inputs; remove missing_rate/missing_model.")
        if c.distribution != "auto" or c.mean is not None or c.std is not None or c.probabilities or c.conditional_on:
            p.append(f"{label}: derived columns cannot also define a distribution, mean/std, probabilities or conditional_on.")
        return p
    for d in c.depends_on or []:
        if d not in by_name:
            p.append(f"{label}: depends_on refers to unknown column {d!r}.")
        elif d == c.name:
            p.append(f"{label}: depends_on refers to itself.")
    if c.missing_model is not None:
        _missing_model_problems(c, p, label, by_name)
    if c.type in NUMERIC_TYPES:
        _numeric_problems(c, p, label)
        _numeric_conditional_problems(c, p, label, by_name)
    elif c.type in TIME_TYPES:
        _time_problems(c, p, label)
    elif c.type == "categorical":
        _categorical_problems(c, p, label, by_name)
    elif c.type == "boolean":
        if c.probability is not None and not (_is_num(c.probability) and 0 <= c.probability <= 1):
            p.append(f"{label}: probability must be between 0 and 1.")
    elif c.type == "identifier":
        if c.unique is False:
            p.append(f"{label}: identifier columns are always unique; remove 'unique': false.")
        if c.nullable is True or c.has_missing:
            p.append(f"{label}: identifier columns cannot have missing values.")
        for k in ("width", "start"):
            v = getattr(c, k)
            if v is not None and (not isinstance(v, int) or isinstance(v, bool) or v < 0):
                p.append(f"{label}: {k} must be a non-negative integer.")
        if c.sequence is not None and c.sequence not in SEQUENCES:
            p.append(f"{label}: sequence must be one of {list(SEQUENCES)}.")
    elif c.type == "string":
        if c.text_kind is not None and c.text_kind not in TEXT_KINDS:
            p.append(f"{label}: unknown text_kind {c.text_kind!r}. Valid: {list(TEXT_KINDS)}")
        if (c.text_kind == "template" or (c.pattern and not c.text_kind)) and not c.pattern:
            p.append(f"{label}: text_kind 'template' requires a 'pattern' such as 'AB-####'.")
    return p


def _missing_model_problems(c: ColumnSpec, p: list[str], label: str, by_name: dict[str, ColumnSpec]) -> None:
    m = c.missing_model
    if not isinstance(m, dict) or m.get("type") not in MISSING_MODEL_TYPES:
        p.append(f"{label}: missing_model needs a 'type' in {list(MISSING_MODEL_TYPES)}.")
        return
    if c.type == "identifier":
        return
    ctrl = by_name.get(m.get("column"))
    if ctrl is None or ctrl.name == c.name:
        p.append(f"{label}: missing_model 'column' must name a different, existing column.")
        return
    def rate_ok(v):
        return _is_num(v) and 0 <= v < 1
    if m["type"] == "group":
        rates = m.get("rates")
        if not isinstance(rates, dict) or not rates or any(not rate_ok(v) for v in rates.values()):
            p.append(f"{label}: missing_model 'rates' must map values of {ctrl.name!r} to rates in [0, 1).")
        if "default" in m and not rate_ok(m["default"]):
            p.append(f"{label}: missing_model 'default' must be in [0, 1).")
    else:
        if ctrl.type not in NUMERIC_TYPES:
            p.append(f"{label}: threshold missing_model needs an integer/float control column.")
        if m.get("op") not in THRESHOLD_OPS or not _is_num(m.get("value")):
            p.append(f"{label}: threshold missing_model needs 'op' in {list(THRESHOLD_OPS)} and a numeric 'value'.")
        for k in ("rate_if_true", "rate_if_false"):
            if not rate_ok(m.get(k)):
                p.append(f"{label}: missing_model '{k}' must be in [0, 1).")


def _numeric_conditional_problems(c: ColumnSpec, p: list[str], label: str, by_name: dict[str, ColumnSpec]) -> None:
    if c.conditional_on is None and c.conditional_params is None:
        return
    parent = by_name.get(c.conditional_on) if c.conditional_on else None
    if parent is None:
        p.append(f"{label}: conditional_on must name an existing column.")
        return
    if parent.type not in ("categorical", "boolean") or parent.is_derived:
        p.append(f"{label}: conditional_on column must be a non-derived categorical or boolean column.")
    if c.categories is not None or c.is_unique:
        p.append(f"{label}: conditional_params cannot be combined with discrete 'categories' or 'unique'.")
    cp = c.conditional_params
    if not isinstance(cp, dict) or not cp:
        p.append(f"{label}: conditional_on needs conditional_params {{parent_value: {{'mean':..,'std':..}}}}.")
        return
    for pv, params in cp.items():
        if not isinstance(params, dict) or not params or set(params) - set(CONDITIONAL_PARAM_KEYS):
            p.append(f"{label}: conditional_params[{pv!r}] may only contain {list(CONDITIONAL_PARAM_KEYS)}.")
        elif "quantiles" in params:
            q = params["quantiles"]
            if not isinstance(q, list) or len(q) < 2 or any(not _is_num(v) for v in q) or q != sorted(q):
                p.append(f"{label}: conditional_params[{pv!r}].quantiles must be a sorted list of >= 2 numbers.")
        elif not (_is_num(params.get("mean")) and _is_num(params.get("std")) and params["std"] >= 0):
            p.append(f"{label}: conditional_params[{pv!r}] needs numeric mean and std >= 0.")
    if c.distribution in ("uniform", "constant"):
        p.append(f"{label}: conditional_params need a normal, lognormal or empirical base distribution.")


def _numeric_problems(c: ColumnSpec, p: list[str], label: str) -> None:
    for k in ("min", "max", "mean", "std"):
        v = getattr(c, k)
        if v is not None and not _is_num(v):
            p.append(f"{label}: '{k}' must be a finite number, got {v!r}.")
            return
    if c.min is not None and c.max is not None and c.min > c.max:
        p.append(f"{label}: min ({c.min}) is greater than max ({c.max}).")
        return
    if c.type == "integer" and c.min is not None and c.max is not None and math.ceil(c.min) > math.floor(c.max):
        p.append(f"{label}: no integer lies between min={c.min} and max={c.max}.")
    if c.std is not None and c.std < 0:
        p.append(f"{label}: std must be >= 0.")
    if c.decimals is not None and (not isinstance(c.decimals, int) or c.decimals < 0):
        p.append(f"{label}: decimals must be a non-negative integer.")
    if c.categories is not None:
        if not c.categories or any(not _is_num(v) for v in c.categories):
            p.append(f"{label}: numeric 'categories' must be a non-empty list of numbers.")
        elif len(set(c.categories)) != len(c.categories):
            p.append(f"{label}: duplicate values in categories.")
        elif c.type == "integer" and any(float(v) != int(v) for v in c.categories):
            p.append(f"{label}: integer column has non-integer allowed values.")
        _check_probabilities(c, p)
        if c.categories and all(_is_num(v) for v in c.categories):
            lo, hi = c.min, c.max
            if (lo is not None and min(c.categories) < lo) or (hi is not None and max(c.categories) > hi):
                p.append(f"{label}: allowed values fall outside min/max.")
        return
    if c.probabilities is not None:
        p.append(f"{label}: 'probabilities' needs 'categories'.")
    d = c.distribution
    if d == "uniform" and (c.min is None or c.max is None):
        p.append(f"{label}: uniform distribution needs both min and max.")
    elif d == "normal" and (c.mean is None or c.std is None) and (c.min is None or c.max is None):
        p.append(f"{label}: normal distribution needs mean and std (or both min and max to derive them).")
    elif d == "lognormal":
        if c.mean is None or c.std is None or c.mean <= 0 or c.std <= 0:
            p.append(f"{label}: lognormal needs mean > 0 and std > 0.")
        if c.min is not None and c.min < 0:
            p.append(f"{label}: lognormal values are positive; min must be >= 0.")
    elif d == "empirical":
        _check_quantiles(c, p, label)
    elif d == "constant" and c.default is None and not (c.min is not None and c.min == c.max):
        p.append(f"{label}: constant distribution needs 'default' (or min == max).")
    elif d == "auto":
        has_info = (c.min is not None and c.max is not None) or (c.mean is not None and c.std is not None) \
            or c.quantiles or c.default is not None
        if not has_info:
            p.append(f"{label}: cannot generate values - provide both min and max, or mean and std, "
                     "or categories, or quantiles, or a default.")
    if c.quantiles is not None and d != "empirical":
        p.append(f"{label}: 'quantiles' are only used with distribution 'empirical'.")
    if c.mean is not None and c.min is not None and c.max is not None and not c.min <= c.mean <= c.max:
        p.append(f"{label}: mean {c.mean} lies outside [{c.min}, {c.max}].")
    if c.default is not None and not _is_num(c.default):
        p.append(f"{label}: default must be a number.")
    elif c.default is not None and ((c.min is not None and c.default < c.min) or (c.max is not None and c.default > c.max)):
        p.append(f"{label}: default lies outside min/max.")


def _check_quantiles(c: ColumnSpec, p: list[str], label: str) -> None:
    q = c.quantiles
    if not isinstance(q, list) or len(q) < 2:
        p.append(f"{label}: empirical distribution needs a 'quantiles' list (>= 2 values). "
                 "Use dataseed.infer_schema on a sample to create one.")
        return
    try:
        vals = [parse_bound(v, c.type) for v in q]
    except (ValueError, TypeError):
        p.append(f"{label}: quantiles contain unparseable values.")
        return
    if any(b < a for a, b in zip(vals, vals[1:])):
        p.append(f"{label}: quantiles must be sorted ascending.")


def _time_problems(c: ColumnSpec, p: list[str], label: str) -> None:
    lo = hi = None
    try:
        lo, hi = parse_bound(c.min, c.type), parse_bound(c.max, c.type)
    except (ValueError, TypeError):
        p.append(f"{label}: min/max must be ISO date strings like '2020-01-31'.")
        return
    if lo is not None and hi is not None and lo > hi:
        p.append(f"{label}: min date is after max date.")
    if c.distribution not in ("auto", "uniform", "empirical", "constant"):
        p.append(f"{label}: date columns support distributions 'uniform' and 'empirical'.")
    if c.distribution == "empirical":
        _check_quantiles(c, p, label)
    elif c.default is None and (lo is None or hi is None):
        p.append(f"{label}: date columns need both min and max (ISO strings).")
    if c.date_format is not None and not isinstance(c.date_format, str):
        p.append(f"{label}: date_format must be a strftime string.")


def _categorical_problems(c: ColumnSpec, p: list[str], label: str, by_name: dict[str, ColumnSpec]) -> None:
    if not c.categories:
        if c.default is None:
            p.append(f"{label}: categorical columns need a non-empty 'categories' list.")
        return
    if len(set(map(str, c.categories))) != len(c.categories):
        p.append(f"{label}: duplicate categories.")
    if any(str(v).strip() == "" for v in c.categories):
        p.append(f"{label}: blank category value.")
    _check_probabilities(c, p)
    if c.conditional_on is None and c.conditional_probabilities is not None:
        p.append(f"{label}: conditional_probabilities needs conditional_on.")
    if c.conditional_on is not None:
        parent = by_name.get(c.conditional_on)
        if parent is None:
            p.append(f"{label}: conditional_on refers to unknown column {c.conditional_on!r}.")
        elif parent.type not in ("categorical", "boolean") or parent.is_derived:
            p.append(f"{label}: conditional_on column must be a non-derived categorical or boolean column.")
        cp = c.conditional_probabilities
        if not isinstance(cp, dict) or not cp:
            p.append(f"{label}: conditional_on needs conditional_probabilities {{parent_value: {{category: p}}}}.")
        else:
            for pv, dist in cp.items():
                if not isinstance(dist, dict) or set(map(str, dist)) - set(map(str, c.categories)):
                    p.append(f"{label}: conditional distribution for {pv!r} uses unknown categories.")
                elif any(not _is_num(v) or v < 0 for v in dist.values()) or abs(sum(dist.values()) - 1) > 1e-3:
                    p.append(f"{label}: conditional probabilities for {pv!r} must be >= 0 and sum to 1.")


def collect_schema_problems(schema: Schema) -> list[str]:
    """Return a list of human-readable problems (empty list = schema is valid)."""
    p: list[str] = []
    if not schema.columns:
        return ["Schema has no columns."]
    names = [c.name for c in schema.columns]
    for n in {n for n in names if names.count(n) > 1}:
        p.append(f"Duplicate column definition: {n!r}.")
    by_name = {c.name: c for c in schema.columns}
    for c in schema.columns:
        p += _column_problems(c, by_name)
    for k in schema.generation:
        if k not in GENERATION_KEYS:
            p.append(f"Unknown generation setting {k!r}. Supported: {list(GENERATION_KEYS)}")
    n = schema.generation.get("n_rows")
    if n is not None and (not isinstance(n, int) or isinstance(n, bool) or n < 1):
        p.append("generation.n_rows must be a positive integer.")
    seen_pairs: dict[frozenset, float] = {}
    for rel in schema.relationships:
        _relationship_problems(rel, by_name, seen_pairs, p)
    for con in schema.constraints:
        _constraint_problems(con, by_name, p)
    return p


def _relationship_problems(rel: dict, by_name: dict, seen: dict, p: list[str]) -> None:
    cols = rel.get("columns")
    if not isinstance(cols, list) or len(cols) != 2 or cols[0] == cols[1]:
        p.append(f"Correlation relationship needs two different columns: {rel!r}")
        return
    v = rel.get("value")
    if not _is_num(v) or not -0.99 <= v <= 0.99:
        p.append(f"Correlation between {cols} must be a number in [-0.99, 0.99], got {v!r}.")
    for name in cols:
        c = by_name.get(name)
        if c is None:
            p.append(f"Correlation refers to unknown column {name!r}.")
        elif c.type not in NUMERIC_LIKE or c.is_derived or c.is_unique or c.categories:
            reason = ("derived" if c.is_derived else "unique" if c.is_unique else
                      "discrete (has categories)" if c.categories else f"of type '{c.type}'")
            p.append(f"Correlation cannot use column {name!r}: it is {reason}. Correlations apply to "
                     "continuous integer/float/date columns that are generated directly.")
    key = frozenset(cols)
    if key in seen and _is_num(v) and abs(seen[key] - v) > 1e-9:
        p.append(f"Conflicting correlation values for {cols}: {seen[key]} and {v}.")
    seen[key] = v if _is_num(v) else 0.0


def _constraint_problems(con: dict, by_name: dict, p: list[str]) -> None:
    t = con.get("type") if isinstance(con, dict) else None
    if t not in CONSTRAINT_TYPES:
        p.append(f"Constraint {con!r} needs a 'type' in {list(CONSTRAINT_TYPES)}.")
        return
    if t == "expression":
        try:
            refs = referenced_columns(con.get("expression"))
        except ExpressionError as exc:
            p.append(f"Constraint {con!r}: {exc}")
            return
        for r in refs:
            if r not in by_name:
                p.append(f"Constraint {con.get('expression')!r} refers to unknown column {r!r}.")
            elif by_name[r].type == "identifier":
                p.append(f"Constraint {con.get('expression')!r} cannot use identifier column {r!r}.")
    else:
        cols = con.get("columns")
        if not isinstance(cols, list) or not cols:
            p.append(f"unique_together constraint needs a non-empty 'columns' list: {con!r}")
        else:
            for name in cols:
                if name not in by_name:
                    p.append(f"unique_together refers to unknown column {name!r}.")
