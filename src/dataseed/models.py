"""Result and settings dataclasses shared by the public API, the CLI and (later) a UI."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import pandas as pd

from .exceptions import GenerationError

if TYPE_CHECKING:  # pragma: no cover
    from .schema import Schema


@dataclass
class GenerationSettings:
    """Tunable behaviour of a generation run.

    Attributes:
        max_rows: Hard upper limit for ``n_rows`` (protects memory). Default 1,000,000.
        max_resample_attempts: Rounds of constraint-driven re-sampling before giving up.
        strict: If True, raise :class:`ConstraintConflictError` when the output violates a
            hard constraint. If False, return the data with a failing validation result.
        missing_policy: ``"schema"`` honours each column's missing_rate; ``"none"`` forces no
            missing values at all.
        infer_correlations / infer_conditionals / infer_missing_patterns / detect_rules: sample-driven
            generation only (conditionals = category-on-category and numeric-by-category dependence).
        min_rows_for_correlation: Sample size below which correlations are not estimated.
        min_rows_for_conditional: Sample size below which conditional category models are skipped.
        overrides: ``{column: {property: value}}`` applied to the schema before generation.
    """

    max_rows: int = 1_000_000
    max_resample_attempts: int = 300
    strict: bool = True
    missing_policy: str = "schema"
    infer_correlations: bool = True
    infer_conditionals: bool = True
    detect_rules: bool = True
    infer_missing_patterns: bool = True
    min_rows_for_correlation: int = 30
    min_rows_for_conditional: int = 60
    overrides: dict[str, dict] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_rows < 1:
            raise GenerationError("max_rows must be >= 1.")
        if self.missing_policy not in ("schema", "none"):
            raise GenerationError("missing_policy must be 'schema' or 'none'.")
        if self.max_resample_attempts < 1:
            raise GenerationError("max_resample_attempts must be >= 1.")


@dataclass
class Decision:
    """One recorded generation decision (method choice, assumption, fallback...)."""

    kind: str          # method | relationship | constraint | missing | fallback | assumption | info
    message: str
    column: str | None = None
    origin: str = "user"   # user | inferred | assumed | default
    confidence: str | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class GenerationReport:
    """Everything the engine decided and warned about during one run."""

    mode: str
    dataset_name: str
    n_rows: int
    seed: int
    decisions: list[Decision] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)

    def add(self, kind: str, message: str, column: str | None = None, origin: str = "user",
            confidence: str | None = None) -> None:
        self.decisions.append(Decision(kind, message, column, origin, confidence))

    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    def methods(self) -> dict[str, str]:
        return {d.column: d.message for d in self.decisions if d.kind == "method" and d.column}

    def to_dict(self) -> dict:
        return {"mode": self.mode, "dataset_name": self.dataset_name, "n_rows": self.n_rows,
                "seed": self.seed, "decisions": [d.to_dict() for d in self.decisions],
                "warnings": list(self.warnings), "unsupported": list(self.unsupported)}


@dataclass
class ValidationCheck:
    """Outcome of a single validation check."""

    name: str
    status: str                 # passed | failed | not_evaluated
    message: str = ""
    column: str | None = None
    severity: str = "error"     # error | warning | info
    count: int | None = None
    examples: list[Any] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v not in (None, [])}


@dataclass
class ValidationResult:
    """All checks run by :func:`dataseed.validate_dataset`."""

    checks: list[ValidationCheck] = field(default_factory=list)
    n_rows: int = 0

    @property
    def passed(self) -> list[ValidationCheck]:
        return [c for c in self.checks if c.status == "passed"]

    @property
    def failed(self) -> list[ValidationCheck]:
        return [c for c in self.checks if c.status == "failed"]

    @property
    def not_evaluated(self) -> list[ValidationCheck]:
        return [c for c in self.checks if c.status == "not_evaluated"]

    @property
    def errors(self) -> list[ValidationCheck]:
        """Failed checks of severity 'error' (hard constraint violations)."""
        return [c for c in self.failed if c.severity == "error"]

    @property
    def is_valid(self) -> bool:
        """True only if no error-level check failed and nothing required was left unevaluated."""
        return not self.errors and not any(c.severity == "error" for c in self.not_evaluated)

    def to_dict(self) -> dict:
        return {
            "is_valid": self.is_valid, "n_rows": self.n_rows,
            "summary": {"passed": len(self.passed), "failed": len(self.failed),
                        "not_evaluated": len(self.not_evaluated), "errors": len(self.errors)},
            "checks": [c.to_dict() for c in self.checks],
        }


@dataclass
class ColumnProfile:
    """Profile of one source column produced by schema inference."""

    name: str
    inferred_type: str
    original_dtype: str
    n_rows: int
    n_missing: int
    n_unique: int
    unique_ratio: float
    confidence: str = "high"
    stats: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def missing_rate(self) -> float:
        return self.n_missing / self.n_rows if self.n_rows else 0.0

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["missing_rate"] = self.missing_rate
        return d


@dataclass
class InferenceResult:
    """Inferred schema plus everything we learned (and are unsure about) about the sample."""

    schema: "Schema"
    profiles: list[ColumnProfile]
    typed_frame: pd.DataFrame
    n_rows: int
    duplicate_rows: int = 0
    conversions: list[str] = field(default_factory=list)
    ambiguities: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    candidate_rules: list[dict] = field(default_factory=list)
    relationship_notes: list[str] = field(default_factory=list)
    correlations: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_rows": self.n_rows, "duplicate_rows": self.duplicate_rows,
            "schema": self.schema.to_dict(), "columns": [p.to_dict() for p in self.profiles],
            "conversions": self.conversions, "ambiguities": self.ambiguities,
            "warnings": self.warnings, "candidate_rules": self.candidate_rules,
            "correlations": self.correlations, "relationship_notes": self.relationship_notes,
        }


@dataclass
class DescriptionParseResult:
    """Result of :func:`dataseed.parse_description` (always review before trusting)."""

    schema: "Schema | None"
    n_rows: int | None = None
    domain: str | None = None
    assumptions: list[str] = field(default_factory=list)
    understood: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return self.schema is not None and bool(self.schema.columns)

    def to_dict(self) -> dict:
        return {"usable": self.usable, "n_rows": self.n_rows, "domain": self.domain,
                "schema": self.schema.to_dict() if self.schema else None,
                "assumptions": self.assumptions, "understood": self.understood,
                "unsupported": self.unsupported, "warnings": self.warnings}


@dataclass
class GenerationResult:
    """Output of a generation run."""

    dataframe: pd.DataFrame
    schema: "Schema"
    report: GenerationReport
    validation: ValidationResult
    inference: InferenceResult | None = None

    @property
    def seed(self) -> int:
        return self.report.seed

    def to_dict(self) -> dict:
        return {"generation": self.report.to_dict(), "validation": self.validation.to_dict(),
                "schema": self.schema.to_dict(),
                "inference": self.inference.to_dict() if self.inference else None}


@dataclass
class QualityReport:
    """Statistical summary of a generated dataset, optionally compared with its source."""

    overview: dict
    columns: dict
    correlations: dict
    comparisons: dict
    constraint_compliance: dict | None
    warnings: list[str]
    metric_notes: dict

    def to_dict(self) -> dict:
        return {"overview": self.overview, "columns": self.columns,
                "correlations": self.correlations, "comparisons": self.comparisons,
                "constraint_compliance": self.constraint_compliance,
                "warnings": self.warnings, "metric_notes": self.metric_notes}
