"""UI-framework-independent helpers used by the Streamlit app (and usable by any other front end).

Nothing here imports Streamlit, so everything is unit-testable. Functions turn DataSeed results into
plain pandas DataFrames / bytes that a UI can display or offer for download.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, BinaryIO

import pandas as pd

from .exceptions import DataSeedError, SchemaError
from .inference import infer_schema
from .io_utils import dataset_to_bytes, list_sheets, load_sample, to_jsonable
from .models import GenerationSettings, InferenceResult, QualityReport, ValidationResult
from .pipeline import PipelineOutput, run_pipeline
from .schema import Schema, schema_from_dict

MAX_UI_ROWS = 1_000_000


def _file_type(file: Any) -> str:
    return Path(getattr(file, "name", "")).suffix.lower().lstrip(".")


def uploaded_sheets(file: BinaryIO) -> list[str]:
    """Worksheet names of an uploaded .xlsx (empty list for CSV)."""
    if _file_type(file) != "xlsx":
        return []
    _rewind(file)
    return list_sheets(file)


def _rewind(file: Any) -> None:
    if hasattr(file, "seek"):
        file.seek(0)


def read_uploaded_sample(file: BinaryIO, sheet: str | None = None) -> pd.DataFrame:
    """Load an uploaded CSV/XLSX file object (e.g. from ``st.file_uploader``)."""
    ft = _file_type(file)
    if ft not in ("csv", "xlsx"):
        raise DataSeedError(f"Unsupported file type '.{ft}'. Upload a .csv or .xlsx file.")
    _rewind(file)
    return load_sample(file, sheet_name=sheet, file_type=ft)


def schema_to_text(schema: Schema) -> str:
    """Pretty JSON for an editable text box."""
    return json.dumps(schema.to_dict(), indent=2, ensure_ascii=False)


def schema_from_text(text: str) -> Schema:
    """Parse + validate schema JSON typed/pasted by the user. Raises :class:`SchemaError`."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SchemaError(f"The schema is not valid JSON (line {exc.lineno}, column {exc.colno}): {exc.msg}") from exc
    return schema_from_dict(data)


def parse_seed(text: str | None) -> int | None:
    """Blank -> None (random seed). Otherwise a non-negative integer or a clear error."""
    if text is None or not str(text).strip():
        return None
    try:
        v = int(str(text).strip())
    except ValueError as exc:
        raise DataSeedError(f"Seed must be a whole number, got {text!r}.") from exc
    if v < 0 or v >= 2 ** 32:
        raise DataSeedError("Seed must be between 0 and 4,294,967,295.")
    return v


def column_overview(schema: Schema) -> pd.DataFrame:
    """One row per column summarising how it will be generated."""
    rows = []
    for c in schema.columns:
        if c.expression:
            gen = f"derived: {c.expression}"
        elif c.type == "identifier":
            gen = f"{c.prefix or ''}<n>{c.suffix or ''} ({c.sequence or 'sequential'})"
        elif c.categories:
            gen = ", ".join(map(str, c.categories[:6])) + (" ..." if len(c.categories) > 6 else "")
        elif c.type in ("integer", "float", "date", "datetime"):
            gen = f"{c.distribution} [{c.min}, {c.max}]"
        elif c.type == "string":
            gen = f"text: {c.text_kind or c.pattern or 'word'}"
        else:
            gen = ""
        notes = []
        if c.conditional_on:
            notes.append(f"depends on {c.conditional_on}")
        if c.missing_model:
            notes.append(f"missing depends on {c.missing_model['column']}")
        elif c.missing_rate:
            notes.append(f"{c.missing_rate:.1%} missing")
        if c.is_unique and c.type != "identifier":
            notes.append("unique")
        rows.append({"column": c.name, "type": c.type, "generation": gen, "notes": "; ".join(notes), "source": c.source})
    return pd.DataFrame(rows)


def profile_table(inf: InferenceResult) -> pd.DataFrame:
    return pd.DataFrame([{"column": p.name, "type": p.inferred_type, "missing %": round(p.missing_rate * 100, 1),
                          "unique values": p.n_unique, "confidence": p.confidence, "notes": "; ".join(p.notes)}
                         for p in inf.profiles])


def methods_table(output: PipelineOutput) -> pd.DataFrame:
    rep = output.result.report
    return pd.DataFrame([{"column": d.column, "method": d.message, "origin": d.origin, "confidence": d.confidence or ""}
                         for d in rep.decisions if d.kind == "method"])


def decisions_table(output: PipelineOutput, kinds: tuple[str, ...]) -> pd.DataFrame:
    rep = output.result.report
    return pd.DataFrame([{"column": d.column or "", "detail": d.message, "origin": d.origin}
                         for d in rep.decisions if d.kind in kinds])


def validation_table(v: ValidationResult) -> pd.DataFrame:
    return pd.DataFrame([{"status": c.status, "severity": c.severity, "check": c.name, "column": c.column or "",
                          "message": c.message, "count": c.count if c.count is not None else ""} for c in v.checks])


def quality_tables(q: QualityReport) -> dict[str, pd.DataFrame]:
    """Display-ready tables: numeric, categories, comparison, correlation (empty frames when N/A)."""
    num, cat = [], []
    for name, c in q.columns.items():
        if "mean" in c:
            num.append({"column": name, **{k: c[k] for k in ("mean", "std", "min", "q25", "median", "q75", "max")},
                        "missing %": c["missing_pct"]})
        elif "frequencies" in c:
            cat += [{"column": name, "category": k, "share": v} for k, v in c["frequencies"].items()]
    cmp_rows = []
    for name, e in q.comparisons.get("columns", {}).items():
        cmp_rows.append({"column": name, "KS distance": e.get("ks_statistic"), "KS 5% critical": e.get("ks_critical_5pct"),
                         "detectable difference": e.get("ks_exceeds_critical"),
                         "mean diff (source std)": e.get("mean_diff_in_source_std"),
                         "category distance (TVD)": e.get("total_variation_distance"),
                         "note": e.get("reliability") or e.get("note") or ""})
    corr = pd.DataFrame(q.correlations.get("generated", {}))
    return {"numeric": pd.DataFrame(num), "categories": pd.DataFrame(cat), "comparison": pd.DataFrame(cmp_rows),
            "correlation": corr}


def run_generation(*, sample: pd.DataFrame | None, schema: Schema | None, n_rows: int, seed: int | None,
                   no_missing: bool = False, strict: bool = True, max_rows: int = MAX_UI_ROWS) -> PipelineOutput:
    """Single entry point for a UI 'Generate' button (sample XOR schema)."""
    settings = GenerationSettings(max_rows=max_rows, missing_policy="none" if no_missing else "schema", strict=strict)
    return run_pipeline(n_rows=n_rows, seed=seed, sample=sample, schema=schema, settings=settings)


def analyse_sample(df: pd.DataFrame) -> InferenceResult:
    return infer_schema(df)


def report_bytes(output: PipelineOutput) -> bytes:
    """The full JSON report (generation decisions, validation, quality) as UTF-8 bytes."""
    return json.dumps(to_jsonable(output.to_dict()), indent=2, ensure_ascii=False).encode("utf-8")


def dataset_bytes(output: PipelineOutput, fmt: str) -> bytes:
    return dataset_to_bytes(output.result.dataframe, fmt, schema=output.result.schema)


def example_files(project_root: Path | None = None) -> dict[str, Path]:
    """Example schema/sample files shipped with the project (empty if not found)."""
    root = project_root or Path(__file__).resolve().parents[2]
    found = {"Student schema": root / "examples" / "student_schema.json",
             "Employee schema": root / "examples" / "employee_schema.json",
             "Student sample (CSV)": root / "examples" / "students_sample.csv"}
    return {k: v for k, v in found.items() if v.exists()}
