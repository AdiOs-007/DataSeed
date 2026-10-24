"""Human-readable text rendering of DataSeed results (used by the CLI; reusable by a UI)."""
from __future__ import annotations

from .io_utils import save_report  # re-exported for convenience
from .models import (DescriptionParseResult, GenerationResult, InferenceResult, QualityReport,
                     ValidationResult)

__all__ = ["save_report", "format_inference", "format_parse_result", "format_generation",
           "format_validation", "format_quality"]


def _bullets(title: str, items: list[str], limit: int | None = None) -> list[str]:
    if not items:
        return []
    shown = items if limit is None else items[:limit]
    out = [f"\n{title}"] + [f"  - {i}" for i in shown]
    if limit is not None and len(items) > limit:
        out.append(f"  ... and {len(items) - limit} more (see the JSON report)")
    return out


def format_inference(inf: InferenceResult) -> str:
    lines = [f"Sample: {inf.n_rows} rows x {len(inf.profiles)} columns; duplicate rows: {inf.duplicate_rows}", "",
             f"{'column':<24}{'type':<12}{'missing':>9}{'unique':>8}  confidence"]
    for p in inf.profiles:
        lines.append(f"{p.name:<24}{p.inferred_type:<12}{p.missing_rate:>8.1%}{p.n_unique:>8}  {p.confidence}")
        for note in p.notes:
            lines.append(f"{'':<24}  . {note}")
    lines += _bullets("Conversions / loading notes:", inf.conversions)
    lines += _bullets("Ambiguities (please review):", inf.ambiguities)
    lines += _bullets("Relationships:", inf.relationship_notes)
    lines += _bullets("Warnings:", inf.warnings)
    return "\n".join(lines)


def format_parse_result(r: DescriptionParseResult) -> str:
    lines = [f"Usable schema: {'yes' if r.usable else 'NO'}   Rows requested: {r.n_rows or 'not stated'}   Domain: {r.domain or 'none'}"]
    if r.schema:
        lines.append("\nInterpreted columns:")
        for c in r.schema.columns:
            extra = f"= {c.expression}" if c.expression else ""
            lines.append(f"  - {c.name} ({c.type}) {extra}".rstrip())
        for rel in r.schema.relationships:
            lines.append(f"  - correlation {rel['columns']} = {rel['value']:+.2f}")
    lines += _bullets("Understood:", r.understood)
    lines += _bullets("Assumptions made by the parser (review them):", r.assumptions)
    lines += _bullets("NOT understood / unsupported:", r.unsupported)
    lines += _bullets("Warnings:", r.warnings)
    return "\n".join(lines)


def format_validation(v: ValidationResult, show_passed: bool = False) -> str:
    s = v.to_dict()["summary"]
    lines = [f"Validation: {'VALID' if v.is_valid else 'INVALID'} - {s['passed']} passed, {s['failed']} failed "
             f"({s['errors']} error-level), {s['not_evaluated']} not evaluated; {v.n_rows} rows"]
    for c in v.failed:
        lines.append(f"  [FAIL/{c.severity}] {c.name}{f' [{c.column}]' if c.column else ''}: {c.message}")
        for ex in c.examples[:3]:
            lines.append(f"        e.g. {ex}")
    for c in v.not_evaluated:
        lines.append(f"  [not evaluated] {c.name}{f' [{c.column}]' if c.column else ''}: {c.message}")
    if show_passed:
        for c in v.passed:
            lines.append(f"  [ok] {c.name}{f' [{c.column}]' if c.column else ''}: {c.message}")
    return "\n".join(lines)


def format_generation(r: GenerationResult, max_warnings: int = 12) -> str:
    rep = r.report
    lines = [f"Generated {len(r.dataframe):,} rows x {r.dataframe.shape[1]} columns "
             f"(mode: {rep.mode}, seed: {rep.seed})", "", "How each column was generated:"]
    for d in rep.decisions:
        if d.kind == "method":
            conf = f" [confidence: {d.confidence}]" if d.confidence else ""
            lines.append(f"  - {d.column}: {d.message} ({d.origin}){conf}")
    lines += _bullets("Relationships:", [d.message for d in rep.decisions if d.kind == "relationship"], 8)
    lines += _bullets("Constraints:", [d.message for d in rep.decisions if d.kind == "constraint"], 8)
    lines += _bullets("Fallbacks / assumptions:", [d.message for d in rep.decisions if d.kind in ("fallback", "assumption")], 8)
    lines += _bullets("Warnings:", rep.warnings, max_warnings)
    lines += ["", format_validation(r.validation)]
    return "\n".join(lines)


def format_quality(q: QualityReport) -> str:
    o = q.overview
    lines = [f"Quality: {o['n_rows']:,} rows, {o['n_columns']} columns, {o['duplicate_rows']} duplicate rows "
             f"({o['duplicate_rows_pct']}%), {o['missing_cells']} missing cells"]
    cmp = q.comparisons.get("columns", {})
    if cmp:
        lines.append("\nSource vs generated (higher distance = bigger difference):")
        for name, e in cmp.items():
            if "ks_statistic" in e:
                lines.append(f"  - {name}: KS={e['ks_statistic']:.3f} (5% critical {e['ks_critical_5pct']:.3f}), "
                             f"mean {e['source_mean']:.3g} -> {e['generated_mean']:.3g}; {e['interpretation']}")
            elif "total_variation_distance" in e:
                lines.append(f"  - {name}: total variation distance {e['total_variation_distance']:.3f}")
    cd = q.comparisons.get("correlation_difference")
    if cd and cd.get("mean_abs_difference") is not None:
        lines.append(f"  - correlations: mean absolute difference {cd['mean_abs_difference']:.3f}")
    eo = q.comparisons.get("exact_row_overlap")
    if eo:
        lines.append(f"  - generated rows identical to a source row: {eo['rows']} ({eo['pct']}%)")
    lines += _bullets("Quality warnings:", q.warnings, 10)
    lines.append("\nNote: these metrics do not prove the data is realistic; see metric_notes in the JSON report.")
    return "\n".join(lines)
