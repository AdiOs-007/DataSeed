import numpy as np
import pandas as pd
import pytest

from dataseed import analyze_quality, generate_from_sample, generate_from_schema
from helpers import schema, student_sample


def test_quality_report_with_source_comparison_and_serialisation(tmp_path):
    from dataseed import save_report
    sample = student_sample(80)
    r = generate_from_sample(sample, 2000, seed=1)
    q = analyze_quality(r.dataframe, source=sample, schema=r.schema, validation=r.validation)
    m = q.comparisons["columns"]["marks"]
    assert m["ks_statistic"] < 0.2 and abs(m["mean_diff_in_source_std"]) < 0.3
    assert q.comparisons["columns"]["gender"]["total_variation_distance"] < 0.1
    assert q.comparisons["correlation_difference"]["mean_abs_difference"] < 0.15
    assert q.comparisons["exact_row_overlap"]["pct"] < 1
    assert q.constraint_compliance["is_valid"] and q.overview["n_rows"] == 2000
    assert "ks_statistic" in q.metric_notes and "general" in q.metric_notes
    save_report(q, tmp_path / "q.json")


def test_a_bad_synthetic_dataset_is_visibly_different():
    sample = student_sample(80)
    s = schema({"name": "marks", "type": "float", "min": 0, "max": 10}, {"name": "gender", "type": "categorical", "categories": ["Other"]},
               {"name": "age", "type": "integer", "min": 18, "max": 25}, {"name": "attendance", "type": "float", "min": 0, "max": 100})
    bad = generate_from_schema(s, 1000, seed=1).dataframe
    q = analyze_quality(bad, source=sample, schema=s)
    assert q.comparisons["columns"]["marks"]["ks_exceeds_critical"] is True
    assert q.comparisons["columns"]["gender"]["total_variation_distance"] == pytest.approx(1.0)
    assert "Other" in q.comparisons["columns"]["gender"]["categories_only_in_generated"]


def test_edge_cases_tiny_source_constant_column_and_no_source():
    s = schema({"name": "k", "type": "integer", "min": 5, "max": 5}, {"name": "v", "type": "float", "min": 0, "max": 1})
    gen = generate_from_schema(s, 100, seed=1).dataframe
    tiny = pd.DataFrame({"k": [5, 5, 5], "v": [0.1, 0.5, 0.9]})
    q = analyze_quality(gen, source=tiny, schema=s)
    assert "note" in q.comparisons["columns"]["k"] or "ks_statistic" in q.comparisons["columns"]["k"]
    assert any("only 3 source values" in w for w in q.warnings)
    q2 = analyze_quality(gen, schema=s)
    assert q2.comparisons == {} and q2.columns["v"]["n_unique"] > 1
    q3 = analyze_quality(gen.head(5))               # no schema: inferred, with a warning
    assert any("inferred" in w for w in q3.warnings)
    with pytest.raises(ValueError):
        analyze_quality(pd.DataFrame())
