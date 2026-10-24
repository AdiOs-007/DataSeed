import numpy as np
import pandas as pd
import pytest

from dataseed import (ColumnSpec, GenerationSettings, generate_from_sample, generate_from_schema,
                      schema_from_dict)
from dataseed.exceptions import ConstraintConflictError, GenerationError
from helpers import schema, student_sample

STUDENTS = schema(
    {"name": "student_id", "type": "identifier", "prefix": "STU-", "width": 6},
    {"name": "age", "type": "integer", "min": 18, "max": 25},
    {"name": "marks", "type": "float", "min": 0, "max": 100, "mean": 60, "std": 15, "distribution": "normal", "decimals": 1},
    {"name": "passed", "type": "boolean", "expression": "marks >= 40"})


def test_exact_row_count_and_column_order():
    for n in (1, 7, 5000):
        r = generate_from_schema(STUDENTS, n, seed=1)
        assert len(r.dataframe) == n and list(r.dataframe.columns) == STUDENTS.column_names
        assert r.validation.is_valid


def test_reproducible_with_same_seed_and_different_with_other_seed():
    a = generate_from_schema(STUDENTS, 300, seed=5).dataframe
    b = generate_from_schema(STUDENTS, 300, seed=5).dataframe
    c = generate_from_schema(STUDENTS, 300, seed=6).dataframe
    pd.testing.assert_frame_equal(a, b)
    assert not a["marks"].equals(c["marks"])


def test_numeric_bounds_without_boundary_piles():
    df = generate_from_schema(STUDENTS, 20000, seed=2).dataframe
    assert df.age.min() == 18 and df.age.max() == 25
    shares = df.age.value_counts(normalize=True)
    assert all(abs(s - 1 / 8) < 0.02 for s in shares)                    # every integer gets its fair share
    assert (df.marks == 100).mean() < 0.01 and (df.marks == 0).mean() < 0.01   # truncation, not clipping
    assert df.marks.mean() == pytest.approx(60, abs=1) and df.marks.std() == pytest.approx(15, abs=1.5)
    assert (df.marks.round(1) == df.marks).all()


def test_uniform_lognormal_discrete_and_constant_numeric():
    s = schema({"name": "u", "type": "float", "min": 5, "max": 6},
               {"name": "ln", "type": "float", "distribution": "lognormal", "mean": 50, "std": 30, "min": 1, "max": 400},
               {"name": "d", "type": "integer", "categories": [1, 2, 3], "probabilities": [0.6, 0.3, 0.1]},
               {"name": "k", "type": "integer", "default": 7})
    df = generate_from_schema(s, 10000, seed=3).dataframe
    assert df.u.between(5, 6).all() and df.ln.between(1, 400).all()
    assert df.ln.mean() == pytest.approx(50, rel=0.15)
    assert df.d.value_counts(normalize=True)[1] == pytest.approx(0.6, abs=0.03)
    assert (df.k == 7).all()


def test_categorical_and_boolean_proportions():
    s = schema({"name": "c", "type": "categorical", "categories": ["a", "b", "c"], "probabilities": [0.5, 0.3, 0.2]},
               {"name": "flag", "type": "boolean", "probability": 0.25},
               {"name": "even", "type": "categorical", "categories": ["x", "y"]})
    df = generate_from_schema(s, 20000, seed=4).dataframe
    p = df.c.value_counts(normalize=True)
    assert p["a"] == pytest.approx(0.5, abs=0.02) and p["c"] == pytest.approx(0.2, abs=0.02)
    assert df.flag.mean() == pytest.approx(0.25, abs=0.02) and df.flag.dtype == bool
    assert df.even.value_counts(normalize=True)["x"] == pytest.approx(0.5, abs=0.02)
    assert set(df.c) <= {"a", "b", "c"}


def test_dates_and_datetimes_respect_bounds_and_formats(tmp_path):
    s = schema({"name": "d", "type": "date", "min": "2020-01-01", "max": "2020-12-31", "date_format": "%d/%m/%Y"},
               {"name": "t", "type": "datetime", "min": "2024-03-01 08:00:00", "max": "2024-03-01 18:00:00"})
    df = generate_from_schema(s, 2000, seed=1).dataframe
    assert df.d.min() >= pd.Timestamp("2020-01-01") and df.d.max() <= pd.Timestamp("2020-12-31")
    assert (df.d == df.d.dt.normalize()).all()
    assert df.t.min() >= pd.Timestamp("2024-03-01 08:00:00") and df.t.max() <= pd.Timestamp("2024-03-01 18:00:00")
    from dataseed import export_dataset
    export_dataset(df, tmp_path / "x.csv")
    first = (tmp_path / "x.csv").read_text().splitlines()[1]
    assert first.count("/") == 2 and ":" in first     # dd/mm/YYYY and a timestamp


def test_identifier_formats_and_uniqueness():
    s = schema({"name": "a", "type": "identifier", "prefix": "STU-", "width": 6},
               {"name": "b", "type": "identifier", "start": 1001},
               {"name": "c", "type": "identifier", "prefix": "X", "suffix": "-Z", "width": 3, "sequence": "shuffled"})
    df = generate_from_schema(s, 1000, seed=1).dataframe
    assert df.a.iloc[0] == "STU-000001" and df.a.iloc[-1] == "STU-001000" and df.a.is_unique
    assert df.b.iloc[0] == 1001 and df.b.is_unique and np.issubdtype(df.b.dtype, np.integer)
    assert df.c.is_unique and df.c.str.match(r"^X\d{3,}-Z$").all() and set(df.c) == {f"X{i:03d}-Z" for i in range(1, 1001)}


def test_unique_numeric_and_string_columns():
    s = schema({"name": "n", "type": "integer", "min": 1, "max": 500, "unique": True, "mean": 250, "std": 40, "distribution": "normal"},
               {"name": "email", "type": "string", "text_kind": "email", "unique": True},
               {"name": "code", "type": "string", "pattern": "AB-####-??"})
    df = generate_from_schema(s, 500, seed=1).dataframe
    assert df.n.is_unique and sorted(df.n) == list(range(1, 501))
    assert df.email.is_unique and df.email.str.contains(r"@example\.(?:com|org|net)$").all()
    assert df.code.str.match(r"^AB-\d{4}-[A-Z]{2}$").all()


def test_missing_value_policies():
    s = schema({"name": "x", "type": "float", "min": 0, "max": 1, "missing_rate": 0.1},
               {"name": "i", "type": "integer", "min": 0, "max": 9, "missing_rate": 0.2},
               {"name": "y", "type": "float", "min": 0, "max": 1, "nullable": False})
    df = generate_from_schema(s, 1000, seed=1).dataframe
    assert df.x.isna().sum() == 100 and df.i.isna().sum() == 200 and df.y.isna().sum() == 0
    assert str(df.i.dtype) == "Int64"
    none = generate_from_schema(s, 1000, seed=1, settings=GenerationSettings(missing_policy="none")).dataframe
    assert none.isna().sum().sum() == 0 and none.i.dtype == np.int64


def test_missing_values_propagate_to_derived_columns():
    s = schema({"name": "a", "type": "float", "min": 0, "max": 100, "missing_rate": 0.1},
               {"name": "ok", "type": "boolean", "expression": "a >= 50", "nullable": True},
               {"name": "twice", "type": "float", "expression": "a * 2", "nullable": True})
    r = generate_from_schema(s, 500, seed=1)
    df = r.dataframe
    assert (df.ok.isna() == df.a.isna()).all() and (df.twice.isna() == df.a.isna()).all()
    assert r.validation.is_valid


def test_derived_columns_arithmetic_and_boolean():
    s = schema({"name": "q", "type": "integer", "min": 1, "max": 9}, {"name": "p", "type": "float", "min": 1, "max": 50, "decimals": 2},
               {"name": "total", "type": "float", "expression": "q * p", "decimals": 2},
               {"name": "big", "type": "boolean", "expression": "total > 100"},
               {"name": "tier", "type": "categorical", "categories": ["lo", "hi"], "expression": "where(total > 100, 'hi', 'lo')"})
    df = generate_from_schema(s, 1000, seed=1).dataframe
    assert np.allclose(df.total, (df.q * df.p).round(2)) and ((df.total > 100) == df.big).all()
    assert (df.tier == np.where(df.big, "hi", "lo")).all()


def test_correlation_is_reproduced_and_reported():
    s = schema({"name": "a", "type": "float", "min": 0, "max": 100, "mean": 50, "std": 15, "distribution": "normal"},
               {"name": "b", "type": "float", "min": 0, "max": 100, "mean": 50, "std": 15, "distribution": "normal"},
               {"name": "c", "type": "float", "min": 0, "max": 100, "mean": 50, "std": 15, "distribution": "normal"},
               relationships=[{"type": "correlation", "columns": ["a", "b"], "value": 0.7},
                              {"type": "correlation", "columns": ["a", "c"], "value": -0.4}])
    r = generate_from_schema(s, 20000, seed=1)
    c = r.dataframe.corr()
    assert c.loc["a", "b"] == pytest.approx(0.7, abs=0.04) and c.loc["a", "c"] == pytest.approx(-0.4, abs=0.04)
    assert abs(c.loc["b", "c"]) < 0.5
    assert any("copula" in d.message for d in r.report.decisions if d.kind == "relationship")


def test_incompatible_user_correlations_raise_clear_error():
    cols = [{"name": n, "type": "float", "min": 0, "max": 1} for n in "abc"]
    with pytest.raises(ConstraintConflictError) as e:
        generate_from_schema(schema(*cols, relationships=[
            {"type": "correlation", "columns": ["a", "b"], "value": 0.9},
            {"type": "correlation", "columns": ["b", "c"], "value": 0.9},
            {"type": "correlation", "columns": ["a", "c"], "value": -0.9}]), 100, seed=1)
    assert "not mutually consistent" in str(e.value)


def test_row_constraints_enforced_by_resampling_with_exact_row_count():
    s = schema({"name": "a", "type": "integer", "min": 0, "max": 100}, {"name": "b", "type": "integer", "min": 0, "max": 100},
               constraints=[{"type": "expression", "expression": "a < b"}])
    r = generate_from_schema(s, 3000, seed=1)
    assert len(r.dataframe) == 3000 and (r.dataframe.a < r.dataframe.b).all()
    assert any("replaced" in d.message for d in r.report.decisions)


def test_unique_together_constraint():
    s = schema({"name": "x", "type": "integer", "min": 1, "max": 10}, {"name": "y", "type": "integer", "min": 1, "max": 10},
               constraints=[{"type": "unique_together", "columns": ["x", "y"]}])
    df = generate_from_schema(s, 90, seed=1).dataframe
    assert not df.duplicated(subset=["x", "y"]).any()
    with pytest.raises(ConstraintConflictError) as e:
        generate_from_schema(s, 101, seed=1)
    assert "100 distinct combinations" in str(e.value)


def test_impossible_constraints_fail_with_explanation_not_fewer_rows():
    s = schema({"name": "a", "type": "integer", "min": 0, "max": 10},
               constraints=[{"type": "expression", "expression": "a > 50"}])
    with pytest.raises(ConstraintConflictError) as e:
        generate_from_schema(s, 100, seed=1)
    assert "a > 50" in str(e.value) and "0.00%" in str(e.value)
    u = schema({"name": "a", "type": "integer", "min": 1, "max": 10, "unique": True})
    with pytest.raises(ConstraintConflictError) as e:
        generate_from_schema(u, 11, seed=1)
    assert "only 10" in str(e.value)


def test_request_validation():
    with pytest.raises(GenerationError):
        generate_from_schema(STUDENTS, 0)
    with pytest.raises(GenerationError):
        generate_from_schema(STUDENTS, -5)
    with pytest.raises(GenerationError):
        generate_from_schema(STUDENTS, None)
    with pytest.raises(GenerationError) as e:
        generate_from_schema(STUDENTS, 11, settings=GenerationSettings(max_rows=10))
    assert "maximum" in str(e.value)
    assert generate_from_schema(schema_from_dict({**STUDENTS.to_dict(), "generation": {"n_rows": 12}}), seed=1).dataframe.shape[0] == 12


def test_conditional_categorical_distribution():
    s = schema({"name": "dept", "type": "categorical", "categories": ["eng", "hr"], "probabilities": [0.5, 0.5]},
               {"name": "level", "type": "categorical", "categories": ["jr", "sr"], "conditional_on": "dept",
                "conditional_probabilities": {"eng": {"jr": 0.9, "sr": 0.1}, "hr": {"jr": 0.1, "sr": 0.9}}})
    df = generate_from_schema(s, 20000, seed=1).dataframe
    assert (df[df.dept == "eng"].level == "jr").mean() == pytest.approx(0.9, abs=0.02)
    assert (df[df.dept == "hr"].level == "jr").mean() == pytest.approx(0.1, abs=0.02)


def test_overrides_and_non_strict_mode():
    r = generate_from_schema(STUDENTS, 500, seed=1, settings=GenerationSettings(overrides={"age": {"min": 30, "max": 31}}))
    assert r.dataframe.age.between(30, 31).all()
    with pytest.raises(GenerationError):
        generate_from_schema(STUDENTS, 5, settings=GenerationSettings(overrides={"ghost": {"min": 1}}))


def test_sample_driven_generation_learns_structure_without_copying_rows():
    sample = student_sample(80)
    r = generate_from_sample(sample, 3000, seed=11)
    df = r.dataframe
    assert len(df) == 3000 and r.validation.is_valid
    assert df.attendance.corr(df.marks) > 0.4                                    # learned correlation
    assert ((df.marks >= 40) == (df.pass_status == "Pass")).all()                # learned rule
    assert df.attendance.between(sample.attendance.min(), sample.attendance.max()).all()
    key = lambda d: set(zip(d.age, d.gender, d.attendance, d.marks))
    assert len(key(df) & key(sample)) <= 2                                       # no copying of source rows
    assert any("copula" in d.message for d in r.report.decisions)


def test_tiny_samples_and_constant_columns_work_and_warn():
    tiny = pd.DataFrame({"a": [1.0, 2.0, 3.5], "c": ["x", "x", "y"], "k": [9, 9, 9]})
    r = generate_from_sample(tiny, 200, seed=1)
    assert len(r.dataframe) == 200 and (r.dataframe.k == 9).all()
    assert r.dataframe.a.between(1.0, 3.5).all()
    assert any("only 3 rows" in w for w in r.report.warnings)
    one = generate_from_sample(pd.DataFrame({"a": [4.0]}), 10, seed=1)
    assert (one.dataframe.a == 4.0).all()


def test_sample_missing_rates_are_learned():
    df = pd.DataFrame({"v": [float(i) if i % 5 else np.nan for i in range(100)], "w": np.arange(100) % 7})
    out = generate_from_sample(df, 1000, seed=1).dataframe
    assert out.v.isna().sum() == 200


def test_text_columns_are_synthetic():
    s = schema({"name": "n", "type": "string", "text_kind": "name"}, {"name": "ph", "type": "string", "text_kind": "phone"},
               {"name": "w", "type": "string"})
    r = generate_from_schema(s, 200, seed=1)
    assert r.dataframe.ph.str.contains("-555-01").all() and r.dataframe.n.str.contains(" ").all()
    assert any("no semantic realism" in w for w in r.report.warnings)


def test_report_records_method_for_every_column():
    r = generate_from_schema(STUDENTS, 100, seed=9)
    assert set(r.report.methods()) == set(STUDENTS.column_names) and r.report.seed == 9
    assert r.to_dict()["generation"]["n_rows"] == 100
