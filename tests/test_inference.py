import numpy as np
import pandas as pd
import pytest

from dataseed import GenerationSettings, infer_schema, load_sample
from dataseed.exceptions import InferenceError
from helpers import student_sample


def types(df, **kw):
    r = infer_schema(df, **kw)
    return {c.name: c.type for c in r.schema.columns}, r


def test_basic_type_inference_from_csv_text(tmp_path):
    p = tmp_path / "s.csv"
    student_sample(60, tmp_csv=p)
    t, r = types(load_sample(p))
    assert t["student_id"] == "identifier" and t["age"] == "integer" and t["attendance"] == "float"
    assert t["gender"] == "categorical" and t["pass_status"] == "categorical"
    assert any("converted to numeric" in c for c in r.conversions)


def test_numeric_id_column_is_identifier_not_a_measurement():
    df = pd.DataFrame({"employee_id": range(1001, 1041), "score": np.random.default_rng(0).normal(50, 5, 40)})
    r = infer_schema(df)
    c = r.schema.column("employee_id")
    assert c.type == "identifier" and c.start == 1001 and c.prefix == ""
    assert r.schema.column("score").type == "float"


def test_boolean_date_text_and_missing_detection():
    n = 40
    df = pd.DataFrame({"active": ["Yes", "No"] * 20,
                       "joined": pd.date_range("2022-01-01", periods=n).strftime("%Y-%m-%d"),
                       "email": [f"user{i}@mail.org" for i in range(n)],
                       "score": [str(i) if i % 10 else "" for i in range(n)]})
    t, r = types(df)
    assert t == {"active": "boolean", "joined": "date", "email": "string", "score": "integer"}
    assert r.schema.column("email").text_kind == "email"
    assert r.schema.column("score").missing_rate == pytest.approx(4 / 40)
    assert any("not reused" in a for a in r.ambiguities)


def test_mixed_values_are_reported_not_silently_dropped():
    df = pd.DataFrame({"x": [str(i * 3) for i in range(30)] + ["oops"]})
    t, r = types(df)
    assert t["x"] == "integer"
    assert any("oops" in w for w in r.warnings)


def test_constant_column_all_empty_column_and_tiny_sample():
    df = pd.DataFrame({"c": [5, 5, 5, 5], "empty": [None] * 4, "v": [1.5, 2.5, 3.5, 9.0]})
    t, r = types(df)
    assert "empty" not in t and any("empty" in a for a in r.ambiguities)
    assert r.schema.column("c").distribution == "constant" and r.schema.column("c").default == 5
    assert any("only 4 rows" in w for w in r.warnings)
    assert r.profiles[2].confidence == "low"


def test_duplicate_rows_counted():
    df = pd.DataFrame({"a": [1, 1, 2, 3, 3, 3], "b": list("xxyzzz")})
    assert infer_schema(df).duplicate_rows == 3


def test_threshold_rule_detected_only_with_perfect_separation():
    df = student_sample(60)
    r = infer_schema(df)
    col = r.schema.column("pass_status")
    assert col.expression == "where(marks >= 40, 'Pass', 'Fail')"
    noisy = df.copy()
    noisy.loc[noisy.index[:3], "pass_status"] = np.where(noisy.loc[noisy.index[:3], "pass_status"] == "Pass", "Fail", "Pass")
    assert infer_schema(noisy).schema.column("pass_status").expression is None
    assert infer_schema(df, settings=GenerationSettings(detect_rules=False)).schema.column("pass_status").expression is None


def test_arithmetic_rule_detected():
    rng = np.random.default_rng(1)
    q, p = rng.integers(1, 9, 40), rng.integers(10, 99, 40)
    r = infer_schema(pd.DataFrame({"qty": q, "price": p, "total": q * p}))
    assert r.schema.column("total").expression in ("qty * price", "price * qty")


def test_correlation_inferred_only_with_enough_rows_and_evidence():
    rng = np.random.default_rng(2)
    x = rng.normal(size=120)
    df = pd.DataFrame({"x": x, "y": 0.8 * x + rng.normal(scale=0.6, size=120), "z": rng.normal(size=120)})
    r = infer_schema(df)
    assert [tuple(c["columns"]) for c in r.correlations] == [("x", "y")]
    assert infer_schema(df.head(20)).correlations == []


def test_empty_input_is_an_error():
    with pytest.raises(InferenceError):
        infer_schema(pd.DataFrame())
