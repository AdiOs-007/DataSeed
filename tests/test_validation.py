import numpy as np
import pandas as pd
import pytest

from dataseed import generate_from_schema, validate_dataset
from helpers import schema

S = schema({"name": "id", "type": "identifier", "prefix": "R-", "width": 4},
           {"name": "x", "type": "float", "min": 0, "max": 10, "missing_rate": 0.1},
           {"name": "c", "type": "categorical", "categories": ["a", "b"]},
           {"name": "d", "type": "date", "min": "2020-01-01", "max": "2020-12-31"},
           {"name": "q", "type": "integer", "min": 1, "max": 5},
           {"name": "big", "type": "boolean", "expression": "q > 3"},
           constraints=[{"type": "expression", "expression": "x < 9.5"}])


def good(n=200):
    return generate_from_schema(S, n, seed=1).dataframe


def status(res, name, column=None):
    return [c.status for c in res.checks if c.name == name and (column is None or c.column == column)]


def test_generated_data_passes_every_check_and_is_not_modified():
    df = good()
    before = df.copy(deep=True)
    res = validate_dataset(df, S, expected_rows=200)
    assert res.is_valid and not res.failed and not res.not_evaluated
    assert {"row_count", "columns", "type", "not_null", "bounds", "allowed_values", "unique", "identifier_format"} <= {c.name for c in res.passed}
    pd.testing.assert_frame_equal(df, before)


def test_each_kind_of_violation_is_detected_with_counts_and_examples():
    df = good()
    df.loc[0, "x"] = 99.0
    df.loc[1, "c"] = "zzz"
    df.loc[2, "id"] = df.loc[3, "id"]
    df.loc[4, "q"] = np.nan
    df.loc[5, "d"] = pd.Timestamp("1999-01-01")
    df.loc[6, "big"] = not df.loc[6, "big"]
    res = validate_dataset(df, S)
    assert not res.is_valid
    names = {(c.name, c.column) for c in res.errors}
    assert {("bounds", "x"), ("allowed_values", "c"), ("unique", "id"), ("not_null", "q"), ("bounds", "d"), ("derived(big)", "big")} <= names
    b = next(c for c in res.errors if c.name == "bounds" and c.column == "x")
    assert b.count >= 1 and b.examples[0]["row"] == 0 and b.examples[0]["value"] == 99.0
    assert any(c.name.startswith("constraint(") for c in res.errors)


def test_row_count_and_column_problems():
    df = good()
    assert status(validate_dataset(df, S, expected_rows=999), "row_count") == ["failed"]
    res = validate_dataset(df.drop(columns=["x"]).assign(extra=1), S)
    assert status(res, "columns") == ["failed"] and status(res, "column_checks") == ["not_evaluated"]
    assert status(validate_dataset(df[list(reversed(df.columns))], S), "column_order") == ["failed"]


def test_checks_that_need_missing_columns_are_not_evaluated_not_passed():
    res = validate_dataset(good().drop(columns=["q"]), S)
    assert status(res, "derived(big)") == ["not_evaluated"]
    assert not res.is_valid or res.not_evaluated


def test_unreadable_values_and_blank_strings():
    raw = good().astype(object)
    raw.loc[0, "q"] = "five"
    raw.loc[1, "x"] = "   "      # blank counts as missing, not as a type error
    res = validate_dataset(raw, S)
    assert [c.count for c in res.failed if c.name == "type" and c.column == "q"] == [1]
    assert not [c for c in res.failed if c.name == "type" and c.column == "x"]


def test_missing_rate_check_and_duplicate_rows_metric():
    df = good(1000)
    df["x"] = np.nan                                     # 100% missing vs configured 10%
    assert status(validate_dataset(df, S), "missing_rate") == ["failed"]
    dup = pd.concat([good(10), good(10)], ignore_index=True)
    res = validate_dataset(dup, S)
    d = next(c for c in res.checks if c.name == "duplicate_rows")
    assert d.count == 10 and d.severity == "info" and res.n_rows == 20


def test_unique_together_and_correlation_checks():
    s = schema({"name": "a", "type": "integer", "min": 1, "max": 3}, {"name": "b", "type": "integer", "min": 1, "max": 3},
               constraints=[{"type": "unique_together", "columns": ["a", "b"]}])
    df = pd.DataFrame({"a": [1, 1, 2], "b": [1, 1, 2]})
    assert validate_dataset(df, s).errors[0].count == 1
    s2 = schema({"name": "u", "type": "float", "min": 0, "max": 1}, {"name": "v", "type": "float", "min": 0, "max": 1},
                relationships=[{"type": "correlation", "columns": ["u", "v"], "value": 0.8}])
    rng = np.random.default_rng(0)
    indep = pd.DataFrame({"u": rng.random(500), "v": rng.random(500)})
    res = validate_dataset(indep, s2)
    corr = next(c for c in res.checks if c.name.startswith("correlation"))
    assert corr.status == "failed" and corr.severity == "warning" and res.is_valid
