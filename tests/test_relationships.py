import numpy as np
import pandas as pd
import pytest

from dataseed.relationships import (build_correlation_matrix, correlated_normals, detect_arithmetic_rules,
                                    detect_threshold_rules, estimate_conditionals, nearest_psd, _round_threshold)
from helpers import schema


def test_consistent_matrix_is_untouched_and_sampling_matches():
    m, info = build_correlation_matrix(["a", "b"], [{"columns": ["a", "b"], "value": 0.6}])
    assert not info["repaired"]
    z = correlated_normals(m, 50000, np.random.default_rng(0))
    assert np.corrcoef(z.T)[0, 1] == pytest.approx(0.6, abs=0.02)


def test_inconsistent_matrix_is_repaired_to_positive_definite():
    pairs = [{"columns": ["a", "b"], "value": 0.9}, {"columns": ["b", "c"], "value": 0.9}, {"columns": ["a", "c"], "value": -0.9}]
    m, info = build_correlation_matrix(["a", "b", "c"], pairs)
    assert info["repaired"] and info["max_adjustment"] > 0.1
    assert np.linalg.eigvalsh(m).min() > 0 and np.allclose(np.diag(m), 1)


def test_nearly_singular_matrix_still_samples():
    m = np.array([[1, 0.999999], [0.999999, 1]])
    z = correlated_normals(m, 100, np.random.default_rng(0))
    assert z.shape == (100, 2) and np.isfinite(z).all()
    assert np.linalg.eigvalsh(nearest_psd(m)).min() > 0


def test_round_threshold_prefers_tidy_numbers():
    assert _round_threshold(38.2, 41.0) == 40 and _round_threshold(39, 40) == 40
    assert _round_threshold(51.2, 52.3) == pytest.approx(52.0)


def test_conditional_dependency_found_only_when_evidence_is_strong():
    rng = np.random.default_rng(0)
    n = 400
    dept = rng.choice(["a", "b"], n)
    dep = np.where(dept == "a", rng.choice(["x", "y"], n, p=[0.9, 0.1]), rng.choice(["x", "y"], n, p=[0.1, 0.9]))
    indep = rng.choice(["u", "v"], n)
    df = pd.DataFrame({"dept": dept, "dep": dep, "indep": indep})
    s = schema({"name": "dept", "type": "categorical", "categories": ["a", "b"]},
               {"name": "dep", "type": "categorical", "categories": ["x", "y"]},
               {"name": "indep", "type": "categorical", "categories": ["u", "v"]})
    models, _ = estimate_conditionals(df, s)
    assert [(m["child"], m["parent"]) for m in models] == [("dep", "dept")]
    assert models[0]["probabilities"]["a"]["x"] > 0.8
    assert estimate_conditionals(df.head(30), s)[0] == []


def test_rule_detection_edge_cases():
    df = pd.DataFrame({"x": np.arange(30.0), "flag": [False] * 12 + [True] * 18})
    s = schema({"name": "x", "type": "float", "min": 0, "max": 30}, {"name": "flag", "type": "boolean"})
    rules = detect_threshold_rules(df, s)
    assert rules[0]["expression"] == "x >= 12" or rules[0]["expression"].startswith("x >=")
    few = df.iloc[[0, 1, 2, 3, 28, 29]]
    assert detect_threshold_rules(few, s) == []                      # below min_rows
    const = pd.DataFrame({"a": [2.0] * 20, "b": [3.0] * 20, "c": [6.0] * 20})
    sc = schema(*[{"name": n, "type": "float", "min": 0, "max": 10} for n in "abc"])
    assert detect_arithmetic_rules(const, sc) == []                  # constants give no evidence
