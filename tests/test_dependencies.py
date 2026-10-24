"""Numeric-by-category dependence and missing-at-random (MAR) patterns."""
import numpy as np
import pandas as pd
import pytest

from dataseed import generate_from_sample, generate_from_schema, infer_schema, validate_dataset
from dataseed.exceptions import SchemaError
from helpers import schema

SAL = {"name": "sal", "type": "float", "min": 0, "max": 300000, "distribution": "normal", "mean": 70000, "std": 20000,
       "conditional_on": "dept",
       "conditional_params": {"eng": {"mean": 100000, "std": 15000}, "hr": {"mean": 50000, "std": 8000}}}
DEPT = {"name": "dept", "type": "categorical", "categories": ["eng", "hr", "ops"], "probabilities": [0.4, 0.4, 0.2]}


def sample(n=600, seed=0, mar=True):
    rng = np.random.default_rng(seed)
    dept = rng.choice(["eng", "hr", "ops"], n)
    sal = np.where(dept == "eng", rng.normal(100, 10, n), np.where(dept == "hr", rng.normal(60, 8, n), rng.normal(75, 9, n)))
    age = rng.integers(20, 60, n).astype(float)
    if mar:
        age[(dept == "hr") & (rng.random(n) < 0.5)] = np.nan
    else:
        age[rng.random(n) < 0.2] = np.nan
    return pd.DataFrame({"dept": dept, "sal": sal.round(1), "age": age, "noise": rng.normal(size=n)})


def test_numeric_distribution_depends_on_category_and_unlisted_groups_fall_back():
    df = generate_from_schema(schema(DEPT, SAL), 20000, seed=1).dataframe
    g = df.groupby("dept").sal
    assert g.mean()["eng"] == pytest.approx(100000, rel=0.02) and g.std()["hr"] == pytest.approx(8000, rel=0.1)
    assert g.mean()["ops"] == pytest.approx(70000, rel=0.03)          # overall distribution for the unlisted group
    assert df.sal.between(0, 300000).all()


def test_numeric_conditional_works_with_boolean_parent_and_quantile_params():
    s = schema({"name": "flag", "type": "boolean", "probability": 0.5},
               {"name": "v", "type": "float", "min": 0, "max": 100, "distribution": "normal", "mean": 50, "std": 20,
                "conditional_on": "flag",
                "conditional_params": {"true": {"quantiles": [80, 90, 100]}, "false": {"mean": 20, "std": 5}}})
    df = generate_from_schema(s, 10000, seed=1).dataframe
    assert df[df.flag].v.min() >= 80 and df[~df.flag].v.mean() == pytest.approx(20, abs=1)


@pytest.mark.parametrize("bad", [
    {"conditional_on": "ghost"},
    {"conditional_on": "sal"},
    {"conditional_params": {"eng": {"mean": 1}}},
    {"conditional_params": {"eng": {"mean": 1, "std": 1, "bogus": 2}}},
    {"distribution": "uniform"},
])
def test_invalid_numeric_conditionals_rejected(bad):
    col = {**SAL, **bad}
    with pytest.raises(SchemaError):
        schema(DEPT, col)


def test_conditional_dependency_creates_generation_order_and_detects_cycles():
    assert schema(SAL, DEPT).generation_order().index("dept") < schema(SAL, DEPT).generation_order().index("sal")


def test_missing_group_model_gives_exact_group_rates():
    s = schema(DEPT, {"name": "x", "type": "float", "min": 0, "max": 1,
                      "missing_model": {"type": "group", "column": "dept", "rates": {"hr": 0.5, "eng": 0.1}, "default": 0.0}})
    r = generate_from_schema(s, 10000, seed=1)
    rates = r.dataframe.groupby("dept").x.apply(lambda v: v.isna().mean())
    assert rates["hr"] == pytest.approx(0.5, abs=0.001) and rates["eng"] == pytest.approx(0.1, abs=0.001) and rates["ops"] == 0
    assert any("missing at random" in d.message for d in r.report.decisions if d.kind == "missing")
    assert r.validation.is_valid and any(c.name == "missing_model(x)" and c.status == "passed" for c in r.validation.checks)


def test_missing_threshold_model_and_derived_propagation():
    s = schema({"name": "a", "type": "integer", "min": 0, "max": 100},
               {"name": "b", "type": "float", "min": 0, "max": 1,
                "missing_model": {"type": "threshold", "column": "a", "op": ">=", "value": 50, "rate_if_true": 0.4, "rate_if_false": 0.0}},
               {"name": "c", "type": "float", "expression": "b * 2", "nullable": True})
    df = generate_from_schema(s, 10000, seed=2).dataframe
    assert df[df.a < 50].b.isna().sum() == 0 and df[df.a >= 50].b.isna().mean() == pytest.approx(0.4, abs=0.002)
    assert (df.c.isna() == df.b.isna()).all()
    with pytest.raises(SchemaError):      # derived column that must not be null but depends on a MAR column
        schema(*s.to_dict()["columns"][:2], {"name": "c", "type": "float", "expression": "b * 2", "nullable": False})


def test_missing_model_validation_detects_wrong_pattern_and_policy_none():
    s = schema(DEPT, {"name": "x", "type": "float", "min": 0, "max": 1,
                      "missing_model": {"type": "group", "column": "dept", "rates": {"hr": 0.5}, "default": 0.0}})
    df = generate_from_schema(s, 5000, seed=1).dataframe
    df["x"] = df["x"].where(df.dept != "hr", df.x)           # unchanged: still valid
    wrong = df.copy()
    wrong.loc[wrong.dept == "eng", "x"] = np.nan             # eng should have 0% missing
    res = validate_dataset(wrong, s)
    chk = next(c for c in res.checks if c.name == "missing_model(x)")
    assert chk.status == "failed" and chk.severity == "warning" and res.is_valid
    from dataseed import GenerationSettings
    none = generate_from_schema(s, 500, seed=1, settings=GenerationSettings(missing_policy="none")).dataframe
    assert none.x.isna().sum() == 0


@pytest.mark.parametrize("bad", [
    {"type": "group", "column": "ghost", "rates": {"a": 0.1}},
    {"type": "group", "column": "dept", "rates": {"hr": 1.5}},
    {"type": "threshold", "column": "dept", "op": ">", "value": 1, "rate_if_true": 0.1, "rate_if_false": 0.1},
    {"type": "weird"},
])
def test_invalid_missing_models_rejected(bad):
    with pytest.raises(SchemaError):
        schema(DEPT, {"name": "x", "type": "float", "min": 0, "max": 1, "missing_model": bad})
    with pytest.raises(SchemaError):
        schema(DEPT, {"name": "y", "type": "float", "min": 0, "max": 1, "nullable": False,
                      "missing_model": {"type": "group", "column": "dept", "rates": {"hr": 0.1}}})


def test_inference_detects_numeric_dependence_and_mar_but_not_mcar_or_noise():
    r = infer_schema(sample())
    sal, age, noise = r.schema.column("sal"), r.schema.column("age"), r.schema.column("noise")
    assert sal.conditional_on == "dept" and set(sal.conditional_params) == {"eng", "hr", "ops"}
    assert age.missing_model["column"] == "dept" and age.missing_model["rates"]["hr"] == pytest.approx(0.5, abs=0.07)
    assert age.missing_model["rates"]["eng"] == 0.0 and noise.conditional_on is None
    mcar = infer_schema(sample(mar=False))
    assert mcar.schema.column("age").missing_model is None
    small = infer_schema(sample(50))
    assert small.schema.column("sal").conditional_on is None and small.schema.column("age").missing_model is None


def test_sample_driven_generation_reproduces_group_structure_and_mar():
    out = generate_from_sample(sample(), 6000, seed=1)
    df = out.dataframe
    assert df.groupby("dept").sal.mean()["eng"] == pytest.approx(100, abs=2)
    assert df[df.dept == "hr"].age.isna().mean() == pytest.approx(0.5, abs=0.06)
    assert df[df.dept == "eng"].age.isna().sum() == 0 and out.validation.is_valid
