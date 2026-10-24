"""The shipped example files must stay valid and usable (the README relies on them)."""
from pathlib import Path

import pytest

from dataseed import generate_from_sample, generate_from_schema, load_sample, load_schema

EX = Path(__file__).resolve().parent.parent / "examples"


def test_student_schema_example_generates_valid_data():
    s = load_schema(EX / "student_schema.json")
    r = generate_from_schema(s, 2000, seed=42)
    df = r.dataframe
    assert r.validation.is_valid and len(df) == 2000
    assert ((df.marks >= 40) == (df.pass_status == "Pass")).all()
    assert df.attendance_percentage.isna().sum() == 40                    # 2% configured missing rate
    assert df.attendance_percentage.corr(df.marks) > 0.4


def test_employee_schema_example_generates_valid_data():
    s = load_schema(EX / "employee_schema.json")
    r = generate_from_schema(s, 2000, seed=7)
    df = r.dataframe
    assert r.validation.is_valid and df.email.is_unique and (df.salary >= 25000).all()
    assert abs(df.bonus - (df.salary * 0.1).round(2)).max() < 0.01
    assert df.years_experience.corr(df.salary) > 0.4
    assert (df[df.department == "Sales"].job_level == "Junior").mean() > 0.4   # conditional distribution
    by = df.groupby("department").salary.mean()
    assert by["Engineering"] > by["Sales"] > by["HR"]                          # numeric depends on category
    miss = df.groupby("department").manager_rating.apply(lambda v: v.isna().mean())
    assert miss["Sales"] == pytest.approx(0.2, abs=0.01) and miss["Engineering"] == pytest.approx(0.02, abs=0.01)


def test_sample_example_learns_correlation_and_pass_rule():
    sample = load_sample(EX / "students_sample.csv")
    r = generate_from_sample(sample, 3000, seed=42)
    df = r.dataframe
    assert r.validation.is_valid and len(df) == 3000
    assert ((df.marks >= 40) == (df.pass_status == "Pass")).all()
    assert df.attendance_percentage.corr(df.marks) > 0.3
    assert df.attendance_percentage.isna().sum() > 0                          # observed missingness reproduced
    assert any("copula" in d.message for d in r.report.decisions if d.kind == "relationship")
