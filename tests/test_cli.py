import json

import pandas as pd

from dataseed.cli import main
from helpers import student_sample

TEXT = ("Create a student dataset with student_id, age, gender and marks. Age must be between 18 and 25 "
        "and marks between 0 and 100.")


def sample_file(tmp_path):
    p = tmp_path / "students.csv"
    student_sample(60, tmp_csv=p)
    return p


def schema_file(tmp_path):
    p = tmp_path / "schema.json"
    p.write_text(json.dumps({"dataset_name": "t", "columns": [
        {"name": "id", "type": "identifier", "prefix": "T-"}, {"name": "age", "type": "integer", "min": 18, "max": 25},
        {"name": "marks", "type": "float", "min": 0, "max": 100, "mean": 60, "std": 15, "distribution": "normal"}]}))
    return p


def test_help_and_usage_errors(capsys):
    assert main(["--help"]) == 0
    assert "generate" in capsys.readouterr().out
    assert main([]) == 2
    assert main(["generate", "--output", "x.csv"]) == 2                       # needs a source
    assert main(["generate", "--input", "a.csv", "--schema", "b.json", "--output", "o.csv"]) == 2


def test_profile_command(tmp_path, capsys):
    out = tmp_path / "s.json"
    assert main(["profile", "--input", str(sample_file(tmp_path)), "--output-schema", str(out)]) == 0
    text = capsys.readouterr().out
    assert "student_id" in text and "identifier" in text and out.exists()


def test_generate_from_sample_writes_dataset_and_report(tmp_path, capsys):
    out = tmp_path / "nested" / "students_out.csv"
    code = main(["generate", "--input", str(sample_file(tmp_path)), "--rows", "500", "--seed", "42", "--output", str(out)])
    assert code == 0
    df = pd.read_csv(out)
    assert len(df) == 500 and list(df.columns)[0] == "student_id"
    report = json.loads((tmp_path / "nested" / "students_out_report.json").read_text())
    assert report["validation"]["is_valid"] and report["generation"]["seed"] == 42 and "quality" in report
    assert "Reproduce with: --seed 42" in capsys.readouterr().out


def test_generate_is_reproducible_and_xlsx_supported(tmp_path):
    s = schema_file(tmp_path)
    for name in ("a.csv", "b.csv"):
        assert main(["generate", "--schema", str(s), "--rows", "200", "--seed", "7", "--output", str(tmp_path / name), "--no-report"]) == 0
    assert (tmp_path / "a.csv").read_bytes() == (tmp_path / "b.csv").read_bytes()
    assert main(["generate", "--schema", str(s), "--rows", "50", "--seed", "7", "--output", str(tmp_path / "x.xlsx")]) == 0
    assert len(pd.read_excel(tmp_path / "x.xlsx")) == 50


def test_describe_saves_schema_and_can_generate(tmp_path, capsys):
    sp = tmp_path / "out" / "schema.json"
    assert main(["describe", "--text", TEXT, "--output-schema", str(sp)]) == 0
    out = capsys.readouterr().out
    assert "Understood" in out and "Assumptions" in out and sp.exists()
    d = tmp_path / "gen.csv"
    assert main(["describe", "--text", TEXT, "--rows", "120", "--seed", "1", "--output", str(d)]) == 0
    assert len(pd.read_csv(d)) == 120
    assert main(["describe", "--text", TEXT, "--output", str(tmp_path / "n.csv")]) == 2        # no row count anywhere
    assert main(["generate", "--text", TEXT + " Create 80 records.", "--seed", "1", "--output", str(tmp_path / "t.csv")]) == 0
    assert main(["describe", "--text", "make me something nice"]) == 1                          # unusable description


def test_validate_command_success_and_failure(tmp_path, capsys):
    s, out = schema_file(tmp_path), tmp_path / "gen.csv"
    main(["generate", "--schema", str(s), "--rows", "100", "--seed", "1", "--output", str(out), "--no-report"])
    rep = tmp_path / "v.json"
    assert main(["validate", "--input", str(out), "--schema", str(s), "--rows", "100", "--output-report", str(rep)]) == 0
    assert json.loads(rep.read_text())["is_valid"] is True
    df = pd.read_csv(out)
    df.loc[0, "age"] = 99
    df.to_csv(tmp_path / "bad.csv", index=False)
    capsys.readouterr()
    assert main(["validate", "--input", str(tmp_path / "bad.csv"), "--schema", str(s), "--output-report", str(rep)]) == 1
    assert "INVALID" in capsys.readouterr().out and json.loads(rep.read_text())["is_valid"] is False
    assert main(["validate", "--input", str(out), "--schema", str(s), "--rows", "101"]) == 1


def test_failure_cases_return_nonzero_with_actionable_messages(tmp_path, capsys):
    s = schema_file(tmp_path)
    assert main(["generate", "--schema", str(s), "--output", str(tmp_path / "o.csv")]) == 2          # --rows missing
    assert "--rows" in capsys.readouterr().err
    assert main(["generate", "--schema", str(s), "--rows", "5", "--output", str(s)]) == 2             # would overwrite input
    assert main(["generate", "--input", str(tmp_path / "nope.csv"), "--rows", "5", "--output", str(tmp_path / "o.csv")]) == 1
    assert "not found" in capsys.readouterr().err
    assert main(["generate", "--schema", str(s), "--rows", "5", "--output", str(tmp_path / "o.parquet")]) == 1
    assert main(["generate", "--schema", str(s), "--rows", "0", "--output", str(tmp_path / "o.csv")]) == 1
    assert main(["generate", "--schema", str(s), "--rows", "50", "--max-rows", "10", "--output", str(tmp_path / "o.csv")]) == 1
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"columns": [{"name": "a", "type": "integer", "min": 9, "max": 1}]}))
    assert main(["generate", "--schema", str(bad), "--rows", "5", "--output", str(tmp_path / "o.csv")]) == 1
    impossible = tmp_path / "imp.json"
    impossible.write_text(json.dumps({"columns": [{"name": "a", "type": "integer", "min": 0, "max": 5}],
                                      "constraints": [{"type": "expression", "expression": "a > 10"}]}))
    capsys.readouterr()
    assert main(["generate", "--schema", str(impossible), "--rows", "10", "--output", str(tmp_path / "o.csv")]) == 1
    assert "a > 10" in capsys.readouterr().err
    assert not (tmp_path / "o.csv").exists()                                                         # nothing partial written
