import io
import json
from pathlib import Path

import pandas as pd
import pytest

from dataseed import generate_from_schema, infer_schema, parse_description
from dataseed import ui_helpers as ui
from dataseed.cli import main
from dataseed.exceptions import DataSeedError, SchemaError
from helpers import student_sample

ROOT = Path(__file__).resolve().parent.parent


class FakeUpload(io.BytesIO):
    """Mimics Streamlit's UploadedFile (a BytesIO with a .name)."""

    def __init__(self, data: bytes, name: str):
        super().__init__(data)
        self.name = name


def csv_upload(n=40):
    return FakeUpload(student_sample(n).to_csv(index=False).encode(), "s.csv")


def test_read_uploaded_csv_and_excel_and_rejects_other_types(tmp_path):
    f = csv_upload()
    f.read()                                   # position moved: helper must rewind
    assert ui.read_uploaded_sample(f).shape == (40, 6) and ui.uploaded_sheets(f) == []
    p = tmp_path / "w.xlsx"
    with pd.ExcelWriter(p) as xw:
        pd.DataFrame({"a": [1, 2]}).to_excel(xw, sheet_name="one", index=False)
        pd.DataFrame({"b": [3, 4, 5]}).to_excel(xw, sheet_name="two", index=False)
    x = FakeUpload(p.read_bytes(), "w.xlsx")
    assert ui.uploaded_sheets(x) == ["one", "two"] and len(ui.read_uploaded_sample(x, "two")) == 3
    with pytest.raises(DataSeedError):
        ui.read_uploaded_sample(FakeUpload(b"x", "notes.txt"))


def test_schema_text_roundtrip_and_errors():
    s = infer_schema(student_sample(60)).schema
    assert ui.schema_from_text(ui.schema_to_text(s)).to_dict() == s.to_dict()
    with pytest.raises(SchemaError) as e:
        ui.schema_from_text('{"columns": [')
    assert "not valid JSON" in str(e.value)
    with pytest.raises(SchemaError):
        ui.schema_from_text('{"columns": [{"name": "a", "type": "integer", "min": 3, "max": 1}]}')


def test_parse_seed():
    assert ui.parse_seed("") is None and ui.parse_seed(None) is None and ui.parse_seed(" 42 ") == 42
    for bad in ("abc", "-1", "4294967296", "1.5"):
        with pytest.raises(DataSeedError):
            ui.parse_seed(bad)


def test_display_tables_for_sample_flow():
    sample = ui.read_uploaded_sample(csv_upload(60))
    inf = ui.analyse_sample(sample)
    assert list(ui.profile_table(inf).column) == [p.name for p in inf.profiles]
    out = ui.run_generation(sample=sample, schema=None, n_rows=500, seed=1)
    assert len(out.result.dataframe) == 500
    m = ui.methods_table(out)
    assert set(m.column) == set(out.result.schema.column_names)
    v = ui.validation_table(out.result.validation)
    assert {"status", "severity", "check", "message"} <= set(v.columns) and (v.status == "failed").sum() == 0
    q = ui.quality_tables(out.quality)
    assert not q["comparison"].empty and not q["numeric"].empty and not q["categories"].empty
    assert "marks" in q["correlation"].columns
    assert not ui.decisions_table(out, ("relationship",)).empty
    overview = ui.column_overview(out.result.schema)
    assert overview.loc[overview.column == "pass_status", "generation"].iloc[0].startswith("derived:")


def test_downloads_are_valid_files_and_report_is_strict_json():
    out = ui.run_generation(sample=None, schema=parse_description("Create 80 customer records with customer_id, age and city.").schema,
                            n_rows=80, seed=3, no_missing=True)
    csv = ui.dataset_bytes(out, "csv").decode()
    assert csv.splitlines()[0] == "customer_id,age,city" and len(csv.splitlines()) == 81
    assert pd.read_excel(io.BytesIO(ui.dataset_bytes(out, "xlsx"))).shape == (80, 3)
    rep = json.loads(ui.report_bytes(out))                 # strict JSON (no NaN literals)
    assert rep["generation"]["seed"] == 3 and "quality" in rep


def test_run_generation_surfaces_user_errors():
    s = parse_description("Create 10 rows with id and age. Age between 20 and 30.").schema
    with pytest.raises(DataSeedError):
        ui.run_generation(sample=None, schema=s, n_rows=50, seed=1, max_rows=10)
    with pytest.raises(ValueError):
        ui.run_generation(sample=None, schema=None, n_rows=5, seed=1)


def test_example_files_exist_in_project():
    ex = ui.example_files(ROOT)
    assert {"Student schema", "Employee schema", "Student sample (CSV)"} <= set(ex)


def test_ui_command_explains_missing_streamlit(capsys):
    import importlib.util
    if importlib.util.find_spec("streamlit") is not None:
        return                                              # real Streamlit present: do not launch a server in tests
    assert main(["ui"]) == 1
    assert "pip install" in capsys.readouterr().err
