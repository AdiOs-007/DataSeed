import json

import pandas as pd
import pytest

import dataseed.io_utils as io_utils
from dataseed import (dataset_to_bytes, export_dataset, generate_from_schema, list_sheets, load_sample, load_schema,
                      save_report, save_schema, validate_dataset)
from dataseed.exceptions import DataIOError, SchemaError
from helpers import schema

S = schema({"name": "n", "type": "integer", "min": 1, "max": 9, "missing_rate": 0.1},
           {"name": "d", "type": "date", "min": "2020-01-01", "max": "2020-03-01", "date_format": "%d/%m/%Y"},
           {"name": "s", "type": "string", "text_kind": "name"})


def test_csv_loading_handles_duplicate_headers_blanks_and_placeholders(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("a,b,a,\n1,x,3,\n2,NA,4,\n,,,\n", encoding="utf-8")
    df = load_sample(p)
    assert list(df.columns) == ["a", "b", "a_2"] and len(df) == 2
    notes = " ".join(df.attrs["load_notes"])
    assert "renamed" in notes and "blank row" in notes and "blank column" in notes and "placeholder" in notes
    assert df.b.isna().tolist() == [False, True]


def test_csv_semicolon_bom_and_latin1(tmp_path):
    (tmp_path / "a.csv").write_bytes("\ufeffx;y\n1;2\n3;4\n".encode("utf-8"))
    assert load_sample(tmp_path / "a.csv").shape == (2, 2)
    (tmp_path / "b.csv").write_bytes("name,city\nJos\xe9,M\xfcnchen\n".encode("latin-1"))
    df = load_sample(tmp_path / "b.csv")
    assert df.name[0] == "José" and any("Latin-1" in n for n in df.attrs["load_notes"])


def test_excel_loading_with_sheet_selection(tmp_path):
    p = tmp_path / "w.xlsx"
    with pd.ExcelWriter(p) as xw:
        pd.DataFrame({"a": [1, 2]}).to_excel(xw, sheet_name="first", index=False)
        pd.DataFrame({"b": [5, 6, 7], "when": pd.to_datetime(["2020-01-01"] * 3)}).to_excel(xw, sheet_name="second", index=False)
    assert list_sheets(p) == ["first", "second"]
    assert list(load_sample(p).columns) == ["a"]
    df = load_sample(p, sheet_name="second")
    assert list(df.columns) == ["b", "when"] and len(df) == 3
    with pytest.raises(DataIOError):
        load_sample(p, sheet_name="nope")


def test_bad_inputs_raise_dataioerror(tmp_path):
    with pytest.raises(DataIOError):
        load_sample(tmp_path / "missing.csv")
    (tmp_path / "empty.csv").write_text("")
    with pytest.raises(DataIOError):
        load_sample(tmp_path / "empty.csv")
    (tmp_path / "hdr.csv").write_text("a,b\n")
    with pytest.raises(DataIOError):
        load_sample(tmp_path / "hdr.csv")
    (tmp_path / "x.txt").write_text("a")
    with pytest.raises(DataIOError) as e:
        load_sample(tmp_path / "x.txt")
    assert "Unsupported" in str(e.value)
    (tmp_path / "bad.xlsx").write_bytes(b"not a workbook")
    with pytest.raises(DataIOError):
        load_sample(tmp_path / "bad.xlsx")
    with pytest.raises(DataIOError):
        load_sample(tmp_path)


def test_export_csv_and_xlsx_roundtrip_with_formats_and_missing(tmp_path):
    df = generate_from_schema(S, 50, seed=1).dataframe
    csv = export_dataset(df, tmp_path / "deep" / "dir" / "o.csv", schema=S)
    back = load_sample(csv)
    assert len(back) == 50 and back.d[0].count("/") == 2
    assert back.n.isna().sum() == 5
    assert validate_dataset(back, S, expected_rows=50).is_valid          # read-back data validates
    xl = export_dataset(df, tmp_path / "o.xlsx", schema=S)
    back2 = load_sample(xl)
    assert len(back2) == 50 and pd.api.types.is_datetime64_any_dtype(back2.d)
    assert validate_dataset(back2, S).is_valid
    assert dataset_to_bytes(df, "csv").startswith(b"n,d,s") and len(dataset_to_bytes(df, "xlsx")) > 100


def test_export_errors(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(DataIOError):
        export_dataset(df, tmp_path / "o.parquet")
    with pytest.raises(DataIOError):
        export_dataset(df, tmp_path)
    export_dataset(df, tmp_path / "keep.csv")
    with pytest.raises(DataIOError):
        export_dataset(df, tmp_path / "keep.csv", overwrite=False)
    old = io_utils.EXCEL_MAX_ROWS
    io_utils.EXCEL_MAX_ROWS = 3
    try:
        with pytest.raises(DataIOError) as e:
            export_dataset(df, tmp_path / "big.xlsx")
        assert "worksheet limit" in str(e.value)
    finally:
        io_utils.EXCEL_MAX_ROWS = old


def test_schema_and_report_json_roundtrip(tmp_path):
    p = save_schema(S, tmp_path / "out" / "s.json")
    assert load_schema(p).to_dict() == S.to_dict()
    r = generate_from_schema(S, 30, seed=2)
    rp = save_report(r.to_dict(), tmp_path / "rep.json")
    data = json.loads(rp.read_text())                  # strict JSON: no NaN literals
    assert data["generation"]["seed"] == 2 and data["validation"]["is_valid"] is True
    save_report({"x": float("nan")}, tmp_path / "nan.json")
    assert json.loads((tmp_path / "nan.json").read_text())["x"] is None


def test_invalid_schema_files(tmp_path):
    (tmp_path / "bad.json").write_text("{not json")
    with pytest.raises(DataIOError):
        load_schema(tmp_path / "bad.json")
    (tmp_path / "inv.json").write_text(json.dumps({"columns": [{"name": "a", "type": "integer", "min": 5, "max": 1}]}))
    with pytest.raises(SchemaError) as e:
        load_schema(tmp_path / "inv.json")
    assert "inv.json" in str(e.value)
    with pytest.raises(DataIOError):
        load_schema(tmp_path / "nope.json")
    with pytest.raises(DataIOError):
        save_schema(S, tmp_path / "s.txt")
