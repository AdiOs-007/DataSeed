"""All file input/output: CSV, XLSX, JSON schemas and reports.

Notes and limitations
---------------------
* CSV files are read and written as UTF-8 (a UTF-8 BOM is accepted on input). If a file is not
  valid UTF-8 it is re-read as Latin-1 and a note is recorded.
* Excel worksheets are limited to 1,048,576 rows (including the header); larger datasets must be
  exported as CSV. Excel (``.xlsx``) is read with OpenPyXL; formulas are read as cached values.
* ``load_sample`` records what it changed (renamed headers, dropped blank rows/columns...) in
  ``df.attrs["load_notes"]`` so nothing is changed silently.
"""
from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import pandas as pd

from .exceptions import DataIOError, SchemaError
from .schema import Schema, schema_from_dict

EXCEL_MAX_ROWS = 1_048_576
NA_TOKENS = {"", "na", "n/a", "nan", "null", "none"}
SUPPORTED_TABLE_EXTS = (".csv", ".xlsx")


def _ext(path: Any, file_type: str | None = None) -> str:
    if file_type:
        return "." + file_type.lower().lstrip(".")
    return Path(str(getattr(path, "name", path))).suffix.lower()


def list_sheets(source: str | Path | BinaryIO) -> list[str]:
    """Return the worksheet names of an ``.xlsx`` file."""
    try:
        with pd.ExcelFile(source, engine="openpyxl") as xl:
            return list(xl.sheet_names)
    except Exception as exc:
        raise DataIOError(f"Could not read Excel workbook {source!r}: {exc}") from exc


def _tidy_header(raw: list, notes: list[str]) -> list[str]:
    names: list[str] = []
    for i, h in enumerate(raw):
        name = "" if h is None or (isinstance(h, float) and math.isnan(h)) else str(h).strip()
        if not name:
            name = f"unnamed_{i + 1}"
            notes.append(f"Column {i + 1} had no header; named it '{name}'.")
        if name in names:
            k = 2
            while f"{name}_{k}" in names:
                k += 1
            notes.append(f"Duplicate column name '{name}' renamed to '{name}_{k}'.")
            name = f"{name}_{k}"
        names.append(name)
    return names


def _read_csv_raw(source, notes: list[str]) -> pd.DataFrame:
    data = source.read() if hasattr(source, "read") else Path(source).read_bytes()
    if isinstance(data, str):
        data = data.encode("utf-8")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("latin-1")
        notes.append("File is not valid UTF-8; decoded as Latin-1 (non-ASCII characters may differ).")
    if not text.strip():
        raise DataIOError("The file is empty.")
    try:
        delim = csv.Sniffer().sniff(text[:5000], delimiters=",;\t|").delimiter
    except csv.Error:
        delim = ","
    if delim != ",":
        notes.append(f"Detected delimiter {delim!r}.")
    try:
        return pd.read_csv(io.StringIO(text), sep=delim, header=None, dtype=str, keep_default_na=False,
                           skip_blank_lines=False)
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise DataIOError(f"Malformed CSV: {exc}") from exc


def load_sample(source: str | Path | BinaryIO, sheet_name: str | int | None = None,
                file_type: str | None = None) -> pd.DataFrame:
    """Load a small sample dataset from CSV or XLSX.

    Args:
        source: Path, or an open binary/text file object (e.g. an uploaded file in a UI; then
            pass ``file_type="csv"`` or ``"xlsx"``).
        sheet_name: Worksheet name/index for Excel (default: first sheet).
        file_type: Overrides the extension check.

    Returns:
        A DataFrame. CSV columns are returned as text (type inference happens in
        :func:`dataseed.infer_schema`); Excel keeps native cell types. Changes made while
        loading are listed in ``df.attrs["load_notes"]``.

    Raises:
        DataIOError: missing/empty/unreadable/unsupported files.
    """
    ext = _ext(source, file_type)
    if ext not in SUPPORTED_TABLE_EXTS:
        raise DataIOError(f"Unsupported file type {ext or '(none)'!r}. Supported: {', '.join(SUPPORTED_TABLE_EXTS)}")
    if not hasattr(source, "read"):
        p = Path(source)
        if not p.exists():
            raise DataIOError(f"File not found: {p}")
        if not p.is_file():
            raise DataIOError(f"Not a file: {p}")
    notes: list[str] = []
    try:
        if ext == ".csv":
            raw = _read_csv_raw(source, notes)
        else:
            raw = pd.read_excel(source, sheet_name=sheet_name if sheet_name is not None else 0,
                                header=None, engine="openpyxl")
    except DataIOError:
        raise
    except ValueError as exc:
        raise DataIOError(f"Could not read {source}: {exc}") from exc
    except Exception as exc:
        raise DataIOError(f"Could not read {source!r}: {exc}") from exc
    if raw.empty:
        raise DataIOError("The file contains no data.")
    raw = raw.astype(object).where(raw.notna(), None)
    header = _tidy_header(list(raw.iloc[0]), notes)
    df = raw.iloc[1:].reset_index(drop=True)
    df.columns = header
    if ext == ".csv":   # normalise missing tokens and whitespace
        n_tok = 0
        for c in df.columns:
            s = df[c].astype(object)
            stripped = s.map(lambda v: v.strip() if isinstance(v, str) else v)
            is_na = stripped.map(lambda v: v is None or (isinstance(v, str) and v.lower() in NA_TOKENS))
            n_tok += int((is_na & s.notna() & (s != "")).sum())
            df[c] = stripped.where(~is_na, None)
        if n_tok:
            notes.append(f"{n_tok} placeholder value(s) such as 'NA'/'null' were treated as missing.")
    blank_rows = df.isna().all(axis=1)
    if blank_rows.any():
        notes.append(f"Dropped {int(blank_rows.sum())} completely blank row(s).")
        df = df.loc[~blank_rows].reset_index(drop=True)
    blank_cols = [c for c in df.columns if df[c].isna().all()]
    if blank_cols:
        notes.append(f"Dropped {len(blank_cols)} completely blank column(s): {blank_cols}.")
        df = df.drop(columns=blank_cols)
    if df.empty or df.shape[1] == 0:
        raise DataIOError("The file has a header but no usable data rows/columns.")
    df = df.reset_index(drop=True)
    if ext != ".csv":
        df = df.infer_objects()
    df.attrs["load_notes"] = notes
    return df


def _excel_format(fmt: str) -> str:
    for a, b in (("%Y", "YYYY"), ("%y", "YY"), ("%m", "MM"), ("%d", "DD"), ("%H", "HH"),
                 ("%M", "mm"), ("%S", "ss"), ("%B", "MMMM"), ("%b", "MMM")):
        fmt = fmt.replace(a, b)
    return fmt


def _date_formats(df: pd.DataFrame, schema: Schema | None) -> dict[str, str]:
    fmts = dict(df.attrs.get("dataseed_date_formats", {}))
    if schema is not None:
        for c in schema.columns:
            f = c.effective_date_format()
            if f and c.name in df.columns:
                fmts[c.name] = f
    return {k: v for k, v in fmts.items() if k in df.columns and pd.api.types.is_datetime64_any_dtype(df[k])}


def _prepare_path(path: str | Path, overwrite: bool) -> Path:
    p = Path(path)
    if p.suffix.lower() not in SUPPORTED_TABLE_EXTS:
        raise DataIOError(f"Unsupported output type {p.suffix or '(none)'!r}. Use .csv or .xlsx.")
    if p.exists() and p.is_dir():
        raise DataIOError(f"Output path is a directory: {p}")
    if p.exists() and not overwrite:
        raise DataIOError(f"Refusing to overwrite existing file {p} (pass overwrite=True).")
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise DataIOError(f"Cannot create output directory {p.parent}: {exc}") from exc
    return p


def export_dataset(df: pd.DataFrame, path: str | Path, schema: Schema | None = None,
                   sheet_name: str = "data", overwrite: bool = True) -> Path:
    """Save ``df`` to ``.csv`` (UTF-8) or ``.xlsx``; creates parent folders. Returns the path.

    Date/datetime columns use the formats from ``schema`` (or the generated frame's metadata).
    """
    p = _prepare_path(path, overwrite)
    fmts = _date_formats(df, schema)
    try:
        if p.suffix.lower() == ".csv":
            out = df.copy()
            for c, f in fmts.items():
                out[c] = out[c].dt.strftime(f)
            out.to_csv(p, index=False, encoding="utf-8")
        else:
            if len(df) + 1 > EXCEL_MAX_ROWS:
                raise DataIOError(f"{len(df):,} rows exceed Excel's worksheet limit of {EXCEL_MAX_ROWS:,} rows. "
                                  "Export as .csv instead.")
            with pd.ExcelWriter(p, engine="openpyxl") as xw:
                df.to_excel(xw, index=False, sheet_name=sheet_name[:31])
                ws = xw.sheets[sheet_name[:31]]
                for c, f in fmts.items():
                    col = ws.cell(row=1, column=list(df.columns).index(c) + 1).column
                    for row in ws.iter_rows(min_row=2, min_col=col, max_col=col):
                        row[0].number_format = _excel_format(f)
    except DataIOError:
        raise
    except OSError as exc:
        raise DataIOError(f"Could not write {p}: {exc}") from exc
    return p


def dataset_to_bytes(df: pd.DataFrame, fmt: str = "csv", schema: Schema | None = None) -> bytes:
    """Serialise ``df`` to bytes (for download buttons in a future UI)."""
    fmt = fmt.lower().lstrip(".")
    if fmt not in ("csv", "xlsx"):
        raise DataIOError("fmt must be 'csv' or 'xlsx'.")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        p = export_dataset(df, Path(tmp) / f"data.{fmt}", schema=schema)
        return p.read_bytes()


# ---------------------------------------------------------------------- JSON
def _clean(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    if o is pd.NaT or o is pd.NA:
        return None
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if hasattr(o, "to_dict") and callable(o.to_dict) and not isinstance(o, pd.DataFrame):
        return _clean(o.to_dict())
    return o


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy/pandas values and NaN to plain JSON-safe Python values."""
    return _clean(obj)


def save_json(obj: Any, path: str | Path) -> Path:
    """Write ``obj`` (dict, dataclass with ``to_dict``...) as UTF-8 JSON; creates parent folders."""
    p = Path(path)
    if p.suffix.lower() != ".json":
        raise DataIOError(f"Expected a .json path, got {p}")
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(to_jsonable(obj), indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        raise DataIOError(f"Could not write {p}: {exc}") from exc
    return p


def load_json(path: str | Path) -> Any:
    p = Path(path)
    if not p.exists():
        raise DataIOError(f"File not found: {p}")
    try:
        return json.loads(p.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise DataIOError(f"{p} is not valid JSON: {exc}") from exc
    except OSError as exc:
        raise DataIOError(f"Could not read {p}: {exc}") from exc


def save_schema(schema: Schema, path: str | Path) -> Path:
    """Save a schema as editable JSON."""
    return save_json(schema.to_dict(), path)


def load_schema(path: str | Path) -> Schema:
    """Load and validate a schema from JSON. Raises DataIOError / SchemaError."""
    data = load_json(path)
    try:
        return schema_from_dict(data)
    except SchemaError as exc:
        raise SchemaError(f"{path}: {exc}", exc.problems) from exc


def save_report(report: Any, path: str | Path) -> Path:
    """Save any report object (generation/validation/quality/inference) as JSON."""
    return save_json(report, path)
