"""DataSeed Streamlit front end.

Run with:   python -m dataseed ui         (or:  streamlit run src/dataseed/streamlit_app.py)

All logic lives in the DataSeed API / ``ui_helpers``; this file only arranges widgets.
"""
from __future__ import annotations

import streamlit as st

from dataseed import __version__, parse_description
from dataseed.exceptions import DataSeedError
from dataseed.io_utils import load_sample
from dataseed.reporting import format_parse_result
from dataseed import ui_helpers as ui

MODES = ["Sample file (CSV / Excel)", "Schema (JSON)", "Plain English"]


def _state_default(key, value):
    if key not in st.session_state:
        st.session_state[key] = value


def _set_schema_text(text: str) -> None:
    """Replace the schema text and force the editor widget to refresh (Streamlit keeps keyed widget state)."""
    st.session_state.schema_text = text
    st.session_state.schema_version = st.session_state.get("schema_version", 0) + 1


def _show_error(exc: Exception) -> None:
    st.error(str(exc))


def _input_sample() -> None:
    file = st.file_uploader("Upload a small sample (.csv or .xlsx)", type=["csv", "xlsx"], key="sample_file")
    examples = ui.example_files()
    if "Student sample (CSV)" in examples and st.button("Use the example student sample"):
        st.session_state.sample = load_sample(examples["Student sample (CSV)"])
        _after_sample()
    if file is not None:
        sheet = None
        try:
            sheets = ui.uploaded_sheets(file)
            if len(sheets) > 1:
                sheet = st.selectbox("Worksheet", sheets)
            key = f"{file.name}|{sheet}|{getattr(file, 'size', 0)}"
            if st.session_state.get("sample_key") != key:
                st.session_state.sample = ui.read_uploaded_sample(file, sheet)
                st.session_state.sample_key = key
                _after_sample()
        except DataSeedError as exc:
            _show_error(exc)


def _after_sample() -> None:
    sample = st.session_state.sample
    inf = ui.analyse_sample(sample)
    st.session_state.inference = inf
    _set_schema_text(ui.schema_to_text(inf.schema))
    st.session_state.default_rows = 5000
    st.session_state.parse_result = None
    st.session_state.output = None


def _show_sample_analysis() -> None:
    inf, sample = st.session_state.get("inference"), st.session_state.get("sample")
    if inf is None or sample is None:
        return
    st.subheader("What DataSeed found in your sample")
    st.caption(f"{inf.n_rows} rows x {len(inf.profiles)} columns; {inf.duplicate_rows} duplicate rows")
    st.dataframe(ui.profile_table(inf), use_container_width=True)
    for title, items in (("Warnings", inf.warnings), ("Ambiguities - please review", inf.ambiguities),
                         ("Relationships found (or not)", inf.relationship_notes), ("Loading / conversion notes", inf.conversions)):
        if items:
            with st.expander(title, expanded=title == "Warnings"):
                for i in items:
                    st.write(f"- {i}")


def _input_schema() -> None:
    examples = ui.example_files()
    cols = st.columns(3)
    for col, name in zip(cols, ("Student schema", "Employee schema")):
        if name in examples and col.button(f"Load: {name}"):
            _set_schema_text(examples[name].read_text(encoding="utf-8"))
            st.session_state.sample = None
            st.session_state.inference = None
            st.session_state.parse_result = None
            st.session_state.output = None
            st.session_state.default_rows = 5000
    up = st.file_uploader("...or upload a schema .json", type=["json"], key="schema_file")
    if up is not None and st.session_state.get("schema_key") != f"{up.name}|{up.size}":
        _set_schema_text(up.getvalue().decode("utf-8-sig"))
        st.session_state.schema_key = f"{up.name}|{up.size}"
        st.session_state.sample = None
        st.session_state.inference = None
        st.session_state.output = None


def _input_text() -> None:
    text = st.text_area("Describe the dataset you want", height=150, key="description",
                        placeholder="Create a student dataset with 5,000 records. Include student ID, age, gender, "
                                    "department, attendance percentage, marks and pass status. Age should be between "
                                    "18 and 25 ... A student passes if their marks are at least 40.")
    if st.button("Interpret description"):
        result = parse_description(text)
        st.session_state.parse_result = result
        st.session_state.sample = None
        st.session_state.inference = None
        st.session_state.output = None
        if result.usable:
            _set_schema_text(ui.schema_to_text(result.schema))
            st.session_state.default_rows = result.n_rows or 5000
    res = st.session_state.get("parse_result")
    if res is not None:
        if res.usable:
            st.success("Description interpreted. Review the schema below before generating.")
        else:
            st.error("The description could not be turned into a schema. See the warnings below.")
        with st.expander("What was understood / assumed / not understood", expanded=True):
            st.text(format_parse_result(res))


def _schema_editor():
    """Show the editable schema; return a valid Schema or None."""
    text = st.session_state.get("schema_text")
    if not text:
        return None
    st.subheader("Review and edit the schema")
    try:
        schema = ui.schema_from_text(text)
        st.dataframe(ui.column_overview(schema), use_container_width=True)
    except DataSeedError as exc:
        schema = None
        _show_error(exc)
    with st.expander("Edit schema JSON", expanded=schema is None):
        edited = st.text_area("Schema JSON", value=text, height=320,
                              key=f"schema_editor_{st.session_state.get('schema_version', 0)}")
        st.session_state.schema_text = edited
        try:
            schema = ui.schema_from_text(edited)
            st.success("Schema is valid.")
        except DataSeedError as exc:
            schema = None
            _show_error(exc)
    return schema


def _generation_form(schema):
    st.subheader("Generate")
    c1, c2, c3 = st.columns(3)
    n_rows = c1.number_input("Rows", min_value=1, max_value=ui.MAX_UI_ROWS, value=int(st.session_state.get("default_rows", 5000)), step=100)
    seed_text = c2.text_input("Seed (blank = random)", value="42")
    no_missing = c3.checkbox("Force zero missing values", value=False)
    strict = st.checkbox("Stop with an error if a constraint cannot be met", value=True)
    use_sample = False
    if st.session_state.get("sample") is not None:
        use_sample = st.radio("Generate from", ["The sample (learn distributions & relationships)",
                                                "The schema above (as edited)"], horizontal=True).startswith("The sample")
    if st.button("Generate dataset", type="primary", disabled=(schema is None and not use_sample)):
        try:
            with st.spinner("Generating, validating and analysing..."):
                st.session_state.output = ui.run_generation(
                    sample=st.session_state.sample if use_sample else None, schema=None if use_sample else schema,
                    n_rows=int(n_rows), seed=ui.parse_seed(seed_text), no_missing=no_missing, strict=strict)
        except DataSeedError as exc:
            st.session_state.output = None
            _show_error(exc)


def _results() -> None:
    out = st.session_state.get("output")
    if out is None:
        return
    res = out.result
    st.divider()
    st.header("Results")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows", f"{len(res.dataframe):,}")
    m2.metric("Columns", res.dataframe.shape[1])
    m3.metric("Seed", res.seed)
    m4.metric("Validation", "VALID" if res.validation.is_valid else "INVALID")
    if res.report.warnings:
        with st.expander(f"{len(res.report.warnings)} warning(s)", expanded=True):
            for w in res.report.warnings:
                st.warning(w)
    tabs = st.tabs(["Data", "How it was generated", "Validation", "Quality", "Download"])
    with tabs[0]:
        st.dataframe(res.dataframe.head(500), use_container_width=True)
        st.caption("First 500 rows shown.")
    with tabs[1]:
        st.dataframe(ui.methods_table(out), use_container_width=True)
        for title, kinds in (("Relationships", ("relationship",)), ("Constraints", ("constraint",)),
                             ("Missing values", ("missing",)), ("Fallbacks and assumptions", ("fallback", "assumption"))):
            t = ui.decisions_table(out, kinds)
            if not t.empty:
                st.markdown(f"**{title}**")
                st.dataframe(t, use_container_width=True)
    with tabs[2]:
        st.dataframe(ui.validation_table(res.validation), use_container_width=True)
    with tabs[3]:
        tables = ui.quality_tables(out.quality)
        for w in out.quality.warnings:
            st.info(w)
        for key, title in (("comparison", "Source vs generated"), ("numeric", "Numeric summary"),
                           ("categories", "Category shares"), ("correlation", "Correlations (generated)")):
            if not tables[key].empty:
                st.markdown(f"**{title}**")
                st.dataframe(tables[key], use_container_width=True)
        st.caption("These metrics describe agreement with the sample; they do not prove the data is realistic or private.")
        with st.expander("What each metric means"):
            for k, v in out.quality.metric_notes.items():
                st.write(f"**{k}** - {v}")
    with tabs[4]:
        d1, d2, d3, d4 = st.columns(4)
        d1.download_button("Dataset (CSV)", ui.dataset_bytes(out, "csv"), "dataseed_output.csv", "text/csv")
        if len(res.dataframe) < 1_048_576:
            d2.download_button("Dataset (Excel)", ui.dataset_bytes(out, "xlsx"), "dataseed_output.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        d3.download_button("Schema (JSON)", ui.schema_to_text(res.schema).encode("utf-8"), "dataseed_schema.json", "application/json")
        d4.download_button("Full report (JSON)", ui.report_bytes(out), "dataseed_report.json", "application/json")


def render() -> None:
    st.set_page_config(page_title="DataSeed", page_icon="🌱", layout="wide")
    for k, v in (("schema_text", ""), ("sample", None), ("inference", None), ("parse_result", None), ("output", None)):
        _state_default(k, v)
    st.title("🌱 DataSeed - Tiny Input, Massive Dataset")
    st.caption(f"v{__version__} - runs locally; nothing is uploaded anywhere. Synthetic data is generated from the model "
               "DataSeed builds - it is not guaranteed to be realistic, anonymous or private.")
    mode = st.sidebar.radio("Choose an input method", MODES)
    st.sidebar.markdown("**Tips**\n- Samples under 30 rows give rough estimates.\n- Always review the schema before generating.\n"
                        "- The same seed always reproduces the same dataset.")
    st.header("Provide the input")
    {0: _input_sample, 1: _input_schema, 2: _input_text}[MODES.index(mode)]()
    if mode == MODES[0]:
        _show_sample_analysis()
    schema = _schema_editor()
    if schema is not None or st.session_state.get("sample") is not None:
        _generation_form(schema)
    _results()


render()
