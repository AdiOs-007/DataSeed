"""Executes src/dataseed/streamlit_app.py against a stub (see fake_streamlit.py). A real-browser check is still advised."""
from pathlib import Path

from fake_streamlit import Up, run

EX = Path(__file__).resolve().parent.parent / "examples"


def sample_upload():
    data = (EX / "students_sample.csv").read_bytes()
    up = Up(data)
    up.name, up.size = "students_sample.csv", len(data)
    return up


def test_sample_flow_generates_valid_output():
    st, _ = run({"Generate dataset"}, 0, upload=sample_upload())
    out = st.session_state.output
    assert out is not None and out.result.validation.is_valid and len(out.result.dataframe) == 600
    assert st.session_state.inference is not None and "pass_status" in st.session_state.schema_text


def test_schema_flow_loads_example_and_generates():
    st, _ = run({"Load: Student schema"}, 1)
    text = st.session_state.schema_text
    assert '"student_id"' in text
    st, _ = run({"Generate dataset"}, 1, state={"schema_text": text, "output": None})
    assert len(st.session_state.output.result.dataframe) == 600


def test_plain_english_flow():
    txt = "Create a student dataset with 300 records. Include student ID, age, gender and marks. Age between 18 and 25."
    st, _ = run({"Interpret description"}, 2, text=txt)
    assert st.session_state.parse_result.usable and st.session_state.default_rows == 300
    st, _ = run({"Generate dataset"}, 2, state={"schema_text": st.session_state.schema_text, "default_rows": 300})
    assert st.session_state.output is not None


def test_bad_schema_text_shows_error_and_does_not_generate():
    st, calls = run({"Generate dataset"}, 1, state={"schema_text": "{bad json"})
    assert st.session_state.output is None
    assert any(name == "error" and "not valid JSON" in msg for name, msg in calls)
