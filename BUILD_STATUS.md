# DataSeed build status

_Last updated: end of iteration 2 (version 0.2.0)_

## Environment note (important)
`/projects/dataseed` was NOT accessible to the assistant. All work was done in a sandbox copy and delivered as `dataseed_v2.zip`.
Nothing was written to your machine by the assistant. To adopt it: unzip over `/projects/dataseed` (keep your `.venv`), then:

    cd /projects/dataseed && source .venv/bin/activate
    python -m pip install -e ".[dev,ui]"
    python -m pytest -q
    python -m dataseed ui

## Completed
Iteration 1: CSV/XLSX loading, profiling + schema inference, JSON schemas, plain-English parser, generation engine
(numeric/categorical/boolean/date/datetime/identifier/text; Gaussian copula; derived columns; constraints by re-sampling;
exact missing counts), validation engine, quality analysis, CSV/XLSX/JSON export, CLI (profile/describe/generate/validate),
examples, README.

Iteration 2 (this delivery):
- Numeric columns that depend on a categorical column (`conditional_on` + `conditional_params`), manual and inferred.
- Missing-at-random patterns (`missing_model`: group / threshold), manual; group-based MAR inferred; validation check.
- Streamlit web interface (`python -m dataseed ui`, `src/dataseed/streamlit_app.py`) + framework-free `ui_helpers.py`.
- `ui` CLI command, `[ui]` extra, version 0.2.0, `GenerationSettings.infer_missing_patterns`.
- Parser dead-code cleanup; richer `employee_schema.json`; README updated.

## Files created / modified in iteration 2
Created: BUILD_STATUS.md, src/dataseed/ui_helpers.py, src/dataseed/streamlit_app.py, tests/test_dependencies.py,
tests/test_ui_helpers.py, tests/test_streamlit_app.py, tests/fake_streamlit.py.
Modified: schema.py, generator.py, relationships.py, inference.py, validation.py, models.py, cli.py, __init__.py,
description_parser.py, pyproject.toml, README.md, examples/employee_schema.json, tests/test_examples.py.

## Tests actually run (sandbox, Python 3.12, pandas 3.0.2, numpy 2.4, scipy 1.17)
- 135 tests, 135 passed - run with a **stand-in runner** (`/tmp/pytest_shim.py`, not shipped) because real pytest could not be
  installed offline. The tests use only plain pytest features (tmp_path, raises, approx, parametrize, capsys).
  **Please run `python -m pytest -q` yourself.**
- Fresh editable install (`pip install -e . --no-deps`), `python -m dataseed --help`, profile, generate (sample, schema,
  CSV and XLSX), describe, validate (exit 0), `ui` (clean error without Streamlit), `examples/run_example.py`: all executed OK.
- Performance: 1,000,000 rows in ~5 s (student schema); 200,000 employee rows in ~2 s.
- **Streamlit was NOT installed in the sandbox**, so the UI has never run in a real Streamlit server/browser. Its logic is
  covered by unit tests of `ui_helpers.py` and by executing `streamlit_app.py` against a recording stub. Widget layout/API
  typos specific to your installed Streamlit version could still appear - see "Known issues".

## Known issues / limitations
- UI untested in a real browser (see above). If a Streamlit call errors on your version (written against >= 1.32 APIs:
  `st.dataframe(use_container_width=...)`, `st.tabs`, `st.download_button`), fix in `streamlit_app.py` only.
- Plain-English parser does not understand numeric-by-category or MAR phrasing; use JSON schema (`conditional_params`,
  `missing_model`) or sample inference for those.
- Numeric group parameters are not checked by validation (truncation shifts group means); MAR rates are checked.
- Correlations are estimated over all rows (can be inflated by group differences).
- Threshold-based MAR is manual only; not-at-random missingness is not modelled.
- No privacy/anonymity guarantee; metrics do not prove realism.

## Next tasks (dependency order)
1. Run `python -m pytest -q` on your machine; report any failure.
2. `python -m pip install -e ".[ui]" && python -m dataseed ui`; click through the three input modes; fix any Streamlit-version issue.
3. Optional: parser phrases for conditional/MAR rules; infer threshold MAR; schema editor as a form instead of raw JSON.
