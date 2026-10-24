# DataSeed — Tiny Input, Massive Dataset

DataSeed is a **local, offline Python backend + command-line tool** that turns a *small* input into a *large* synthetic dataset. It needs no API key, cloud service, database or internet connection at runtime.

Three ways to describe what you want:

| Input | Example |
|---|---|
| **Sample file** (CSV / Excel) | 40 student records → 5,000 new ones |
| **Manual schema** (JSON) | columns, types, ranges, categories, correlations, rules |
| **Plain English** | "Create a student dataset with 5,000 records. Age between 18 and 25 …" |

DataSeed then generates the rows, **validates** them, **measures quality** against the source, and exports CSV/XLSX plus JSON reports.

> **Honesty note.** DataSeed does *not* claim its output is "realistic", private, or anonymous. It preserves the structure it was told about or could measure (ranges, proportions, some correlations, rules) and reports how reliable that evidence is. A 20-row sample cannot reveal real-world distributions — you will see warnings saying so.

There is a **command-line interface** and a **Streamlit web interface** (`python -m dataseed ui`). Both are thin layers over the same Python API, so you can also call DataSeed from your own code or notebooks.

---

## 1. Folder structure

```text
/projects/dataseed
├── pyproject.toml            package, dependencies, extras [dev] and [ui], CLI entry point
├── BUILD_STATUS.md           progress log: what is done, tested, unfinished, how to resume
├── README.md
├── .gitignore
├── .vscode/settings.json     points VS Code at .venv
├── src/dataseed/
│   ├── __init__.py           public API (everything a UI needs is importable from here)
│   ├── __main__.py           enables `python -m dataseed`
│   ├── cli.py                thin command-line layer (argparse)
│   ├── streamlit_app.py      Streamlit web interface (widgets only)
│   ├── ui_helpers.py         framework-independent UI logic (tables, uploads, downloads); unit-tested
│   ├── pipeline.py           generate + validate + analyse in one call (CLI and UI share it)
│   ├── models.py             dataclasses: settings, results, reports
│   ├── schema.py             ColumnSpec / Schema, validation, JSON (de)serialisation
│   ├── expressions.py        SAFE expression evaluator (no eval/exec)
│   ├── distributions.py      marginal distributions (bounded, no boundary piles)
│   ├── relationships.py      Gaussian copula, conditional categories, rule detection
│   ├── inference.py          profiling + schema inference from a sample
│   ├── description_parser.py plain-English -> schema (deterministic, no LLM)
│   ├── generator.py          the generation engine
│   ├── textgen.py            synthetic names / emails / phones / text
│   ├── validation.py         post-generation validation engine
│   ├── analysis.py           statistical quality analysis + source comparison
│   ├── reporting.py          text rendering of results
│   ├── io_utils.py           CSV / XLSX / JSON input & output
│   └── exceptions.py
├── tests/                    pytest suite (135 tests; local fixtures only)
├── examples/
│   ├── students_sample.csv   40-row sample (correlated marks/attendance, pass rule, 2 missing values)
│   ├── student_schema.json   ranges, categories, identifier, derived column, correlation, constraint
│   ├── employee_schema.json  conditional categories, lognormal salary, derived bonus, datetime, unique emails
│   └── run_example.py        runnable end-to-end demo
└── outputs/                  generated files (git-ignored)
```

---

## 2. Setup on Pop!_OS (VS Code)

### 2.1 System packages (only needed once; these are system setup, not project dependencies)

```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip
```

### 2.2 Create and activate the virtual environment

```bash
cd /projects/dataseed
python3 -m venv .venv            # skip if .venv already exists
source /projects/dataseed/.venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,ui]"     # dev = pytest, ui = Streamlit (use ".[dev]" if you do not want the web UI)
```

Your prompt should now start with `(.venv)`. Never use `sudo pip`; every install goes into `.venv`.

### 2.3 VS Code

1. Open the folder: `cd /projects/dataseed && code .` (if the `code` command is missing, use *File → Open Folder…* and choose `/projects/dataseed`).
2. Install Microsoft's **Python** extension if prompted.
3. Press `Ctrl+Shift+P` → **Python: Select Interpreter** → choose `/projects/dataseed/.venv/bin/python`.
4. Open a new integrated terminal (``Ctrl+` ``). It should auto-activate `.venv`; if not, run the `source` command above.

---

## 3. Verify it works (first commands)

```bash
python -m dataseed --help
python -m pytest -q
python -m dataseed profile --input examples/students_sample.csv
```

## 4. Generate your first 5,000-row dataset

```bash
python -m dataseed generate \
  --input examples/students_sample.csv \
  --rows 5000 \
  --seed 42 \
  --output outputs/students.csv
```

Files appear in `/projects/dataseed/outputs/`:
`students.csv` (data) and `students_report.json` (generation decisions + validation + quality). The same seed always gives the same data; change `--seed 7` for a different dataset. If you omit `--seed` a random one is chosen and printed ("Reproduce with: --seed N").

---

## 4b. Use the web interface

```bash
python -m dataseed ui            # opens http://localhost:8501
```

Pick an input method in the sidebar (sample file, schema JSON, or plain English), review/edit the inferred schema, choose rows and seed, press **Generate**, then browse *Data / How it was generated / Validation / Quality* and download CSV, Excel, the schema and the full JSON report. Requires `pip install -e ".[ui]"`.

## 5. Command reference

All commands: `python -m dataseed <command> --help`. Exit codes: **0** success · **1** error or failed validation · **2** invalid usage · **3** unexpected bug (add `--debug` before the command for a traceback).

### `profile` — inspect a sample
```bash
python -m dataseed profile --input examples/students_sample.csv
python -m dataseed profile --input mydata.xlsx --sheet Sheet2 --output-schema outputs/inferred_schema.json
```
Shows inferred types, missing rates, confidence, conversions, ambiguities, detected rules/correlations and warnings.

### `generate` — from a sample, a schema, or text
```bash
# from a sample
python -m dataseed generate --input examples/students_sample.csv --rows 5000 --seed 42 --output outputs/students.csv

# from a schema (CSV or Excel output)
python -m dataseed generate --schema examples/student_schema.json --rows 5000 --seed 42 --output outputs/students.xlsx
python -m dataseed generate --schema examples/employee_schema.json --rows 2000 --seed 7 --output outputs/employees.csv

# from a plain-English description (the interpreted schema is printed first)
python -m dataseed generate --text "Create 500 customer records with customer_id, age, city, membership_type and annual_spending. Age must be between 18 and 80, and membership_type can be Basic, Silver or Gold." --seed 1 --output outputs/customers.csv
```
Useful options: `--output-schema PATH` (save the schema used), `--report PATH` / `--no-report`, `--no-missing` (force zero missing values), `--max-rows N` (safety limit, default 1,000,000), `--non-strict` (return data even if a constraint check fails), `--sheet NAME` (Excel input).
DataSeed refuses to write over its own input files.

### `describe` — plain English → editable schema
```bash
python -m dataseed describe \
  --text "Create a student dataset with student_id, age, gender and marks. Age must be between 18 and 25 and marks between 0 and 100." \
  --output-schema outputs/student_schema.json
```
Prints what was understood, the assumptions made, what was **not** understood and warnings. Edit the JSON, then generate with `--schema`. Add `--rows N --output outputs/x.csv` to generate immediately.

### `ui` — launch the web interface
```bash
python -m dataseed ui --port 8501
```

### `validate` — check a file against a schema
```bash
python -m dataseed generate --schema examples/student_schema.json --rows 5000 --seed 42 --output outputs/students.csv
python -m dataseed validate --input outputs/students.csv --schema examples/student_schema.json --output-report outputs/validation_report.json
```
Exit code 0 means no error-level check failed. (Validating a file against a *different* schema — e.g. a sample-derived file against `student_schema.json` — correctly reports failures.)

---

## 6. Python API

```python
from dataseed import (load_sample, infer_schema, schema_from_dict, parse_description,
                      generate_from_sample, generate_from_schema, validate_dataset,
                      analyze_quality, export_dataset, GenerationSettings)

sample = load_sample("examples/students_sample.csv")          # path or uploaded file object
inference = infer_schema(sample)                              # schema + profiles + warnings
result = generate_from_sample(sample, n_rows=5000, seed=42)
print(result.dataframe.head())
print(result.report.warnings)                                 # every decision: result.report.decisions
print(result.validation.is_valid)

quality = analyze_quality(result.dataframe, source=sample, schema=result.schema, validation=result.validation)
export_dataset(result.dataframe, "outputs/students.csv", schema=result.schema)

schema = schema_from_dict({"dataset_name": "students", "columns": [
    {"name": "student_id", "type": "identifier", "prefix": "STU-", "unique": True, "nullable": False},
    {"name": "age", "type": "integer", "min": 18, "max": 25},
    {"name": "marks", "type": "float", "min": 0, "max": 100}], "relationships": []})
result = generate_from_schema(schema, n_rows=5000, seed=42)

parsed = parse_description("Create 100 employee records with employee_id, age and salary.")
if parsed.usable:
    generate_from_schema(parsed.schema, parsed.n_rows, seed=1)
```
Everything returned is a documented dataclass with `.to_dict()`. Run `python examples/run_example.py` for a complete demo.

---

## 7. How generation works

**Pipeline:** validate request & schema → choose a marginal distribution per column → (optional) Gaussian copula for correlations → generate complete rows in dependency order → replace rows that break constraints → assign identifiers → apply missing values → convert dtypes → validate → report.

* **Schema-driven vs sample-driven.** Sample-driven generation first *infers a schema* (`infer_schema`) and then runs the same engine, so the inferred schema is inspectable, savable (`--output-schema`) and editable. Schema-driven uses exactly what you wrote; nothing is inferred.
* **Numeric columns:** truncated normal, uniform, lognormal, smoothed empirical (interpolated stored quantiles), discrete (few distinct values) or constant. Bounds are applied by *truncating the distribution*, not clipping, so no artificial spikes at min/max; integers get unbiased rounding. Sample-inferred choice: <8 values → normal (low confidence); otherwise normal if a Shapiro test doesn't reject it, lognormal if clearly right-skewed, else empirical.
* **Categorical / boolean:** observed or given probabilities; optional **conditional** distributions (`conditional_on`) — inferred only with ≥60 rows, strong chi-square evidence (p<0.01) and Cramér's V ≥ 0.25.
* **Numeric columns that depend on a category (new):** `conditional_on` + `conditional_params` give each group its own parameters (`{"Engineering": {"mean": 85000, "std": 30000}}`, or `{"quantiles": [...]}`); groups not listed use the column's overall distribution. Inferred from a sample only with ≥60 rows, Kruskal-Wallis p<0.01 and a non-trivial effect size (ε² ≥ 0.06), with ≥10 rows in at least two groups. Works together with copula correlations (groups shift/scale the same correlated uniforms).
* **Dates / datetimes:** uniform or empirical between bounds, exported with your `date_format`.
* **Identifiers:** `prefix + zero-padded number + suffix` (sequential or shuffled), unique, never modelled as measurements. Detected by column name *and* values.
* **Text:** synthetic names, `@example.com` emails, fictional `555-01XX` phones, addresses, companies, cities, words, or `pattern` templates (`#` digit, `?` letter). Source text is never reused. Small vocabularies → unique text gets counters appended (reported).
* **Relationships:**
  * *Correlations* — Gaussian copula: correlated normals → uniforms → each column's quantile function. Estimated from normal scores of ranks (needs ≥30 rows; entries below ~2/√n are treated as noise). Inconsistent matrices are repaired to positive-definite; a *user-specified* set that must change by >0.05 raises an error. Requested value is the latent correlation (exact Pearson for normal marginals, approximate otherwise); validation compares achieved vs requested.
  * *Derived columns* — `passed = marks >= 40`, `total = quantity * unit_price`, `where(cond, 'a', 'b')`. Sample inference also **detects** perfect threshold rules and arithmetic identities (and tells you, so you can verify they are real).
  * *Constraints* — row expressions, `unique_together`, column `unique`. Violating rows are **re-sampled** (targeted rejection sampling), never deleted, so the row count is exact. Impossible constraints raise `ConstraintConflictError` explaining the acceptance rate.
* **Missing values:** exact counts (`round(rate × rows)`) completely at random (MCAR) by default, or **missing-at-random (MAR)** via `missing_model` (new): `{"type":"group","column":"department","rates":{"Sales":0.2},"default":0.03}` or `{"type":"threshold","column":"age","op":">=","value":60,"rate_if_true":0.2,"rate_if_false":0.02}`. Group MAR is inferred from a sample only with ≥60 rows, ≥8 missing values, chi-square p<0.01 and Cramér's V ≥ 0.2; otherwise MCAR is kept (missingness that merely correlates with a group in a small sample is not promoted to a rule). The controlling column's *complete* generated values are used. Validation compares observed per-group rates with the model (warning severity). Rates are exact per group; derived columns are missing wherever an input is; non-nullable columns are never missing; `--no-missing` overrides.
* **Safe expressions:** parsed with `ast` and interpreted by a whitelist evaluator — **no `eval`/`exec`**. Supported: numbers, strings, `True/False`, `+ - * / // % **`, comparisons (chained OK), `in [...]`, `and/or/not`, `a if c else b`, and `abs round min max sqrt log exp floor ceil clip where isnull notnull year month days_between date`. Column names must be valid identifiers.

### Schema JSON cheat-sheet

Types: `integer, float, categorical, boolean, date, datetime, identifier, string`.

| Property | Applies to | Meaning |
|---|---|---|
| `nullable`, `missing_rate` | all but identifier | allow nulls; fraction missing (0–<1) |
| `min`, `max` | numeric, date | bounds (dates as ISO strings) |
| `distribution` | numeric, date | `auto, normal, uniform, lognormal, empirical, constant` |
| `mean`, `std`, `decimals` | numeric | parameters / rounding |
| `categories`, `probabilities` | categorical (and discrete numeric) | list; list or `{value: p}` |
| `conditional_on`, `conditional_probabilities` | categorical | `{parent_value: {category: p}}` |
| `conditional_on`, `conditional_params` | integer / float | `{parent_value: {"mean":..,"std":..}}` or `{"quantiles":[..]}` per group (normal / lognormal / empirical base distribution) |
| `missing_model` | any non-derived, non-identifier | MAR pattern: `group` or `threshold` (see above) |
| `probability` | boolean | P(True) |
| `unique` | most | no duplicates (identifiers always unique) |
| `prefix`, `suffix`, `width`, `start`, `sequence` | identifier | `sequence`: `sequential` / `shuffled` |
| `text_kind`, `pattern`, `domain` | string | `name, email, phone, address, company, city, word, sentence, username, first_name, last_name, template` |
| `expression`, `depends_on` | any | derived column / ordering hint |
| `date_format` | date | strftime used on export |
| `default` | any | constant value when nothing else defines the column |

Top level: `dataset_name`, `description`, `generation` (`n_rows`, `seed`), `columns`, `relationships` (`{"type":"correlation","columns":[a,b],"value":0.6}`; also `derived` / `dependency` shortcuts), `constraints` (`{"type":"expression","expression":"..."}`, `{"type":"unique_together","columns":[...]}`). See `examples/*.json`.

---

## 8. Plain-English parsing — how it works and where it fails

Deterministic regex/heuristic parsing (no LLM). It finds the record count, a domain (student, employee, sales, customer, fitness), explicit column lists ("with a, b and c", "including …"), then column rules: ranges, bounds, categories ("can be A, B or C"), uniqueness, missing %, mean/std, derived rules ("passes if marks are at least 40", `total = quantity * unit_price`) and correlations ("A and B are strongly positively correlated"). Unknown column names get name-based defaults (`age` → 18–65, `*_id` → identifier, …) **recorded as assumptions**.

It **fails** on: unusual phrasing, negation, conditions spanning sentences, per-group rules, free-form constraints. Those sentences are listed under *NOT understood*, not silently ignored. Always review the interpreted schema (`describe --output-schema`).

## 9. Limitations (read before trusting results)

* Tiny samples (<30 rows) give rough distributions; correlations/conditionals are not inferred from them. Warnings say so.
* The observed min/max of a sample are used as bounds when inferring; widen them in the schema if reality is wider.
* Correlations are linear/rank-based via a Gaussian copula; no non-linear or multi-column dependencies are learned beyond the detected rules. Correlations are estimated over all rows, so strong group differences can inflate a correlation that disappears *within* groups.
* Missingness is completely-at-random unless a `missing_model` is given or detected; only group-based MAR is inferred (threshold models are manual), and not-at-random mechanisms (missingness depending on the missing value itself) are not modelled. Rows with rule/constraint rejection are drawn from the model *conditioned* on the constraints.
* No privacy guarantee: output rows are generated, not copied (the quality report counts exact overlaps with the source), but this is **not** anonymisation or differential privacy.
* Quality metrics (KS distance, total-variation distance, correlation difference) say how closely summary statistics agree; they do not prove realism. `metric_notes` in the report explains each.
* Excel worksheets hold at most 1,048,576 rows (export CSV beyond that). CSV is UTF-8. Default maximum is 1,000,000 rows (`--max-rows`).
* Parsed boolean labels (Yes/No, 0/1) are exported as True/False.

## 10. How the Streamlit UI is built (and how to extend it)

`streamlit_app.py` only arranges widgets; everything else is reusable API:

* `ui_helpers.read_uploaded_sample(file, sheet)` / `load_sample(file, file_type=...)` accept Streamlit upload objects; `uploaded_sheets()` fills a worksheet picker.
* `parse_description`, `infer_schema` and `Schema.from_dict/to_dict` give editable, JSON-ready structures (the UI shows the schema as a table plus an editable JSON box that is re-validated on every edit).
* `ui_helpers.run_generation(...)` → `run_pipeline(...)` returns generation + validation + quality in one object.
* `dataset_to_bytes` / `report_bytes` feed `st.download_button`.
* Everything raises `DataSeedError` subclasses with user-readable messages; the UI shows them with `st.error`.

To add a page or widget, add a function to `streamlit_app.py` that calls these helpers; put any non-widget logic in `ui_helpers.py` and unit-test it (`tests/test_ui_helpers.py`).

**Testing status:** the helper logic is unit-tested. The app script itself is executed in tests against a recording *stub* of Streamlit (`tests/fake_streamlit.py`), which checks the flows run and reach the right states; it is not a browser test. Open the UI once and click through the three input methods to confirm the layout in a real browser.

## 11. Troubleshooting (Pop!_OS)

| Problem | Fix |
|---|---|
| `The virtual environment was not created successfully… ensurepip is not available` | `sudo apt install python3-venv python3-pip`, delete `.venv`, recreate it |
| `externally-managed-environment` error | you are outside the venv — run `source /projects/dataseed/.venv/bin/activate` first |
| `No module named dataseed` | venv not active or not installed: `python -m pip install -e ".[dev]"` |
| `No module named pytest` | install the dev extra (command above) |
| `Streamlit is not installed` | `python -m pip install -e ".[ui]"` |
| UI shows stale schema after loading an example | refresh the page (`R`) — session state is per browser tab |
| VS Code runs the wrong Python | `Ctrl+Shift+P` → *Python: Select Interpreter* → `/projects/dataseed/.venv/bin/python` |
| `Error: --rows is required` | add `--rows N` (or set `generation.n_rows` in the schema) |
| `Constraints cannot be satisfied …` | the message names the constraint; widen ranges, relax it, or request fewer rows (for `unique`) |
| `Refusing to overwrite the input file` | choose a different `--output` |
| Excel row-limit error | export to `.csv` |
| `python3 -m venv` works but `python` isn't found | activate the venv; inside it `python` is available |

Tests: `python -m pytest -q`.
