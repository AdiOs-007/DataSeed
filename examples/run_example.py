"""End-to-end demonstration: sample -> generate -> validate -> quality -> export.

Run from the project root (with the virtual environment active):
    python examples/run_example.py
Outputs go to the ``outputs/`` folder.
"""
from pathlib import Path

from dataseed import (analyze_quality, export_dataset, generate_from_sample, generate_from_schema, infer_schema,
                      load_sample, load_schema, save_report, validate_dataset)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"

# 1) Sample-driven generation -------------------------------------------------
sample = load_sample(ROOT / "examples" / "students_sample.csv")
print(f"Loaded sample: {sample.shape[0]} rows x {sample.shape[1]} columns")
inference = infer_schema(sample)
print("Inferred types:", {c.name: c.type for c in inference.schema.columns})

result = generate_from_sample(sample, n_rows=5000, seed=42)
df = result.dataframe
assert len(df) == 5000, "exact row count expected"
print(df.head())
print("Warnings:", *result.report.warnings, sep="\n  - ")

validation = validate_dataset(df, result.schema, expected_rows=5000)
print("Valid:", validation.is_valid, "| failed checks:", len(validation.failed))

quality = analyze_quality(df, source=sample, schema=result.schema, validation=validation)
print("Mean |corr difference| vs sample:", quality.comparisons["correlation_difference"]["mean_abs_difference"])

export_dataset(df, OUT / "example_students_from_sample.csv", schema=result.schema)
save_report(quality, OUT / "example_quality_report.json")

# 2) Schema-driven generation -------------------------------------------------
schema = load_schema(ROOT / "examples" / "student_schema.json")
result2 = generate_from_schema(schema, n_rows=5000, seed=42)
print(result2.dataframe.head())
export_dataset(result2.dataframe, OUT / "example_students_from_schema.xlsx", schema=schema)
print("Schema-driven valid:", result2.validation.is_valid)
print("Files written to", OUT)
