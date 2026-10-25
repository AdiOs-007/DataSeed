"""Command-line interface: ``python -m dataseed <command>``.

Commands: profile, describe, generate, validate, ui.  Exit codes: 0 success; 1 error or failed
validation; 2 invalid command-line usage; 3 unexpected internal error.
"""
from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path

from .description_parser import parse_description
from .exceptions import DataSeedError
from .inference import infer_schema
from .io_utils import (export_dataset, load_sample, load_schema, save_report, save_schema)
from .models import GenerationSettings
from .pipeline import run_pipeline
from .reporting import format_generation, format_inference, format_parse_result, format_quality, format_validation
from .schema import Schema
from .validation import validate_dataset

EXIT_OK, EXIT_FAIL, EXIT_USAGE, EXIT_BUG = 0, 1, 2, 3


class UsageError(DataSeedError):
    """Bad combination of command-line options."""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dataseed", description="DataSeed - Tiny Input, Massive Dataset")
    p.add_argument("--debug", action="store_true", help="show full tracebacks")
    sub = p.add_subparsers(dest="command", required=True)

    pr = sub.add_parser("profile", help="profile a sample CSV/XLSX and show the inferred schema")
    pr.add_argument("--input", required=True, help="sample .csv or .xlsx")
    pr.add_argument("--sheet", help="Excel worksheet name (default: first)")
    pr.add_argument("--output-schema", help="save the inferred schema as JSON")
    pr.add_argument("--output-report", help="save the full profile as JSON")

    de = sub.add_parser("describe", help="turn a plain-English description into a schema (and optionally data)")
    g = de.add_mutually_exclusive_group(required=True)
    g.add_argument("--text", help="the description")
    g.add_argument("--text-file", help="a UTF-8 text file containing the description")
    de.add_argument("--output-schema", help="save the interpreted schema as JSON")
    de.add_argument("--rows", type=int, help="rows to generate (default: the count stated in the text)")
    de.add_argument("--seed", type=int)
    de.add_argument("--output", help="if given, also generate data to this .csv/.xlsx")

    ge = sub.add_parser("generate", help="generate a synthetic dataset")
    src = ge.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="sample .csv/.xlsx to learn from")
    src.add_argument("--schema", help="schema .json")
    src.add_argument("--text", help="plain-English description")
    ge.add_argument("--sheet", help="Excel worksheet name")
    ge.add_argument("--rows", type=int, help="number of rows")
    ge.add_argument("--seed", type=int, help="random seed (default: random; always printed)")
    ge.add_argument("--output", required=True, help="output .csv or .xlsx")
    ge.add_argument("--output-schema", help="also save the schema that was used")
    ge.add_argument("--report", help="JSON report path (default: <output>_report.json)")
    ge.add_argument("--no-report", action="store_true", help="do not write a JSON report")
    ge.add_argument("--no-missing", action="store_true", help="force zero missing values")
    ge.add_argument("--max-rows", type=int, default=1_000_000, help="safety limit (default 1,000,000)")
    ge.add_argument("--non-strict", action="store_true", help="return data even if a constraint check fails")

    va = sub.add_parser("validate", help="validate a dataset file against a schema")
    va.add_argument("--input", required=True)
    va.add_argument("--schema", required=True)
    va.add_argument("--sheet")
    va.add_argument("--rows", type=int, help="expected number of rows")
    va.add_argument("--output-report", help="save the validation result as JSON")
    va.add_argument("--show-passed", action="store_true")
    ui = sub.add_parser("ui", help="launch the Streamlit web interface (needs: pip install -e '.[ui]')")
    ui.add_argument("--port", type=int, default=8501)
    return p


def _same(a: str | None, b: str | None) -> bool:
    return bool(a and b) and Path(a).resolve() == Path(b).resolve()


def _text_arg(args) -> str:
    if getattr(args, "text_file", None):
        try:
            return Path(args.text_file).read_text(encoding="utf-8")
        except OSError as exc:
            raise UsageError(f"Cannot read {args.text_file}: {exc}") from exc
    return args.text


def _cmd_profile(a) -> int:
    inf = infer_schema(load_sample(a.input, sheet_name=a.sheet))
    print(format_inference(inf))
    for path, obj, label in ((a.output_schema, inf.schema, "schema"), (a.output_report, inf, "profile")):
        if path:
            if _same(path, a.input):
                raise UsageError("Refusing to overwrite the input file.")
            print(f"\nSaved {label}: {save_schema(obj, path) if label == 'schema' else save_report(obj, path)}")
    return EXIT_OK


def _generate_and_save(schema: Schema | None, sample, a, n_rows: int | None, out: str) -> int:
    settings = GenerationSettings(max_rows=getattr(a, "max_rows", 1_000_000),
                                  missing_policy="none" if getattr(a, "no_missing", False) else "schema",
                                  strict=not getattr(a, "non_strict", False))
    out_p = Path(out)
    for other in (getattr(a, "input", None), getattr(a, "schema", None)):
        if _same(out, other):
            raise UsageError(f"--output must differ from the input file ({other}); refusing to overwrite it.")
    run = run_pipeline(n_rows=n_rows, seed=a.seed, sample=sample, schema=schema, settings=settings)
    res = run.result
    path = export_dataset(res.dataframe, out_p, schema=res.schema)
    print(format_generation(res))
    print("\n" + format_quality(run.quality))
    print(f"\nDataset saved: {path.resolve()}  ({len(res.dataframe):,} rows)")
    if getattr(a, "output_schema", None):
        if _same(a.output_schema, getattr(a, "schema", None)):
            raise UsageError("--output-schema must differ from --schema.")
        print(f"Schema saved: {save_schema(res.schema, a.output_schema).resolve()}")
    if not getattr(a, "no_report", False):
        rp = Path(a.report) if getattr(a, "report", None) else out_p.with_name(out_p.stem + "_report.json")
        print(f"Report saved: {save_report(run.to_dict(), rp).resolve()}")
    print(f"Reproduce with: --seed {res.seed}")
    return EXIT_OK


def _cmd_describe(a) -> int:
    parsed = parse_description(_text_arg(a))
    print(format_parse_result(parsed))
    if not parsed.usable:
        return EXIT_FAIL
    if a.output_schema:
        print(f"\nSchema saved: {save_schema(parsed.schema, a.output_schema).resolve()}  (edit it, then use --schema)")
    if a.output:
        n = a.rows if a.rows is not None else parsed.n_rows
        if n is None:
            raise UsageError("No row count: add '--rows N' or state it in the text (e.g. '5,000 records').")
        print()
        return _generate_and_save(parsed.schema, None, a, n, a.output)
    return EXIT_OK


def _cmd_generate(a) -> int:
    if a.input:
        sample = load_sample(a.input, sheet_name=a.sheet)
        if a.rows is None:
            raise UsageError("--rows is required with --input.")
        return _generate_and_save(None, sample, a, a.rows, a.output)
    if a.schema:
        schema = load_schema(a.schema)
        if a.rows is None and not schema.n_rows:
            raise UsageError("--rows is required (the schema has no generation.n_rows).")
        return _generate_and_save(schema, None, a, a.rows, a.output)
    parsed = parse_description(a.text)
    print(format_parse_result(parsed) + "\n")
    if not parsed.usable:
        return EXIT_FAIL
    n = a.rows if a.rows is not None else parsed.n_rows
    if n is None:
        raise UsageError("No row count: add --rows N or state it in the text.")
    return _generate_and_save(parsed.schema, None, a, n, a.output)


def _cmd_validate(a) -> int:
    schema = load_schema(a.schema)
    df = load_sample(a.input, sheet_name=a.sheet)
    v = validate_dataset(df, schema, expected_rows=a.rows)
    print(format_validation(v, show_passed=a.show_passed))
    if a.output_report:
        if _same(a.output_report, a.input):
            raise UsageError("Refusing to overwrite the input file.")
        print(f"\nReport saved: {save_report(v, a.output_report).resolve()}")
    return EXIT_OK if v.is_valid else EXIT_FAIL


def _cmd_ui(a) -> int:
    import importlib.util
    import subprocess
    if importlib.util.find_spec("streamlit") is None:
        raise DataSeedError("Streamlit is not installed in this environment. Run:\n"
                            "  python -m pip install -e \".[ui]\"\nthen retry: python -m dataseed ui")
    app = Path(__file__).with_name("streamlit_app.py")
    print(f"Starting DataSeed UI on http://localhost:{a.port} (Ctrl+C to stop)")
    theme = ["--theme.base", "dark", "--theme.primaryColor", "#6effc8", "--theme.backgroundColor", "#05080f",
             "--theme.secondaryBackgroundColor", "#0a1220", "--theme.textColor", "#d8e6ee"]
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app), "--server.port", str(a.port), *theme])


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns the process exit code."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.WARNING, format="%(levelname)s: %(message)s")
    handlers = {"profile": _cmd_profile, "describe": _cmd_describe, "generate": _cmd_generate, "validate": _cmd_validate, "ui": _cmd_ui}
    try:
        return handlers[args.command](args)
    except BrokenPipeError:   # e.g. output piped to `head`
        return EXIT_OK
    except UsageError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except DataSeedError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        if args.debug:
            traceback.print_exc()
        return EXIT_FAIL
    except Exception as exc:  # unexpected: show a short message, full trace with --debug
        print(f"Unexpected error ({type(exc).__name__}): {exc}\nRe-run with --debug for details.", file=sys.stderr)
        if args.debug:
            traceback.print_exc()
        return EXIT_BUG


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
