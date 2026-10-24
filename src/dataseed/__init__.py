"""DataSeed - Tiny Input, Massive Dataset.

Local, deterministic synthetic data generation from a small sample, a JSON schema or a
plain-English description, with validation and quality analysis. No network or API keys.
"""
from .analysis import analyze_quality
from .description_parser import parse_description
from .exceptions import (CircularDependencyError, ConstraintConflictError, DataIOError, DataSeedError,
                         DescriptionParseError, ExpressionError, GenerationError, InferenceError, SchemaError)
from .generator import generate_from_sample, generate_from_schema
from .inference import infer_schema, profile_dataframe
from .io_utils import (dataset_to_bytes, export_dataset, list_sheets, load_json, load_sample, load_schema,
                       save_json, save_report, save_schema)
from .models import (DescriptionParseResult, GenerationReport, GenerationResult, GenerationSettings,
                     InferenceResult, QualityReport, ValidationCheck, ValidationResult)
from .pipeline import PipelineOutput, run_pipeline
from .schema import ColumnSpec, Schema, schema_from_dict, schema_to_dict
from .validation import validate_dataset

__version__ = "0.2.0"

__all__ = [
    "load_sample", "list_sheets", "infer_schema", "profile_dataframe", "schema_from_dict", "schema_to_dict",
    "parse_description", "generate_from_sample", "generate_from_schema", "validate_dataset", "analyze_quality",
    "export_dataset", "dataset_to_bytes", "save_schema", "load_schema", "save_report", "save_json", "load_json",
    "run_pipeline", "PipelineOutput", "Schema", "ColumnSpec", "GenerationSettings", "GenerationResult",
    "GenerationReport", "InferenceResult", "DescriptionParseResult", "ValidationResult", "ValidationCheck",
    "QualityReport", "DataSeedError", "SchemaError", "CircularDependencyError", "ExpressionError",
    "InferenceError", "DescriptionParseError", "GenerationError", "ConstraintConflictError", "DataIOError",
]
