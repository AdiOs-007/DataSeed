"""High-level helpers combining generation + validation + quality analysis in one call.

The CLI uses these functions, and so can a future Streamlit app: they take in-memory objects and
return plain dataclasses (no printing, no files).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .analysis import analyze_quality
from .generator import generate_from_sample, generate_from_schema
from .models import GenerationResult, GenerationSettings, QualityReport
from .schema import Schema


@dataclass
class PipelineOutput:
    """Generation result plus quality report; ``to_dict()`` gives one JSON-ready report."""

    result: GenerationResult
    quality: QualityReport
    source: pd.DataFrame | None = None

    def to_dict(self) -> dict:
        d = self.result.to_dict()
        d["quality"] = self.quality.to_dict()
        return d


def run_pipeline(*, n_rows: int | None, seed: int | None = None, sample: pd.DataFrame | None = None,
                 schema: Schema | None = None, settings: GenerationSettings | None = None) -> PipelineOutput:
    """Generate from a ``sample`` XOR a ``schema`` and analyse the result."""
    if (sample is None) == (schema is None):
        raise ValueError("Provide exactly one of sample= or schema=.")
    if sample is not None:
        result = generate_from_sample(sample, n_rows, seed, settings)
    else:
        result = generate_from_schema(schema, n_rows, seed, settings)
    quality = analyze_quality(result.dataframe, source=sample, schema=result.schema, validation=result.validation)
    return PipelineOutput(result=result, quality=quality, source=sample)
