"""Exception types used across DataSeed. All derive from :class:`DataSeedError`."""
from __future__ import annotations


class DataSeedError(Exception):
    """Base class for all errors raised deliberately by DataSeed."""


class SchemaError(DataSeedError):
    """A schema is malformed or internally inconsistent. ``problems`` lists every issue found."""

    def __init__(self, message: str, problems: list[str] | None = None):
        self.problems = list(problems or [])
        if self.problems and message:
            message = message + "\n" + "\n".join(f"  - {p}" for p in self.problems)
        super().__init__(message)


class CircularDependencyError(SchemaError):
    """Columns depend on each other in a cycle."""


class ExpressionError(DataSeedError):
    """An expression is syntactically invalid, unsafe, or cannot be evaluated."""


class InferenceError(DataSeedError):
    """Schema inference from a sample failed."""


class DescriptionParseError(DataSeedError):
    """A plain-English description could not be interpreted at all."""


class GenerationError(DataSeedError):
    """Generation could not be performed (bad request, unsupported settings...)."""


class ConstraintConflictError(GenerationError):
    """Constraints conflict or cannot be satisfied for the requested number of rows."""


class DataIOError(DataSeedError):
    """Reading or writing a file failed."""
