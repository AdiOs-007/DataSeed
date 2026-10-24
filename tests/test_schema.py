import json

import pytest

from dataseed import ColumnSpec, Schema, schema_from_dict
from dataseed.exceptions import CircularDependencyError, SchemaError
from helpers import schema

GOOD = {"dataset_name": "s", "columns": [
    {"name": "id", "type": "identifier", "prefix": "S-", "width": 4},
    {"name": "age", "type": "int", "min": 18, "max": 25},
    {"name": "marks", "type": "float", "min": 0, "max": 100, "mean": 60, "std": 10, "distribution": "normal"},
    {"name": "passed", "type": "boolean", "expression": "marks >= 40"}],
    "relationships": [{"type": "correlation", "columns": ["age", "marks"], "value": 0.3}]}


def test_valid_schema_roundtrips_through_json():
    s = schema_from_dict(GOOD)
    assert s.column("age").type == "integer"            # alias normalised
    again = schema_from_dict(json.loads(s.to_json()))
    assert again.to_dict() == s.to_dict()
    assert s.column("id").is_unique and not s.column("age").is_nullable


@pytest.mark.parametrize("mutation,needle", [
    (lambda d: d["columns"][1].update(min=30), "greater than max"),
    (lambda d: d["columns"][1].update(type="blob"), "invalid type"),
    (lambda d: d["columns"].append(dict(d["columns"][1])), "Duplicate column"),
    (lambda d: d["relationships"][0].update(columns=["age", "ghost"]), "unknown column"),
    (lambda d: d["columns"][3].update(expression="ghost > 1"), "unknown column"),
    (lambda d: d["columns"][1].update(banana=1), "unknown properties"),
    (lambda d: d["columns"][2].update(missing_rate=1.5), "missing_rate"),
    (lambda d: d["columns"][1].update(prefix="x"), "not valid for type"),
    (lambda d: d["relationships"][0].update(value=1.5), "[-0.99, 0.99]"),
])
def test_invalid_schemas_are_rejected_with_clear_message(mutation, needle):
    d = json.loads(json.dumps(GOOD))
    mutation(d)
    with pytest.raises(SchemaError) as e:
        schema_from_dict(d)
    assert needle in str(e.value)


def test_malformed_categorical_probabilities():
    with pytest.raises(SchemaError):
        schema({"name": "c", "type": "categorical", "categories": ["a", "b"], "probabilities": [0.9, 0.9]})
    with pytest.raises(SchemaError):
        schema({"name": "c", "type": "categorical", "categories": ["a", "b"], "probabilities": [1.0]})
    with pytest.raises(SchemaError):
        schema({"name": "c", "type": "categorical", "categories": ["a", "a"]})
    s = schema({"name": "c", "type": "categorical", "probabilities": {"a": 0.25, "b": 0.75}})
    assert s.column("c").categories == ["a", "b"]


def test_nullable_conflicts_and_identifier_rules():
    with pytest.raises(SchemaError):
        schema({"name": "x", "type": "float", "min": 0, "max": 1, "nullable": False, "missing_rate": 0.1})
    with pytest.raises(SchemaError):
        schema({"name": "i", "type": "identifier", "missing_rate": 0.1})
    with pytest.raises(SchemaError) as e:   # derived non-nullable column fed by a nullable input
        schema({"name": "a", "type": "float", "min": 0, "max": 1, "missing_rate": 0.1},
               {"name": "b", "type": "boolean", "expression": "a > 0.5", "nullable": False})
    assert "not nullable" in str(e.value)


def test_circular_dependency_detected_with_cycle_in_message():
    with pytest.raises(CircularDependencyError) as e:
        schema({"name": "a", "type": "float", "expression": "b + 1"}, {"name": "b", "type": "float", "expression": "a + 1"})
    assert "a -> b -> a" in str(e.value) or "b -> a -> b" in str(e.value)
    with pytest.raises(SchemaError):
        schema({"name": "a", "type": "float", "expression": "a + 1"})


def test_generation_order_respects_dependencies():
    s = schema({"name": "total", "type": "float", "expression": "q * p"},
               {"name": "q", "type": "integer", "min": 1, "max": 5}, {"name": "p", "type": "float", "min": 1, "max": 2})
    assert s.generation_order().index("total") == 2


def test_relationship_shortcuts_merge_into_columns():
    s = schema_from_dict({"columns": [{"name": "a", "type": "integer", "min": 1, "max": 3},
                                      {"name": "b", "type": "boolean"}],
                          "relationships": [{"type": "derived", "column": "b", "expression": "a > 1"}]})
    assert s.column("b").expression == "a > 1" and s.relationships == []


def test_numeric_column_needs_enough_information_and_no_identifier_in_expressions():
    with pytest.raises(SchemaError):
        schema({"name": "x", "type": "float", "min": 0})
    with pytest.raises(SchemaError):
        schema({"name": "id", "type": "identifier"}, {"name": "f", "type": "boolean", "expression": "id > 1"})


def test_columnspec_python_construction_and_copy():
    s = Schema("p", [ColumnSpec("a", "integer", min=1, max=3)])
    s.validate()
    assert s.copy().to_dict() == s.to_dict()
