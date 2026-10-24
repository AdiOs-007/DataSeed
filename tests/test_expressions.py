import pandas as pd
import pytest

from dataseed.exceptions import ExpressionError
from dataseed.expressions import evaluate_expression, referenced_columns

DF = pd.DataFrame({"marks": [10.0, 40.0, 90.0], "q": [1, 2, 3], "p": [2.0, 2.0, 2.0], "d": ["A", "B", "A"]})


def test_arithmetic_and_comparison():
    assert evaluate_expression("q * p", DF).tolist() == [2.0, 4.0, 6.0]
    assert evaluate_expression("marks >= 40", DF).tolist() == [False, True, True]


def test_where_chained_and_membership():
    assert evaluate_expression("where(marks >= 40, 'Pass', 'Fail')", DF).tolist() == ["Fail", "Pass", "Pass"]
    assert evaluate_expression("0 <= marks <= 50 and d in ['A']", DF).tolist() == [True, False, False]
    assert evaluate_expression("not (marks > 20)", DF).tolist() == [True, False, False]


def test_referenced_columns_excludes_functions():
    assert set(referenced_columns("round(q * p, 1) + marks")) == {"q", "p", "marks"}


@pytest.mark.parametrize("bad", ["__import__('os').system('x')", "marks.__class__", "[x for x in q]",
                                 "lambda: 1", "open('f')", "q if", "exec('1')", "(1).real"])
def test_unsafe_or_invalid_expressions_rejected(bad):
    with pytest.raises(ExpressionError):
        evaluate_expression(bad, DF)


def test_unknown_column_and_huge_power_are_errors_not_hangs():
    with pytest.raises(ExpressionError):
        evaluate_expression("nope + 1", DF)
    evaluate_expression("9 ** 9 ** 9", DF)   # float arithmetic: returns inf quickly, never hangs


def test_round_with_digits_and_functions():
    assert evaluate_expression("round(marks / 3, 1)", DF).tolist() == [3.3, 13.3, 30.0]
    assert evaluate_expression("clip(marks, 20, 50)", DF).tolist() == [20.0, 40.0, 50.0]
    assert evaluate_expression("max(q, 2)", DF).tolist() == [2.0, 2.0, 3.0]
