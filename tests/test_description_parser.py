import pytest

from dataseed import generate_from_schema, parse_description

STUDENT = ("Create a student dataset with 5,000 records. Include student ID, age, gender, department, attendance "
           "percentage, marks, and pass status. Age should be between 18 and 25, attendance between 0 and 100, and "
           "marks between 0 and 100. A student passes if their marks are at least 40.")
CUSTOMER = ("Create customer records with customer_id, age, city, membership_type and annual_spending. Age must be "
            "between 18 and 80, and membership_type can be Basic, Silver or Gold.")


def test_student_example_is_fully_interpreted():
    r = parse_description(STUDENT)
    assert r.usable and r.n_rows == 5000 and r.domain == "student"
    s = r.schema
    assert s.column_names == ["student_id", "age", "gender", "department", "attendance_percentage", "marks", "pass_status"]
    assert (s.column("age").min, s.column("age").max) == (18, 25)
    assert (s.column("marks").min, s.column("marks").max) == (0, 100)   # the pass rule must not become a bound
    assert s.column("pass_status").expression == "where(marks >= 40, 'Pass', 'Fail')"
    assert s.n_rows == 5000 and not r.unsupported


def test_arbitrary_columns_and_categories():
    r = parse_description(CUSTOMER)
    s = r.schema
    assert s.column("customer_id").type == "identifier" and s.column("customer_id").prefix == "CUS-"
    assert s.column("membership_type").categories == ["Basic", "Silver", "Gold"]
    assert (s.column("age").min, s.column("age").max) == (18, 80)
    assert any("record count" in w.lower() for w in r.warnings)
    assert not any("placeholder" in w for w in r.warnings)


def test_unknown_column_name_gets_warning_not_silence():
    r = parse_description("Make data with zorblax, age and city. zorblax between 5 and 10.")
    assert r.schema.column("zorblax").min == 5 and r.schema.column("zorblax").max == 10
    r2 = parse_description("Make data with zorblax and age.")
    assert any("zorblax" in w and "could not be inferred" in w for w in r2.warnings)


def test_derived_arithmetic_missing_and_correlation_rules():
    r = parse_description("Generate 300 orders with order_id, quantity, unit_price and total. total = quantity * unit_price. "
                          "10% of quantity values are missing. unit_price and quantity are strongly negatively correlated.")
    s = r.schema
    assert s.column("total").expression == "quantity * unit_price"
    assert s.column("quantity").missing_rate == pytest.approx(0.10) and s.column("total").is_nullable
    assert s.relationships[0]["value"] == pytest.approx(-0.7)
    assert generate_from_schema(s, seed=1).dataframe.shape == (300, 4)


def test_unsupported_instructions_are_reported():
    r = parse_description("Create 100 rows with id, price and city. Prices must always rise on Mondays.")
    assert any("Mondays" in u for u in r.unsupported)


def test_domain_template_when_no_columns_given():
    r = parse_description("Employee dataset of 1k employees")
    assert r.n_rows == 1000 and "salary" in r.schema.column_names
    assert any("template" in a for a in r.assumptions)
    assert r.schema.relationships and generate_from_schema(r.schema, 50, seed=1).dataframe.shape[0] == 50


def test_gibberish_and_empty_fail_gracefully():
    for text in ("make me something nice", "   "):
        r = parse_description(text)
        assert not r.usable and r.warnings
    with pytest.raises(TypeError):
        parse_description(None)


def test_parsed_schema_can_be_saved_and_reloaded(tmp_path):
    from dataseed import load_schema, save_schema
    r = parse_description(STUDENT)
    p = save_schema(r.schema, tmp_path / "s.json")
    assert load_schema(p).to_dict() == r.schema.to_dict()
