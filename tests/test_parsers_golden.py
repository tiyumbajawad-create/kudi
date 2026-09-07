"""Golden-file tests: fixture in, expected canonical values out (§11).

All five fixtures encode the *same two* underlying transactions (a $4.75
coffee and a $2600 payroll deposit on the same two dates), each dressed
in that format's own quirks. This is the cross-format convergence test
from §8/§11: every parser must agree on amount, date, and description
regardless of how differently the source file is shaped.
"""

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from kudi.ingest.parsers import boa, capital_one, chase, generic_credit, ofx
from kudi.ingest.parsers.base import ParseResult

FIXTURES = Path(__file__).parent / "fixtures" / "ingest"

EXPECTED = [
    {"date": date(2026, 1, 15), "amount": Decimal("-4.75"), "description_contains": "BLUE BOTTLE"},
    {
        "date": date(2026, 1, 16),
        "amount": Decimal("2600.00"),
        "description_contains": "ACME CORP PAYROLL",
    },
]


def _assert_matches_expected(result: ParseResult) -> None:
    assert not result.failures, result.failures
    assert len(result.records) == len(EXPECTED)
    for record, expected in zip(result.records, EXPECTED, strict=True):
        assert record.posted_date == expected["date"]
        assert record.amount == expected["amount"]
        assert expected["description_contains"] in record.raw_description


def test_chase_golden() -> None:
    text = (FIXTURES / "chase_sample.csv").read_text()
    _assert_matches_expected(chase.parse(text))


def test_boa_golden_skips_preamble() -> None:
    text = (FIXTURES / "boa_sample.csv").read_text()
    result = boa.parse(text)
    assert result.skipped == 5, "5 preamble lines before the real header"
    _assert_matches_expected(result)


def test_capital_one_golden_ignores_category_hint() -> None:
    text = (FIXTURES / "capital_one_sample.csv").read_text()
    result = capital_one.parse(text)
    _assert_matches_expected(result)
    # the bank's category hint is kept as a hint, but it's not trusted --
    # the fixture deliberately supplies wrong ones.
    assert result.records[0].category_hint == "Travel-Airfare"


def test_generic_credit_golden_strips_separators_and_skips_total() -> None:
    text = (FIXTURES / "generic_credit_sample.csv").read_text()
    result = generic_credit.parse(text)
    assert result.skipped == 1, "trailing TOTAL row is not a transaction"
    _assert_matches_expected(result)


def test_ofx_golden_carries_fitid_and_account() -> None:
    text = (FIXTURES / "sample.ofx").read_text()
    result = ofx.parse(text)
    _assert_matches_expected(result)
    assert result.records[0].account_id == "chase-checking"
    assert result.records[0].fitid == "fitid-0001"
    assert result.records[1].fitid == "fitid-0002"


@pytest.mark.parametrize(
    "fixture,parser",
    [
        ("chase_sample.csv", chase.parse),
        ("boa_sample.csv", boa.parse),
        ("capital_one_sample.csv", capital_one.parse),
        ("generic_credit_sample.csv", generic_credit.parse),
        ("sample.ofx", ofx.parse),
    ],
)
def test_all_formats_converge_on_same_canonical_values(fixture: str, parser) -> None:
    """The actual cross-format golden test: every one of the 5 fixtures,
    despite completely different byte-level shapes, parses to the exact
    same (date, amount, description-fragment) pairs."""
    text = (FIXTURES / fixture).read_text()
    _assert_matches_expected(parser(text))
