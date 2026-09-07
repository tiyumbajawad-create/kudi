"""Format detection tests (§4.2): correct identification of all 5
formats, and a loud refusal-to-guess on unrecognizable input."""

from pathlib import Path

from kudi.ingest.detect import CONFIDENCE_THRESHOLD, detect_format

FIXTURES = Path(__file__).parent / "fixtures" / "ingest"


def test_detects_chase() -> None:
    result = detect_format((FIXTURES / "chase_sample.csv").read_text())
    assert result.format_name == "chase_csv"
    assert result.confidence == 1.0


def test_detects_boa_despite_preamble() -> None:
    result = detect_format((FIXTURES / "boa_sample.csv").read_text())
    assert result.format_name == "boa_csv"
    assert result.confidence == 1.0


def test_detects_capital_one() -> None:
    result = detect_format((FIXTURES / "capital_one_sample.csv").read_text())
    assert result.format_name == "capital_one_csv"
    assert result.confidence == 1.0


def test_detects_generic_credit_not_confused_with_chase() -> None:
    """Regression test: generic_credit_csv's 4-column header was once
    scoring a false 1.0 against chase_csv's 5-column header because the
    naive comma-joined fuzzy match ignored column count entirely."""
    result = detect_format((FIXTURES / "generic_credit_sample.csv").read_text())
    assert result.format_name == "generic_credit_csv"


def test_detects_ofx() -> None:
    result = detect_format((FIXTURES / "sample.ofx").read_text())
    assert result.format_name == "ofx"
    assert result.confidence == 1.0


def test_refuses_to_guess_on_unknown_format() -> None:
    result = detect_format("Foo,Bar,Baz\n1,2,3\n")
    assert result.format_name is None
    assert result.confidence < CONFIDENCE_THRESHOLD
    assert result.headers_seen == ["Foo", "Bar", "Baz"]


def test_refuses_to_guess_on_empty_input() -> None:
    result = detect_format("")
    assert result.format_name is None
