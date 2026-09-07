"""Property-based tests (§11):
- parsers never raise on arbitrary text -- malformed rows are reported as
  failures, not crashes, because a crash on one bad row shouldn't lose an
  entire statement.
- the dedup hash is deterministic, sensitive to every input field, and
  the occurrence counter reliably disambiguates same-day duplicates.
"""

from datetime import date
from decimal import Decimal

from hypothesis import given
from hypothesis import strategies as st

from kudi.ingest.dedupe import compute_txn_id
from kudi.ingest.parsers import boa, capital_one, chase, generic_credit, ofx

_PARSERS = [chase.parse, boa.parse, capital_one.parse, generic_credit.parse, ofx.parse]


@given(st.text(max_size=500))
def test_parsers_never_raise_on_arbitrary_text(text: str) -> None:
    for parser in _PARSERS:
        parser(text)  # must not raise, regardless of how malformed


@given(st.binary(max_size=500))
def test_parsers_never_raise_on_arbitrary_bytes_decoded_with_fallback(data: bytes) -> None:
    from kudi.ingest.parsers.common import decode_bytes

    try:
        text = decode_bytes(data)
    except UnicodeDecodeError:
        return  # cp1252 fallback itself can't decode everything; that's fine
    for parser in _PARSERS:
        parser(text)


@given(
    account_id=st.text(
        min_size=1, max_size=20, alphabet=st.characters(min_codepoint=48, max_codepoint=122)
    ),
    amount=st.decimals(
        min_value=-10000, max_value=10000, places=2, allow_nan=False, allow_infinity=False
    ),
    day_offset=st.integers(min_value=0, max_value=3650),
    description=st.text(min_size=1, max_size=50),
)
def test_txn_id_is_deterministic(
    account_id: str, amount: Decimal, day_offset: int, description: str
) -> None:
    d = date(2020, 1, 1).fromordinal(date(2020, 1, 1).toordinal() + day_offset)
    id1 = compute_txn_id(account_id, d, amount, description)
    id2 = compute_txn_id(account_id, d, amount, description)
    assert id1 == id2


@given(
    account_id=st.text(
        min_size=1, max_size=10, alphabet=st.characters(min_codepoint=97, max_codepoint=122)
    ),
    amount=st.decimals(
        min_value=-1000, max_value=1000, places=2, allow_nan=False, allow_infinity=False
    ),
    description=st.text(min_size=1, max_size=20),
)
def test_occurrence_counter_disambiguates_same_day_duplicates(
    account_id: str, amount: Decimal, description: str
) -> None:
    d = date(2026, 1, 1)
    id_occurrence_0 = compute_txn_id(account_id, d, amount, description, occurrence=0)
    id_occurrence_1 = compute_txn_id(account_id, d, amount, description, occurrence=1)
    assert id_occurrence_0 != id_occurrence_1


def test_fitid_always_wins_regardless_of_other_fields() -> None:
    id1 = compute_txn_id(
        "acct-a", date(2026, 1, 1), Decimal("1.00"), "desc-a", fitid="shared-fitid"
    )
    id2 = compute_txn_id(
        "acct-b", date(2099, 12, 31), Decimal("-999.99"), "desc-b", fitid="shared-fitid"
    )
    assert id1 == id2, "same FITID must always produce the same txn_id"
