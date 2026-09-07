"""Integration tests for the full pipeline (design doc §4.4, M2 DoD:
'overlapping re-ingest is a no-op'). Covers the three cases §4.4 calls
out explicitly: exact re-ingest, overlapping export windows, and
same-day duplicate transactions that must NOT collide.
"""

from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from kudi.ingest.pipeline import ingest_file
from kudi.schema import Transaction
from kudi.store.db import create_all, get_engine, get_session_factory
from kudi.store.repository import count_transactions, upsert_transactions

FIXTURES = Path(__file__).parent / "fixtures" / "ingest"


@pytest.fixture
def session(tmp_path) -> Session:
    engine = get_engine(str(tmp_path / "test.db"))
    create_all(engine)
    factory = get_session_factory(engine)
    s = factory()
    yield s
    s.close()


def test_exact_reingest_is_a_noop(session: Session) -> None:
    path = FIXTURES / "chase_sample.csv"
    first = ingest_file(path, session, account_id="chase-checking")
    second = ingest_file(path, session, account_id="chase-checking")

    assert first.rows_deduped == 0
    assert second.rows_parsed == first.rows_parsed
    assert second.rows_deduped == first.rows_parsed
    assert count_transactions(session, "chase-checking") == first.rows_parsed


def test_overlapping_export_windows_are_a_noop_for_shared_rows(tmp_path, session: Session) -> None:
    """A 30-day export and a 90-day export of the same account: the rows
    they share must not be double-counted."""
    wide = (
        "Details,Posting Date,Description,Amount,Type\n"
        'DEBIT,01/01/2026,"Coffee Shop",-4.00,DEBIT\n'
        'DEBIT,01/15/2026,"Grocery Store",-50.00,DEBIT\n'
        'DEBIT,02/01/2026,"Gas Station",-30.00,DEBIT\n'
    )
    narrow = (
        "Details,Posting Date,Description,Amount,Type\n"
        'DEBIT,01/15/2026,"Grocery Store",-50.00,DEBIT\n'
        'DEBIT,02/01/2026,"Gas Station",-30.00,DEBIT\n'
    )
    wide_path = tmp_path / "wide.csv"
    narrow_path = tmp_path / "narrow.csv"
    wide_path.write_text(wide)
    narrow_path.write_text(narrow)

    r1 = ingest_file(wide_path, session, account_id="chase-checking")
    r2 = ingest_file(narrow_path, session, account_id="chase-checking")

    assert r1.rows_parsed == 3
    assert r1.rows_deduped == 0
    assert r2.rows_parsed == 2
    assert r2.rows_deduped == 2, "both overlapping rows must be recognized as already present"
    assert count_transactions(session, "chase-checking") == 3


def test_same_day_identical_transactions_do_not_collide(tmp_path, session: Session) -> None:
    """Two legitimately identical transactions on the same day (two
    identical coffees) must both be kept, not deduped into one."""
    text = (
        "Details,Posting Date,Description,Amount,Type\n"
        'DEBIT,01/15/2026,"Blue Bottle Coffee",-4.75,DEBIT\n'
        'DEBIT,01/15/2026,"Blue Bottle Coffee",-4.75,DEBIT\n'
    )
    path = tmp_path / "same_day.csv"
    path.write_text(text)

    report = ingest_file(path, session, account_id="chase-checking")

    assert report.rows_parsed == 2
    assert report.rows_deduped == 0
    assert count_transactions(session, "chase-checking") == 2


def test_cross_format_same_underlying_ledger_dedupes(session: Session) -> None:
    """chase_sample.csv and boa_sample.csv encode the same two
    transactions in two different file formats -- ingesting both must
    not double the row count."""
    r1 = ingest_file(FIXTURES / "chase_sample.csv", session, account_id="chase-checking")
    r2 = ingest_file(FIXTURES / "boa_sample.csv", session, account_id="chase-checking")

    assert r1.rows_deduped == 0
    assert r2.rows_deduped == r2.rows_parsed
    assert count_transactions(session, "chase-checking") == r1.rows_parsed


def test_unrecognized_format_fails_loudly_not_silently(session: Session, tmp_path) -> None:
    path = tmp_path / "mystery.csv"
    path.write_text("Foo,Bar,Baz\n1,2,3\n")

    report = ingest_file(path, session, account_id="some-account")

    assert report.rows_parsed == 0
    assert report.rows_failed == 1
    assert "confidence" in report.failures[0]
    assert count_transactions(session, "some-account") == 0


def test_csv_format_without_account_id_fails_per_row(session: Session) -> None:
    """CSV formats don't self-identify their account; omitting
    account_id should fail those rows explicitly, not silently drop or
    misattribute them."""
    report = ingest_file(FIXTURES / "chase_sample.csv", session, account_id=None)
    assert report.rows_failed == 2
    assert all("account_id" in f for f in report.failures)
    assert count_transactions(session) == 0


def test_ofx_self_identifies_account_without_explicit_param(session: Session) -> None:
    report = ingest_file(FIXTURES / "sample.ofx", session, account_id=None)
    assert report.rows_parsed == 2
    assert report.rows_failed == 0
    assert count_transactions(session, "chase-checking") == 2


def test_upsert_preserves_existing_rows_on_conflict(session: Session) -> None:
    """Re-upserting a txn_id that already exists must not error or
    duplicate -- upsert on txn_id, never blind insert (§4.4)."""
    txn = Transaction(
        txn_id="fixed-id-1",
        account_id="acct",
        posted_date=date(2026, 1, 1),
        amount=Decimal("-10.00"),
        raw_description="Test Merchant",
        source_format="chase_csv",
        ingested_at=datetime.now(UTC),
    )
    inserted1, present1 = upsert_transactions(session, [txn])
    inserted2, present2 = upsert_transactions(session, [txn])

    assert (inserted1, present1) == (1, 0)
    assert (inserted2, present2) == (0, 1)
    assert count_transactions(session, "acct") == 1
