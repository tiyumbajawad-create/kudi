"""Smoke test for M0: verifies the package imports and the canonical
schema round-trips. Keeps CI green from the very first commit."""

from datetime import date, datetime
from decimal import Decimal

from kudi.schema import IngestReport, Transaction


def test_transaction_round_trip() -> None:
    txn = Transaction(
        txn_id="abc123",
        account_id="chase-checking",
        posted_date=date(2026, 1, 15),
        amount=Decimal("-4.75"),
        raw_description="SQ *BLUE BOTTLE COF 4155551234 CA",
        source_format="chase_csv",
        ingested_at=datetime(2026, 1, 16, 9, 0, 0),
    )
    dumped = txn.model_dump()
    restored = Transaction.model_validate(dumped)
    assert restored == txn


def test_ingest_report_defaults() -> None:
    report = IngestReport(source_file="chase_export.csv")
    assert report.rows_parsed == 0
    assert report.failures == []
