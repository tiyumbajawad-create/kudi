"""Tests for the shared service layer (§9): the single place both the
CLI and API pull enrichment logic from."""

from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from kudi import service
from kudi.store.db import create_all, get_engine, get_session_factory
from kudi.store.repository import get_corrections, get_transactions, upsert_correction

FIXTURES = Path(__file__).parent / "fixtures" / "ingest"


@pytest.fixture
def session(tmp_path) -> Session:
    engine = get_engine(str(tmp_path / "test.db"))
    create_all(engine)
    s = get_session_factory(engine)()
    yield s
    s.close()


def test_ingest_paths_infers_account_from_filename_stem(session: Session) -> None:
    reports = service.ingest_paths(session, [FIXTURES / "chase_sample.csv"])
    assert len(reports) == 1
    assert reports[0].rows_parsed == 2
    rows = get_transactions(session, account_id="chase_sample")
    assert len(rows) == 2


def test_enrich_all_populates_category_and_merchant_norm(session: Session) -> None:
    path = FIXTURES / "chase_sample.csv"
    service.ingest_paths(session, [path], {path: "acct"})
    summary = service.enrich_all(session)
    assert summary["transactions"] == 2

    rows = get_transactions(session, account_id="acct")
    payroll = next(r for r in rows if "PAYROLL" in r.raw_description)
    assert payroll.category == "Income>Salary"
    assert payroll.merchant_norm == "Acme Corp Payroll"


def test_enrich_all_is_idempotent(session: Session) -> None:
    path = FIXTURES / "chase_sample.csv"
    service.ingest_paths(session, [path], {path: "acct"})
    service.enrich_all(session)
    first = {r.txn_id: r.category for r in get_transactions(session, account_id="acct")}

    service.enrich_all(session)
    second = {r.txn_id: r.category for r in get_transactions(session, account_id="acct")}
    assert first == second


def test_enrich_all_respects_existing_user_corrections(session: Session) -> None:
    path = FIXTURES / "chase_sample.csv"
    service.ingest_paths(session, [path], {path: "acct"})
    service.enrich_all(session)

    rows = get_transactions(session, account_id="acct")
    payroll = next(r for r in rows if "PAYROLL" in r.raw_description)
    upsert_correction(session, payroll.merchant_norm, "Other>Uncategorized")

    service.enrich_all(session)
    rows2 = get_transactions(session, account_id="acct")
    payroll2 = next(r for r in rows2 if "PAYROLL" in r.raw_description)
    assert payroll2.category == "Other>Uncategorized"
    assert payroll2.category_source == "user"


def test_enrich_all_on_empty_store_returns_zeros(session: Session) -> None:
    summary = service.enrich_all(session)
    assert summary == {"accounts": 0, "transactions": 0, "transfers": 0, "recurring_groups": 0}


def test_monthly_summary_excludes_transfers(session: Session) -> None:
    path = FIXTURES / "chase_sample.csv"
    service.ingest_paths(session, [path], {path: "acct"})
    service.enrich_all(session)
    summary = service.monthly_summary(session, "acct")
    for month_totals in summary.values():
        assert not any(cat.startswith("Transfers>") for cat in month_totals)


def test_get_recurring_alerts_on_empty_account_returns_empty_shape(session: Session) -> None:
    alerts = service.get_recurring_alerts(session, "nonexistent-account")
    assert alerts == {"audit": [], "price_hikes": [], "missed_charges": [], "duplicate_billing": []}


def test_load_categorizer_includes_current_corrections(session: Session) -> None:
    upsert_correction(session, "Kroger", "Shopping>General Merchandise")
    categorizer = service.load_categorizer(session)
    assert categorizer.corrections.get("Kroger") == "Shopping>General Merchandise"
    assert get_corrections(session) == {"Kroger": "Shopping>General Merchandise"}
