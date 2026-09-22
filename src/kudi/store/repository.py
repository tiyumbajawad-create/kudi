"""All database queries live here (design doc §2) -- nothing outside
this module issues SQL directly."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from kudi.schema import Transaction
from kudi.store.db import CorrectionORM, TransactionORM


def upsert_transactions(session: Session, transactions: list[Transaction]) -> tuple[int, int]:
    """Insert new transactions, silently skip ones whose txn_id already
    exists (design doc §4.4: re-ingesting the same file, or an
    overlapping export, is a no-op). Returns (inserted, already_present).
    """
    if not transactions:
        return (0, 0)

    existing_ids = set(
        session.scalars(
            select(TransactionORM.txn_id).where(
                TransactionORM.txn_id.in_([t.txn_id for t in transactions])
            )
        )
    )

    rows = [
        {
            "txn_id": t.txn_id,
            "account_id": t.account_id,
            "posted_date": t.posted_date,
            "amount": t.amount,
            "currency": t.currency,
            "raw_description": t.raw_description,
            "merchant_norm": t.merchant_norm,
            "category": t.category,
            "category_source": t.category_source,
            "category_confidence": t.category_confidence,
            "is_recurring": t.is_recurring,
            "recurring_group_id": t.recurring_group_id,
            "anomaly_score": t.anomaly_score,
            "anomaly_reasons_json": json.dumps(t.anomaly_reasons),
            "source_format": t.source_format,
            "ingested_at": t.ingested_at,
        }
        for t in transactions
        if t.txn_id not in existing_ids
    ]

    if rows:
        stmt = (
            sqlite_insert(TransactionORM)
            .values(rows)
            .on_conflict_do_nothing(index_elements=["txn_id"])
        )
        session.execute(stmt)
        session.commit()

    return (len(rows), len(transactions) - len(rows))


def get_transactions(
    session: Session,
    account_id: str | None = None,
    min_anomaly_score: float | None = None,
) -> list[TransactionORM]:
    stmt = select(TransactionORM)
    if account_id is not None:
        stmt = stmt.where(TransactionORM.account_id == account_id)
    if min_anomaly_score is not None:
        stmt = stmt.where(TransactionORM.anomaly_score >= min_anomaly_score)
    stmt = stmt.order_by(TransactionORM.posted_date)
    return list(session.scalars(stmt))


def count_transactions(session: Session, account_id: str | None = None) -> int:
    return len(get_transactions(session, account_id=account_id))


def bulk_update_enrichment(session: Session, updates: list[dict[str, Any]]) -> int:
    """Writes categorization/recurring/anomaly results back onto
    already-persisted rows. `updates` is a list of dicts, each keyed by
    `txn_id` plus whichever enrichment fields it's setting.

    A transaction whose current category_source is "user" is never
    overwritten here (design doc §3: 'category_source="user" never
    gets clobbered') -- corrections always win over a later rule/model
    pass, even a re-run of the same pipeline.
    """
    if not updates:
        return 0

    txn_ids = [u["txn_id"] for u in updates]
    existing = {
        row.txn_id: row.category_source
        for row in session.scalars(select(TransactionORM).where(TransactionORM.txn_id.in_(txn_ids)))
    }

    n_updated = 0
    for update in updates:
        txn_id = update["txn_id"]
        if existing.get(txn_id) == "user":
            continue  # user corrections are never overwritten
        row = session.get(TransactionORM, txn_id)
        if row is None:
            continue
        for key, value in update.items():
            if key == "txn_id":
                continue
            if key == "anomaly_reasons":
                row.anomaly_reasons_json = json.dumps(value)
            else:
                setattr(row, key, value)
        n_updated += 1

    session.commit()
    return n_updated


def upsert_correction(session: Session, merchant_norm: str, category: str) -> None:
    """Records a user correction (design doc §3, §9): future ingests
    and re-enrichment runs use this to override rules/model output for
    this merchant, permanently, until the user changes it again."""
    stmt = (
        sqlite_insert(CorrectionORM)
        .values(
            merchant_norm=merchant_norm,
            category=category,
            corrected_at=datetime.now(UTC),
        )
        .on_conflict_do_update(
            index_elements=["merchant_norm"],
            set_={"category": category, "corrected_at": datetime.now(UTC)},
        )
    )
    session.execute(stmt)
    session.commit()


def get_corrections(session: Session) -> dict[str, str]:
    rows = session.scalars(select(CorrectionORM))
    return {row.merchant_norm: row.category for row in rows}


def ack_anomaly(session: Session, txn_id: str) -> bool:
    """Suppresses re-alerting for this transaction (design doc §6.4).
    Returns False if the txn_id doesn't exist."""
    row = session.get(TransactionORM, txn_id)
    if row is None:
        return False
    row.anomaly_acked = True
    session.commit()
    return True


def get_anomalies(
    session: Session,
    account_id: str | None = None,
    min_score: float = 0.0,
    include_acked: bool = False,
    limit: int = 50,
) -> list[TransactionORM]:
    stmt = select(TransactionORM).where(TransactionORM.anomaly_score >= min_score)
    if account_id is not None:
        stmt = stmt.where(TransactionORM.account_id == account_id)
    if not include_acked:
        stmt = stmt.where(TransactionORM.anomaly_acked.is_(False))
    stmt = stmt.order_by(TransactionORM.anomaly_score.desc()).limit(limit)
    return list(session.scalars(stmt))
