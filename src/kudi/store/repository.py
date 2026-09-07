"""All database queries live here (design doc §2) -- nothing outside
this module issues SQL directly."""

from __future__ import annotations

import json

from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from kudi.schema import Transaction
from kudi.store.db import TransactionORM


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
