"""FastAPI serving layer (design doc §9). Every endpoint calls the same
service-layer functions the CLI uses -- no logic is duplicated between
the two entry points.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Generator
from datetime import date
from decimal import Decimal
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from kudi import service
from kudi.schema import IngestReport
from kudi.store.db import TransactionORM, create_all, get_engine, get_session_factory
from kudi.store.repository import (
    ack_anomaly,
    get_anomalies,
    get_transactions,
    upsert_correction,
)

DB_PATH = os.environ.get("KUDI_DB_PATH", "kudi.db")

app = FastAPI(
    title="Kudi API",
    version=pkg_version("kudi"),
    description="Transaction categorization, anomaly detection, and recurring-charge inference.",
)


def get_session() -> Generator[Session, None, None]:
    engine = get_engine(DB_PATH)
    create_all(engine)
    session = get_session_factory(engine)()
    try:
        yield session
    finally:
        session.close()


class TransactionOut(BaseModel):
    txn_id: str
    account_id: str
    posted_date: date
    amount: Decimal
    raw_description: str
    merchant_norm: str | None = None
    category: str | None = None
    category_source: str | None = None
    category_confidence: float | None = None
    is_recurring: bool = False
    recurring_group_id: str | None = None
    anomaly_score: float | None = None
    anomaly_reasons: list[str] = []

    @classmethod
    def from_orm_row(cls, row: TransactionORM) -> TransactionOut:
        return cls(
            txn_id=row.txn_id,
            account_id=row.account_id,
            posted_date=row.posted_date,
            amount=row.amount,
            raw_description=row.raw_description,
            merchant_norm=row.merchant_norm,
            category=row.category,
            category_source=row.category_source,
            category_confidence=row.category_confidence,
            is_recurring=row.is_recurring,
            recurring_group_id=row.recurring_group_id,
            anomaly_score=row.anomaly_score,
            anomaly_reasons=json.loads(row.anomaly_reasons_json),
        )


class CategoryCorrection(BaseModel):
    category: str


class RecurringGroupOut(BaseModel):
    merchant_norm: str
    period_name: str
    monthly_equivalent: float
    annualized_cost: float
    confidence: float


class PriceHikeOut(BaseModel):
    merchant_norm: str
    old_amount: float
    new_amount: float
    pct_increase: float
    detected_date: date


class MissedChargeOut(BaseModel):
    merchant_norm: str
    expected_date: date
    days_overdue: int


class DuplicateBillingOut(BaseModel):
    merchant_norm: str
    first_date: date
    second_date: date
    amount: float


class RecurringResponse(BaseModel):
    audit: list[RecurringGroupOut]
    price_hikes: list[PriceHikeOut]
    missed_charges: list[MissedChargeOut]
    duplicate_billing: list[DuplicateBillingOut]


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/version")
def get_version() -> dict[str, Any]:
    model_path = service.MODELS_DIR / "categorizer.joblib"
    return {
        "version": pkg_version("kudi"),
        "categorizer_model": "loaded" if model_path.exists() else "not found (rules-only)",
    }


@app.post("/ingest")
async def ingest(
    files: list[UploadFile],
    account_id: str | None = None,
    session: Session = Depends(get_session),
) -> list[IngestReport]:
    """Multipart file upload -> IngestReport per file, then runs the
    full enrichment pipeline across every account in the store.

    Without an explicit `account_id`, it's inferred from each upload's
    *original* filename stem (e.g. `chase-checking.csv` ->
    `chase-checking`) -- not the temp file's randomly-generated name,
    which carries no account information at all.
    """
    tmp_paths: list[Path] = []
    overrides: dict[Path, str] = {}
    try:
        for f in files:
            original_name = f.filename or "upload"
            suffix = Path(original_name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(await f.read())
                tmp_path = Path(tmp.name)
                tmp_paths.append(tmp_path)
                overrides[tmp_path] = account_id or Path(original_name).stem

        reports = service.ingest_paths(session, tmp_paths, overrides)
        service.enrich_all(session)
        return reports
    finally:
        for p in tmp_paths:
            p.unlink(missing_ok=True)


@app.get("/transactions")
def list_transactions(
    account_id: str | None = None,
    category: str | None = None,
    min_anomaly_score: float | None = None,
    limit: int = 100,
    offset: int = 0,
    session: Session = Depends(get_session),
) -> list[TransactionOut]:
    rows = get_transactions(session, account_id=account_id, min_anomaly_score=min_anomaly_score)
    if category is not None:
        rows = [r for r in rows if r.category == category]
    page = rows[offset : offset + limit]
    return [TransactionOut.from_orm_row(r) for r in page]


@app.post("/transactions/{txn_id}/category")
def correct_category(
    txn_id: str, correction: CategoryCorrection, session: Session = Depends(get_session)
) -> TransactionOut:
    row = session.get(TransactionORM, txn_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"transaction {txn_id!r} not found")
    if not row.merchant_norm:
        raise HTTPException(
            status_code=422, detail="transaction has no merchant_norm; re-ingest first"
        )
    upsert_correction(session, row.merchant_norm, correction.category)
    service.enrich_all(session)
    session.refresh(row)
    return TransactionOut.from_orm_row(row)


@app.get("/anomalies")
def list_anomalies(
    account_id: str | None = None,
    min_score: float = 0.9,
    limit: int = 50,
    session: Session = Depends(get_session),
) -> list[TransactionOut]:
    rows = get_anomalies(session, account_id=account_id, min_score=min_score, limit=limit)
    return [TransactionOut.from_orm_row(r) for r in rows]


@app.post("/anomalies/{txn_id}/ack")
def ack(txn_id: str, session: Session = Depends(get_session)) -> dict[str, bool]:
    ok = ack_anomaly(session, txn_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"transaction {txn_id!r} not found")
    return {"acked": True}


@app.get("/recurring")
def recurring(account_id: str, session: Session = Depends(get_session)) -> RecurringResponse:
    alerts = service.get_recurring_alerts(session, account_id)
    return RecurringResponse(
        audit=[
            RecurringGroupOut(
                merchant_norm=r.merchant_norm,
                period_name=r.period_name,
                monthly_equivalent=r.monthly_equivalent,
                annualized_cost=r.annualized_cost,
                confidence=r.confidence,
            )
            for r in alerts["audit"]
        ],
        price_hikes=[
            PriceHikeOut(
                merchant_norm=h.merchant_norm,
                old_amount=h.old_amount,
                new_amount=h.new_amount,
                pct_increase=h.pct_increase,
                detected_date=h.detected_date,
            )
            for h in alerts["price_hikes"]
        ],
        missed_charges=[
            MissedChargeOut(
                merchant_norm=m.merchant_norm,
                expected_date=m.expected_date,
                days_overdue=m.days_overdue,
            )
            for m in alerts["missed_charges"]
        ],
        duplicate_billing=[
            DuplicateBillingOut(
                merchant_norm=d.merchant_norm,
                first_date=d.first_date,
                second_date=d.second_date,
                amount=d.amount,
            )
            for d in alerts["duplicate_billing"]
        ],
    )


@app.get("/insights/summary")
def insights_summary(
    account_id: str, session: Session = Depends(get_session)
) -> dict[str, dict[str, float]]:
    return service.monthly_summary(session, account_id)
