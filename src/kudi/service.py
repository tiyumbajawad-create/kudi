"""Service layer (design doc §9): the single place that turns raw
ingested rows into fully-enriched ones -- categorized, transfer-tagged,
recurring-tagged, anomaly-scored. Both the CLI and the API call these
functions; neither duplicates this logic.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from kudi.detect.alerts import (
    DuplicateBillingAlert,
    MissedChargeAlert,
    PriceHikeAlert,
    SubscriptionAuditRow,
    detect_duplicate_billing,
    detect_missed_charge,
    detect_price_hikes,
    subscription_audit,
)
from kudi.detect.anomaly import detect_anomalies
from kudi.detect.recurring import RecurringGroupResult, infer_recurring_groups
from kudi.enrich.categorize import Categorizer
from kudi.enrich.classifier import load_model
from kudi.enrich.rules import RuleTable
from kudi.enrich.transfers import TRANSFER_CATEGORY, TransferCandidate, detect_transfer_pairs
from kudi.ingest.pipeline import ingest_file
from kudi.schema import IngestReport
from kudi.store.db import TransactionORM
from kudi.store.repository import bulk_update_enrichment, get_corrections, get_transactions

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
DEFAULT_CONFIDENCE_THRESHOLD = 0.6


def load_categorizer(session: Session) -> Categorizer:
    rules = RuleTable.load()
    model_path = MODELS_DIR / "categorizer.joblib"
    model = load_model(model_path) if model_path.exists() else None
    corrections = get_corrections(session)
    return Categorizer(
        rules=rules,
        model=model,
        confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
        corrections=corrections,
    )


def ingest_paths(
    session: Session, paths: list[Path], account_overrides: dict[Path, str] | None = None
) -> list[IngestReport]:
    """Ingests files. Without an override, the account is inferred from
    the filename stem (e.g. `chase-checking.csv` -> `chase-checking`),
    matching the generator's own naming convention -- OFX files
    self-identify their account regardless."""
    reports = []
    for path in paths:
        account_id = (account_overrides or {}).get(path) or path.stem
        reports.append(ingest_file(path, session, account_id=account_id))
    return reports


def _rows_to_indexed_df(rows: list[TransactionORM]) -> pd.DataFrame:
    df = pd.DataFrame(
        [
            {
                "txn_id": r.txn_id,
                "account_id": r.account_id,
                "posted_date": r.posted_date,
                "amount": float(r.amount),
                "raw_description": r.raw_description,
            }
            for r in rows
        ]
    )
    return df.set_index("txn_id", drop=False)


def enrich_all(session: Session) -> dict[str, int]:
    """Runs the full enrichment pipeline across every account in the
    store: transfer detection (needs cross-account visibility) ->
    categorization -> recurring detection -> anomaly detection ->
    write-back. Returns a small summary dict for reporting.

    Safe to re-run: user corrections (category_source="user") are
    never overwritten (design doc §3), and re-running produces the
    same enrichment for everything else since every stage here is
    deterministic given the same data.
    """
    all_rows = get_transactions(session)
    if not all_rows:
        return {"accounts": 0, "transactions": 0, "transfers": 0, "recurring_groups": 0}

    account_ids = sorted({r.account_id for r in all_rows})
    df = _rows_to_indexed_df(all_rows)

    # 1. transfer detection needs visibility across ALL accounts at once
    candidates = [
        TransferCandidate(r.txn_id, r.account_id, r.posted_date, r.amount, r.raw_description)
        for r in all_rows
    ]
    pairs = detect_transfer_pairs(candidates)
    transfer_ids = {tid for pair in pairs for tid in pair}

    categorizer = load_categorizer(session)
    updates: list[dict[str, Any]] = []

    for txn_id, row in df.iterrows():
        result = categorizer.categorize_one(
            row["raw_description"], row["amount"], row["posted_date"]
        )
        if txn_id in transfer_ids:
            updates.append(
                {
                    "txn_id": txn_id,
                    "merchant_norm": result.merchant_norm,
                    "category": TRANSFER_CATEGORY,
                    "category_source": "rule",
                    "category_confidence": 1.0,
                }
            )
        else:
            updates.append(
                {
                    "txn_id": txn_id,
                    "merchant_norm": result.merchant_norm,
                    "category": result.category,
                    "category_source": result.category_source,
                    "category_confidence": result.category_confidence,
                }
            )

    df["merchant_norm"] = df["txn_id"].map({u["txn_id"]: u["merchant_norm"] for u in updates})
    df["category"] = df["txn_id"].map({u["txn_id"]: u["category"] for u in updates})

    # 2. recurring + anomaly detection run per-account, excluding transfers
    recurring_updates_by_id: dict[str, dict[str, Any]] = {}
    anomaly_updates_by_id: dict[str, dict[str, Any]] = {}
    n_recurring_groups = 0

    for account_id in account_ids:
        acct_df = df[(df["account_id"] == account_id) & (~df["txn_id"].isin(transfer_ids))]
        if acct_df.empty:
            continue

        recurring_input = acct_df.reset_index(drop=True)
        groups = infer_recurring_groups(recurring_input)
        n_recurring_groups += len(groups)
        known_recurring_merchants = {g.merchant_norm for g in groups}

        for g in groups:
            for pos in g.transaction_indices:
                txn_id = str(recurring_input.loc[pos, "txn_id"])
                recurring_updates_by_id[txn_id] = {
                    "is_recurring": True,
                    "recurring_group_id": f"{account_id}:{g.merchant_norm}",
                }

        anomaly_input = acct_df.assign(category=acct_df["category"].fillna("Other>Uncategorized"))
        anomaly_input = anomaly_input.set_index("txn_id", drop=False)
        anomaly_result = detect_anomalies(
            anomaly_input, known_recurring_merchants=known_recurring_merchants
        )
        for txn_id_raw, score in anomaly_result.scores.items():
            txn_id = str(txn_id_raw)
            anomaly_updates_by_id[txn_id] = {
                "anomaly_score": float(score),
                "anomaly_reasons": anomaly_result.reasons.get(txn_id_raw, []),
            }

    for u in updates:
        u.update(recurring_updates_by_id.get(u["txn_id"], {}))
        u.update(anomaly_updates_by_id.get(u["txn_id"], {}))

    n_updated = bulk_update_enrichment(session, updates)

    return {
        "accounts": len(account_ids),
        "transactions": n_updated,
        "transfers": len(transfer_ids),
        "recurring_groups": n_recurring_groups,
    }


def monthly_summary(session: Session, account_id: str) -> dict[str, dict[str, float]]:
    """Monthly spend totals by category, excluding transfers (they're
    not spend -- they're money moving between the user's own
    accounts). Shared by the CLI's `report` command and the API's
    `/insights/summary` endpoint so the two never drift apart.
    """
    rows = get_transactions(session, account_id=account_id)
    totals: dict[str, dict[str, float]] = {}
    for r in rows:
        if r.category is None or r.category.startswith("Transfers>"):
            continue
        month = r.posted_date.strftime("%Y-%m")
        totals.setdefault(month, {})
        totals[month][r.category] = totals[month].get(r.category, 0.0) + abs(float(r.amount))
    return totals


def get_recurring_groups_for_account(
    session: Session, account_id: str
) -> list[RecurringGroupResult]:
    rows = get_transactions(session, account_id=account_id)
    non_transfer = [r for r in rows if r.category != TRANSFER_CATEGORY]
    df = pd.DataFrame(
        [
            {
                "txn_id": r.txn_id,
                "account_id": r.account_id,
                "posted_date": r.posted_date,
                "amount": float(r.amount),
                "merchant_norm": r.merchant_norm or "",
            }
            for r in non_transfer
        ]
    )
    if df.empty:
        return []
    return infer_recurring_groups(df)


def get_recurring_alerts(
    session: Session, account_id: str, as_of: _dt.date | None = None
) -> dict[str, list[Any]]:
    """`as_of` defaults to the account's own most recent transaction
    date, not real wall-clock "today" -- this is batch/historical
    analysis, and a dataset that doesn't extend to the present (a
    demo, an old export, a closed account) shouldn't have every single
    recurring charge misreported as wildly overdue just because
    real-world time has moved on since the data was captured.
    """
    rows = get_transactions(session, account_id=account_id)
    non_transfer = [r for r in rows if r.category != TRANSFER_CATEGORY]
    df = pd.DataFrame(
        [
            {
                "txn_id": r.txn_id,
                "account_id": r.account_id,
                "posted_date": r.posted_date,
                "amount": float(r.amount),
                "merchant_norm": r.merchant_norm or "",
            }
            for r in non_transfer
        ]
    )
    if df.empty:
        return {"audit": [], "price_hikes": [], "missed_charges": [], "duplicate_billing": []}

    groups = infer_recurring_groups(df)
    as_of_date = as_of or df["posted_date"].max()

    price_hikes: list[PriceHikeAlert] = []
    missed_charges: list[MissedChargeAlert] = []
    duplicate_billing: list[DuplicateBillingAlert] = []

    for g in groups:
        price_hikes.extend(detect_price_hikes(g, df))
        duplicate_billing.extend(detect_duplicate_billing(g, df))
        missed = detect_missed_charge(g, as_of_date)
        if missed:
            missed_charges.append(missed)

    audit: list[SubscriptionAuditRow] = subscription_audit(groups, df)

    return {
        "audit": audit,
        "price_hikes": price_hikes,
        "missed_charges": missed_charges,
        "duplicate_billing": duplicate_billing,
    }
