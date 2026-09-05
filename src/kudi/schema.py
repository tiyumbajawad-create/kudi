"""Canonical data model. Every downstream component consumes only this."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field


class Transaction(BaseModel):
    """A single normalized bank transaction.

    All ingestion parsers converge on this schema (see design doc §3-4).
    """

    txn_id: str
    account_id: str
    posted_date: date
    amount: Decimal
    currency: str = "USD"
    raw_description: str

    merchant_norm: str | None = None
    category: str | None = None
    category_source: Literal["rule", "model", "user"] | None = None
    category_confidence: float | None = None

    is_recurring: bool = False
    recurring_group_id: str | None = None

    anomaly_score: float | None = None
    anomaly_reasons: list[str] = Field(default_factory=list)

    source_format: str
    ingested_at: datetime


class IngestReport(BaseModel):
    """Per-file ingestion summary. Ingestion is never a black box (§4.5)."""

    source_file: str
    rows_parsed: int = 0
    rows_skipped: int = 0
    rows_deduped: int = 0
    rows_failed: int = 0
    failures: list[str] = Field(default_factory=list)
